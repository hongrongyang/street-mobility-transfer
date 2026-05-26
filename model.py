import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn import RMSNorm
from torch_geometric.nn import GCNConv


def decode_in_chunks(decoder, memory, pad_mask, chunk=8192):
    # process edges in chunks to avoid OOM on large graphs
    T, Nbt, H = memory.shape
    outs = []
    for s in range(0, Nbt, chunk):
        e = min(s + chunk, Nbt)
        mem_chunk = memory[:, s:e, :]
        pm_chunk = pad_mask[s:e, :]

        tgt_valid = ~pm_chunk[:, -1]
        if not tgt_valid.any():
            continue

        mem_sub = mem_chunk[:, tgt_valid, :]
        tgt_sub = mem_sub[-1:, :, :].contiguous()
        pm_sub = pm_chunk[tgt_valid, :]

        out = decoder(
            memory=mem_sub, tgt=tgt_sub,
            tgt_mask=None,
            memory_key_padding_mask=pm_sub,
            tgt_key_padding_mask=None
        )
        outs.append(out[0])

    if len(outs) == 0:
        return memory.new_zeros((0, decoder.fc_edge_out.out_features))
    return torch.cat(outs, dim=0)


# ---------------------- Temporal encoding ----------------------

class TemporalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalEncoding, self).__init__()

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        encoding = encoding.unsqueeze(0).unsqueeze(0)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x, time_steps):
        enc = self.encoding[:, :, :time_steps, :].permute(0, 1, 3, 2)
        return x + enc


# ---------------------- Temporal conv ----------------------

class TemporalConv(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=256, dropout_rate=0.4, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size, padding=kernel_size // 2, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = RMSNorm(hidden_dim1)
        self.norm2 = RMSNorm(hidden_dim2)

    def forward(self, x):
        batch_size, num_nodes, input_dim, time_steps = x.shape
        x = x.reshape(batch_size * num_nodes, input_dim, time_steps)  # [B*N, C, T]

        x = self.conv1(x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.reshape(batch_size, num_nodes, -1, time_steps)  # [B, N, H, T]
        return x


# ---------------------- TC-GCN encoder ----------------------

class TCGCNEncoder(nn.Module):
    def __init__(self, input_dim, temporal_hidden_dim1, temporal_hidden_dim2, temporal_dropout_rate, kernel_size,
                 gcn_hidden_dim1, gcn_hidden_dim2, gcn_dropout_rate):
        super(TCGCNEncoder, self).__init__()
        self.temporal_conv = TemporalConv(input_dim=input_dim, hidden_dim1=temporal_hidden_dim1,
                                         hidden_dim2=temporal_hidden_dim2, dropout_rate=temporal_dropout_rate,
                                         kernel_size=kernel_size)
        self.gcn1 = GCNConv(temporal_hidden_dim2, gcn_hidden_dim1)
        self.gcn2 = GCNConv(gcn_hidden_dim1, gcn_hidden_dim2)
        self.fc_output = nn.Linear(3 * gcn_hidden_dim2, gcn_hidden_dim2)
        self.proj1 = nn.Linear(temporal_hidden_dim2, gcn_hidden_dim1)
        self.proj2 = nn.Linear(gcn_hidden_dim1, gcn_hidden_dim2)

        self.norm_in = RMSNorm(temporal_hidden_dim2)
        self.norm1 = RMSNorm(gcn_hidden_dim1)
        self.norm2 = RMSNorm(gcn_hidden_dim2)

        self.dropout = nn.Dropout(gcn_dropout_rate)

    def forward(self, x, edge_index):
        batch_size, num_nodes, input_dim, time_steps = x.shape
        x = self.temporal_conv(x)  # [B, N, H, T]

        # split edge_index by (batch, time) slice and pad to uniform size
        edge_indices = []
        max_num_edges = 0
        for b in range(batch_size):
            for t in range(time_steps):
                start_idx = (b * time_steps + t) * num_nodes
                end_idx = start_idx + num_nodes
                edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & \
                            (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
                edge_index_b = edge_index[:, edge_mask] - start_idx
                num_edges = edge_index_b.size(1)
                if num_edges > max_num_edges:
                    max_num_edges = num_edges
                edge_indices.append(edge_index_b)

        edge_outputs_padded = torch.zeros(
            batch_size, time_steps, max_num_edges, self.fc_output.out_features,
            dtype=x.dtype, device=x.device
        )
        edge_masks_padded = torch.zeros(
            batch_size, time_steps, max_num_edges,
            dtype=torch.bool, device=x.device
        )

        for b in range(batch_size):
            for t in range(time_steps):
                edge_index_b = edge_indices[b * time_steps + t]
                num_edges = edge_index_b.size(1)
                x_t = x[b, :, :, t]  # [N, H]

                # two-layer GCN with residual connections
                h = self.norm_in(x_t)
                x_t = self.proj1(x_t) + self.dropout(F.relu(self.gcn1(h, edge_index_b)))

                h = self.norm1(x_t)
                x_t = self.proj2(x_t) + self.dropout(F.relu(self.gcn2(h, edge_index_b)))

                # edge feature: concat(src, dst, |src-dst|)
                src = x_t[edge_index_b[0]]
                tgt = x_t[edge_index_b[1]]
                edge_features = self.fc_output(torch.cat([src, tgt, torch.abs(src - tgt)], dim=-1))

                edge_outputs_padded[b, t, :num_edges] = edge_features
                edge_masks_padded[b, t, :num_edges] = True

        # reshape to [T, B*E, H] for transformer input
        edge_outputs_padded = edge_outputs_padded.permute(1, 0, 2, 3)
        edge_outputs_padded = edge_outputs_padded.reshape(time_steps, -1, edge_outputs_padded.size(-1))
        edge_masks_padded = edge_masks_padded.permute(1, 0, 2).reshape(time_steps, -1).transpose(0, 1)

        return edge_outputs_padded, edge_masks_padded


# ---------------------- Transformer decoder ----------------------

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attention_dropout, ffn_dropout,
                 activation="relu", norm_first=False):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn_res1 = nn.Dropout(ffn_dropout)
        self.dropout_attn_res2 = nn.Dropout(ffn_dropout)
        self.dropout_ffn = nn.Dropout(ffn_dropout)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            # pre-norm: normalize before each sub-layer
            x = x + self.dropout_attn_res1(
                self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)[0])
            x = x + self.dropout_attn_res2(
                self.multihead_attn(self.norm2(x), memory, memory,
                                    attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0])
            x = x + self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm3(x)))))
        else:
            x = self.norm1(x + self.dropout_attn_res1(
                self.self_attn(x, x, x, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)[0]))
            x = self.norm2(x + self.dropout_attn_res2(
                self.multihead_attn(x, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]))
            x = self.norm3(x + self.linear2(self.dropout_ffn(self.activation(self.linear1(x)))))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, edge_output_dim,
                 decoder_dropout_rate, attention_dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                attention_dropout=attention_dropout_rate,
                ffn_dropout=decoder_dropout_rate,
                activation="relu", norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.fc_edge_out = nn.Linear(hidden_dim, edge_output_dim)

    def forward(self, memory, tgt, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory,
                      tgt_mask=tgt_mask,
                      memory_mask=None,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)

        if hasattr(self, "hook_store") and self.hook_store is not None:
            self.hook_store(x)

        return self.fc_edge_out(x)


# ---------------------- Full model ----------------------

class TCGCNTransformer(nn.Module):
    def __init__(self, input_dim, temporal_hidden_dim1, temporal_hidden_dim2, temporal_dropout_rate, kernel_size,
                 gcn_hidden_dim1, gcn_hidden_dim2, gcn_dropout_rate, decoder_hidden_dim, edge_output_dim, num_heads,
                 num_layers, decoder_dropout_rate, num_poi_types=456, embed_dim=8, emb_dropout=0,
                 attention_dropout_rate=0.1):
        super(TCGCNTransformer, self).__init__()

        self.poi_embedding = nn.Embedding(num_poi_types + 1, embed_dim)
        self.input_projection = nn.Linear(6 + embed_dim, input_dim)
        self.emb_drop = nn.Dropout(p=emb_dropout)
        self.temporal_encoding = TemporalEncoding(d_model=input_dim)
        self.tc_gcn = TCGCNEncoder(input_dim, temporal_hidden_dim1, temporal_hidden_dim2,
                                   temporal_dropout_rate, kernel_size,
                                   gcn_hidden_dim1, gcn_hidden_dim2, gcn_dropout_rate)
        self.transformer_decoder = TransformerDecoder(decoder_hidden_dim, num_heads, num_layers, edge_output_dim,
                                                      decoder_dropout_rate, attention_dropout_rate)
        self.edge_output_dim = edge_output_dim
        self.num_heads = num_heads
        self.eps = 1e-6
        self._debug_hidden_list = []

    def forward(self, x, edge_index, tgt_mask=None):
        """x: [B, N, F, T] — channel 1 is POI index; returns λ > 0 for last-step edges"""
        self._debug_hidden_list = []
        self.transformer_decoder.hook_store = lambda h: self._debug_hidden_list.append(h.detach().cpu())

        batch_size, num_nodes, _, time_steps = x.shape

        # fuse POI embedding with numeric features
        poi_idx = x[:, :, 1, :].long()
        other = torch.cat([x[:, :, 0:1, :], x[:, :, 2:, :]], dim=2)
        poi_embed = self.emb_drop(self.poi_embedding(poi_idx).permute(0, 1, 3, 2))
        x = torch.cat([other, poi_embed], dim=2)
        x = self.input_projection(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.temporal_encoding(x, time_steps)

        memory, pad_mask = self.tc_gcn(x, edge_index)  # [T, B*E, H], [B*E, T]
        pad_mask = ~pad_mask.bool()

        last = decode_in_chunks(self.transformer_decoder, memory, pad_mask, chunk=64512)
        lam = F.softplus(last.squeeze(-1)) + self.eps  # λ > 0

        self.transformer_decoder.hook_store = None
        return lam