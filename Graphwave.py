import argparse
import time
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from torch_geometric.nn import GCNConv

from graph_data_loader_slide_LA import get_dataloader

# ==== import shared logic from main training script (pre_training_ztp) ====
from pre_training_ztp import (
    make_autocast,
    make_scaler,
    ztp_nll_vector,
    ztp_mean,
    safe_softplus_to_lambda,
    _last_step_edge_mask,   # mask for last timestep edges
    early_stopping,
    count_parameters
)

# ================== Global accel / TF32 ==================
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ================== GraphWaveNet model ==================
class GraphWaveNet(nn.Module):
    """
    Edge-level GraphWaveNet-style model:
    - Temporal gated dilated convolutions (causal TCN)
    - Single GCNConv over (B*T*N) flattened node space
    - Edge decoder predicting lambda (>0) for each edge at the LAST timestep
    """
    def __init__(
        self,
        input_dim: int,            # fused in_feat after POI emb + numeric features
        hidden_dim: int,
        num_nodes: int,
        kernel_size: int = 2,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        num_poi_types: int = 456,
        embed_dim: int = 16,
        emb_dropout: float = 0.08
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # POI embedding + numeric features -> fuse -> input_dim
        self.poi_emb  = nn.Embedding(num_poi_types + 1, embed_dim)
        self.emb_drop = nn.Dropout(p=emb_dropout)
        self.fuse     = nn.Linear(6 + embed_dim, input_dim)

        # Temporal gated dilated TCN stacks
        # Layer i: dilation = wztp_l1**i
        # First layer: input_dim -> hidden_dim
        # Later layers: hidden_dim -> hidden_dim
        self.temporal_convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = input_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            dilation = 2 ** i
            self.temporal_convs.append(
                nn.ModuleDict({
                    'filter':   nn.Conv1d(in_ch,  out_ch, kernel_size, dilation=dilation, padding=0),
                    'gate':     nn.Conv1d(in_ch,  out_ch, kernel_size, dilation=dilation, padding=0),
                    'residual': nn.Conv1d(in_ch,  out_ch, kernel_size=1),
                    'norm':     nn.LayerNorm(out_ch, eps=1e-3),
                })
            )

        # One GCNConv over flattened (B*T*N) node space
        self.gcn = GCNConv(hidden_dim, hidden_dim)

        # Edge decoder: [h_u, h_v, |h_u-h_v|] -> scalar log-rate -> lambda via softplus
        self.edge_out = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                last_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, N, F, T]  (F includes poi_type at dim=wztp0 channel)
        edge_index: [wztp_l1, E_total]  over flattened (B*T*N) node indices
        last_mask: [E_total] boolean mask for "last timestep edges".
                   If provided, we'll ONLY decode those edges.
        return: lambda (>0) per selected edge, shape [E_selected]
        """
        device = x.device
        B, N, F_in, T = x.shape

        # ----- fuse POI embedding + numeric features -----
        poi_idx = x[:, :, 1, :].long()                           # [B,N,T]
        poi_e   = self.poi_emb(poi_idx).permute(0, 1, 3, 2)      # [B,N,embed_dim,T]
        poi_e   = self.emb_drop(poi_e)

        numeric = torch.cat([x[:, :, :1, :], x[:, :, 2:, :]], dim=2)  # [B,N,6,T]
        fused   = torch.cat([numeric, poi_e], dim=2)                  # [B,N,6+embed,T]

        # shape -> [B,N,T,6+embed] -> [B*N, T, 6+embed]
        fused   = fused.permute(0, 1, 3, 2).reshape(-1, T, 6 + self.poi_emb.embedding_dim)
        xh      = self.fuse(fused).transpose(1, 2)                    # [B*N, input_dim, T]

        # ----- temporal gated conv stack -----
        skips = []
        for i, layer in enumerate(self.temporal_convs):
            res = layer['residual'](xh)  # [B*N, H, T] (or [B*N, hidden_dim, T] after broadcast)

            pad = (self.kernel_size - 1) * (2 ** i)
            x_pad = F.pad(xh, (pad, 0))  # causal padding on time dim

            filt = torch.tanh(layer['filter'](x_pad))
            gate = torch.sigmoid(layer['gate'](x_pad))
            xh = filt * gate                   # gated activation
            xh = xh[:, :, -res.size(-1):]      # crop back to T
            # norm across channel dimension via LayerNorm on [T, C] transpose trick
            xh = xh.transpose(1, 2)            # [B*N, T, H]
            xh = layer['norm'](xh)
            xh = xh.transpose(1, 2)            # [B*N, H, T]
            xh = self.dropout(xh)

            # residual connection (aligned in time)
            res_t = res[:, :, -xh.size(-1):]
            xh = xh + res_t

            skips.append(xh)

        # sum of skip connections
        xh = torch.stack(skips, dim=0).sum(dim=0)   # [B*N, H, T]
        # flatten all time+batch+nodes to node states in BTN space
        xh = xh.permute(0, 2, 1).reshape(-1, self.hidden_dim)  # [B*N*T, H]

        # ----- spatial graph conv over flattened BT node space -----
        xh = self.gcn(xh, edge_index)               # [B*N*T, H]
        xh = F.relu(xh, inplace=True)

        # ----- decode only last-step edges if mask provided -----
        if last_mask is not None:
            idx0 = edge_index[0][last_mask]
            idx1 = edge_index[1][last_mask]
        else:
            idx0 = edge_index[0]
            idx1 = edge_index[1]

        src = xh[idx0]                               # [E_sel, H]
        dst = xh[idx1]                               # [E_sel, H]
        e_feat = torch.cat([src, dst, torch.abs(src - dst)], dim=-1)  # [E_sel, 3H]

        raw = self.edge_out(e_feat).squeeze(-1)      # [E_sel]
        # GraphWaveNet baseline directly outputs λ>0
        lam = F.softplus(raw) + 1e-6
        return lam


# ================== train / eval loops (aligned) ==================
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    tail_alpha: float,
    w_max: float,
    weight_mode: str,
    tail_min_count: int,
    tail_max_count: int,
    amp_enabled: bool,
    amp_dtype,
):
    """
    Train one epoch:
    - only last timestep edges
    - loss = weighted ZTP NLL (same ztp_nll_vector from pre_training_ztp)
    - AMP usage same as main model
    - logs step-level train_loss_step_ztp
    """
    model.train()
    total_loss_sum, total_edges = 0.0, 0
    scaler = make_scaler(enabled_fp16=(amp_enabled and amp_dtype == torch.float16))

    for step, (x_batch, edge_index_batch, edge_attr_batch) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        x_batch          = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch  = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)
        if not last_mask.any():
            continue

        # forward under autocast (model only)
        with make_autocast(enabled=amp_enabled and (device.type == "cuda"), dtype=amp_dtype):
            lam_last = model(x_batch, edge_index_batch, last_mask=last_mask)  # [E_last], already λ>0

        y_true_last = edge_attr_batch[last_mask]  # [E_last]

        if lam_last.numel() == 0 or lam_last.numel() != y_true_last.numel():
            wandb.log({
                "mismatch_last_step": 1,
                "n_pred": int(lam_last.numel()),
                "n_tgt": int(y_true_last.numel())
            })
            continue

        # loss in FP32 via shared ztp_nll_vector (now unified with main model)
        loss_vec = ztp_nll_vector(
            y_true_last, lam_last,
            tail_alpha=tail_alpha,
            w_max=w_max,
            weight_mode=weight_mode,
            tail_min_count=tail_min_count,
            tail_max_count=tail_max_count,
            # discount_factor=0.5
        )
        loss_mean = loss_vec.mean()
        loss_sum  = loss_vec.sum().item()

        # backward with scaler for fp16, or normal for bf16/fp32
        if not amp_enabled or amp_dtype == torch.bfloat16:
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss_mean).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        n_edges = int(y_true_last.numel())
        total_loss_sum += loss_sum
        total_edges    += n_edges

        # step-level log with same key naming as main model
        wandb.log({"train_loss_step_ztp": loss_sum / max(1, n_edges)})

        del x_batch, edge_index_batch, edge_attr_batch, y_true_last, lam_last, loss_vec
        gc.collect()

    return total_loss_sum / max(1, total_edges)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    tail_alpha: float,
    w_max: float,
    weight_mode: str,
    tail_min_count: int,
    tail_max_count: int,
    amp_enabled: bool,
    amp_dtype,
):
    """
    Validation loop aligned with main script:
    - computes both unweighted and weighted NLL
    - logs val_loss_step_ztp / val_loss_step_wztp
    - returns (avg_nll_unweighted, avg_nll_weighted)
    """
    model.eval()
    total_edges = 0

    sum_nll_weighted = 0.0
    sum_nll_unweight = 0.0
    sum_mae_mu = 0.0
    sum_mse_mu = 0.0

    for step, (x_batch, edge_index_batch, edge_attr_batch) in enumerate(dataloader):
        x_batch          = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch  = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)
        if not last_mask.any():
            continue

        with make_autocast(enabled=amp_enabled and (device.type == "cuda"), dtype=amp_dtype):
            lam_last = model(x_batch, edge_index_batch, last_mask=last_mask)  # [E_last], λ>0

        y_true_last = edge_attr_batch[last_mask]  # [E_last]

        if lam_last.numel() == 0 or lam_last.numel() != y_true_last.numel():
            wandb.log({
                "mismatch_last_step": 1,
                "n_pred": int(lam_last.numel()),
                "n_tgt": int(y_true_last.numel())
            })
            continue

        # weighted
        nll_vec_weighted = ztp_nll_vector(
            y_true_last, lam_last,
            tail_alpha=tail_alpha,
            w_max=w_max,
            weight_mode=weight_mode,
            tail_min_count=tail_min_count,
            tail_max_count=tail_max_count,
        )

        # unweighted (baseline ZTP)
        nll_vec_unweight = ztp_nll_vector(
            y_true_last, lam_last,
            tail_alpha=0.0,
            w_max=0.0,
            weight_mode='none',
            tail_min_count=1,
            tail_max_count=300
        )

        n = int(y_true_last.numel())
        step_nll_unw = nll_vec_unweight.mean().item()
        step_nll_w   = nll_vec_weighted.mean().item()

        # step-level val logs with same keys
        wandb.log({
            "val_loss_step_ztp": step_nll_unw,
            "val_loss_step_wztp": step_nll_w
        })

        sum_nll_weighted += nll_vec_weighted.sum().item()
        sum_nll_unweight += nll_vec_unweight.sum().item()
        total_edges      += n

        # extra metrics for monitoring
        mu   = ztp_mean(lam_last)
        diff = (mu - y_true_last.float())
        sum_mae_mu += diff.abs().sum().item()
        sum_mse_mu += (diff.pow(2)).sum().item()

        del x_batch, edge_index_batch, edge_attr_batch, y_true_last, lam_last
        del nll_vec_weighted, nll_vec_unweight, mu, diff
        gc.collect()

    total_edges = max(1, total_edges)
    avg_nll_unw = sum_nll_unweight / total_edges
    avg_nll_w   = sum_nll_weighted / total_edges
    avg_mae_mu  = sum_mae_mu / total_edges
    avg_mse_mu  = sum_mse_mu / total_edges

    # epoch-level extra logs (optional, parallel to pre_training_ztp)
    wandb.log({
        "NLL_ZTP_weighted": avg_nll_w,
        "MAE_mu": avg_mae_mu,
        "MSE_mu": avg_mse_mu
    })

    return avg_nll_unw, avg_nll_w


# ================== Main loop (aligned with TCGCN / STGCN) ==================
def main():
    parser = argparse.ArgumentParser()

    # ----- model/data shape -----
    parser.add_argument("--input_dim", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_nodes_per_graph", type=int, default=38628)

    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)

    # ----- training -----
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience_lr", type=int, default=3)
    parser.add_argument("--patience_stop", type=int, default=8)

    # ----- POI embedding params -----
    parser.add_argument("--num_poi_types", type=int, default=456)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--embed_dropout", type=float, default=0.1)

    # ----- dirs -----
    parser.add_argument("--train_dir", type=str, default="./graph_data/LA/train")
    parser.add_argument("--val_dir", type=str, default="./graph_data/LA/val")
    parser.add_argument("--project", type=str, default="Paper1_GraphWaveNet_ZTP")

    # ----- tail weighting (must MATCH main script semantics) -----
    parser.add_argument("--tail_alpha", type=float, default=2.2)
    parser.add_argument("--w_max", type=float, default=4.5)
    parser.add_argument("--weight_mode", type=str, default="log", choices=["none", "log", "power"])
    parser.add_argument("--tail_min_count", type=int, default=3)
    parser.add_argument("--tail_max_count", type=int, default=8)

    # ----- AMP mode (same choices as others) -----
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["off", "fp16", "bf16", "auto"],
                        help="mixed precision: off/fp16/bf16/auto(bf16 if available else fp16)")

    args = parser.parse_args()

    # ====== AMP global config (replicate main script logic) ======
    global AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL
    if args.amp == 'off':
        AMP_ENABLED_GLOBAL = False
        AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    elif args.amp == 'fp16':
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.float16
    elif args.amp == 'bf16':
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.bfloat16
    else:
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # ====== wandb init ======
    wandb.init(project=args.project)
    wandb.config.update(vars(args) | {
        "amp_enabled": AMP_ENABLED_GLOBAL,
        "amp_dtype": str(AMP_DTYPE_GLOBAL).replace("torch.", "")
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== DataLoaders (same signature as main script) ======
    train_dataloader = get_dataloader(
        gpickle_dir=[args.train_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )
    val_dataloader = get_dataloader(
        gpickle_dir=[args.val_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )

    # ====== Model ======
    model = GraphWaveNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes_per_graph,
        kernel_size=2,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,
        emb_dropout=args.embed_dropout,
    ).to(device)

    print(f"GraphWaveNet trainable parameters: {count_parameters(model)}")

    # ====== optimizer param groups (match main script: decay / no_decay / emb no_decay) ======
    def _is_no_decay_param(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim < 2:
            return True
        ln = name.lower()
        return (
            ln.endswith(".bias")
            or "norm" in ln
            or "bn" in ln
            or "layernorm" in ln
            or "ln" in ln
        )

    decay_params, no_decay_params, emb_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        if "poi_emb" in ln or "poi_embedding" in ln:
            emb_params.append(p)        # embeddings: no weight decay
        elif _is_no_decay_param(n, p):
            no_decay_params.append(p)   # norm/bias: no decay
        else:
            decay_params.append(p)      # everything else: decay

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": args.weight_decay},
            {"params": no_decay_params,"weight_decay": 0.0},
            {"params": emb_params,     "weight_decay": 0.0},
        ],
        lr=args.learning_rate
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor,
        patience=args.patience_lr, verbose=True
    )

    # ====== checkpoint bookkeeping (aligned to main + STGCN) ======
    os.makedirs("./model/compare", exist_ok=True)

    eval_losses = []
    best_val_loss_w      = float('inf')   # best weighted val loss overall

    for epoch in range(args.num_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, device,
            tail_alpha=args.tail_alpha,
            w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL,
            amp_dtype=AMP_DTYPE_GLOBAL,
        )

        val_loss, val_loss_w = evaluate(
            model, val_dataloader, device,
            tail_alpha=args.tail_alpha,
            w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL,
            amp_dtype=AMP_DTYPE_GLOBAL,
        )
        eval_losses.append(val_loss)

        if val_loss_w < best_val_loss_w:
            best_val_loss_w = val_loss_w
            torch.save(model.state_dict(), "model/compare/gw_wztp.pth")
            print("Saved: ./model/compare/gw_wztp.pth")


        torch.cuda.empty_cache()
        gc.collect()

        # ----- epoch-level logs (align keys exactly with main script) -----
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_w": val_loss_w
        })
        print(f"Epoch {epoch + 1}/{args.num_epochs} | train {train_loss:.8f} | val {val_loss:.8f} | {time.time()-t0:.1f}s")

        # lr scheduling
        scheduler.step(val_loss_w)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Learning Rate: {current_lr}")

        # early stopping (same helper as main script)
        if early_stopping(args.patience_stop, eval_losses):
            torch.cuda.empty_cache()
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
