import argparse
import time
import gc
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from graph_data_loader_slide_LA import get_dataloader
from pre_training_ztp import (
    make_autocast,
    make_scaler,
    ztp_nll_vector,
    ztp_mean,
    safe_softplus_to_lambda,
    _last_step_edge_mask,
    early_stopping,
    count_parameters,
)


try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------------- DCGRU Cell (Diffusion GRU) ----------------------
class DCGRUCell(nn.Module):
    """
    Simplified DCRNN cell:
    x_t: [B, N, Cx], h_prev: [B, N, H], eidx_bt: [wztp_l1, E_t*B]
    """
    def __init__(self, in_dim: int, hidden_dim: int, cheb_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # linear projections from input
        self.lin_x_r = nn.Linear(in_dim, hidden_dim)
        self.lin_x_z = nn.Linear(in_dim, hidden_dim)
        self.lin_x_n = nn.Linear(in_dim, hidden_dim)

        # graph convs on hidden
        self.cheb_h_r = ChebConv(hidden_dim, hidden_dim, K=cheb_k, normalization='sym')
        self.cheb_h_z = ChebConv(hidden_dim, hidden_dim, K=cheb_k, normalization='sym')
        self.cheb_h_n = ChebConv(hidden_dim, hidden_dim, K=cheb_k, normalization='sym')

        self.norm_r = nn.LayerNorm(hidden_dim)
        self.norm_z = nn.LayerNorm(hidden_dim)
        self.norm_n = nn.LayerNorm(hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, eidx_bt: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape

        xr = self.lin_x_r(x_t)  # [B,N,H]
        xz = self.lin_x_z(x_t)
        xn = self.lin_x_n(x_t)

        h_flat = h_prev.reshape(B * N, -1)  # [B*N,H]
        r_h = self.cheb_h_r(h_flat, eidx_bt)  # [B*N,H]
        z_h = self.cheb_h_z(h_flat, eidx_bt)
        n_h = self.cheb_h_n(h_flat, eidx_bt)

        r = torch.sigmoid(self.norm_r(xr + r_h.view(B, N, -1)))
        z = torch.sigmoid(self.norm_z(xz + z_h.view(B, N, -1)))
        n_hat = torch.tanh(self.norm_n(xn + n_h.view(B, N, -1)))

        h_t = (1 - z) * n_hat + z * h_prev
        return self.dropout(h_t)


# ---------------------- DCRNN Encoder ----------------------
class DCRNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1, cheb_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(in_dim if l == 0 else hidden_dim, hidden_dim,
                      cheb_k=cheb_k, dropout=dropout)
            for l in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor, eidx_bt_list: List[torch.Tensor], h0=None):
        """
        x_seq: [B, T, N, Cx]
        eidx_bt_list: list len T with [wztp_l1, E_t*B] each
        returns:
            H_all: [T, B, N, H]  (top-layer hidden at each t)
        """
        B, T, N, _ = x_seq.shape
        nl = len(self.layers)

        if h0 is None:
            h_states = [x_seq.new_zeros(B, N, self.layers[0].hidden_dim) for _ in range(nl)]
        else:
            h_states = h0

        outs = []
        for t in range(T):
            x_t = x_seq[:, t]       # [B,N,C]
            e_t = eidx_bt_list[t]   # [wztp_l1,E_t*B]
            for li, cell in enumerate(self.layers):
                h_states[li] = cell(x_t, h_states[li], e_t)
                x_t = h_states[li]
            outs.append(h_states[-1].unsqueeze(0))
        H_all = torch.cat(outs, dim=0)  # [T,B,N,H]
        return H_all, h_states


# ---------------------- DCRNNForEdges ----------------------
class DCRNNForEdges(nn.Module):
    """
    Pipeline:
      - split raw x into numeric(6) + poi index -> fuse -> in_feat
      - DCRNNEncoder over time
      - use embed_16 hidden state HT (last step) as node embedding
      - edge decoder: concat(src,dst,|src-dst|) -> raw -> softplus -> λ
      - return λ for all edges in BT graph (all timesteps' edges pooled),
        flattened as [E_total].
    """
    def __init__(self,
                 num_nodes: int,
                 in_feat: int,
                 num_poi_types: int = 456,
                 embed_dim: int = 16,
                 dcrnn_hidden: int = 256,
                 dcrnn_layers: int = 2,
                 cheb_k: int = 3,
                 dropout: float = 0.2,
                 emb_dropout: float = 0.08):
        super().__init__()
        self.N = num_nodes

        # POI emb + dropout
        self.poi_emb  = nn.Embedding(num_poi_types + 1, embed_dim)
        self.emb_drop = nn.Dropout(p=emb_dropout)

        # fuse 6 numeric + emb_dim -> in_feat
        self.fuse = nn.Linear(6 + embed_dim, in_feat)

        # DCRNN encoder
        self.encoder = DCRNNEncoder(
            in_dim=in_feat,
            hidden_dim=dcrnn_hidden,
            num_layers=dcrnn_layers,
            cheb_k=cheb_k,
            dropout=dropout
        )

        # edge decoder
        edge_in = 3 * dcrnn_hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, dcrnn_hidden),
            nn.GELU(),
            nn.Linear(dcrnn_hidden, 1)
        )

    @staticmethod
    def _split_edges_by_time(edge_index: torch.Tensor, B: int, T: int, N: int, device: torch.device):
        """
        Return:
          eidx_bt_list[t]: edge_index replicated across batch for time t  [wztp_l1, E_t*B]
          eidx_raw_list[t]: raw per-time edge_index (0..N-wztp0)             [wztp_l1, E_t]
        """
        eidx_bt_list = []
        eidx_raw_list = []
        for t in range(T):
            t0, t1 = t * N, (t + 1) * N
            mask = (edge_index[0] >= t0) & (edge_index[0] < t1) & \
                   (edge_index[1] >= t0) & (edge_index[1] < t1)
            eidx_t = edge_index[:, mask] - t0  # [wztp_l1, E_t]
            eidx_raw_list.append(eidx_t)

            if B == 1:
                eidx_bt = eidx_t
            else:
                E_t = eidx_t.size(1)
                eidx_bt = eidx_t.repeat(1, B)
                offset = (torch.arange(B, device=device) * N).repeat_interleave(E_t)
                eidx_bt = eidx_bt + torch.stack([offset, offset], dim=0)
            eidx_bt_list.append(eidx_bt)
        return eidx_bt_list, eidx_raw_list

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,F,T]  (channel wztp0 is poi idx)
        edge_index: [wztp_l1,E_total] over BxT "stacked graphs"
        returns λ>0 for all edges pooled over all timesteps, shape [E_total]
        """
        B, N, F_in, T = x.shape
        device = x.device
        assert N == self.N, f"num_nodes mismatch: {N} vs {self.N}"

        # wztp0. fuse POI emb + numeric
        poi_idx = x[:, :, 1, :].long()                       # [B,N,T]
        poi_e   = self.poi_emb(poi_idx).permute(0,1,3,2)     # [B,N,embed_dim,T]
        poi_e   = self.emb_drop(poi_e)

        other   = torch.cat([x[:, :, :1, :], x[:, :, 2:, :]], dim=2)  # [B,N,6,T]
        fused   = torch.cat([other, poi_e], dim=2)                    # [B,N,6+emb,T]
        fused   = fused.permute(0,1,3,2).contiguous()                 # [B,N,T,6+emb]
        fused   = self.fuse(fused)                                    # [B,N,T,in_feat]
        x_seq   = fused.permute(0,2,1,3).contiguous()                 # [B,T,N,in_feat]

        # wztp_l1. build time-split edge indices
        eidx_bt_list, eidx_raw_list = self._split_edges_by_time(edge_index, B, T, N, device)

        # wztp_final. run DCRNN
        H_all, _ = self.encoder(x_seq, eidx_bt_list, h0=None)  # [T,B,N,H]
        HT = H_all[-1]                                        # last step hidden, [B,N,H]

        # 4. edge decode FROM LAST NODE STATE across ALL time slices' edges
        outs = []
        for t in range(T):
            eidx_raw = eidx_raw_list[t]
            if eidx_raw.numel() == 0:
                continue
            src = HT[:, eidx_raw[0], :]               # [B,E_t,H]
            dst = HT[:, eidx_raw[1], :]               # [B,E_t,H]
            e_feat = torch.cat([src, dst, torch.abs(src - dst)], dim=-1)  # [B,E_t,3H]
            raw = self.edge_mlp(e_feat).squeeze(-1)   # [B,E_t]
            lam_t = F.softplus(raw) + 1e-6            # λ>0
            outs.append(lam_t)

        if len(outs) == 0:
            return x.new_zeros((0,), dtype=x.dtype, device=x.device)

        lam_all = torch.cat(outs, dim=1)              # [B,E_total_agg]
        return lam_all.reshape(-1)                    # [E_total]


# ---------------------- train / val loops ----------------------
def train_epoch(
    model,
    loader,
    optimizer,
    device,
    tail_alpha,
    w_max,
    weight_mode,
    tail_min_count,
    tail_max_count,
    amp_enabled,
    amp_dtype,
    log_prefix="dcrnn",
):
    model.train()
    total_loss_sum = 0.0
    total_edges = 0

    scaler = make_scaler(enabled_fp16=(amp_enabled and amp_dtype == torch.float16))

    for step, (x_b, eidx_b, y_b) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        x_b    = x_b.to(device, non_blocking=True)
        eidx_b = eidx_b.to(device, non_blocking=True)
        y_b    = y_b.to(device, non_blocking=True)

        B, N, _, T = x_b.shape
        last_mask = _last_step_edge_mask(eidx_b, B, N, T, device)  # bool[E_total]

        if not last_mask.any():
            continue

        # forward under AMP
        with make_autocast(enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
            lam_all = model(x_b, eidx_b)  # [E_total], already softplus in forward
        # select last-step edges
        lam_last = lam_all[last_mask]

        y_last   = y_b[last_mask]        # [E_last]

        if lam_last.numel() == 0 or lam_last.numel() != y_last.numel():
            wandb.log({
                f"{log_prefix}_mismatch_last_step": 1,
                f"{log_prefix}_n_pred": int(lam_last.numel()),
                f"{log_prefix}_n_tgt": int(y_last.numel()),
            })
            continue

        # convert to λ in FP32-style safe path (consistency w/ main script)
        lam_last = safe_softplus_to_lambda(lam_last, already_positive=True)

        # weighted tail-aware NLL (same as main pre_training_ztp)
        loss_vec = ztp_nll_vector(
            y_last,
            lam_last,
            tail_alpha=tail_alpha,
            w_max=w_max,
            weight_mode=weight_mode,
            tail_min_count=tail_min_count,
            tail_max_count=tail_max_count,
        )
        loss_mean = loss_vec.mean()
        loss_sum  = loss_vec.sum().item()

        # backward / step with or w/o scaler
        if not amp_enabled or amp_dtype == torch.bfloat16:
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss_mean).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        n_edges = int(y_last.numel())
        total_edges += n_edges
        total_loss_sum += loss_sum

        wandb.log({f"train_loss_step_ztp": loss_sum / max(1, n_edges)})

        del x_b, eidx_b, y_b, lam_all, lam_last, y_last, loss_vec, loss_mean
        gc.collect()

    return total_loss_sum / max(1, total_edges)


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    tail_alpha,
    w_max,
    weight_mode,
    tail_min_count,
    tail_max_count,
    amp_enabled,
    amp_dtype,
    log_prefix="dcrnn",
):
    model.eval()
    total_edges = 0

    sum_nll_weighted = 0.0
    sum_nll_unweight = 0.0
    sum_mae_mu = 0.0
    sum_mse_mu = 0.0

    for step, (x_b, eidx_b, y_b) in enumerate(loader):
        x_b    = x_b.to(device, non_blocking=True)
        eidx_b = eidx_b.to(device, non_blocking=True)
        y_b    = y_b.to(device, non_blocking=True)

        B, N, _, T = x_b.shape
        last_mask = _last_step_edge_mask(eidx_b, B, N, T, device)
        if not last_mask.any():
            continue

        with make_autocast(enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
            lam_all = model(x_b, eidx_b)

        lam_last = lam_all[last_mask]
        y_last   = y_b[last_mask]

        if lam_last.numel() == 0 or lam_last.numel() != y_last.numel():
            wandb.log({
                f"{log_prefix}_mismatch_last_step": 1,
                f"{log_prefix}_n_pred": int(lam_last.numel()),
                f"{log_prefix}_n_tgt": int(y_last.numel()),
            })
            continue

        # ensure consistent lambda numeric handling
        lam_last = safe_softplus_to_lambda(lam_last, already_positive=True)

        # weighted and unweighted NLLs
        nll_vec_weighted = ztp_nll_vector(
            y_last,
            lam_last,
            tail_alpha=tail_alpha,
            w_max=w_max,
            weight_mode=weight_mode,
            tail_min_count=tail_min_count,
            tail_max_count=tail_max_count,
        )
        nll_vec_unweight = ztp_nll_vector(
            y_last,
            lam_last,
            tail_alpha=0.0,
            w_max=0.0,
            weight_mode="none",
            tail_min_count=1,
            tail_max_count=300,
        )

        n = int(y_last.numel())
        wandb.log({
            "val_loss_step_ztp":  nll_vec_unweight.mean().item(),
            "val_loss_step_wztp": nll_vec_weighted.mean().item(),
        })

        sum_nll_weighted += nll_vec_weighted.sum().item()
        sum_nll_unweight += nll_vec_unweight.sum().item()
        total_edges      += n

        # compute μ under ZTP
        mu = ztp_mean(lam_last)
        diff = (mu - y_last.float())
        sum_mae_mu += diff.abs().sum().item()
        sum_mse_mu += (diff.pow(2)).sum().item()

        del x_b, eidx_b, y_b, lam_all, lam_last, y_last
        del nll_vec_weighted, nll_vec_unweight, mu, diff
        gc.collect()

    total_edges = max(1, total_edges)
    avg_nll_unw = sum_nll_unweight / total_edges
    avg_nll_w   = sum_nll_weighted / total_edges
    avg_mae_mu  = sum_mae_mu / total_edges
    avg_mse_mu  = sum_mse_mu / total_edges

    wandb.log({
        "NLL_ZTP_weighted": avg_nll_w,
        "MAE_mu": avg_mae_mu,
        "MSE_mu": avg_mse_mu,
    })

    return avg_nll_unw, avg_nll_w


# ---------------------- main ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data / model dims
    parser.add_argument("--input_dim", type=int, default=11)     # in_feat after fuse
    parser.add_argument("--num_nodes", type=int, default=38628)
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--num_poi_types", type=int, default=456)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--embed_dropout", type=float, default=0.1)

    parser.add_argument("--dcrnn_hidden", type=int, default=256)
    parser.add_argument("--dcrnn_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    # training / sched
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience_stop", type=int, default=8)
    parser.add_argument("--patience_lr", type=int, default=3)
    parser.add_argument("--factor", type=float, default=0.5)

    # AMP config (align with main script)
    parser.add_argument("--amp", type=str, default="bf16",
                        choices=["off", "fp16", "bf16", "auto"],
                        help="mixed precision mode")

    # paths / logging
    parser.add_argument("--project", type=str, default="Paper1_DCRNN_ZTP")
    parser.add_argument("--train_dir", type=str, default="./graph_data/LA/train")
    parser.add_argument("--val_dir", type=str, default="./graph_data/LA/val")

    # tail reweight (align with main script)
    parser.add_argument("--tail_alpha", type=float, default=2.2)
    parser.add_argument("--w_max", type=float, default=4.5)
    parser.add_argument("--weight_mode", type=str, default="log", choices=["none", "log", "power"])
    parser.add_argument("--tail_min_count", type=int, default=3)
    parser.add_argument("--tail_max_count", type=int, default=8)

    args = parser.parse_args()

    # ----- AMP global policy (match pre_training_ztp logic) -----
    if args.amp == "off":
        AMP_ENABLED_GLOBAL = False
        AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    elif args.amp == "fp16":
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.float16
    elif args.amp == "bf16":
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.bfloat16
    else:  # auto
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    wandb.init(project=args.project)
    wandb.config.update(vars(args) | {
        "amp_enabled": AMP_ENABLED_GLOBAL,
        "amp_dtype": str(AMP_DTYPE_GLOBAL).replace("torch.", "")
    })

    # reproducibility (optional, consistent with other baselines)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(
        gpickle_dir=[args.train_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )
    val_loader = get_dataloader(
        gpickle_dir=[args.val_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )

    model = DCRNNForEdges(
        num_nodes=args.num_nodes,
        in_feat=args.input_dim,
        num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,
        dcrnn_hidden=args.dcrnn_hidden,
        dcrnn_layers=args.dcrnn_layers,
        cheb_k=args.cheb_k,
        dropout=args.dropout,
        emb_dropout=args.embed_dropout,
    ).to(device)

    print("DCRNN params:", count_parameters(model))

    # param groups (match style from pre_training_ztp.py)
    def _is_no_decay_param(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim < 2:
            return True
        ln = name.lower()
        return (ln.endswith(".bias")
                or "norm" in ln
                or "bn" in ln
                or "layernorm" in ln
                or "ln" in ln)

    decay_params, no_decay_params, emb_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "poi_emb" in n:
            emb_params.append(p)
        elif _is_no_decay_param(n, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": args.weight_decay},
            {"params": no_decay_params,"weight_decay": 0.0},
            {"params": emb_params,     "weight_decay": 0.0},
        ],
        lr=args.learning_rate
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience_lr,
        verbose=True
    )

    os.makedirs("./model/compare", exist_ok=True)

    eval_losses = []
    best_val_loss_after6 = float('inf')
    best_val_loss = float('inf')
    best_val_loss_w = float('inf')

    for epoch in range(args.num_epochs):
        t0 = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            tail_alpha=args.tail_alpha,
            w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL,
            amp_dtype=AMP_DTYPE_GLOBAL,
            log_prefix="dcrnn_train",
        )

        val_loss, val_loss_w = eval_epoch(
            model,
            val_loader,
            device,
            tail_alpha=args.tail_alpha,
            w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL,
            amp_dtype=AMP_DTYPE_GLOBAL,
            log_prefix="dcrnn_val",
        )

        eval_losses.append(val_loss_w)


        if val_loss_w < best_val_loss_w:
            best_val_loss_w = val_loss_w
            torch.save(model.state_dict(), "models/compare/dcrnn_wztp.pth")
            print("Saved: ./model/compare/dcrnn_wztp.pth")


        torch.cuda.empty_cache()
        gc.collect()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_w": val_loss_w,
        })
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train {train_loss:.6f} | Val {val_loss:.6f} | {time.time()-t0:.1f}s")

        scheduler.step(val_loss_w)
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})
        print(f"Learning Rate: {current_lr}")

        if early_stopping(args.patience_stop, eval_losses):
            torch.cuda.empty_cache()
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete.")
