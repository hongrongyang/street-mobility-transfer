import argparse
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from typing import Tuple, List

from pre_training_ztp import (
    make_autocast, make_scaler, ztp_nll_vector, ztp_mean,
    safe_softplus_to_lambda, _last_step_edge_mask,
    early_stopping, count_parameters,
)
from graph_data_loader_slide_LA import get_dataloader

EPS = 1e-6

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------------- AMP setup ----------------------

def _default_amp_dtype():
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16

AMP_ENABLED_GLOBAL = True
AMP_DTYPE_GLOBAL   = _default_amp_dtype()


# ---------------------- Model ----------------------

class TemporalGLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv    = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, k), padding=(0, 0), bias=True)
        self.k       = k
        self.dropout = nn.Dropout(dropout)
        self.c_out   = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = F.pad(x, (self.k - 1, 0, 0, 0))   # causal pad
        y       = self.conv(x)                        # [B, 2*C_out, N, T]
        P, Q    = torch.split(y, self.c_out, dim=1)
        return self.dropout(torch.tanh(P) * torch.sigmoid(Q))


class STBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_temporal: int = 3, cheb_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.temp1    = TemporalGLU(c_in, c_out, k=k_temporal, dropout=dropout)
        self.cheb     = ChebConv(c_out, c_out, K=cheb_k, normalization='sym')
        self.norm     = nn.LayerNorm(c_out)
        self.temp2    = TemporalGLU(c_out, c_out, k=k_temporal, dropout=dropout)
        self.res_proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None

    def forward(self, x: torch.Tensor, edge_index_bt: List[torch.Tensor], N: int) -> torch.Tensor:
        B, _, _, T = x.shape
        res = x if self.res_proj is None else self.res_proj(x)
        x   = self.temp1(x)
        C   = x.size(1)

        xs: List[torch.Tensor] = []
        for t in range(T):
            x_t    = x[:, :, :, t].permute(0, 2, 1).reshape(B * N, C)  # [B*N, C]
            x_t    = F.relu(self.cheb(x_t, edge_index_bt[t]))
            x_t    = self.norm(x_t.view(B, N, C)).view(B * N, C)
            x_t    = x_t.view(B, N, C).permute(0, 2, 1).unsqueeze(-1)
            xs.append(x_t)
        x = torch.cat(xs, dim=-1)  # [B, C, N, T]

        return self.temp2(x) + res


class STGCNForEdges(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 in_feat: int,
                 hid_channels: int = 128,
                 num_blocks: int = 2,
                 k_temporal: int = 3,
                 cheb_k: int = 2,
                 dropout: float = 0.2,
                 num_poi_types: int = 456,
                 embed_dim: int = 8,
                 emb_dropout: float = 0.08):
        super().__init__()
        self.N        = num_nodes
        self.poi_emb  = nn.Embedding(num_poi_types + 1, embed_dim)
        self.emb_drop = nn.Dropout(p=emb_dropout)
        self.fuse     = nn.Linear(6 + embed_dim, in_feat)

        blocks = [STBlock(in_feat, hid_channels, k_temporal, cheb_k, dropout)]
        for _ in range(num_blocks - 1):
            blocks.append(STBlock(hid_channels, hid_channels, k_temporal, cheb_k, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hid_channels, hid_channels),
            nn.GELU(),
            nn.Linear(hid_channels, 1),
        )

    @staticmethod
    def _split_edges_by_time(
        edge_index: torch.Tensor, B: int, T: int, N: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        eidx_bt_list:  List[torch.Tensor] = []
        eidx_raw_list: List[torch.Tensor] = []
        for t in range(T):
            t0, t1 = t * N, (t + 1) * N
            mask   = (edge_index[0] >= t0) & (edge_index[0] < t1) & \
                     (edge_index[1] >= t0) & (edge_index[1] < t1)
            eidx_t = edge_index[:, mask] - t0   # [2, E_t]
            eidx_raw_list.append(eidx_t)

            if B == 1:
                eidx_bt = eidx_t
            else:
                E_t     = eidx_t.size(1)
                eidx_bt = eidx_t.repeat(1, B)
                offset  = (torch.arange(B, device=device) * N).repeat_interleave(E_t)
                eidx_bt = eidx_bt + torch.stack([offset, offset], dim=0)
            eidx_bt_list.append(eidx_bt)
        return eidx_bt_list, eidx_raw_list

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [B, N, F, T] — channel 1 is POI index; returns λ [E_total]"""
        B, N, _, T = x.shape
        assert N == self.N, f"num_nodes mismatch: got {N}, expected {self.N}"
        device = x.device

        # fuse POI embedding with numeric features
        poi_e  = self.emb_drop(self.poi_emb(x[:, :, 1, :].long()).permute(0, 1, 3, 2))  # [B,N,emb,T]
        other  = torch.cat([x[:, :, :1, :], x[:, :, 2:, :]], dim=2)                     # [B,N,6,T]
        fused  = torch.cat([other, poi_e], dim=2)                                        # [B,N,6+emb,T]
        xh     = self.fuse(fused.permute(0, 1, 3, 2)).permute(0, 3, 1, 2).contiguous()  # [B,C,N,T]

        eidx_bt_list, eidx_raw_list = self._split_edges_by_time(edge_index, B, T, N, device)

        for blk in self.blocks:
            xh = blk(xh, eidx_bt_list, N)  # [B, C, N, T]

        outs: List[torch.Tensor] = []
        for t in range(T):
            eidx_raw = eidx_raw_list[t]
            if eidx_raw.numel() == 0:
                continue
            h_t   = xh[:, :, :, t].permute(0, 2, 1)                          # [B, N, C]
            src   = h_t[:, eidx_raw[0], :]                                    # [B, E_t, C]
            dst   = h_t[:, eidx_raw[1], :]
            e_feat = torch.cat([src, dst, torch.abs(src - dst)], dim=-1)      # [B, E_t, 3C]
            lam_t  = F.softplus(self.edge_mlp(e_feat).squeeze(-1)) + EPS      # [B, E_t]
            outs.append(lam_t)

        if not outs:
            return x.new_zeros((0,), dtype=x.dtype, device=device)
        return torch.cat(outs, dim=1).view(-1)  # [E_total]


# ---------------------- train / eval loops ----------------------

def train(model: nn.Module, loader, optim, device: torch.device,
          tail_alpha: float, w_max: float, weight_mode: str,
          tail_min_count: int, tail_max_count: int,
          amp_enabled: bool, amp_dtype) -> float:

    model.train()
    total_loss_sum, total_edges = 0.0, 0
    scaler = make_scaler(enabled_fp16=(amp_enabled and amp_dtype == torch.float16))

    for x, eidx, y in loader:
        optim.zero_grad(set_to_none=True)

        x    = x.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        B, N, _, T = x.shape
        mask = _last_step_edge_mask(eidx, B, N, T, device)
        if not mask.any():
            continue

        with make_autocast(amp_enabled and device.type == "cuda", dtype=amp_dtype):
            lam_all = model(x, eidx)  # [E_total]

        lam    = safe_softplus_to_lambda(lam_all[mask], already_positive=True)
        y_last = y[mask]
        if lam.numel() != y_last.numel():
            wandb.log({"mismatch_last_step": 1, "n_pred": int(lam.numel()), "n_tgt": int(y_last.numel())})
            continue

        loss_vec  = ztp_nll_vector(y_last, lam, tail_alpha=tail_alpha, w_max=w_max,
                                   weight_mode=weight_mode, tail_min_count=tail_min_count,
                                   tail_max_count=tail_max_count)
        loss_mean = loss_vec.mean()
        loss_sum  = loss_vec.sum().item()

        if not amp_enabled or amp_dtype == torch.bfloat16:
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optim.step()
        else:
            scaler.scale(loss_mean).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optim)
            scaler.update()

        n_edges         = int(y_last.numel())
        total_loss_sum += loss_sum
        total_edges    += n_edges
        wandb.log({"train_loss_step_ztp": loss_sum / max(1, n_edges)})

        del x, eidx, y, lam_all, lam, y_last, loss_vec
        gc.collect()

    return total_loss_sum / max(1, total_edges)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             tail_alpha: float, w_max: float, weight_mode: str,
             tail_min_count: int, tail_max_count: int,
             amp_enabled: bool, amp_dtype):

    model.eval()
    total_edges      = 0
    sum_nll_weighted = 0.0
    sum_nll_unweight = 0.0
    sum_mae_mu       = 0.0
    sum_mse_mu       = 0.0

    for x, eidx, y in loader:
        x    = x.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        B, N, _, T = x.shape
        mask = _last_step_edge_mask(eidx, B, N, T, device)
        if not mask.any():
            continue

        with make_autocast(amp_enabled and device.type == "cuda", dtype=amp_dtype):
            lam_all = model(x, eidx)

        lam    = safe_softplus_to_lambda(lam_all[mask], already_positive=True)
        y_last = y[mask]
        if lam.numel() != y_last.numel():
            wandb.log({"mismatch_last_step": 1, "n_pred": int(lam.numel()), "n_tgt": int(y_last.numel())})
            continue

        nll_w = ztp_nll_vector(y_last, lam, tail_alpha=tail_alpha, w_max=w_max,
                               weight_mode=weight_mode, tail_min_count=tail_min_count,
                               tail_max_count=tail_max_count)
        nll_u = ztp_nll_vector(y_last, lam, tail_alpha=0.0, w_max=0.0,
                               weight_mode='none', tail_min_count=1, tail_max_count=300)

        n = int(y_last.numel())
        wandb.log({
            "val_loss_step_ztp":  nll_u.sum().item() / max(1, n),
            "val_loss_step_wztp": nll_w.sum().item() / max(1, n),
        })

        sum_nll_weighted += nll_w.sum().item()
        sum_nll_unweight += nll_u.sum().item()
        total_edges      += n

        mu   = ztp_mean(lam)
        diff = mu - y_last.float()
        sum_mae_mu += diff.abs().sum().item()
        sum_mse_mu += diff.pow(2).sum().item()

        del x, eidx, y, lam_all, lam, y_last, nll_w, nll_u, mu, diff
        gc.collect()

    total_edges = max(1, total_edges)
    avg_nll_unw = sum_nll_unweight / total_edges
    avg_nll_w   = sum_nll_weighted  / total_edges
    avg_mae_mu  = sum_mae_mu / total_edges
    avg_mse_mu  = sum_mse_mu / total_edges

    wandb.log({"ZTP_epoch": avg_nll_unw, "wZTP_epoch": avg_nll_w,
               "MAE_mu_epoch": avg_mae_mu, "MSE_mu_epoch": avg_mse_mu})
    return avg_nll_unw, avg_nll_w


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim",      type=int,   default=11)
    parser.add_argument("--hidden",         type=int,   default=128)
    parser.add_argument("--num_blocks",     type=int,   default=2)
    parser.add_argument("--k_temporal",     type=int,   default=3)
    parser.add_argument("--cheb_k",         type=int,   default=2)
    parser.add_argument("--dropout",        type=float, default=0.2)
    parser.add_argument("--num_nodes",      type=int,   default=38628)
    parser.add_argument("--window_size",    type=int,   default=12)
    parser.add_argument("--batch_size",     type=int,   default=1)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=4e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--patience_stop",  type=int,   default=9)
    parser.add_argument("--patience_lr",    type=int,   default=3)
    parser.add_argument("--factor",         type=float, default=0.5)
    parser.add_argument("--num_poi_types",  type=int,   default=456)
    parser.add_argument("--embed_dim",      type=int,   default=5)
    parser.add_argument("--embed_dropout",  type=float, default=0.1)
    parser.add_argument("--project",        type=str,   default="Paper1_STGCN_ZTP")
    parser.add_argument("--train_dir",      type=str,   default="./graph_data/LA/train")
    parser.add_argument("--val_dir",        type=str,   default="./graph_data/LA/val")
    parser.add_argument("--tail_alpha",     type=float, default=2.2)
    parser.add_argument("--w_max",          type=float, default=4.5)
    parser.add_argument("--weight_mode",    type=str,   default="log",
                        choices=["none", "log", "power"])
    parser.add_argument("--tail_min_count", type=int,   default=3)
    parser.add_argument("--tail_max_count", type=int,   default=8)
    parser.add_argument("--amp",            type=str,   default="bf16",
                        choices=["off", "fp16", "bf16", "auto"])
    args = parser.parse_args()

    global AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL
    if args.amp == "off":
        AMP_ENABLED_GLOBAL = False
    elif args.amp == "fp16":
        AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL = True, torch.float16
    elif args.amp == "bf16":
        AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL = True, torch.bfloat16
    else:
        AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL = True, _default_amp_dtype()

    wandb.init(project=args.project)
    wandb.config.update(vars(args) | {
        "amp_enabled": AMP_ENABLED_GLOBAL,
        "amp_dtype":   str(AMP_DTYPE_GLOBAL).replace("torch.", ""),
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(
        gpickle_dir=[args.train_dir], batch_size=args.batch_size,
        input_dim=args.input_dim, window_sizes=[args.window_size], num_workers=1,
    )
    val_loader = get_dataloader(
        gpickle_dir=[args.val_dir], batch_size=args.batch_size,
        input_dim=args.input_dim, window_sizes=[args.window_size], num_workers=1,
    )

    model = STGCNForEdges(
        num_nodes=args.num_nodes, in_feat=args.input_dim,
        hid_channels=args.hidden, num_blocks=args.num_blocks,
        k_temporal=args.k_temporal, cheb_k=args.cheb_k,
        dropout=args.dropout, num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim, emb_dropout=args.embed_dropout,
    ).to(device)

    print("STGCN params:", count_parameters(model))

    def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim < 2:
            return True
        ln = name.lower()
        return ln.endswith(".bias") or any(k in ln for k in ("norm", "bn", "layernorm", "ln"))

    decay_params, no_decay_params, emb_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "poi_emb" in n or "poi_embedding" in n:
            emb_params.append(p)
        elif _no_decay(n, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optim = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": emb_params,      "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=args.factor,
                                  patience=args.patience_lr, verbose=True)

    eval_losses = []
    best_val_loss_w = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train(
            model, train_loader, optim, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max, weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count, tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL, amp_dtype=AMP_DTYPE_GLOBAL,
        )
        val_loss, val_loss_w = evaluate(
            model, val_loader, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max, weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count, tail_max_count=args.tail_max_count,
            amp_enabled=AMP_ENABLED_GLOBAL, amp_dtype=AMP_DTYPE_GLOBAL,
        )
        eval_losses.append(val_loss_w)

        if val_loss_w < best_val_loss_w:
            best_val_loss_w = val_loss_w
            torch.save(model.state_dict(), "model/compare/stgcn_wztp.pth")
            print("Saved: model/compare/stgcn_wztp.pth")

        torch.cuda.empty_cache()
        gc.collect()

        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "val_loss": val_loss, "val_loss_w": val_loss_w})
        print(f"[{epoch:03d}] train {train_loss:.8f} | val {val_loss:.8f} | {time.time()-t0:.1f}s")

        scheduler.step(val_loss_w)
        current_lr = optim.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Learning Rate: {current_lr}")

        if early_stopping(args.patience_stop, eval_losses):
            torch.cuda.empty_cache()
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()