import argparse
import time
import gc
import torch
import torch.nn.functional as F
import wandb
import math
from contextlib import nullcontext
from torch.optim.lr_scheduler import ReduceLROnPlateau
from graph_data_loader_slide_LA import get_dataloader
# from graph_data_loader_slide_SF import get_dataloader
# from graph_data_loader_slide_FRE import get_dataloader
from model import TCGCNTransformer
from torch.special import gammaln

EPS = 1e-6

AMP_ENABLED_GLOBAL = True
AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16


# ---------------------- AMP utilities ----------------------

def make_autocast(enabled: bool, dtype=None):
    if not enabled:
        return nullcontext()
    dt = dtype or AMP_DTYPE_GLOBAL
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dt)
    from torch.cuda.amp import autocast as old_autocast
    return old_autocast(dtype=dt)


def make_scaler(enabled_fp16: bool):
    if not enabled_fp16:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    from torch.cuda.amp import GradScaler as OldGradScaler
    return OldGradScaler(enabled=True)


# ---------------------- ZTP loss and helpers ----------------------

def safe_softplus_to_lambda(x_raw_or_lambda: torch.Tensor, already_positive: bool) -> torch.Tensor:
    # always compute in fp32 to avoid underflow in exp(-lam)
    with torch.autocast(device_type="cuda", enabled=False):
        x = x_raw_or_lambda.float()
        lam = x if already_positive else F.softplus(x)
        lam = lam.clamp_min(EPS)
        return lam


def ztp_mean(lam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # E[Y | Y>0] = lam / (1 - exp(-lam))
    with torch.autocast(device_type="cuda", enabled=False):
        lam32 = lam.float()
        den = (1.0 - torch.exp(-lam32)).clamp_min(eps)
        return lam32 / den


def ztp_nll_vector(
    y, lam,
    tail_alpha: float = 0.0,
    w_max: float = 6.0,
    weight_mode: str = "log",   # 'none' | 'log' | 'power'
    tail_min_count: int = 3,
    tail_max_count: int = 10,
    w_1: float = 0.15,
    w_2: float = 0.15,
):
    with torch.autocast(device_type="cuda", enabled=False):
        y   = y.float().clamp_min(1.0)
        lam = lam.float().clamp_min(EPS)

        # Poisson log-pmf + ZTP normalisation constant
        log_p = y * torch.log(lam) - lam - gammaln(y + 1.0)
        norm  = -torch.log1p(-torch.exp(-lam).clamp_max(1 - 1e-6))
        nll   = -(log_p + norm)

        if tail_alpha and tail_alpha > 0 and weight_mode != "none":
            mask_range = (y >= float(tail_min_count)) & (y <= float(tail_max_count))

            if mask_range.any():
                w = torch.ones_like(y)

                if weight_mode == "log":
                    w_range = 1.0 + tail_alpha * torch.log1p(y[mask_range] - 1.0)
                    w[mask_range] = torch.clamp(w_range, 1.0, w_max)

                elif weight_mode == "power":
                    w_range = torch.pow(y[mask_range], tail_alpha)
                    w[mask_range] = torch.clamp(w_range, 1.0, w_max)

                else:
                    raise ValueError(f"unknown weight_mode={weight_mode}")

                # down-weight y=1 and y=2 to offset ZTP mass concentration
                w[y < 2]  *= w_1
                w[y == 2] *= w_2
                nll = nll * w

        return nll


def _last_step_edge_mask(edge_index_batch, batch_size, num_nodes, time_steps, device):
    # keep only edges belonging to the final time slice of each batch item
    mask_list = []
    for b in range(batch_size):
        t = time_steps - 1
        start_idx = (b * time_steps + t) * num_nodes
        end_idx   = start_idx + num_nodes
        m = (edge_index_batch[0] >= start_idx) & (edge_index_batch[0] < end_idx) & \
            (edge_index_batch[1] >= start_idx) & (edge_index_batch[1] < end_idx)
        mask_list.append(m)
    return torch.stack(mask_list, dim=0).any(dim=0).to(device)


def early_stopping(patience, eval_losses_input):
    if len(eval_losses_input) > patience:
        recent = eval_losses_input[-patience:]
        return all(recent[i] >= recent[i - 1] for i in range(1, patience))
    return False


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------- train / eval loops ----------------------

def train(model, dataloader, optimizer, device,
          tail_alpha=2.5, w_max=7.0, weight_mode="log",
          tail_min_count=3, tail_max_count=10,
          model_outputs_raw: bool = False,
          w_1: float = 0.15,
          w_2: float = 0.15):

    model.train()
    total_loss_sum, total_edges = 0.0, 0

    AMP_ENABLED_LOCAL = AMP_ENABLED_GLOBAL
    AMP_DTYPE_LOCAL   = AMP_DTYPE_GLOBAL
    scaler = make_scaler(enabled_fp16=(AMP_ENABLED_LOCAL and AMP_DTYPE_LOCAL == torch.float16))

    for step, (x_batch, edge_index_batch, edge_attr_batch) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        x_batch          = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch  = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)
        y_true = edge_attr_batch[last_mask]  # [E_last]

        with make_autocast(enabled=AMP_ENABLED_LOCAL, dtype=AMP_DTYPE_LOCAL):
            pred = model(x_batch, edge_index_batch, tgt_mask=None)  # [E_total]

        if pred.numel() == 0:
            continue
        if pred.numel() != y_true.numel():
            wandb.log({"mismatch_last_step": 1,
                       "n_pred": int(pred.numel()),
                       "n_tgt":  int(y_true.numel())})
            continue

        lam = safe_softplus_to_lambda(pred, already_positive=(not model_outputs_raw))  # [E_last]

        loss_vec  = ztp_nll_vector(y_true, lam,
                                   tail_alpha=tail_alpha, w_max=w_max,
                                   weight_mode=weight_mode,
                                   tail_min_count=tail_min_count,
                                   tail_max_count=tail_max_count,
                                   w_1=w_1, w_2=w_2)
        loss_mean = loss_vec.mean()
        loss_sum  = loss_vec.sum().item()

        if not AMP_ENABLED_LOCAL or AMP_DTYPE_LOCAL == torch.bfloat16:
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        else:
            scaler.scale(loss_mean).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        n_edges         = int(y_true.numel())
        total_loss_sum += loss_sum
        total_edges    += n_edges

        wandb.log({"train_loss_step_ztp": loss_sum / max(1, n_edges)})

        del x_batch, edge_index_batch, edge_attr_batch, y_true, pred, lam, loss_vec
        gc.collect()

    return total_loss_sum / max(1, total_edges)


@torch.no_grad()
def evaluate(model, dataloader, device,
             tail_alpha: float = 2.0, w_max: float = 7.0,
             weight_mode: str = "log",
             tail_min_count: int = 2, tail_max_count: int = 10,
             model_outputs_raw: bool = False,
             w_1: float = 0.15,
             w_2: float = 0.15):

    model.eval()
    total_edges      = 0
    sum_nll_weighted = 0.0
    sum_nll_unweight = 0.0
    sum_mae_mu       = 0.0
    sum_mse_mu       = 0.0

    AMP_ENABLED_LOCAL = AMP_ENABLED_GLOBAL
    AMP_DTYPE_LOCAL   = AMP_DTYPE_GLOBAL

    for step, (x_batch, edge_index_batch, edge_attr_batch) in enumerate(dataloader):
        x_batch          = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch  = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)
        y_true = edge_attr_batch[last_mask]  # [E_last]

        with make_autocast(enabled=AMP_ENABLED_LOCAL, dtype=AMP_DTYPE_LOCAL):
            pred = model(x_batch, edge_index_batch, tgt_mask=None)  # [E_total]

        if pred.numel() == 0 or pred.numel() != y_true.numel():
            wandb.log({"mismatch_last_step": 1,
                       "n_pred": int(pred.numel()),
                       "n_tgt":  int(y_true.numel())})
            del x_batch, edge_index_batch, edge_attr_batch, last_mask, y_true, pred
            gc.collect()
            continue

        lam = safe_softplus_to_lambda(pred, already_positive=(not model_outputs_raw))

        nll_vec_weighted = ztp_nll_vector(y_true, lam,
                                          tail_alpha=tail_alpha, w_max=w_max,
                                          weight_mode=weight_mode,
                                          tail_min_count=tail_min_count,
                                          tail_max_count=tail_max_count,
                                          w_1=w_1, w_2=w_2)

        n = int(y_true.numel())
        wandb.log({"val_loss_step_wztp": nll_vec_weighted.mean().item()})

        sum_nll_weighted += nll_vec_weighted.sum().item()
        total_edges      += n

        mu   = ztp_mean(lam)
        diff = mu - y_true.float()
        sum_mae_mu += diff.abs().sum().item()
        sum_mse_mu += diff.pow(2).sum().item()

        del x_batch, edge_index_batch, edge_attr_batch, last_mask, y_true, pred, lam
        del nll_vec_weighted, mu, diff
        gc.collect()

    total_edges  = max(1, total_edges)
    avg_nll_unw  = sum_nll_unweight / total_edges
    avg_nll_w    = sum_nll_weighted  / total_edges
    avg_mae_mu   = sum_mae_mu / total_edges
    avg_mse_mu   = sum_mse_mu / total_edges

    wandb.log({"NLL_ZTP_weighted": avg_nll_w,
               "MAE_mu": avg_mae_mu,
               "MSE_mu": avg_mse_mu})

    return avg_nll_unw, avg_nll_w


# ---------------------- main ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=11)
    parser.add_argument('--temporal_hidden_dim1', type=int, default=128)
    parser.add_argument('--temporal_hidden_dim2', type=int, default=256)
    parser.add_argument('--temporal_dropout_rate', type=float, default=0.1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--gcn_hidden_dim1', type=int, default=512)
    parser.add_argument('--gcn_hidden_dim2', type=int, default=256)
    parser.add_argument('--gcn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--decoder_hidden_dim', type=int, default=256)
    parser.add_argument('--edge_output_dim', type=int, default=1)
    parser.add_argument('--decoder_dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.05)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--patience_stop', type=int, default=7)
    parser.add_argument('--patience_lr', type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_poi_types', type=int, default=456)
    parser.add_argument('--embed_dim', type=int, default=5)
    parser.add_argument('--w_1', type=float, default=0.15)
    parser.add_argument('--w_2', type=float, default=0.15)
    parser.add_argument('--tail_alpha', type=float, default=2.2)
    parser.add_argument('--w_max', type=float, default=4.5)
    parser.add_argument('--weight_mode', type=str, default='log', choices=['none', 'log', 'power'])
    parser.add_argument('--tail_min_count', type=int, default=3)
    parser.add_argument('--tail_max_count', type=int, default=10)
    parser.add_argument('--embed_dropout', type=float, default=0.1)

    parser.add_argument('--amp', type=str, default='bf16', choices=['auto', 'off', 'fp16', 'bf16'],
                        help="auto/off/fp16/bf16")
    parser.add_argument('--model_outputs_raw', type=bool, default=False,
                        help="If the model's forward pass returns raw logits (not soft-plus), set to True; otherwise,"
                             " set to False to indicate that λ is output directly.")

    args = parser.parse_args()

    if args.amp == 'off':
        AMP_ENABLED_GLOBAL = False
    elif args.amp == 'fp16':
        AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL = True, torch.float16
    elif args.amp == 'bf16':
        AMP_ENABLED_GLOBAL, AMP_DTYPE_GLOBAL = True, torch.bfloat16
    else:
        AMP_ENABLED_GLOBAL = True
        AMP_DTYPE_GLOBAL   = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
                             else torch.float16

    wandb.init(project="Paper1_pre_training")
    wandb.config.update(vars(args) | {
        "amp_enabled": AMP_ENABLED_GLOBAL,
        "amp_dtype":   str(AMP_DTYPE_GLOBAL).replace("torch.", "")
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(
        gpickle_dir=["./graph_data/LA/train"],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[12],
        num_workers=1,
    )
    val_dataloader = get_dataloader(
        gpickle_dir=["./graph_data/LA/val"],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[12],
        num_workers=1,
    )

    model = TCGCNTransformer(
        input_dim=args.input_dim,
        temporal_hidden_dim1=args.temporal_hidden_dim1,
        temporal_hidden_dim2=args.temporal_hidden_dim2,
        temporal_dropout_rate=args.temporal_dropout_rate,
        kernel_size=args.kernel_size,
        gcn_hidden_dim1=args.gcn_hidden_dim1,
        gcn_hidden_dim2=args.gcn_hidden_dim2,
        gcn_dropout_rate=args.gcn_dropout_rate,
        decoder_hidden_dim=args.decoder_hidden_dim,
        edge_output_dim=args.edge_output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_dropout_rate=args.decoder_dropout_rate,
        num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,
        emb_dropout=args.embed_dropout,
        attention_dropout_rate=args.attention_dropout_rate,
    ).to(device)

    # separate embedding params from weight-decayed params
    def _is_no_decay_param(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim < 2:
            return True
        ln = name.lower()
        return ln.endswith(".bias") or any(k in ln for k in ("norm", "bn", "layernorm", "ln"))

    decay_params, no_decay_params, emb_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        if "poi_embedding" in ln or "poi_emb" in ln:
            emb_params.append(p)
        elif "fc_edge_out" in ln or _is_no_decay_param(n, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": emb_params,      "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    print(f"Total trainable parameters: {count_parameters(model)}")

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor,
        patience=args.patience_lr, verbose=True,
    )

    eval_losses  = []
    best_val_loss_w = float('inf')

    for epoch in range(args.num_epochs):
        t0 = time.time()

        train_loss = train(
            model, train_dataloader, optimizer, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            model_outputs_raw=args.model_outputs_raw,
        )
        val_loss, val_loss_w = evaluate(
            model, val_dataloader, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max,
            weight_mode=args.weight_mode,
            tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            model_outputs_raw=args.model_outputs_raw,
        )
        eval_losses.append(val_loss_w)

        if val_loss_w < best_val_loss_w:
            best_val_loss_w = val_loss_w
            save_model(model, "final_model/pre_la.pth")

        torch.cuda.empty_cache()
        gc.collect()

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                   "val_loss": val_loss, "val_loss_w": val_loss_w})
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train {train_loss:.8f} | Val {val_loss:.8f} | WVal {val_loss_w:.8f} | "
              f"{time.time()-t0:.1f}s")

        scheduler.step(val_loss_w)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Learning Rate: {current_lr}")

        if early_stopping(args.patience_stop, eval_losses):
            torch.cuda.empty_cache()
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete.")