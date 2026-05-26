import argparse
import math
import torch
import gc

from DCRNN import DCRNNForEdges  # reuse exact same model class
from graph_data_loader_slide_LA import get_dataloader  # swap if testing SF loader
from pre_training_ztp import (
    ztp_mean,
    ztp_nll_vector,
    _last_step_edge_mask,
    safe_softplus_to_lambda,
)


def load_dcrnn_checkpoint(model, path, device="cpu", strict=True):
    """
    Robust loader:
    - works if you saved state_dict() directly
    - also works if you saved {"model_state_dict": ...}
    """
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model weights from {path}")
    return model


@torch.no_grad()
def evaluate_dcrnn(
    model,
    dataloader,
    device,
    clamp_nonneg: bool = False,
    verbose: bool = True,
):
    """
    Return metrics dict:
      - overall_MSE / overall_MAE
      - top10_MSE / top10_MAE
      - top5_MSE / top5_MAE
      - top1_MSE / top1_MAE
      - overall_ZTP_NLL
      - top5_ZTP_NLL
      - top1_ZTP_NLL
      - q90 / q95 / q99
    All computed on LAST timestep edges only, same as main model eval.
    """
    model.eval()

    ys = []
    preds = []
    nlls = []
    total_edges = 0

    for x_b, eidx_b, y_b in dataloader:
        x_b    = x_b.to(device, non_blocking=True)
        eidx_b = eidx_b.to(device, non_blocking=True)
        y_b    = y_b.to(device, non_blocking=True)

        B, N, _, T = x_b.shape
        last_mask = _last_step_edge_mask(eidx_b, B, N, T, device)  # bool[E_total]

        if not last_mask.any():
            # window has no valid last-step edges (should be rare but safe to skip)
            del x_b, eidx_b, y_b, last_mask
            gc.collect()
            continue

        # forward
        lam_all = model(x_b, eidx_b)  # [E_total], already softplus + eps in forward
        lam_all = safe_softplus_to_lambda(lam_all, already_positive=True)  # FP32 clamp safety

        lam_last = lam_all[last_mask]      # [E_last]
        y_last   = y_b[last_mask]          # [E_last]

        if lam_last.numel() != y_last.numel():
            if verbose:
                print(f"[WARN] mismatch last step: pred={lam_last.numel()} vs tgt={y_last.numel()} -> skip batch")
            del x_b, eidx_b, y_b, last_mask, lam_all, lam_last, y_last
            gc.collect()
            continue

        # point prediction = conditional ZTP mean μ = λ / (wztp0 - e^{-λ})
        y_pred_last = ztp_mean(lam_last)

        if clamp_nonneg:
            y_pred_last = y_pred_last.clamp_min(0.0)

        # unweighted ZTP NLL for metrics logging (no tail upweighting)
        nll_vec_unweight = ztp_nll_vector(
            y_last,
            lam_last,
            tail_alpha=0.0,
            w_max=0.0,
            weight_mode="none",
            tail_min_count=1,
            tail_max_count=300,
        )

        ys.append(y_last.detach().float().cpu())
        preds.append(y_pred_last.detach().float().cpu())
        nlls.append(nll_vec_unweight.detach().float().cpu())
        total_edges += int(y_last.numel())

        del x_b, eidx_b, y_b, last_mask, lam_all, lam_last, y_last, y_pred_last, nll_vec_unweight
        gc.collect()

    # Handle empty dataset edge case
    if total_edges == 0:
        return {
            "overall_MSE": float("nan"),
            "overall_MAE": float("nan"),
            "top10_MSE": float("nan"),
            "top10_MAE": float("nan"),
            "top5_MSE": float("nan"),
            "top5_MAE": float("nan"),
            "top1_MSE": float("nan"),
            "top1_MAE": float("nan"),
            "overall_ZTP_NLL": float("nan"),
            "top5_ZTP_NLL": float("nan"),
            "top1_ZTP_NLL": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
        }

    # concat all batches
    y_all    = torch.cat(ys,    dim=0)  # [E_total]
    pred_all = torch.cat(preds, dim=0)  # [E_total]
    nll_all  = torch.cat(nlls,  dim=0)  # [E_total]

    # overall metrics
    diff = pred_all - y_all
    overall_MSE = torch.mean(diff ** 2).item()
    overall_MAE = torch.mean(torch.abs(diff)).item()
    overall_ZTP_NLL = torch.mean(nll_all).item()

    # helper: build mask for top-k% edges by true y
    def topk_mask_by_y(y: torch.Tensor, pct: float) -> torch.Tensor:
        M = y.numel()
        k = max(1, int(math.ceil(pct * M)))
        idx = torch.topk(y, k, largest=True, sorted=False).indices
        mask = torch.zeros(M, dtype=torch.bool)
        mask[idx] = True
        return mask

    # mask slices
    mask_top1  = topk_mask_by_y(y_all, 0.01)  # top wztp0%

    # utility to compute MAE/MSE on subset
    def mae_mse_on_mask(y, p, mask):
        if mask.sum() == 0:
            return float("nan"), float("nan")
        d = p[mask] - y[mask]
        mse = torch.mean(d ** 2).item()
        mae = torch.mean(torch.abs(d)).item()
        return mae, mse


    top1_MAE,  top1_MSE  = mae_mse_on_mask(y_all, pred_all, mask_top1)

    # tail NLLs for interpretability
    top1_ZTP_NLL = (
        torch.mean(nll_all[mask_top1]).item() if mask_top1.any() else float("nan")
    )



    return {
        "overall_MSE": overall_MSE,
        "overall_MAE": overall_MAE,
        "top1_MSE": top1_MSE,
        "top1_MAE": top1_MAE,
        "overall_ZTP_NLL": overall_ZTP_NLL,
        "top1_ZTP_NLL": top1_ZTP_NLL,
    }


def main():
    parser = argparse.ArgumentParser()

    # -------- model hyperparams (must match training checkpoint) --------
    parser.add_argument("--input_dim", type=int, default=11)
    parser.add_argument("--num_nodes", type=int, default=38628)

    parser.add_argument("--num_poi_types", type=int, default=456)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--embed_dropout", type=float, default=0)

    parser.add_argument("--dcrnn_hidden", type=int, default=256)
    parser.add_argument("--dcrnn_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)

    # -------- data --------
    parser.add_argument("--test_dir", type=str, default="./graph_data/LA/SF_test")
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)

    # -------- checkpoint path --------
    parser.add_argument("--model_path", type=str, default="./model/compare/dcrnn_wztp.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader for SF_test split
    test_loader = get_dataloader(
        gpickle_dir=[args.test_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )

    # build model with same shape hyperparams as training run
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

    # load checkpoint
    model = load_dcrnn_checkpoint(model, args.model_path, device=device, strict=True)

    # evaluate
    m = evaluate_dcrnn(model, test_loader, device, verbose=True)

    print(f"Overall   MSE: {m['overall_MSE']:.8f}  MAE: {m['overall_MAE']:.8f}  NLL: {m['overall_ZTP_NLL']:.8f}")
    print(f"Top-1%    MSE: {m['top1_MSE']:.8f}  MAE: {m['top1_MAE']:.8f}  NLL: {m['top1_ZTP_NLL']:.8f}")


if __name__ == "__main__":
    main()
