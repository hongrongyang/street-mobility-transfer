import argparse
import math
import torch
import torch.nn.functional as F
import gc

from Graphwave import GraphWaveNet  # reuse exact same model class
from graph_data_loader_slide_LA import get_dataloader  # swap to *_SF if you SF_test SF
from pre_training_ztp import (
    ztp_mean,
    ztp_nll_vector,
    _last_step_edge_mask,
)


def load_model_generic(model, path, device="cpu", strict=True):
    """
    Robust loader:
    - accepts either plain state_dict() or {"model_state_dict": ...}
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=strict)
    print(f"Model loaded from {path}")
    return model


@torch.no_grad()
def evaluate_gwn(
    model,
    dataloader,
    device,
    clamp_nonneg: bool = False,
    verbose: bool = True,
):
    """
    Returns a dict of:
      - overall_MSE / overall_MAE
      - top10_MSE / top10_MAE  (top 10% true flow edges)
      - top5_MSE / top5_MAE    (top 5%)
      - top1_MSE / top1_MAE    (top wztp0%)
      - overall_ZTP_NLL        (unweighted)
      - top5_ZTP_NLL
      - top1_ZTP_NLL
      - q90 / q95 / q99        (flow thresholds for tail)
    """
    model.eval()

    ys = []
    preds = []
    nlls = []
    total_edges = 0

    for x_batch, edge_index_batch, edge_attr_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)
        if not last_mask.any():
            # no edges in last step for this window; skip cleanly
            del x_batch, edge_index_batch, edge_attr_batch, last_mask
            gc.collect()
            continue

        # GraphWaveNet.forward already returns λ>0 for selected edges
        lam = model(x_batch, edge_index_batch, last_mask=last_mask)  # [E_last]
        lam = lam.clamp_min(1e-6)

        # last-step edge ground truth
        y_true = edge_attr_batch[last_mask]  # [E_last]

        # sanity check shape match
        if lam.numel() != y_true.numel():
            if verbose:
                print(
                    f"[WARN][EVAL] last-step mismatch: "
                    f"pred={lam.numel()} vs tgt={y_true.numel()}. Skip."
                )
            del x_batch, edge_index_batch, edge_attr_batch, last_mask, y_true, lam
            gc.collect()
            continue

        # predicted conditional mean under ZTP (λ / (wztp0 - e^{-λ}))
        y_pred = ztp_mean(lam)

        if clamp_nonneg:
            y_pred = y_pred.clamp_min(0.0)

        # unweighted ZTP NLL per edge
        nll_vec_unweight = ztp_nll_vector(
            y_true,
            lam,
            tail_alpha=0.0,
            w_max=0.0,
            weight_mode="none",
            tail_min_count=1,
            tail_max_count=300,
        )

        # stash CPU copies
        ys.append(y_true.detach().float().cpu())
        preds.append(y_pred.detach().float().cpu())
        nlls.append(nll_vec_unweight.detach().float().cpu())
        total_edges += int(y_true.numel())

        # cleanup
        del x_batch, edge_index_batch, edge_attr_batch, last_mask
        del y_true, y_pred, lam, nll_vec_unweight
        gc.collect()

    # handle empty dataloader case gracefully
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

    # concat everything
    y_all = torch.cat(ys, dim=0)        # [E_total]
    pred_all = torch.cat(preds, dim=0)  # [E_total]
    nll_all = torch.cat(nlls, dim=0)    # [E_total]

    # ---- overall metrics ----
    diff = pred_all - y_all
    overall_MSE = torch.mean(diff ** 2).item()
    overall_MAE = torch.mean(torch.abs(diff)).item()
    overall_ZTP_NLL = torch.mean(nll_all).item()

    # ---- helper: select top-k% by y ----
    def topk_mask_by_y(y: torch.Tensor, pct: float) -> torch.Tensor:
        M = y.numel()
        k = max(1, int(math.ceil(pct * M)))
        idx = torch.topk(y, k, largest=True, sorted=False).indices
        mask = torch.zeros(M, dtype=torch.bool)
        mask[idx] = True
        return mask

    def mae_mse_on_mask(y, p, mask):
        if mask.sum() == 0:
            return float("nan"), float("nan")
        d = p[mask] - y[mask]
        mse = torch.mean(d ** 2).item()
        mae = torch.mean(torch.abs(d)).item()
        return mae, mse

    # masks for tail slices
    mask_top1 = topk_mask_by_y(y_all, 0.01)   # top wztp0%

    top1_MAE, top1_MSE = mae_mse_on_mask(y_all, pred_all, mask_top1)

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

    # --- model hyperparams (must match training or load will fail) ---
    parser.add_argument("--input_dim", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0)

    parser.add_argument("--num_nodes_per_graph", type=int, default=38628)

    # POI embedding setup
    parser.add_argument("--num_poi_types", type=int, default=456)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--embed_dropout", type=float, default=0)

    # --- data ---
    parser.add_argument("--test_dir", type=str, default="./graph_data/LA/SF_test")
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)

    # --- model checkpoint path ---
    parser.add_argument("--model_path", type=str, default="./model/compare/gw_wztp.pth")

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

    # build model with EXACT SAME SHAPES as training
    model = GraphWaveNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes_per_graph,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,
        emb_dropout=args.embed_dropout,
    ).to(device)

    # load weights
    model = load_model_generic(model, args.model_path, device=device, strict=True)

    # evaluate
    metrics = evaluate_gwn(model, test_loader, device, verbose=True)

    m = metrics
    print(f"Overall   MSE: {m['overall_MSE']:.8f}  MAE: {m['overall_MAE']:.8f}  NLL: {m['overall_ZTP_NLL']:.8f}")
    print(f"Top-1%    MSE: {m['top1_MSE']:.8f}  MAE: {m['top1_MAE']:.8f}  NLL: {m['top1_ZTP_NLL']:.8f}")


if __name__ == "__main__":
    main()
