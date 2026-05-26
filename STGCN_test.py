import argparse
import math
import torch
from torch.special import gammaln

from STGCN import STGCNForEdges
from graph_data_loader_slide_LA import get_dataloader

EPS = 1e-6


# ---------------------- model loading ----------------------

def load_model(model, path, device, strict=False):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys:
        print("[load_model] missing keys (first 10):", incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        print("[load_model] unexpected keys (first 10):", incompatible.unexpected_keys[:10])
    print(f"Weights loaded from {path}")
    return model


# ---------------------- ZTP helpers ----------------------

def ztp_mean(lam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # E[Y | Y>0] = lam / (1 - exp(-lam))
    return lam / (1.0 - torch.exp(-lam)).clamp_min(eps)


def ztp_nll(y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    # unweighted per-edge ZTP NLL
    y   = y.float().clamp_min(1.0)
    lam = lam.float().clamp_min(EPS)
    log_p = y * torch.log(lam) - lam - gammaln(y + 1.0)
    norm  = -torch.log1p(-torch.exp(-lam).clamp_max(1 - 1e-6))
    return -(log_p + norm)


def last_step_mask(edge_index, batch_size, num_nodes, time_steps, device):
    # keep only edges from the final time slice of each batch item
    masks = []
    for b in range(batch_size):
        t     = time_steps - 1
        start = (b * time_steps + t) * num_nodes
        end   = start + num_nodes
        m = (edge_index[0] >= start) & (edge_index[0] < end) & \
            (edge_index[1] >= start) & (edge_index[1] < end)
        masks.append(m)
    return torch.stack(masks, dim=0).any(dim=0).to(device)


# ---------------------- evaluation ----------------------

@torch.no_grad()
def evaluate(model, dataloader, device, verbose=True):
    model.eval()
    ys, mus, nlls = [], [], []

    for x, eidx, y_attr in dataloader:
        x      = x.to(device, non_blocking=True)       # [B, N, F, T]
        eidx   = eidx.to(device, non_blocking=True)    # [2, E_all]
        y_attr = y_attr.to(device, non_blocking=True)  # [E_all]

        B, N, _, T = x.shape
        mask = last_step_mask(eidx, B, N, T, device)

        lam_all  = model(x, eidx)                      # [E_all]
        lam_last = lam_all[mask].clamp_min(EPS)        # [E_last]
        y_last   = y_attr[mask]                        # [E_last]

        if lam_last.numel() != y_last.numel():
            if verbose:
                print(f"[WARN] mismatch: pred={lam_last.numel()} vs tgt={y_last.numel()} — skipping")
            continue

        mu      = ztp_mean(lam_last)
        nll_vec = ztp_nll(y_last, lam_last)

        ys.append(y_last.float().cpu())
        mus.append(mu.float().cpu())
        nlls.append(nll_vec.float().cpu())

        del x, eidx, y_attr, mask, lam_all, lam_last, y_last, mu, nll_vec

    if not ys:
        nan = float("nan")
        return {k: nan for k in (
            "overall_MSE", "overall_MAE",
            "top10_MSE", "top10_MAE", "top5_MSE", "top5_MAE", "top1_MSE", "top1_MAE",
            "overall_NLL", "top10_NLL", "top5_NLL", "top1_NLL",
            "q90", "q95", "q99",
        )}

    y_all   = torch.cat(ys)
    mu_all  = torch.cat(mus)
    nll_all = torch.cat(nlls)

    diff = mu_all - y_all
    overall_MSE = diff.pow(2).mean().item()
    overall_MAE = diff.abs().mean().item()
    overall_NLL = nll_all.mean().item()

    def topk_mask(y, pct):
        k = max(1, math.ceil(pct * y.numel()))
        m = torch.zeros(y.numel(), dtype=torch.bool)
        m[torch.topk(y, k, largest=True, sorted=False).indices] = True
        return m

    def metrics_on(mask):
        if not mask.any():
            return float("nan"), float("nan"), float("nan")
        d = mu_all[mask] - y_all[mask]
        return d.abs().mean().item(), d.pow(2).mean().item(), nll_all[mask].mean().item()

    m10, m05, m01 = topk_mask(y_all, 0.10), topk_mask(y_all, 0.05), topk_mask(y_all, 0.01)

    top10_MAE, top10_MSE, top10_NLL = metrics_on(m10)
    top5_MAE,  top5_MSE,  top5_NLL  = metrics_on(m05)
    top1_MAE,  top1_MSE,  top1_NLL  = metrics_on(m01)

    return {
        "overall_MSE": overall_MSE, "overall_MAE": overall_MAE,
        "top1_MSE":  top1_MSE,      "top1_MAE":  top1_MAE,
        "overall_NLL": overall_NLL,
        "top1_NLL": top1_NLL,
    }


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim",     type=int,   default=11)
    parser.add_argument("--num_nodes",     type=int,   default=38628)
    parser.add_argument("--window_size",   type=int,   default=12)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--num_poi_types", type=int,   default=456)
    parser.add_argument("--embed_dim",     type=int,   default=5)
    parser.add_argument("--hidden",        type=int,   default=128)
    parser.add_argument("--num_blocks",    type=int,   default=2)
    parser.add_argument("--k_temporal",    type=int,   default=3)
    parser.add_argument("--cheb_k",        type=int,   default=2)
    parser.add_argument("--dropout",       type=float, default=0)
    parser.add_argument("--embed_dropout", type=float, default=0)
    parser.add_argument("--model_path",    type=str,   default="./model/compare/stgcn_wztp.pth")
    parser.add_argument("--test_dir",      type=str,   default="./graph_data/LA/SF_test")
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STGCNForEdges(
        num_nodes=args.num_nodes,   in_feat=args.input_dim,
        hid_channels=args.hidden,   num_blocks=args.num_blocks,
        k_temporal=args.k_temporal, cheb_k=args.cheb_k,
        dropout=args.dropout,       num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,   emb_dropout=args.embed_dropout,
    ).to(device)
    model = load_model(model, args.model_path, device, strict=False)

    test_loader = get_dataloader(
        gpickle_dir=[args.test_dir],
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[args.window_size],
        num_workers=1,
    )

    m = evaluate(model, test_loader, device)

    print(f"Overall   MSE: {m['overall_MSE']:.8f}  MAE: {m['overall_MAE']:.8f}  NLL: {m['overall_NLL']:.8f}")
    print(f"Top-1%    MSE: {m['top1_MSE']:.8f}  MAE: {m['top1_MAE']:.8f}  NLL: {m['top1_NLL']:.8f}")


if __name__ == "__main__":
    main()