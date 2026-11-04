import argparse
import math
from pathlib import Path
import torch
from torch.special import gammaln
from graph_data_loader_slide_SF import get_dataloader
from model import TCGCNTransformer

EPS = 1e-6
ARCH = dict(input_dim=11, temporal_hidden_dim1=128, temporal_hidden_dim2=256, temporal_dropout_rate=0.1, kernel_size=3,
            gcn_hidden_dim1=512, gcn_hidden_dim2=256, gcn_dropout_rate=0.2, decoder_hidden_dim=256, edge_output_dim=1,
            decoder_dropout_rate=0.08, attention_dropout_rate=0.04, num_heads=4, num_layers=2, embed_dim=5, emb_dropout=0.05)
MODELS = [("LA pretrained, zero-shot", "pre_la.pth", (3.849, 12.696, 2.008, 0.248, 0.274, 0.227)),
          ("Cold start, 9 days", "cold_start_sf_9d.pth", (2.388, 6.893, 1.450, 0.225, 0.339, 0.218)),
          ("Cold start + RL, 9 days", "rl_sf_9d.pth", (2.336, 6.772, 1.441, 0.240, 0.375, 0.237)),
          ("Supervised, full month", "sl_sf_full.pth", (2.349, 7.141, 1.415, 0.290, 0.385, 0.291))]
KEYS = ["top1_NLL", "top1_MSE", "top1_MAE", "overall_NLL", "overall_MSE", "overall_MAE"]


def ztp_nll(y, lam):
    y = y.float().clamp_min(1.0)
    lam = lam.float().clamp_min(EPS)
    log_p = y * torch.log(lam) - lam - gammaln(y + 1.0)
    norm = -torch.log1p(-torch.exp(-lam).clamp_max(1 - 1e-6))
    return -(log_p + norm)


def ztp_mean(lam):
    lam = lam.float()
    return lam / (1.0 - torch.exp(-lam)).clamp_min(EPS)


def last_step_mask(edge_index, B, N, T):
    masks = []
    for b in range(B):
        s = (b * T + T - 1) * N
        e = s + N
        masks.append((edge_index[0] >= s) & (edge_index[0] < e) & (edge_index[1] >= s) & (edge_index[1] < e))
    return torch.stack(masks, 0).any(0)


def metrics(y, lam):
    mu = ztp_mean(lam)
    nll = ztp_nll(y, lam)
    d = mu - y
    out = {"overall_NLL": nll.mean().item(), "overall_MSE": (d ** 2).mean().item(), "overall_MAE": d.abs().mean().item()}
    k = max(1, int(math.ceil(0.01 * y.numel())))
    idx = torch.topk(y, k, largest=True, sorted=False).indices
    out.update({"top1_NLL": nll[idx].mean().item(), "top1_MSE": (d[idx] ** 2).mean().item(), "top1_MAE": d[idx].abs().mean().item(),
                "n_edges": int(y.numel()), "top1_threshold": y[idx].min().item()})
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, lams = [], []
    for x, ei, ea in loader:
        x, ei, ea = x.to(device), ei.to(device), ea.to(device)
        B, N, _, T = x.shape
        y = ea[last_step_mask(ei, B, N, T)]
        lam = model(x, ei, tgt_mask=None)
        if lam.numel() == 0 or lam.numel() != y.numel():
            continue
        ys.append(y.float().cpu())
        lams.append(lam.float().clamp_min(EPS).cpu())
    return metrics(torch.cat(ys), torch.cat(lams))


def find_graph_dir(root):
    root = Path(root)
    best, n_best = None, 0
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        n = len(list(d.glob("*.gpickle")))
        if n > n_best:
            best, n_best = d, n
    if best is None:
        raise SystemExit(f"no .gpickle files found under {root}")
    files = sorted(best.glob("*.gpickle"))
    print(f"test graphs: {best}  ({n_best} files, {files[0].name} .. {files[-1].name})")
    return str(best)


def find_model(model_dir, name):
    hits = sorted(Path(model_dir).rglob(name))
    return hits[0] if hits else None


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="./SF_test")
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--num_poi_types", type=int, default=456)
    ap.add_argument("--window", type=int, default=12)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdir = find_graph_dir(a.test_dir)
    loader = get_dataloader(gpickle_dir=[gdir], batch_size=1, input_dim=ARCH["input_dim"], window_sizes=[a.window], num_workers=1)
    print(f"{'model':26s} {'metric':12s} {'this run':>10s} {'paper':>10s} {'diff':>8s}")
    for name, fname, ref in MODELS:
        path = find_model(a.model_dir, fname)
        if path is None:
            print(f"{name:26s} missing: {fname} not found under {a.model_dir}")
            continue
        model = TCGCNTransformer(num_poi_types=a.num_poi_types, **ARCH).to(device)
        sd = load_state(path)
        n_rows = sd["poi_embedding.weight"].shape[0] if "poi_embedding.weight" in sd else None
        if n_rows is not None and n_rows != a.num_poi_types + 1:
            model = TCGCNTransformer(num_poi_types=n_rows - 1, **ARCH).to(device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"{name}: missing keys {list(missing)[:5]} unexpected keys {list(unexpected)[:5]}")
        m = evaluate(model, loader, device)
        for k, r in zip(KEYS, ref):
            print(f"{name:26s} {k:12s} {m[k]:10.3f} {r:10.3f} {100 * (m[k] / r - 1):+7.1f}%")
        print(f"{'':26s} edges {m['n_edges']:,}  top-1% threshold {m['top1_threshold']:.0f}")
        print()


if __name__ == "__main__":
    main()
