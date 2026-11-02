import argparse
import json
import math
import random
import numpy as np
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
from graph_data_loader_slide_LA import get_dataloader

EPS = 1e-6


class EdgeMLPForEdges(nn.Module):
    def __init__(self, num_poi_types=456, embed_dim=5, hidden=256, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.N = 1
        self.poi_emb = nn.Embedding(num_poi_types + 1, embed_dim)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.node_mlp = nn.Linear(6 * 3 + embed_dim, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def node_features(self, x):
        poi = self.emb_drop(self.poi_emb(x[:, :, 1, -1].long()))
        num = torch.cat([x[:, :, :1, :], x[:, :, 2:, :]], dim=2)
        feats = torch.cat([num[..., -1], num.mean(-1), num.max(-1).values, poi], dim=-1)
        return F.gelu(self.node_mlp(feats))

    def forward(self, x, edge_index):
        B, N, _, T = x.shape
        h = self.node_features(x)
        outs = []
        for t in range(T):
            t0, t1 = t * N, (t + 1) * N
            m = (edge_index[0] >= t0) & (edge_index[0] < t1) & (edge_index[1] >= t0) & (edge_index[1] < t1)
            e = edge_index[:, m] - t0
            if e.numel() == 0:
                continue
            src, dst = h[:, e[0], :], h[:, e[1], :]
            raw = self.edge_mlp(torch.cat([src, dst, (src - dst).abs()], dim=-1)).squeeze(-1)
            outs.append(F.softplus(raw) + EPS)
        if not outs:
            return x.new_zeros((0,))
        return torch.cat(outs, dim=1).reshape(-1)


def infer_kwargs(sd):
    emb = sd["poi_emb.weight"]
    hidden = sd["edge_mlp.0.weight"].shape[0]
    return dict(num_poi_types=emb.shape[0] - 1, embed_dim=emb.shape[1], hidden=hidden)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ztp_nll(y, lam):
    with torch.autocast(device_type="cuda", enabled=False):
        y = y.float().clamp_min(1.0)
        lam = lam.float().clamp_min(EPS)
        log_p = y * torch.log(lam) - lam - gammaln(y + 1.0)
        norm = -torch.log1p(-torch.exp(-lam).clamp_max(1 - 1e-6))
        return -(log_p + norm)


def edge_weights(y, tail_alpha, w_max, tail_min_count, tail_max_count, w_y1, w_y2):
    y = y.float()
    w = torch.ones_like(y)
    m = (y >= tail_min_count) & (y <= tail_max_count)
    if m.any():
        w[m] = torch.clamp(1.0 + tail_alpha * torch.log1p(y[m] - 1.0), 1.0, w_max)
    w[y < 2] *= w_y1
    w[y == 2] *= w_y2
    return w


def wztp_nll(y, lam, **loss_cfg):
    return ztp_nll(y, lam) * edge_weights(y, **loss_cfg)


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


class LastStep(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, ei):
        B, N, _, T = x.shape
        self.base.N = N
        lam_grouped = self.base(x, ei)
        step = torch.div(ei[0], N, rounding_mode="floor")
        order = torch.argsort(step, stable=True)
        lam = torch.empty_like(lam_grouped)
        lam[order] = lam_grouped
        return lam[last_step_mask(ei, B, N, T)]


def build_optimizer(model, lr, weight_decay):
    decay, no_decay, emb = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        if "poi_emb" in ln:
            emb.append(p)
        elif p.ndim < 2 or ln.endswith(".bias") or "norm" in ln:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0},
                              {"params": emb, "weight_decay": 0.0}], lr=lr)


def train_epoch(model, loader, opt, device, loss_cfg, clip=2.0):
    model.train()
    tot, n_tot = 0.0, 0
    for x, ei, ea in loader:
        opt.zero_grad(set_to_none=True)
        x, ei, ea = x.to(device), ei.to(device), ea.to(device)
        B, N, _, T = x.shape
        y = ea[last_step_mask(ei, B, N, T)]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lam = model(x, ei)
        if lam.numel() == 0 or lam.numel() != y.numel():
            continue
        lv = wztp_nll(y, lam.float().clamp_min(EPS), **loss_cfg)
        lv.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += lv.sum().item()
        n_tot += y.numel()
    return tot / max(1, n_tot)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, lams = [], []
    for x, ei, ea in loader:
        x, ei, ea = x.to(device), ei.to(device), ea.to(device)
        B, N, _, T = x.shape
        y = ea[last_step_mask(ei, B, N, T)]
        lam = model(x, ei)
        if lam.numel() == 0 or lam.numel() != y.numel():
            continue
        ys.append(y.float().cpu())
        lams.append(lam.float().clamp_min(EPS).cpu())
    return torch.cat(ys), torch.cat(lams)


def metrics(y, lam, pcts=(0.10, 0.05, 0.01, 0.005, 0.001)):
    mu = ztp_mean(lam)
    nll = ztp_nll(y, lam)
    d = mu - y
    out = {"n_edges": int(y.numel()), "overall_MSE": (d ** 2).mean().item(),
           "overall_MAE": d.abs().mean().item(), "overall_NLL": nll.mean().item()}
    M = y.numel()
    for p in pcts:
        k = max(1, int(math.ceil(p * M)))
        idx = torch.topk(y, k, largest=True, sorted=False).indices
        tag = f"top{p * 100:g}"
        out[f"{tag}_MSE"] = (d[idx] ** 2).mean().item()
        out[f"{tag}_MAE"] = d[idx].abs().mean().item()
        out[f"{tag}_NLL"] = nll[idx].mean().item()
        out[f"{tag}_threshold"] = y[idx].min().item()
    return out


def evaluate(model, loader, device):
    y, lam = collect_predictions(model, loader, device)
    return metrics(y, lam)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def fit(model, train_loader, val_loader, device, loss_cfg, lr, weight_decay, epochs, patience_lr,
        patience_stop, factor, best_path, last_path, tag):
    opt = build_optimizer(model, lr, weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=factor, patience=patience_lr)
    best_w, best_epoch, bad, history = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, opt, device, loss_cfg)
        y, lam = collect_predictions(model, val_loader, device)
        v_unw = ztp_nll(y, lam).mean().item()
        v_w = wztp_nll(y, lam, **loss_cfg).mean().item()
        lr_now = opt.param_groups[0]["lr"]
        sched.step(v_w)
        improved = v_w < best_w - 1e-8
        if improved:
            best_w, best_epoch, bad = v_w, ep, 0
            torch.save(model.base.state_dict(), best_path)
        else:
            bad += 1
        torch.save(model.base.state_dict(), last_path)
        history.append({"epoch": ep, "train_wztp": tl, "val_ztp": v_unw, "val_wztp": v_w, "lr": lr_now})
        print(f"[{tag}] epoch {ep}/{epochs}  train wZTP {tl:.6f}  val ZTP {v_unw:.6f}  val wZTP {v_w:.6f}"
              f"{' *' if improved else ''}  lr {lr_now:.2e}", flush=True)
        if bad >= patience_stop:
            print(f"[{tag}] early stop at epoch {ep}; best epoch {best_epoch}", flush=True)
            break
    return best_epoch, best_w, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="./graph_data/LA/train")
    ap.add_argument("--val_dir", default="./graph_data/LA/val")
    ap.add_argument("--out", default="./final_model/compare/edgemlp_wztp.pth")
    ap.add_argument("--num_poi_types", type=int, default=456)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--w_y1", type=float, default=0.15)
    ap.add_argument("--w_y2", type=float, default=0.15)
    ap.add_argument("--tail_alpha", type=float, default=2.2)
    ap.add_argument("--w_max", type=float, default=4.5)
    ap.add_argument("--tail_min_count", type=int, default=3)
    ap.add_argument("--tail_max_count", type=int, default=10)
    ap.add_argument("--patience_lr", type=int, default=3)
    ap.add_argument("--patience_stop", type=int, default=7)
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--emb_dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk = lambda d: get_dataloader(gpickle_dir=[d], batch_size=1, input_dim=11, window_sizes=[a.window], num_workers=1)
    train_loader, val_loader = mk(a.train_dir), mk(a.val_dir)
    base = EdgeMLPForEdges(num_poi_types=a.num_poi_types, embed_dim=5, hidden=a.hidden, dropout=a.dropout,
                           emb_dropout=a.emb_dropout)
    model = LastStep(base).to(device)
    print(f"[edgemlp] parameters: {sum(p.numel() for p in base.parameters()):,}", flush=True)
    loss_cfg = dict(tail_alpha=a.tail_alpha, w_max=a.w_max, tail_min_count=a.tail_min_count,
                    tail_max_count=a.tail_max_count, w_y1=a.w_y1, w_y2=a.w_y2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_epoch, best_w, history = fit(model, train_loader, val_loader, device, loss_cfg, a.lr, a.weight_decay,
                                         a.epochs, a.patience_lr, a.patience_stop, a.factor, out,
                                         out.with_name(out.stem + "_last.pth"), "edgemlp")
    save_json({"config": vars(a), "best_epoch": best_epoch, "best_val_wztp": best_w, "history": history,
                  "wall_time_s": time.time() - t0}, out.with_suffix(".json"))
    print(f"[edgemlp] best checkpoint: {out} (epoch {best_epoch}, val wZTP {best_w:.6f})", flush=True)


if __name__ == "__main__":
    main()
