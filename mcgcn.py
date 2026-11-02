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
from STGCN import STBlock, STGCNForEdges
from graph_data_loader_slide_LA import get_dataloader

EPS = 1e-6
NUMERIC = [0, 2, 3, 4, 5, 6]
TEMPORAL = [0, 4, 5, 6]


class LearnedGraph(nn.Module):

    def __init__(self, num_nodes, dim=40, k=50, alpha=3.0, chunk=1024):
        super().__init__()
        self.emb1 = nn.Embedding(num_nodes, dim)
        self.emb2 = nn.Embedding(num_nodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.k, self.alpha, self.chunk = k, alpha, chunk

    def forward(self):
        v1 = torch.tanh(self.alpha * self.lin1(self.emb1.weight)).float()
        v2 = torch.tanh(self.alpha * self.lin2(self.emb2.weight)).float()
        N = v1.shape[0]
        k = min(self.k, N - 1)
        cols = []
        with torch.no_grad():
            for s in range(0, N, self.chunk):
                a = v1[s:s + self.chunk] @ v2.T - v2[s:s + self.chunk] @ v1.T
                a = F.relu(torch.tanh(self.alpha * a))
                cols.append(a.topk(k, dim=1).indices)
        col = torch.cat(cols, 0)
        row = torch.arange(N, device=col.device).unsqueeze(1).expand_as(col)
        row, col = row.reshape(-1), col.reshape(-1)
        a = (v1[row] * v2[col]).sum(-1) - (v2[row] * v1[col]).sum(-1)
        w = F.relu(torch.tanh(self.alpha * a))
        return row, col, w


class _SpMM(torch.autograd.Function):
    CHUNK = 400_000

    @staticmethod
    def forward(ctx, X, dst, src, w):
        ctx.save_for_backward(X, dst, src, w)
        out = torch.zeros_like(X)
        for s in range(0, dst.numel(), _SpMM.CHUNK):
            d, u, ww = dst[s:s + _SpMM.CHUNK], src[s:s + _SpMM.CHUNK], w[s:s + _SpMM.CHUNK]
            out.index_add_(0, d, X[u] * ww.to(X.dtype).unsqueeze(1))
        return out

    @staticmethod
    def backward(ctx, g):
        X, dst, src, w = ctx.saved_tensors
        gX = torch.zeros_like(X) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(w) if ctx.needs_input_grad[3] else None
        for s in range(0, dst.numel(), _SpMM.CHUNK):
            d, u, ww = dst[s:s + _SpMM.CHUNK], src[s:s + _SpMM.CHUNK], w[s:s + _SpMM.CHUNK]
            gd = g[d]
            if gX is not None:
                gX.index_add_(0, u, gd * ww.to(g.dtype).unsqueeze(1))
            if gw is not None:
                gw[s:s + _SpMM.CHUNK] = (gd.float() * X[u].float()).sum(1)
        return gX, None, None, gw


def propagate(x, dst, src, w, N):
    B, C, _, T = x.shape
    X = x.permute(2, 0, 1, 3).reshape(N, -1)
    return _SpMM.apply(X, dst, src, w).reshape(N, B, C, T).permute(1, 2, 0, 3)


class MixProp(nn.Module):

    def __init__(self, c_in, c_out, gdep=2, beta=0.05, dropout=0.2):
        super().__init__()
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1))
        self.gdep, self.beta = gdep, beta
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dst, src, w, N):
        deg = torch.ones(N, device=x.device, dtype=torch.float32).index_add_(0, dst, w.float())
        wn = w / deg[dst]
        self_w = (1.0 / deg).view(1, 1, N, 1).to(x.dtype)
        h, out = x, [x]
        for _ in range(self.gdep):
            h = self.beta * x + (1 - self.beta) * (h * self_w + propagate(h, dst, src, wn, N))
            out.append(h)
        return self.drop(self.mlp(torch.cat(out, dim=1)))


class GatedCausalConv(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.filt = nn.Conv2d(c, c, kernel_size=(1, 2), dilation=(1, dilation))
        self.gate = nn.Conv2d(c, c, kernel_size=(1, 2), dilation=(1, dilation))
        self.pad = dilation

    def forward(self, x):
        x = F.pad(x, (self.pad, 0, 0, 0))
        return torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))


class MCGCNForEdges(nn.Module):
    def __init__(self, num_nodes, num_poi_types=456, embed_dim=8, ctx_dim=32, hid_channels=128,
                 num_blocks=2, k_temporal=3, cheb_k=2, node_emb_dim=40, topk=50, gdep=2,
                 prop_alpha=0.05, tanh_alpha=3.0, grid=64, dilations=(1, 2),
                 dropout=0.2, emb_dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self._N = num_nodes
        c = ctx_dim
        self.register_buffer("_cfg", torch.tensor([topk, gdep, grid, len(dilations)], dtype=torch.long))
        self.register_buffer("_cfg_f", torch.tensor([prop_alpha, tanh_alpha], dtype=torch.float32))
        self.v_proj = nn.Conv2d(1, c, kernel_size=(1, 1))
        self.poi_emb = nn.Embedding(num_poi_types + 1, embed_dim)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.s_node = nn.Sequential(nn.Linear(embed_dim + 2, c), nn.GELU(), nn.Linear(c, c))
        self.s_fuse = nn.Linear(2 * c, c)
        self.grid = grid
        self.graph = LearnedGraph(num_nodes, node_emb_dim, topk, tanh_alpha)
        self.t_start = nn.Conv2d(len(TEMPORAL), c, kernel_size=(1, 1))
        self.t_tcn = nn.ModuleList([GatedCausalConv(c, d) for d in dilations])
        self.t_gc1 = nn.ModuleList([MixProp(c, c, gdep, prop_alpha, dropout) for _ in dilations])
        self.t_gc2 = nn.ModuleList([MixProp(c, c, gdep, prop_alpha, dropout) for _ in dilations])
        self.t_norm = nn.ModuleList([nn.LayerNorm(c) for _ in dilations])
        self.att_W = nn.Conv2d(c, c, kernel_size=(1, 1))
        self.att_w = nn.Conv2d(c, 1, kernel_size=(1, 1))
        blocks = [STBlock(c, hid_channels, k_temporal, cheb_k, dropout)]
        for _ in range(num_blocks - 1):
            blocks.append(STBlock(hid_channels, hid_channels, k_temporal, cheb_k, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * hid_channels, hid_channels), nn.GELU(),
                                      nn.Linear(hid_channels, 1))

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, n):
        if n != self.num_nodes:
            raise ValueError(f"MCGCN is defined for the {self.num_nodes} nodes of its training city "
                             f"(learned graph indexed by node identity); got a graph with {n} nodes")
        self._N = n

    def param_report(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        graph_emb = self.graph.emb1.weight.numel() + self.graph.emb2.weight.numel()
        return {"total": n(self), "node_identity_embeddings": graph_emb,
                "temporal_context_branch_excl_embeddings": n(self.graph) - graph_emb + n(self.t_start)
                + n(self.t_tcn) + n(self.t_gc1) + n(self.t_gc2) + n(self.t_norm),
                "spatial_context_branch": n(self.poi_emb) + n(self.s_node) + n(self.s_fuse),
                "fusion": n(self.att_W) + n(self.att_w) + n(self.v_proj),
                "stgcn_predictor": n(self.blocks), "edge_mlp": n(self.edge_mlp)}

    def spatial_context(self, x):
        B, N = x.shape[:2]
        poi = self.emb_drop(self.poi_emb(x[:, :, 1, -1].long()))
        latlon = x[:, :, 2:4, -1]
        s = self.s_node(torch.cat([poi, latlon], dim=-1))
        G = self.grid
        lo, hi = latlon.amin(dim=1, keepdim=True), latlon.amax(dim=1, keepdim=True)
        cell = ((latlon - lo) / (hi - lo + 1e-9) * G).floor().clamp_(0, G - 1).long()
        cid = cell[..., 0] * G + cell[..., 1]
        pooled = []
        for b in range(B):
            sf = s[b].float()
            summ = torch.zeros(G * G, sf.shape[-1], device=s.device).index_add_(0, cid[b], sf)
            cnt = torch.zeros(G * G, device=s.device).index_add_(0, cid[b], torch.ones(N, device=s.device))
            pooled.append((summ / cnt.clamp_min(1).unsqueeze(1))[cid[b]].to(s.dtype))
        s = self.s_fuse(torch.cat([s, torch.stack(pooled, 0)], dim=-1))
        return s.permute(0, 2, 1).unsqueeze(-1)

    def temporal_context(self, x):
        B, N, _, T = x.shape
        dst, src, w = self.graph()
        h = self.t_start(x[:, :, TEMPORAL, :].permute(0, 2, 1, 3))
        for tcn, gc1, gc2, norm in zip(self.t_tcn, self.t_gc1, self.t_gc2, self.t_norm):
            res = h
            z = tcn(h)
            z = gc1(z, dst, src, w, N) + gc2(z, src, dst, w, N)
            h = norm((z + res).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return h

    def fuse(self, v, s, t):
        T = v.shape[-1]
        z = torch.stack([v, s.expand(-1, -1, -1, T), t], dim=0)
        e = torch.stack([self.att_w(torch.tanh(self.att_W(zk))) for zk in z], dim=0)
        a = torch.softmax(e, dim=0)
        return (a * z).sum(0)

    def forward(self, x, edge_index):
        B, N, _, T = x.shape
        if N != self.num_nodes:
            raise ValueError(f"MCGCN: expected {self.num_nodes} nodes, got {N}")
        v = self.v_proj(x[:, :, 0:1, :].permute(0, 2, 1, 3))
        s = self.spatial_context(x)
        t = self.temporal_context(x)
        xh = self.fuse(v, s, t)
        eidx_bt_list, eidx_raw_list = STGCNForEdges._split_edges_by_time(edge_index, B, T, N, x.device)
        for blk in self.blocks:
            xh = blk(xh, eidx_bt_list, N)
        outs = []
        for ti in range(T):
            e = eidx_raw_list[ti]
            if e.numel() == 0:
                continue
            h_t = xh[:, :, :, ti].permute(0, 2, 1).contiguous()
            src_h, dst_h = h_t[:, e[0], :], h_t[:, e[1], :]
            raw = self.edge_mlp(torch.cat([src_h, dst_h, (src_h - dst_h).abs()], dim=-1)).squeeze(-1)
            outs.append(F.softplus(raw) + EPS)
        if not outs:
            return x.new_zeros((0,))
        return torch.cat(outs, dim=1).reshape(-1)


def infer_kwargs(sd):
    cfg, cfg_f = sd["_cfg"].tolist(), sd["_cfg_f"].tolist()
    emb = sd["poi_emb.weight"]
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    return dict(num_nodes=sd["graph.emb1.weight"].shape[0], num_poi_types=emb.shape[0] - 1,
                embed_dim=emb.shape[1], ctx_dim=sd["v_proj.weight"].shape[0],
                hid_channels=sd["edge_mlp.0.weight"].shape[0], num_blocks=n_blocks,
                k_temporal=sd["blocks.0.temp1.conv.weight"].shape[-1],
                cheb_k=len({k.split(".")[4] for k in sd if k.startswith("blocks.0.cheb.lins.")}),
                node_emb_dim=sd["graph.emb1.weight"].shape[1], topk=cfg[0], gdep=cfg[1], grid=cfg[2],
                dilations=tuple(2 ** i for i in range(cfg[3])), prop_alpha=cfg_f[0], tanh_alpha=cfg_f[1])



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
    ap.add_argument("--out", default="./final_model/compare/mcgcn_wztp.pth")
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
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emb_dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk = lambda d: get_dataloader(gpickle_dir=[d], batch_size=1, input_dim=11, window_sizes=[a.window], num_workers=1)
    train_loader, val_loader = mk(a.train_dir), mk(a.val_dir)
    num_nodes = next(iter(train_loader))[0].shape[1]
    base = MCGCNForEdges(num_nodes=num_nodes, num_poi_types=a.num_poi_types, embed_dim=8, ctx_dim=32,
                         hid_channels=128, num_blocks=2, k_temporal=3, cheb_k=2, node_emb_dim=40, topk=50,
                         gdep=2, prop_alpha=0.05, tanh_alpha=3.0, grid=64, dropout=a.dropout,
                         emb_dropout=a.emb_dropout)
    model = LastStep(base).to(device)
    print("[mcgcn] parameters: " + ", ".join(f"{k} {v:,}" for k, v in base.param_report().items()), flush=True)
    loss_cfg = dict(tail_alpha=a.tail_alpha, w_max=a.w_max, tail_min_count=a.tail_min_count,
                    tail_max_count=a.tail_max_count, w_y1=a.w_y1, w_y2=a.w_y2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best_epoch, best_w, history = fit(model, train_loader, val_loader, device, loss_cfg, a.lr, a.weight_decay,
                                         a.epochs, a.patience_lr, a.patience_stop, a.factor, out,
                                         out.with_name(out.stem + "_last.pth"), "mcgcn")
    save_json({"config": vars(a), "num_nodes": num_nodes, "best_epoch": best_epoch, "best_val_wztp": best_w,
                  "history": history, "wall_time_s": time.time() - t0}, out.with_suffix(".json"))
    print(f"[mcgcn] best checkpoint: {out} (epoch {best_epoch}, val wZTP {best_w:.6f})", flush=True)


if __name__ == "__main__":
    main()
