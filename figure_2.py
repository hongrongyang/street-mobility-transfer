import os
import csv
import math
import pickle
from pathlib import Path
from os.path import normpath
from typing import List
from math import lgamma

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import rcParams

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily not installed. Basemap disabled.")

from model import TCGCNTransformer


# =====================
# Manual layout (0~1 figure coords)
# =====================
LAYOUT = {
    "a": [0.04, 0.64, 0.42, 0.334],
    "b": [0.54, 0.64, 0.42, 0.334],

    "c1": [0.04, 0.45, 0.19, 0.135],
    "c2": [0.27, 0.45, 0.19, 0.135],
    "c3": [0.54, 0.45, 0.19, 0.135],
    "c4": [0.77, 0.45, 0.19, 0.135],

    "d1": [0.04, 0.2475, 0.19, 0.135],
    "d2": [0.27, 0.2475, 0.19, 0.135],
    "d3": [0.54, 0.2475, 0.19, 0.135],
    "d4": [0.77, 0.2475, 0.19, 0.135],

    "e1": [0.04, 0.045, 0.19, 0.135],
    "e2": [0.27, 0.045, 0.19, 0.135],
    "e3": [0.54, 0.045, 0.19, 0.135],
    "e4": [0.77, 0.045, 0.19, 0.135],
}

# =====================
# Style
# =====================
plt.style.use("seaborn-v0_8-whitegrid")

rcParams["font.family"]      = "sans-serif"
rcParams["font.size"]        = 12.5
rcParams["figure.dpi"]       = 600
rcParams["axes.linewidth"]   = 0.6
rcParams["axes.facecolor"]   = "white"
rcParams["figure.facecolor"] = "white"
rcParams["grid.color"]       = "#e8e8e8"
rcParams["grid.alpha"]       = 0.6
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6

FIGSIZE = (16, 16)

# =====================
# Paths
# =====================
save_prefix = "./SF"

gdir       = Path("./graph_data/SF_test")
window_size = 12
end_file_a  = gdir / "graph_20230329_200000.gpickle"
end_file_b  = gdir / "graph_20230330_050000.gpickle"

model_path = "./model/rl_sf_9d.pth"

# CS+RL predictions (9d transfer from LA)
csv1_rl = "./pre_data/rl_sf_9d/pair_182_27121_20230329_hourly.csv"
csv2_rl = "./pre_data/rl_sf_9d/pair_8932_23804_20230329_hourly.csv"
csv3_rl = "./pre_data/rl_sf_9d/pair_24161_26926_20230329_hourly.csv"

# SL baseline predictions (full SF supervision)
csv1_sl9 = "./pre_data/sl_sf_full/pair_182_27121_20230329_hourly.csv"
csv2_sl9 = "./pre_data/sl_sf_full/pair_8932_23804_20230329_hourly.csv"
csv3_sl9 = "./pre_data/sl_sf_full/pair_24161_26926_20230329_hourly.csv"

# =====================
# Pair metadata
# =====================
PAIR_LABELS = {
    (182, 27121):   "school ↔ sports center",
    (8932, 23804):  "bus stop ↔ school",
    (24161, 26926): "fuel ↔ parking plot",
}

PAIRS_INFO = [
    ((182, 27121),   pd.read_csv(csv1_rl), pd.read_csv(csv1_sl9),
     "school (A) ↔ sports center (B)"),
    ((8932, 23804),  pd.read_csv(csv2_rl), pd.read_csv(csv2_sl9),
     "bus stop (A) ↔ school (B)"),
    ((24161, 26926), pd.read_csv(csv3_rl), pd.read_csv(csv3_sl9),
     "fuel (A) ↔ parking plot (B)"),
]

# =====================
# Colors
# =====================
true_colors    = ["#1f77b4", "#2ca02c", "#6C5CE7"]
pred_color_rl  = "#C0392B"   # CS+RL solid line
pred_color_sl9 = "#555555"   # SL dashed line

cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Purples]
for i in range(len(cmaps)):
    cm = cmaps[i].copy()
    cm.set_bad("white")
    cm.set_under("white")
    cmaps[i] = cm

COLOR_MATCH         = "#9EABB5"
COLOR_MISMATCH      = "#D4854A"
EDGE_ALPHA_MATCH    = 0.50
EDGE_ALPHA_MISMATCH = 0.45
EDGE_WIDTH_MATCH    = 0.22
EDGE_WIDTH_MISMATCH = 0.20

NODE_COLOR   = "#8A9BB0"
NODE_ALPHA   = 0.28
NODE_SIZE_BG = 0.08
NODE_SIZE_HL = 22

HIGHLIGHT_PAIRS = [(182, 27121), (8932, 23804), (24161, 26926)]
PAIR_COLORS = {
    (182, 27121):   "#1f77b4",
    (8932, 23804):  "#2ca02c",
    (24161, 26926): "#6C5CE7",
}

ELLIPSE_COLOR = "#2C3E50"

TOL = 1
EPS = 1e-6


# =====================
# Config / normalisation constants (must match training)
# =====================
with open("./POI_data/poi_type_mapping_la_to_sf.pkl", "rb") as f:
    poi_types = pickle.load(f)

GLOBAL_BOUNDS = (37.7000836, 37.8619592203417, -123.0577788, -122.351631)
FEATURE_RANGES = {
    'num_people':    {'min': 0.0,   'max': 2116.0},
    'temperature':   {'min': 1.330, 'max': 23.009},
    'precipitation': {'min': 0.0,   'max': 5.761},
    'wind_speed':    {'min': 0.044, 'max': 13.653},
}


# =====================
# Helper functions
# =====================

def ztp_mean(lam: torch.Tensor) -> torch.Tensor:
    lam = lam.clamp_min(EPS)
    return lam / (1.0 - torch.exp(-lam).clamp_max(1 - 1e-6))


def ztp_pmf(y, lam):
    if lam <= 1e-12 or np.isnan(lam):
        return 0.0
    lam = max(float(lam), 1e-12)
    log_p = y * np.log(lam) - lam - lgamma(y + 1)
    log_norm = -np.log(1 - np.exp(-lam))
    return float(np.exp(log_p + log_norm))


def restore_coords(norm_lat, norm_lon, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    lat = norm_lat * (lat_max - lat_min) + lat_min
    lon = norm_lon * (lon_max - lon_min) + lon_min
    return float(lat), float(lon)


def compute_downtown_bbox_from_coords(coords_last, bins=40, pad_bins=1, extra_pad_frac=0.20):
    if coords_last is None or len(coords_last) < 5:
        return None
    lats = np.array([c[0] for c in coords_last], dtype=float)
    lons = np.array([c[1] for c in coords_last], dtype=float)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=bins)
    if H.size == 0:
        return None
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    ix0 = max(0, ix - pad_bins); ix1 = min(H.shape[0]-1, ix+pad_bins)
    iy0 = max(0, iy - pad_bins); iy1 = min(H.shape[1]-1, iy+pad_bins)
    x0, x1 = xedges[ix0], xedges[ix1+1]
    y0, y1 = yedges[iy0], yedges[iy1+1]
    xr = max(x1-x0, 1e-12); yr = max(y1-y0, 1e-12)
    x0 -= extra_pad_frac*xr; x1 += extra_pad_frac*xr
    y0 -= extra_pad_frac*yr; y1 += extra_pad_frac*yr
    return x0, x1, y0, y1


def load_window_files(gdir: Path, end_file: Path, window_size: int) -> List[Path]:
    all_files = sorted(Path(gdir).glob("*.gpickle"), key=lambda p: p.name)
    all_files = [Path(normpath(str(p))) for p in all_files]
    end_file  = Path(normpath(str(end_file)))
    if end_file not in all_files:
        raise ValueError(f"{end_file} not found in {gdir}")
    end_idx   = all_files.index(end_file)
    start_idx = end_idx - window_size + 1
    if start_idx < 0:
        raise ValueError(f"Not enough files before {end_file.name} for window_size={window_size}")
    return all_files[start_idx:end_idx + 1]


def build_inputs_from_files(files: List[Path]):
    T = len(files)
    graphs = []
    for fp in files:
        with open(fp, "rb") as f:
            graphs.append(pickle.load(f))

    node_ids = sorted(list(graphs[0].nodes()))
    id2idx   = {pid: i for i, pid in enumerate(node_ids)}
    N        = len(node_ids)

    X_seq, eidx_bt_list, eidx_raw_list, eattr_list = [], [], [], []
    coords_last = []

    for t, g in enumerate(graphs):
        feats = []
        for pid in node_ids:
            attr    = g.nodes[pid]
            poi_idx = poi_types.get(attr.get("poi_type", "unknown"), 456)
            nlat    = (attr.get("lat", 0.0) - GLOBAL_BOUNDS[0]) / (GLOBAL_BOUNDS[1] - GLOBAL_BOUNDS[0])
            nlon    = (attr.get("lon", 0.0) - GLOBAL_BOUNDS[2]) / (GLOBAL_BOUNDS[3] - GLOBAL_BOUNDS[2])
            npeople = (float(attr.get("num_people", 0.0)) - FEATURE_RANGES['num_people']['min']) / \
                      (FEATURE_RANGES['num_people']['max'] - FEATURE_RANGES['num_people']['min'])
            ntemp   = (float(attr.get("temperature", 0.0)) - FEATURE_RANGES['temperature']['min']) / \
                      (FEATURE_RANGES['temperature']['max'] - FEATURE_RANGES['temperature']['min'])
            precipitation = min(float(attr.get("precipitation", 0.0)), 50.0)
            logp    = float(torch.log1p(torch.tensor(precipitation)).item()) if precipitation >= 0 else 0.0
            nprec   = min(1.0, max(0.0, (logp - FEATURE_RANGES['precipitation']['min']) /
                                   (FEATURE_RANGES['precipitation']['max'] - FEATURE_RANGES['precipitation']['min'])))
            nwind   = (float(attr.get("wind_speed", 0.0)) - FEATURE_RANGES['wind_speed']['min']) / \
                      (FEATURE_RANGES['wind_speed']['max'] - FEATURE_RANGES['wind_speed']['min'])
            feats.append([npeople, poi_idx, nlat, nlon, ntemp, nprec, nwind])
            if t == T - 1:
                coords_last.append(restore_coords(nlat, nlon, GLOBAL_BOUNDS))

        X_seq.append(torch.tensor(feats, dtype=torch.float))

        edges = list(g.edges(data=True))
        if len(edges) == 0:
            eidx_raw = torch.empty((2, 0), dtype=torch.long)
            eattr    = torch.empty((0,),   dtype=torch.float)
        else:
            uv       = torch.tensor([(id2idx[u], id2idx[v]) for (u, v, _) in edges],
                                    dtype=torch.long).t().contiguous()
            eidx_raw = uv
            eattr    = torch.tensor([float(attr.get("population_flow", 0.0))
                                     for (_, _, attr) in edges], dtype=torch.float)

        eidx_raw_list.append(eidx_raw)
        eattr_list.append(eattr)
        eidx_bt_list.append(eidx_raw + t * N)

    x              = torch.stack(X_seq, dim=-1).unsqueeze(0)
    edge_index_bt  = torch.cat(eidx_bt_list, dim=1) \
                     if (len(eidx_bt_list) and eidx_bt_list[0].numel() > 0) \
                     else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index_bt, eidx_raw_list[-1], eattr_list[-1], coords_last, node_ids, id2idx


def compute_last_frame_for_endfile(model, device, gdir, end_file, window_size):
    files = load_window_files(gdir, end_file, window_size)
    x, eidx_bt, eidx_last_raw, edge_attr_last, coords_last, node_ids, id2idx = \
        build_inputs_from_files(files)
    x      = x.to(device)
    eidx_bt = eidx_bt.to(device)
    y_last  = edge_attr_last.to(device)
    with torch.no_grad():
        lam_last  = model(x, eidx_bt, tgt_mask=None).clamp_min(EPS)
        mu_last   = ztp_mean(lam_last)
        pred_last = torch.round(mu_last).clamp(min=0)
        assert pred_last.shape == y_last.shape
    return (pred_last.cpu().numpy(), y_last.cpu().numpy(),
            coords_last, eidx_last_raw.cpu().numpy(), node_ids, id2idx)


def edge_exists_in_last(eidx_last_raw_np, id2idx, a, b):
    if eidx_last_raw_np.size == 0:
        return False
    if (a not in id2idx) or (b not in id2idx):
        return False
    a0, b0 = id2idx[a], id2idx[b]
    uv = eidx_last_raw_np.T
    return bool((((uv[:, 0] == a0) & (uv[:, 1] == b0)) |
                 ((uv[:, 0] == b0) & (uv[:, 1] == a0))).any())


# =====================
# Network plot
# =====================
def plot_network_on_ax(
    ax,
    pred_last, true_last, coords_last, node_ids,
    eidx_last_raw_np, id2idx,
    panel_letter,
    title_text=None,
    pair_lw=1.10,
    pair_alpha=0.92,
    ellipse_lw=0.80,
    panel_xy=(-0.05, 1.04),
    panel_fs=32,
    title_xy=(0.5, 1.01),
    title_ha="center",
    title_va="bottom",
    title_fs=14,
    legend_loc="lower left",
    legend_anchor=(0.05, 0.95),
    legend_ncol=1,
    legend_fs=12,
    show_legend=True,
    add_basemap=True,
    basemap_source=None,
    crop_bottom_ratio=0.0,
    crop_top_ratio=0.0,
):
    SPECIAL_PAIR = (182, 27121)

    if panel_letter:
        ax.text(panel_xy[0], panel_xy[1], panel_letter, transform=ax.transAxes,
                fontsize=panel_fs, fontweight="bold", ha="left", va="top")
    ax.axis("off")

    diff          = pred_last - true_last
    match_mask    = (np.abs(diff) <= TOL)
    mismatch_mask = ~match_mask

    G = nx.Graph()
    for pid, (lat, lon) in zip(node_ids, coords_last):
        G.add_node(pid, pos=(lon, lat))

    idx_pairs      = eidx_last_raw_np.T
    e_match_idx    = [tuple(e.tolist()) for e in idx_pairs[match_mask]]    if idx_pairs.shape[0] else []
    e_mismatch_idx = [tuple(e.tolist()) for e in idx_pairs[mismatch_mask]] if idx_pairs.shape[0] else []

    idx2id     = node_ids
    e_match    = [(idx2id[u], idx2id[v]) for (u, v) in e_match_idx]
    e_mismatch = [(idx2id[u], idx2id[v]) for (u, v) in e_mismatch_idx]

    for (u, v) in e_match + e_mismatch:
        G.add_edge(u, v)

    pos = nx.get_node_attributes(G, "pos")

    edge_xs, edge_ys = [], []
    for u, v in e_match + e_mismatch:
        if u in pos and v in pos:
            edge_xs.extend([pos[u][0], pos[v][0]])
            edge_ys.extend([pos[u][1], pos[v][1]])
    for (a, b) in HIGHLIGHT_PAIRS:
        if (a in pos) and (b in pos):
            edge_xs.extend([pos[a][0], pos[b][0]])
            edge_ys.extend([pos[a][1], pos[b][1]])

    if (SPECIAL_PAIR[0] in pos) and (SPECIAL_PAIR[1] in pos):
        xa, ya = pos[SPECIAL_PAIR[0]]; xb, yb = pos[SPECIAL_PAIR[1]]
        mx, my = (xa+xb)/2.0, (ya+yb)/2.0
        dx, dy = (xb-xa), (yb-ya)
        L      = float(np.hypot(dx, dy) + 1e-12)
        angle  = float(np.degrees(np.arctan2(dy, dx)))
        theta  = np.deg2rad(angle)
        width_sp = 2.60 * L; height_sp = 0.16 * L
        a_ell = 0.5 * width_sp; b_ell = 0.5 * height_sp
        dx_ext = float(np.sqrt((a_ell*np.cos(theta))**2 + (b_ell*np.sin(theta))**2))
        dy_ext = float(np.sqrt((a_ell*np.sin(theta))**2 + (b_ell*np.cos(theta))**2))
        edge_xs.extend([mx-dx_ext, mx+dx_ext])
        edge_ys.extend([my-dy_ext, my+dy_ext])

    if edge_xs and edge_ys:
        x_min, x_max = min(edge_xs), max(edge_xs)
        y_min, y_max = min(edge_ys), max(edge_ys)
        x_range = x_max - x_min; y_range = y_max - y_min
        x_pad = x_range * 0.10;  y_pad = y_range * 0.10
        display_xmin = x_min - x_pad; display_xmax = x_max + x_pad
        display_ymin = y_min - y_pad; display_ymax = y_max + y_pad
    else:
        display_xmin, display_xmax = GLOBAL_BOUNDS[2], GLOBAL_BOUNDS[3]
        display_ymin, display_ymax = GLOBAL_BOUNDS[0], GLOBAL_BOUNDS[1]

    y_range = display_ymax - display_ymin
    if crop_bottom_ratio > 0:
        display_ymin += y_range * crop_bottom_ratio
    if crop_top_ratio > 0:
        display_ymax -= y_range * crop_top_ratio

    fig  = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_aspect   = bbox.width / bbox.height
    data_aspect = (display_xmax - display_xmin) / (display_ymax - display_ymin)

    if data_aspect > ax_aspect:
        new_height = (display_xmax - display_xmin) / ax_aspect
        dy = (new_height - (display_ymax - display_ymin)) / 2.0
        display_ymin -= dy; display_ymax += dy
    else:
        new_width = (display_ymax - display_ymin) * ax_aspect
        dx = (new_width - (display_xmax - display_xmin)) / 2.0
        display_xmin -= dx; display_xmax += dx

    ax.set_xlim(display_xmin, display_xmax)
    ax.set_ylim(display_ymin, display_ymax)
    ax.set_aspect("equal", adjustable="datalim")

    if add_basemap and HAS_CONTEXTILY:
        try:
            source = basemap_source or ctx.providers.CartoDB.Positron
            ctx.add_basemap(ax, source=source, crs='EPSG:4326',
                            attribution=False, interpolation="bilinear")
        except Exception as e:
            print(f"Basemap failed: {e}")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    highlight_node_ids = set()
    for (a_id, b_id) in HIGHLIGHT_PAIRS:
        highlight_node_ids.add(a_id)
        highlight_node_ids.add(b_id)

    bg_nodes = [n for n in G.nodes() if n not in highlight_node_ids]
    hl_nodes_by_pair = {}
    for (a_id, b_id) in HIGHLIGHT_PAIRS:
        pair_nodes = [n for n in [a_id, b_id] if n in G.nodes()]
        if pair_nodes:
            hl_nodes_by_pair[(a_id, b_id)] = pair_nodes

    if bg_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=bg_nodes,
                               node_size=NODE_SIZE_BG, node_color=NODE_COLOR, alpha=NODE_ALPHA)

    for (a_id, b_id), pair_nodes in hl_nodes_by_pair.items():
        col = PAIR_COLORS[(a_id, b_id)]
        for n in pair_nodes:
            if n in pos:
                ax.plot(pos[n][0], pos[n][1], marker="o", markersize=2.5,
                        color=col, markeredgecolor="white", markeredgewidth=0.3,
                        alpha=0.95, zorder=20, linestyle="none")

    if e_match:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=e_match,
                               edge_color=COLOR_MATCH, width=EDGE_WIDTH_MATCH, alpha=EDGE_ALPHA_MATCH)
    if e_mismatch:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=e_mismatch,
                               edge_color=COLOR_MISMATCH, width=EDGE_WIDTH_MISMATCH, alpha=EDGE_ALPHA_MISMATCH)

    highlight_handles = []
    ls_exist  = "-"
    ls_absent = (0, (0.8, 1.4))
    view_xr   = max(display_xmax - display_xmin, 1e-12)
    view_yr   = max(display_ymax - display_ymin, 1e-12)

    for (a_id, b_id) in HIGHLIGHT_PAIRS:
        if (a_id not in pos) or (b_id not in pos):
            continue

        exists    = edge_exists_in_last(eidx_last_raw_np, id2idx, a_id, b_id)
        linestyle = ls_exist if exists else ls_absent

        xa, ya = pos[a_id]; xb, yb = pos[b_id]
        ax.plot([xa, xb], [ya, yb],
                color=PAIR_COLORS[(a_id, b_id)],
                linewidth=pair_lw * (1.1 if not exists else 1.0),
                alpha=0.9 if not exists else pair_alpha,
                linestyle=linestyle,
                solid_capstyle="butt", dash_capstyle="butt",
                zorder=15)

        mx, my = (xa+xb)/2.0, (ya+yb)/2.0
        dx, dy = (xb-xa), (yb-ya)
        L      = float(np.hypot(dx, dy) + 1e-12)
        angle  = float(np.degrees(np.arctan2(dy, dx)))

        if (a_id, b_id) == SPECIAL_PAIR or (b_id, a_id) == SPECIAL_PAIR:
            raw_w = 1.09 * L;  raw_h = 0.08 * L
            min_w = 1.09 * L;  min_h = 0.010 * view_yr
            max_w = np.inf;    max_h = 0.050 * view_yr
            ellipse_lw_this    = 0.8
            ellipse_face_alpha = 0.06
        else:
            raw_w = 1.85 * L;  raw_h = 0.38 * L
            min_w = 0.055 * view_xr;  min_h = 0.028 * view_yr
            max_w = 0.20  * view_xr;  max_h = 0.10  * view_yr
            ellipse_lw_this    = ellipse_lw
            ellipse_face_alpha = 0.09

        width  = float(np.clip(raw_w, min_w, max_w))
        height = float(np.clip(raw_h, min_h, max_h))

        ax.add_patch(Ellipse((mx, my), width=width, height=height, angle=angle,
                             fill=True, facecolor=ELLIPSE_COLOR, alpha=ellipse_face_alpha,
                             edgecolor=ELLIPSE_COLOR, linewidth=ellipse_lw_this, zorder=11))
        ax.add_patch(Ellipse((mx, my), width=width, height=height, angle=angle,
                             fill=False, edgecolor=ELLIPSE_COLOR, linewidth=ellipse_lw_this,
                             alpha=0.82, zorder=12))

        label_txt = PAIR_LABELS.get((a_id, b_id), f"pair ({a_id},{b_id})")
        highlight_handles.append(
            Line2D([0], [0], color=PAIR_COLORS[(a_id, b_id)], lw=pair_lw,
                   linestyle=linestyle, label=label_txt)
        )

    legend_handles = highlight_handles + [
        Line2D([0], [0], color=COLOR_MATCH,    lw=2, label="Match"),
        Line2D([0], [0], color=COLOR_MISMATCH, lw=2, label="Mismatch"),
    ]
    if show_legend:
        leg = ax.legend(handles=legend_handles, loc=legend_loc,
                        bbox_to_anchor=legend_anchor, bbox_transform=ax.transAxes,
                        frameon=True, fontsize=legend_fs, ncol=legend_ncol,
                        handlelength=1.8, handleheight=0.9,
                        borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.78)
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_boxstyle("round,pad=0.35")

    if title_text is not None:
        n_poi   = len(node_ids)
        n_edges = eidx_last_raw_np.shape[1] if eidx_last_raw_np.ndim == 2 else 0
        ax.text(title_xy[0], title_xy[1], f"{title_text} ({n_poi:,} POIs, {n_edges:,} edges)",
                transform=ax.transAxes, fontsize=title_fs, fontweight="bold",
                ha=title_ha, va=title_va)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_axis_on()


# =====================
# Pair row plot (subplots c/d/e)
# =====================
def plot_pair_row(fig, axes_row, row_letter, pair_title,
                  df_rl, df_sl9,
                  true_color, cmap,
                  y_values=np.arange(1, 7), y_buffer=1.2,
                  true_alpha=0.85, pred_alpha=0.80, colorbar_pad=0.02):

    ax_ab = axes_row[0]
    if row_letter:
        ax_ab.text(-0.187, 1.16, row_letter, transform=ax_ab.transAxes,
                   fontsize=30, fontweight="bold", ha="left", va="top")
    ax_ab.text(0.7, 1.12, pair_title, transform=ax_ab.transAxes,
               fontsize=14, fontweight="bold", ha="left", va="top")

    hours      = df_rl["hour"].values
    true_ab    = df_rl["true_ab"].values
    pred_ab_rl = df_rl["pred_ab"].values
    lam_ab_rl  = df_rl["lambda_ab"].values
    true_ba    = df_rl["true_ba"].values
    pred_ba_rl = df_rl["pred_ba"].values
    lam_ba_rl  = df_rl["lambda_ba"].values

    pred_ab_sl9 = df_sl9["pred_ab"].values
    pred_ba_sl9 = df_sl9["pred_ba"].values

    hours_ext = np.arange(0, 25)

    def ext(arr):
        return np.append(arr, arr[-1])

    true_ab_ext     = ext(true_ab)
    pred_ab_rl_ext  = ext(pred_ab_rl)
    pred_ab_sl9_ext = ext(pred_ab_sl9)

    # A→B step plot
    ax_ab.step(hours_ext, true_ab_ext,    label="True",       color=true_color,    linewidth=2.0, where="post", alpha=true_alpha)
    ax_ab.step(hours_ext, pred_ab_rl_ext, label="CS+RL (9d)", color=pred_color_rl, linewidth=1.8, where="post", alpha=pred_alpha, linestyle="-")
    ax_ab.step(hours_ext, pred_ab_sl9_ext,label="SL (full)",  color=pred_color_sl9,linewidth=1.1, where="post", alpha=0.55,       linestyle="--")

    dir_handle = Line2D([0], [0], color="none", label="A→B")
    handles, labels = ax_ab.get_legend_handles_labels()
    ax_ab.legend(handles + [dir_handle], labels + ["A→B"],
                 fontsize=9.5, loc="upper left", frameon=False,
                 handlelength=1.5, labelspacing=0.2)

    ax_ab.set_xlim(0, 24)
    ax_ab.set_xticks(range(0, 25, 4))
    ax_ab.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], rotation=45)
    ax_ab.grid(True, alpha=0.3)

    y_upper = max(1, int(np.ceil(max(np.nanmax(true_ab_ext),
                                     np.nanmax(pred_ab_rl_ext),
                                     np.nanmax(pred_ab_sl9_ext)) * y_buffer)))
    ax_ab.set_ylim(0, y_upper)
    ax_ab.set_yticks(range(0, y_upper + 1))
    ax_ab.set_ylabel("Flow count", fontweight="bold")

    # A→B ZTP heatmap
    ax_hm_ab = axes_row[1]
    y_min_h, y_max_h = y_values.min() - 0.5, y_values.max() + 0.5
    Z = np.array([[ztp_pmf(y, lam_ab_rl[h]) for y in y_values]
                  if (not np.isnan(lam_ab_rl[h]) and lam_ab_rl[h] > 1e-12)
                  else np.zeros_like(y_values)
                  for h in range(24)]).T
    Zm = np.ma.masked_array(Z, mask=(Z == 0))
    im = ax_hm_ab.imshow(Zm, cmap=cmap, origin="lower", aspect="auto",
                          extent=[0, 24, y_min_h, y_max_h],
                          vmin=0, vmax=1, interpolation="nearest")
    ax_hm_ab.set_xlim(0, 24)
    ax_hm_ab.set_xticks(range(0, 25, 4))
    ax_hm_ab.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], rotation=45)
    ax_hm_ab.set_yticks(y_values)
    ax_hm_ab.set_yticklabels([str(v) for v in y_values])
    ax_hm_ab.set_ylabel("Flow count", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_hm_ab, fraction=0.046, pad=colorbar_pad)
    cbar.set_label("Probability", fontweight="bold")

    # B→A step plot
    ax_ba = axes_row[2]
    true_ba_ext     = ext(true_ba)
    pred_ba_rl_ext  = ext(pred_ba_rl)
    pred_ba_sl9_ext = ext(pred_ba_sl9)

    ax_ba.step(hours_ext, true_ba_ext,    label="True",       color=true_color,    linewidth=2.0, where="post", alpha=true_alpha)
    ax_ba.step(hours_ext, pred_ba_rl_ext, label="CS+RL (9d)", color=pred_color_rl, linewidth=1.8, where="post", alpha=pred_alpha, linestyle="-")
    ax_ba.step(hours_ext, pred_ba_sl9_ext,label="SL (full)",  color=pred_color_sl9,linewidth=1.1, where="post", alpha=0.55,       linestyle="--")

    dir_handle2 = Line2D([0], [0], color="none", label="B→A")
    handles2, labels2 = ax_ba.get_legend_handles_labels()
    ax_ba.legend(handles2 + [dir_handle2], labels2 + ["B→A"],
                 fontsize=9.5, loc="upper left", frameon=False,
                 handlelength=1.5, labelspacing=0.2)

    ax_ba.set_xlim(0, 24)
    ax_ba.set_xticks(range(0, 25, 4))
    ax_ba.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], rotation=45)
    ax_ba.grid(True, alpha=0.3)

    y_upper2 = max(1, int(np.ceil(max(np.nanmax(true_ba_ext),
                                      np.nanmax(pred_ba_rl_ext),
                                      np.nanmax(pred_ba_sl9_ext)) * y_buffer)))
    ax_ba.set_ylim(0, y_upper2)
    ax_ba.set_yticks(range(0, y_upper2 + 1))
    ax_ba.set_ylabel("Flow count", fontweight="bold")

    # B→A ZTP heatmap
    ax_hm_ba = axes_row[3]
    Z2 = np.array([[ztp_pmf(y, lam_ba_rl[h]) for y in y_values]
                   if (not np.isnan(lam_ba_rl[h]) and lam_ba_rl[h] > 1e-12)
                   else np.zeros_like(y_values)
                   for h in range(24)]).T
    Zm2 = np.ma.masked_array(Z2, mask=(Z2 == 0))
    im2 = ax_hm_ba.imshow(Zm2, cmap=cmap, origin="lower", aspect="auto",
                           extent=[0, 24, y_min_h, y_max_h],
                           vmin=0, vmax=1, interpolation="nearest")
    ax_hm_ba.set_xlim(0, 24)
    ax_hm_ba.set_xticks(range(0, 25, 4))
    ax_hm_ba.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], rotation=45)
    ax_hm_ba.set_yticks(y_values)
    ax_hm_ba.set_yticklabels([str(v) for v in y_values])
    ax_hm_ba.set_ylabel("Flow count", fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax_hm_ba, fraction=0.046, pad=colorbar_pad)
    cbar2.set_label("Probability", fontweight="bold")

    for ax in axes_row:
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)
        ax.xaxis.labelpad = 4
        ax.yaxis.labelpad = 4


# =====================
# Main
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TCGCNTransformer(
        input_dim=11,
        temporal_hidden_dim1=128, temporal_hidden_dim2=256, temporal_dropout_rate=0.0,
        kernel_size=3, gcn_hidden_dim1=512, gcn_hidden_dim2=256, gcn_dropout_rate=0.0,
        decoder_hidden_dim=256, edge_output_dim=1,
        num_heads=4, num_layers=2, decoder_dropout_rate=0.0,
        num_poi_types=456, embed_dim=5, emb_dropout=0.0, attention_dropout_rate=0.0,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pred_a, true_a, coords_a, eidx_a, node_ids_a, id2idx_a = \
        compute_last_frame_for_endfile(model, device, gdir, end_file_a, window_size)
    pred_b, true_b, coords_b, eidx_b, node_ids_b, id2idx_b = \
        compute_last_frame_for_endfile(model, device, gdir, end_file_b, window_size)

    SF_CENTER_LON = -122.4194
    SF_CENTER_LAT =  37.7749
    MAX_DIST_KM   = 12

    def haversine(lon1, lat1, lon2, lat2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2-lon1; dlat = lat2-lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    def filter_network_data(coords, eidx, pred, true, node_ids, max_dist_km):
        dists      = [haversine(SF_CENTER_LON, SF_CENTER_LAT, lon, lat) for (lat, lon) in coords]
        keep_nodes = [i for i, d in enumerate(dists) if d <= max_dist_km]
        keep_set   = set(keep_nodes)
        new_node_ids = [node_ids[i] for i in keep_nodes]
        new_coords   = [coords[i]   for i in keep_nodes]
        old2new      = {old: new for new, old in enumerate(keep_nodes)}
        keep_edges   = []; edge_keep_indices = []
        for i in range(eidx.shape[1]):
            u, v = eidx[0, i], eidx[1, i]
            if u in keep_set and v in keep_set:
                keep_edges.append([old2new[u], old2new[v]])
                edge_keep_indices.append(i)
        new_eidx = np.array(keep_edges).T if keep_edges else np.empty((2, 0), dtype=int)
        return new_coords, new_eidx, pred[edge_keep_indices], true[edge_keep_indices], new_node_ids

    coords_a, eidx_a, pred_a, true_a, node_ids_a = \
        filter_network_data(coords_a, eidx_a, pred_a, true_a, node_ids_a, MAX_DIST_KM)
    coords_b, eidx_b, pred_b, true_b, node_ids_b = \
        filter_network_data(coords_b, eidx_b, pred_b, true_b, node_ids_b, MAX_DIST_KM)

    id2idx_a = {pid: i for i, pid in enumerate(node_ids_a)}
    id2idx_b = {pid: i for i, pid in enumerate(node_ids_b)}

    print(f"After filtering — a: {len(node_ids_a)} nodes, {len(pred_a)} edges")
    print(f"               — b: {len(node_ids_b)} nodes, {len(pred_b)} edges")

    fig = plt.figure(figsize=FIGSIZE)

    ax_a = fig.add_axes(LAYOUT["a"])
    ax_b = fig.add_axes(LAYOUT["b"])

    pair_axes = [
        [fig.add_axes(LAYOUT["c1"]), fig.add_axes(LAYOUT["c2"]),
         fig.add_axes(LAYOUT["c3"]), fig.add_axes(LAYOUT["c4"])],
        [fig.add_axes(LAYOUT["d1"]), fig.add_axes(LAYOUT["d2"]),
         fig.add_axes(LAYOUT["d3"]), fig.add_axes(LAYOUT["d4"])],
        [fig.add_axes(LAYOUT["e1"]), fig.add_axes(LAYOUT["e2"]),
         fig.add_axes(LAYOUT["e3"]), fig.add_axes(LAYOUT["e4"])],
    ]

    plot_network_on_ax(
        ax_a, pred_a, true_a, coords_a, node_ids_a, eidx_a, id2idx_a,
        panel_letter="a", panel_fs=30,
        title_text="Prediction at 13:00, 29 Mar 2023",
        panel_xy=(-0.082, 1.075), title_xy=(0.50, 1.02),
        legend_loc="lower left", legend_anchor=(0, 0.755),
        ellipse_lw=0.8, add_basemap=True,
        basemap_source=ctx.providers.CartoDB.PositronNoLabels,
        crop_bottom_ratio=0.3, crop_top_ratio=0.15,
    )

    plot_network_on_ax(
        ax_b, pred_b, true_b, coords_b, node_ids_b, eidx_b, id2idx_b,
        panel_letter="", panel_fs=30,
        title_text="Prediction at 22:00, 29 Mar 2023",
        panel_xy=(-0.082, 1.075), title_xy=(0.50, 1.02),
        legend_loc="lower left", legend_anchor=(0, 0.75),
        show_legend=False, ellipse_lw=0.8, add_basemap=True,
        basemap_source=ctx.providers.CartoDB.PositronNoLabels,
        crop_bottom_ratio=0.3, crop_top_ratio=0.15,
    )

    letters = ["b", "", ""]
    for i, ((_, df_rl, df_sl9, title), axes_row) in enumerate(zip(PAIRS_INFO, pair_axes)):
        plot_pair_row(
            fig=fig, axes_row=axes_row,
            row_letter=letters[i], pair_title=title,
            df_rl=df_rl, df_sl9=df_sl9,
            true_color=true_colors[i], cmap=cmaps[i],
            y_values=np.arange(1, 7), y_buffer=1.2,
            true_alpha=0.85, pred_alpha=0.80, colorbar_pad=0.02,
        )

    for j in range(4):
        pair_axes[-1][j].set_xlabel("Hour", fontweight="bold")

    plt.savefig(f"{save_prefix}.pdf", dpi=600, pad_inches=0.02)
    plt.savefig(f"{save_prefix}.png", dpi=600, pad_inches=0.02)
    plt.show()
    print(f"Saved: {save_prefix}.pdf / .png")


if __name__ == "__main__":
    main()