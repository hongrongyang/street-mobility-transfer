import csv
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import pytz
import torch
import networkx as nx

from model import TCGCNTransformer

# ---------------------- config ----------------------

# SF bounding box for coordinate normalisation
GLOBAL_BOUNDS = (37.7000836, 37.8619592203417, -123.0577788, -122.351631)

FEATURE_RANGES = {
    'num_people':  {'min': 0.0,   'max': 2116.0},
    'temperature': {'min': 1.330, 'max': 23.009},
    'precipitation': {'min': 0.0, 'max': 5.761},
    'wind_speed':  {'min': 0.044, 'max': 13.653},
}

POI_MAP_PATH = "./POI_data/poi_type_mapping_la_to_sf.pkl"
EPS = 1e-6

# node pairs used in the case study (1-indexed node IDs)
DEFAULT_PAIRS: List[Tuple[int, int]] = [
    (8932,  23804),  # school <-> bus stop
    (182,   27121),  # school <-> sports centre
    (24161, 26926),  # fuel   <-> parking lot
]


# ---------------------- time utilities ----------------------

def local_date_to_utc_hour_files(local_date_str: str, tz_str="America/Los_Angeles") -> List[str]:
    # convert a local date (e.g. 20230329 PDT) to 24 UTC timestamp strings
    tz = pytz.timezone(tz_str)
    base_date = datetime.strptime(local_date_str, "%Y%m%d")
    utc_list = []
    for h in range(24):
        local_dt = tz.localize(base_date + timedelta(hours=h))
        utc_dt = local_dt.astimezone(pytz.utc)
        utc_list.append(utc_dt.strftime("%Y%m%d_%H%M%S"))
    return utc_list


def hour_files_for_local_day(data_dir: Path, local_date_str: str) -> List[Path]:
    utc_list = local_date_to_utc_hour_files(local_date_str)
    files = []
    for ts in utc_list:
        day, hm = ts.split("_")
        f = data_dir / f"graph_{day}_{hm}.gpickle"
        files.append(f if f.exists() else None)
    return files


def hour_files_prev_day_tail(data_dir: Path, date_str: str, k: int) -> List[Path]:
    prev = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    tail = []
    for h in range(24 - k, 24):
        f = data_dir / f"graph_{prev}_{h:02d}0000.gpickle"
        tail.append(f if f.exists() else None)
    return tail


def load_window_with_backfill(data_dir: Path, date_str: str,
                              files_today: List[Path], end_idx: int, T: int) -> List[Path]:
    # build a T-length window ending at end_idx, borrowing from the previous day if needed
    s = end_idx - T + 1
    if s >= 0:
        win = files_today[s:end_idx + 1]
        return [] if any(p is None for p in win) else win
    prev_tail = hour_files_prev_day_tail(data_dir, date_str, -s)
    win = prev_tail + files_today[0:end_idx + 1]
    return [] if any(p is None for p in win) else win


# ---------------------- graph / feature utilities ----------------------

def ztp_mean(lam: torch.Tensor) -> torch.Tensor:
    # E[Y | Y>0] = lam / (1 - exp(-lam))
    lam = lam.clamp_min(EPS)
    return lam / (1.0 - torch.exp(-lam).clamp_max(1 - 1e-6))


def coerce_int(x) -> int:
    if x is None:
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


def get_edge_flow(attr: dict, flow_keys: List[str]) -> int:
    for k in flow_keys:
        if k in attr:
            return coerce_int(attr[k])
    return 0


def sum_or_nan(x, y):
    return math.nan if any(math.isnan(v) for v in [x, y]) else (x + y)


def build_inputs_from_files(files: List[Path], poi_types: Dict[str, int]):
    T = len(files)
    graphs = [pickle.load(open(fp, "rb")) for fp in files]
    N = graphs[0].number_of_nodes()

    X_seq, eidx_bt_list, eidx_raw_list = [], [], []
    for t, g in enumerate(graphs):
        feats = []
        for _, attr in g.nodes(data=True):
            poi_idx = poi_types.get(attr.get("poi_type", "unknown"), 456)
            nlat = (attr.get("lat", 0.0) - GLOBAL_BOUNDS[0]) / (GLOBAL_BOUNDS[1] - GLOBAL_BOUNDS[0])
            nlon = (attr.get("lon", 0.0) - GLOBAL_BOUNDS[2]) / (GLOBAL_BOUNDS[3] - GLOBAL_BOUNDS[2])
            npeople = (float(attr.get("num_people", 0.0)) - FEATURE_RANGES['num_people']['min']) / \
                      (FEATURE_RANGES['num_people']['max'] - FEATURE_RANGES['num_people']['min'])
            ntemp = (float(attr.get("temperature", 0.0)) - FEATURE_RANGES['temperature']['min']) / \
                    (FEATURE_RANGES['temperature']['max'] - FEATURE_RANGES['temperature']['min'])
            precipitation = min(float(attr.get("precipitation", 0.0)), 50.0)
            logp = float(torch.log1p(torch.tensor(precipitation)).item()) if precipitation >= 0 else 0.0
            nprec = min(1.0, max(0.0, (logp - FEATURE_RANGES['precipitation']['min']) /
                                 (FEATURE_RANGES['precipitation']['max'] - FEATURE_RANGES['precipitation']['min'])))
            nwind = (float(attr.get("wind_speed", 0.0)) - FEATURE_RANGES['wind_speed']['min']) / \
                    (FEATURE_RANGES['wind_speed']['max'] - FEATURE_RANGES['wind_speed']['min'])
            feats.append([npeople, poi_idx, nlat, nlon, ntemp, nprec, nwind])
        X_seq.append(torch.tensor(feats, dtype=torch.float))

        edges = list(g.edges(data=True))
        if len(edges) == 0:
            eidx_raw = torch.empty((2, 0), dtype=torch.long)
        else:
            eidx_raw = torch.tensor(
                [(u - 1, v - 1) for (u, v, _) in edges], dtype=torch.long
            ).t().contiguous()
        eidx_raw_list.append(eidx_raw)
        eidx_bt_list.append(eidx_raw + t * N)

    x = torch.stack(X_seq, dim=-1).unsqueeze(0)
    edge_index_bt = torch.cat(eidx_bt_list, dim=1) if eidx_bt_list else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index_bt, eidx_raw_list[-1]


def true_flow_for_pair(graph_path: Path, a: int, b: int, flow_keys: List[str]) -> Tuple[int, int]:
    G = nx.read_gpickle(graph_path)
    a2b = b2a = 0
    for u, v, d in G.edges(data=True):
        if u == a and v == b:
            a2b += get_edge_flow(d, flow_keys)
        elif u == b and v == a:
            b2a += get_edge_flow(d, flow_keys)
    return a2b, b2a


def predict_pair(model, device, files_window, pair_ab_0based, poi_types):
    x, eidx_bt, eidx_last_raw = build_inputs_from_files(files_window, poi_types)
    x = x.to(device)
    eidx_bt = eidx_bt.to(device)

    with torch.no_grad():
        lam_last = model(x, eidx_bt).clamp_min(EPS)
        mu_last = ztp_mean(lam_last)
        pred_last = torch.round(mu_last).clamp(min=0).cpu().numpy()
        lam_np = lam_last.cpu().numpy()

    uv = eidx_last_raw.cpu().numpy().T
    a0, b0 = pair_ab_0based
    mask_ab = (uv[:, 0] == a0) & (uv[:, 1] == b0)
    mask_ba = (uv[:, 0] == b0) & (uv[:, 1] == a0)

    def pick(arr, mask):
        return float(arr[mask][0]) if mask.any() else 0.0

    return (
        pick(pred_last, mask_ab), pick(pred_last, mask_ba),
        pick(lam_np,   mask_ab), pick(lam_np,   mask_ba),
    )


# ---------------------- main ----------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date",        type=str, default="20230329")
    p.add_argument("--data_dir",    type=str, default="./graph_data/SF_test")

    p.add_argument("--model_path",  type=str, default="./model/rl_sf_9d.pth")
    p.add_argument("--out_dir",     type=str, default="./pre_data/rl_sf_9d")
    # p.add_argument("--model_path", type=str, default="./model/sl_sf_full.pth")
    # p.add_argument("--out_dir", type=str, default="./pre_data/sl_sf_full")

    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--flow_keys",   type=str, nargs="+", default=["population_flow"])
    p.add_argument("--pairs",       type=int, nargs="+", default=None,
                   help="flat list of node-id pairs, e.g. --pairs 8932 23804 182 27121")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # parse --pairs if provided, otherwise use DEFAULT_PAIRS
    if args.pairs:
        ids = args.pairs
        pairs = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
    else:
        pairs = DEFAULT_PAIRS

    files_24 = hour_files_for_local_day(data_dir, args.date)
    if not any(f is not None for f in files_24):
        print(f"[Error] No hourly files found for {args.date}.")
        return

    with open(POI_MAP_PATH, "rb") as f:
        poi_types = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TCGCNTransformer(
        input_dim=11, temporal_hidden_dim1=128, temporal_hidden_dim2=256,
        temporal_dropout_rate=0.0, kernel_size=3, gcn_hidden_dim1=512,
        gcn_hidden_dim2=256, gcn_dropout_rate=0.0, decoder_hidden_dim=256,
        edge_output_dim=1, num_heads=4, num_layers=2, decoder_dropout_rate=0.0,
        num_poi_types=456, embed_dim=5, emb_dropout=0.0, attention_dropout_rate=0.0,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    hours_labels = [f"{h:02d}:00" for h in range(24)]
    CSV_HEADER = ["date", "hour", "a_id", "b_id",
                  "true_ab", "true_ba", "true_sum",
                  "pred_ab", "pred_ba", "pred_sum",
                  "lambda_ab", "lambda_ba"]

    combined_rows = []

    for (a, b) in pairs:
        rows = []
        for h in range(24):
            fpath = files_24[h]
            if fpath is None:
                t_ab = t_ba = p_ab = p_ba = lam_ab = lam_ba = math.nan
            else:
                t_ab, t_ba = true_flow_for_pair(fpath, a, b, args.flow_keys)
                window = load_window_with_backfill(data_dir, args.date, files_24, h, args.window_size)
                if not window:
                    p_ab = p_ba = lam_ab = lam_ba = math.nan
                else:
                    p_ab, p_ba, lam_ab, lam_ba = predict_pair(
                        model=model, device=device, files_window=window,
                        pair_ab_0based=(a - 1, b - 1), poi_types=poi_types,
                    )

            row = {
                "date": args.date, "hour": hours_labels[h],
                "a_id": a, "b_id": b,
                "true_ab": t_ab, "true_ba": t_ba, "true_sum": sum_or_nan(t_ab, t_ba),
                "pred_ab": p_ab, "pred_ba": p_ba, "pred_sum": sum_or_nan(p_ab, p_ba),
                "lambda_ab": lam_ab, "lambda_ba": lam_ba,
            }
            rows.append(row)
            combined_rows.append(row)

        out_csv = out_dir / f"pair_{a}_{b}_{args.date}_hourly.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {out_csv}")

    all_csv = out_dir / f"pairs_all_{args.date}_hourly.csv"
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(combined_rows)
    print(f"Saved: {all_csv}")


if __name__ == "__main__":
    main()