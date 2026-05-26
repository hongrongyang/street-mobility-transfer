import torch
from torch.utils.data import DataLoader, IterableDataset
import pickle
from pathlib import Path
import gc

with open("./POI_data/poi_type_mapping_la_to_sf_plus_fre.pkl", "rb") as f:
    poi_types = pickle.load(f)

class CombinedSlidingWindowIterableDataset(IterableDataset):
    def __init__(self, gpickle_dirs, input_dim=7, window_sizes=None, batch_size=1, global_lat_lon_bounds=None):
        self.gpickle_dirs = gpickle_dirs
        self.input_dim = input_dim
        self.window_sizes = window_sizes or [6]
        self.batch_size = batch_size
        self.global_lat_min, self.global_lat_max, self.global_lon_min, self.global_lon_max = global_lat_lon_bounds

        self.feature_ranges = {
            'num_people': {'min': 0.0, 'max': 2116.0},
            'temperature': {'min': 1.330, 'max': 23.009},
            'precipitation': {'min': 0.0, 'max': 5.761},
            'wind_speed': {'min': 0.044, 'max': 13.653}
        }

    def __len__(self):
        total_windows = 0
        for gpickle_dir, window_size in zip(self.gpickle_dirs, self.window_sizes):
            gpickle_files = list(Path(gpickle_dir).glob("*.gpickle"))
            if len(gpickle_files) < window_size:
                continue
            total_windows += len(gpickle_files) - window_size + 1
        return (total_windows + self.batch_size - 1) // self.batch_size if total_windows > 0 else 0

    def __iter__(self):
        for interval_id, (gpickle_dir, window_size) in enumerate(zip(self.gpickle_dirs, self.window_sizes)):
            gpickle_files = sorted(Path(gpickle_dir).glob("*.gpickle"))
            batch_node_window, batch_edge_index_window, batch_edge_attr_window = [], [], []
            cumulative_node_count = 0

            for i in range(0, len(gpickle_files) - window_size + 1):
                chunk_files = gpickle_files[i:i + window_size]
                x_window, edge_index_window, edge_attr_window = [], [], []

                for gpickle_file in chunk_files:
                    with open(gpickle_file, "rb") as f:
                        graph = pickle.load(f)

                    node_features = []
                    for _, attr in graph.nodes(data=True):
                        poi_type_idx = poi_types.get(attr.get("poi_type"), 463)
                        normalized_lat = (attr.get("lat", 0) - self.global_lat_min) / (
                            self.global_lat_max - self.global_lat_min)
                        normalized_lon = (attr.get("lon", 0) - self.global_lon_min) / (
                            self.global_lon_max - self.global_lon_min)

                        num_people = attr.get("num_people", 0)
                        normalized_num_people = (
                            (num_people - self.feature_ranges['num_people']['min']) /
                            (self.feature_ranges['num_people']['max'] - self.feature_ranges['num_people']['min'])
                        )

                        temperature = attr.get("temperature", 0)
                        normalized_temperature = (
                            (temperature - self.feature_ranges['temperature']['min']) /
                            (self.feature_ranges['temperature']['max'] - self.feature_ranges['temperature']['min'])
                        )
                        precipitation = min(attr.get("precipitation", 0), 50)
                        log_precipitation = torch.log1p(torch.tensor(precipitation)).item() if precipitation >= 0 else 0
                        normalized_precipitation = min(1.0, max(0.0, (
                            (log_precipitation - self.feature_ranges['precipitation']['min']) /
                            (self.feature_ranges['precipitation']['max'] - self.feature_ranges['precipitation']['min'])
                        )))
                        wind_speed = attr.get("wind_speed", 0)
                        normalized_wind_speed = (
                            (wind_speed - self.feature_ranges['wind_speed']['min']) /
                            (self.feature_ranges['wind_speed']['max'] - self.feature_ranges['wind_speed']['min'])
                        )

                        feature = [
                            normalized_num_people,
                            poi_type_idx,
                            normalized_lat,
                            normalized_lon,
                            normalized_temperature,
                            normalized_precipitation,
                            normalized_wind_speed
                        ]
                        node_features.append(feature)

                    x = torch.tensor(node_features, dtype=torch.float)

                    edges = list(graph.edges(data=True))
                    if len(edges) == 0:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_attr = torch.empty((0, 1), dtype=torch.float)
                    else:
                        uv = [(u, v) for (u, v, _) in edges]
                        edge_index = torch.tensor(uv, dtype=torch.long).t().contiguous()
                        edge_index -= 1
                        edge_index += cumulative_node_count

                        edge_features = [[attr.get("population_flow")] for (_, _, attr) in edges]
                        edge_attr = torch.tensor(edge_features, dtype=torch.float)

                    x_window.append(x)
                    edge_index_window.append(edge_index)
                    edge_attr_window.append(edge_attr)
                    cumulative_node_count += len(graph.nodes)

                batch_node_window.append(x_window)
                batch_edge_index_window.append(torch.cat(edge_index_window, dim=1))
                batch_edge_attr_window.append(torch.cat(edge_attr_window, dim=0).squeeze(-1))
                if batch_edge_index_window[-1].size(1) == 0:
                    batch_node_window.pop()
                    batch_edge_index_window.pop()
                    batch_edge_attr_window.pop()
                    continue

                if len(batch_node_window) == self.batch_size:
                    batch_node_window = torch.stack([
                        torch.stack(window, dim=0) for window in batch_node_window
                    ], dim=0).permute(0, 2, 3, 1)
                    batch_edge_index_window = torch.cat(batch_edge_index_window, dim=1)
                    batch_edge_attr_window = torch.cat(batch_edge_attr_window, dim=0)
                    if batch_edge_index_window.size(1) == 0:
                        batch_node_window, batch_edge_index_window, batch_edge_attr_window = [], [], []
                        cumulative_node_count = 0
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    yield batch_node_window, batch_edge_index_window, batch_edge_attr_window
                    batch_node_window, batch_edge_index_window, batch_edge_attr_window = [], [], []
                    cumulative_node_count = 0
                    torch.cuda.empty_cache()
                    gc.collect()

            if batch_node_window:
                batch_node_window = torch.stack([
                    torch.stack(window, dim=0) for window in batch_node_window
                ], dim=0).permute(0, 2, 3, 1)
                batch_edge_index_window = torch.cat(batch_edge_index_window, dim=1)
                batch_edge_attr_window = torch.cat(batch_edge_attr_window, dim=0)

                if batch_edge_index_window.size(1) == 0:
                    del batch_node_window, batch_edge_index_window, batch_edge_attr_window
                    torch.cuda.empty_cache()
                    gc.collect()
                    return

                yield batch_node_window, batch_edge_index_window, batch_edge_attr_window
                del batch_node_window, batch_edge_index_window, batch_edge_attr_window
                torch.cuda.empty_cache()
                gc.collect()

def custom_collate_fn(batch):
    return batch[0]

def get_dataloader(gpickle_dir, batch_size=12, input_dim=7, window_sizes=[8], num_workers=1):
    global_lat_lon_bounds = (36.67740214528253, 36.90482235413335, -119.91876568732503, -119.65546786772028)
    dataset = CombinedSlidingWindowIterableDataset(
        gpickle_dirs=gpickle_dir,
        input_dim=input_dim,
        window_sizes=window_sizes,
        batch_size=batch_size,
        global_lat_lon_bounds=global_lat_lon_bounds
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=False
    )
    return dataloader



