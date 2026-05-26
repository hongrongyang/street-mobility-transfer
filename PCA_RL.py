import torch
import numpy as np
import json
from sklearn.decomposition import PCA

from graph_data_loader_slide_SF_RLFT import get_dataloader
# from graph_data_loader_slide_FRE_RLFT import get_dataloader
from model import TCGCNTransformer
from cold_start import load_model


def collect_hidden_embeddings(model, dataloader, device, max_samples=10000):

    model.eval()
    collected = []

    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, edge_attr = batch
            x = x.to(device)
            edge_index = edge_index.to(device)

            model(x, edge_index)

            for h in model._debug_hidden_list:
                h = h.squeeze(0)
                E = h.shape[0]

                take = min(100, E)
                idx = torch.randperm(E)[:take]
                sampled = h[idx].cpu().numpy()

                collected.append(sampled)

            if sum(a.shape[0] for a in collected) >= max_samples:

                continue

    hidden_mat = np.concatenate(collected, axis=0)
    print("Collected hidden shape:", hidden_mat.shape)  #
    return hidden_mat


def compute_pca_frequency_groups(hidden_mat, save_path="./sf_frequency_groups.json"):

    print("Running PCA on hidden matrix...")

    pca = PCA(n_components=256)
    pca.fit(hidden_mat)

    importance = np.sum(np.abs(pca.components_), axis=0)

    sorted_idx = np.argsort(importance)

    n = len(sorted_idx)
    low = sorted_idx[: n // 3].tolist()
    mid = sorted_idx[n // 3: 2 * n // 3].tolist()
    high = sorted_idx[2 * n // 3:].tolist()

    groups = {
        "low": low,
        "mid": mid,
        "high": high
    }

    with open(save_path, "w") as f:
        json.dump(groups, f, indent=4)

    print(f"[Saved] frequency groups to {save_path}")
    print("low group:", len(low))
    print("mid group:", len(mid))
    print("high group:", len(high))

    return groups


def run_pca_grouping(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCGCNTransformer(
        input_dim=11,
        temporal_hidden_dim1=128,
        temporal_hidden_dim2=256,
        temporal_dropout_rate=0,
        kernel_size=3,
        gcn_hidden_dim1=512,
        gcn_hidden_dim2=256,
        gcn_dropout_rate=0,
        decoder_hidden_dim=256,
        edge_output_dim=1,
        num_heads=4,
        num_layers=2,
        decoder_dropout_rate=0,
        num_poi_types=463,
        embed_dim=5,
        emb_dropout=0,
        attention_dropout_rate=0
    ).to(device)

    model = load_model(model, model_path)
    print("Loaded model:", model_path)

    gpickle_dir = ["./graph_data/LA/CS 9d/full"]
    window_sizes = [12]

    dataloader = get_dataloader(
        gpickle_dir=gpickle_dir,
        batch_size=1,
        input_dim=7,
        window_sizes=window_sizes,
        num_workers=1,
        stride=1
    )


    hidden_mat = collect_hidden_embeddings(
        model, dataloader, device, max_samples=20000
    )


    groups = compute_pca_frequency_groups(
        hidden_mat,
        save_path="./final_model/PCA_results/sf_rl_9d.json"
    )

    print("PCA grouping finished.")
    return groups


if __name__ == "__main__":
    model_path = "final_model/cold_start_sf_9d.pth"
    run_pca_grouping(model_path)

