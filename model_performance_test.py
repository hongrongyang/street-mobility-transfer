from model import TCGCNTransformer
# from graph_data_loader_slide_FRE import get_dataloader
from graph_data_loader_slide_SF import get_dataloader
# from graph_data_loader_slide_LA import get_dataloader
from pre_training_ztp import ztp_mean, ztp_nll_vector
import argparse
import math
import torch


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model

def load_model_ft(model, path):
    # checkpoint = torch.load(path)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {path}")
    return model


def _last_step_edge_mask(edge_index_batch, batch_size, num_nodes, time_steps, device):
    mask_list = []
    for b in range(batch_size):
        t = time_steps - 1
        start_idx = (b * time_steps + t) * num_nodes
        end_idx   = start_idx + num_nodes
        m = (edge_index_batch[0] >= start_idx) & (edge_index_batch[0] < end_idx) & \
            (edge_index_batch[1] >= start_idx) & (edge_index_batch[1] < end_idx)
        mask_list.append(m)
    return torch.stack(mask_list, dim=0).any(dim=0).to(device)  # [E_all]


@torch.no_grad()
def evaluate(model,
             dataloader,
             device,
             clamp_nonneg: bool=False,
             verbose: bool=True
             ):

    model.eval()

    ys, preds, nlls = [], [], []
    total_edges = 0

    for x_batch, edge_index_batch, edge_attr_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        edge_attr_batch  = edge_attr_batch.to(device, non_blocking=True)

        B, N, _, T = x_batch.shape
        last_mask = _last_step_edge_mask(edge_index_batch, B, N, T, device)

        y_true = edge_attr_batch[last_mask]
        lam    = model(x_batch, edge_index_batch, tgt_mask=None)
        lam    = lam.clamp_min(1e-6)

        y_pred = ztp_mean(lam)

        if y_pred.numel() != y_true.numel():
            if verbose:
                print(f"[WARN][EVAL] mismatch: pred={y_pred.numel()} vs tgt={y_true.numel()}")
            continue

        if clamp_nonneg:
            y_pred = y_pred.clamp_min(0.)

        nll_vec_unweight = ztp_nll_vector(
            y_true, lam,
            tail_alpha=0.0,
            w_max=0.0,
            weight_mode='none'
        )

        ys.append(y_true.detach().float().cpu())
        preds.append(y_pred.detach().float().cpu())
        nlls.append(nll_vec_unweight.detach().float().cpu())
        total_edges += int(y_true.numel())

    if total_edges == 0:
        return {}

    y_all    = torch.cat(ys,    dim=0)
    pred_all = torch.cat(preds, dim=0)
    nll_all  = torch.cat(nlls,  dim=0)


    diff = pred_all - y_all
    overall_MSE = torch.mean(diff**2).item()
    overall_MAE = torch.mean(torch.abs(diff)).item()
    overall_ZTP_NLL = torch.mean(nll_all).item()

    def topk_mask_by_y(y, pct):
        M = y.numel()
        k = max(1, int(math.ceil(pct * M)))
        idx = torch.topk(y, k, largest=True, sorted=False).indices
        m = torch.zeros(M, dtype=torch.bool)
        m[idx] = True
        return m

    def compute_metrics(mask):
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        d = pred_all[mask] - y_all[mask]
        mse = torch.mean(d**2).item()
        mae = torch.mean(torch.abs(d)).item()
        nll = torch.mean(nll_all[mask]).item()
        return mse, mae, nll

    mask_top10 = topk_mask_by_y(y_all, 0.10)
    mask_top1  = topk_mask_by_y(y_all, 0.01)

    top10_MSE, top10_MAE, _ = compute_metrics(mask_top10)
    top1_MSE,  top1_MAE,  top1_ZTP_NLL = compute_metrics(mask_top1)

    mask_top01 = topk_mask_by_y(y_all, 0.001)   # 0.1%

    top01_MSE, top01_MAE, top01_ZTP_NLL = compute_metrics(mask_top01)


    return {
        "overall_MSE": overall_MSE,
        "overall_MAE": overall_MAE,
        "overall_ZTP_NLL": overall_ZTP_NLL,

        "top1_MSE": top1_MSE,
        "top1_MAE": top1_MAE,
        "top1_ZTP_NLL": top1_ZTP_NLL,

        "top01_MSE": top01_MSE,
        "top01_MAE": top01_MAE,
        "top01_ZTP_NLL": top01_ZTP_NLL,
    }



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=11, help='Input dimension (e.g., num_people, poi_type, etc.)')
    parser.add_argument('--temporal_hidden_dim1', type=int, default=128, help='First hidden dimension for temporal convolution')
    parser.add_argument('--temporal_hidden_dim2', type=int, default=256, help='Second hidden dimension for temporal convolution')
    parser.add_argument('--temporal_dropout_rate', type=float, default=0, help='Dropout rate for temporal convolution')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for temporal convolution')
    parser.add_argument('--gcn_hidden_dim1', type=int, default=512, help='First hidden dimension for GCN layers')
    parser.add_argument('--gcn_hidden_dim2', type=int, default=256, help='Second hidden dimension for GCN layers')
    parser.add_argument('--gcn_dropout_rate', type=float, default=0, help='Dropout rate for GCN layers')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='Hidden dimension for Transformer decoder')
    parser.add_argument('--edge_output_dim', type=int, default=1, help='Output dimension for edges (e.g., population transfer)')
    parser.add_argument('--decoder_dropout_rate', type=float, default=0, help='Dropout rate for decoder')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in Transformer decoder')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in Transformer decoder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_poi_types', type=float, default=456, help='Number of POI types')
    parser.add_argument('--embed_dim', type=float, default=5, help='Embedding dimension for the POI types')
    # parser.add_argument('--model_path', type=str, default="./model/cold_start_sf_9d.pth", help='Path to the saved model')
    # parser.add_argument('--model_path', type=str, default="./model/rl_sf_9d.pth", help='Path to the saved model')
    parser.add_argument('--model_path', type=str, default="./model/pre_la.pth", help='Path to the saved model')

    parser.add_argument('--embed_dropout', type=float, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCGCNTransformer(
        input_dim=args.input_dim,
        temporal_hidden_dim1=args.temporal_hidden_dim1,
        temporal_hidden_dim2=args.temporal_hidden_dim2,
        temporal_dropout_rate=args.temporal_dropout_rate,
        kernel_size=args.kernel_size,
        gcn_hidden_dim1=args.gcn_hidden_dim1,
        gcn_hidden_dim2=args.gcn_hidden_dim2,
        gcn_dropout_rate=args.gcn_dropout_rate,
        decoder_hidden_dim=args.decoder_hidden_dim,
        edge_output_dim=args.edge_output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_dropout_rate=args.decoder_dropout_rate,
        num_poi_types=args.num_poi_types,
        embed_dim=args.embed_dim,
        emb_dropout=args.embed_dropout
    ).to(device)

    model = load_model_ft(model, args.model_path)

    gpickle_dirs_test_new_city = ["./graph_data/SF_test"]


    test_dataloader_new_city = get_dataloader(
        gpickle_dir=gpickle_dirs_test_new_city,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=[12],
        num_workers=1
    )

    m = evaluate(model, test_dataloader_new_city, device, verbose=True)

    print(f"Overall   MSE: {m['overall_MSE']:.8f}  MAE: {m['overall_MAE']:.8f}  NLL: {m['overall_ZTP_NLL']:.8f}")
    print(f"Top-1%    MSE: {m['top1_MSE']:.8f}  MAE: {m['top1_MAE']:.8f}  NLL: {m['top1_ZTP_NLL']:.8f}")
    print(f"Top-0.1%  MSE: {m['top01_MSE']:.8f}  MAE: {m['top01_MAE']:.8f}  NLL: {m['top01_ZTP_NLL']:.8f}")



