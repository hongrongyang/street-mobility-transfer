import argparse
import torch
from graph_data_loader_slide_LA import get_dataloader
from mcgcn import MCGCNForEdges, infer_kwargs, LastStep, evaluate, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="./graph_data/LA/test")
    ap.add_argument("--ckpt", default="./final_model/compare/mcgcn_wztp.pth")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--out", default="./results/mcgcn_LA_test.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_dataloader(gpickle_dir=[a.test_dir], batch_size=1, input_dim=11, window_sizes=[a.window], num_workers=1)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    base = MCGCNForEdges(**infer_kwargs(sd))
    base.load_state_dict(sd, strict=True)
    model = LastStep(base).to(device)
    m = evaluate(model, loader, device)
    for k in ("top1_NLL", "top1_MSE", "top1_MAE", "overall_NLL", "overall_MSE", "overall_MAE"):
        print(f"{k}: {m[k]:.6f}")
    save_json({"ckpt": a.ckpt, "test_dir": a.test_dir, "metrics": m}, a.out)


if __name__ == "__main__":
    main()
