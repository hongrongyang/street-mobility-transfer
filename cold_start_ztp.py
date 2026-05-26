import argparse
import time
import gc
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pre_training_ztp as pre
from pre_training_ztp import (
    count_parameters, save_model, evaluate, train, early_stopping
)
from model import TCGCNTransformer
from graph_data_loader_slide_FRE import get_dataloader


def load_model(model, path, verbose=True):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint)
    if verbose:
        print(f"Model loaded from {path}")
    return model


def freeze_layers_cold_start(model, num_frozen_layers=1, shared_poi_index_file=None):

    if hasattr(model.tc_gcn.temporal_conv, "conv1"):
        for p in model.tc_gcn.temporal_conv.conv1.parameters():
            p.requires_grad = False

    if hasattr(model.tc_gcn, "gcn1"):
        for p in model.tc_gcn.gcn1.parameters():
            p.requires_grad = False

    if hasattr(model, "transformer_decoder") and hasattr(model.transformer_decoder, "layers"):
        layers = getattr(model.transformer_decoder.layers, "layers", model.transformer_decoder.layers)
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                p.requires_grad = (i >= num_frozen_layers)

    # (Optional) Train embedding vectors for non-shared POIs only
    '''if shared_poi_index_file:
        try:
            with open(shared_poi_index_file, "rb") as f:
                shared_poi_indices = pickle.load(f)
            model.shared_poi_indices = torch.tensor(shared_poi_indices, dtype=torch.long, 
            device=next(model.parameters()).device)
            print(f"[Info] Loaded shared POI indices: {len(shared_poi_indices)}")
        except Exception as e:
            print(f"[Warn] shared_poi_index_file load failed: {e}")'''

    print(f"Frozen the first {num_frozen_layers} decoder layers for cold start.")
    print("Layers frozen for cold start.")


def _is_no_decay_param(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim < 2:
        return True
    ln = name.lower()
    return (ln.endswith(".bias") or "norm" in ln or "bn" in ln or "layernorm" in ln or "ln" in ln)


def build_adamw_like_pretraining(model, lr: float, weight_decay: float):
    decay_params, no_decay_params, emb_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "poi_embedding" in n or "poi_emb" in n:
            emb_params.append(p)
        elif _is_no_decay_param(n, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": emb_params, "weight_decay": 0.0},
        ],
        lr=lr
    )
    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dim', type=int, default=11)
    parser.add_argument('--temporal_hidden_dim1', type=int, default=128)
    parser.add_argument('--temporal_hidden_dim2', type=int, default=256)
    parser.add_argument('--temporal_dropout_rate', type=float, default=0.1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--gcn_hidden_dim1', type=int, default=512)
    parser.add_argument('--gcn_hidden_dim2', type=int, default=256)
    parser.add_argument('--gcn_dropout_rate', type=float, default=0.2)
    parser.add_argument('--decoder_hidden_dim', type=int, default=256)
    parser.add_argument('--edge_output_dim', type=int, default=1)
    parser.add_argument('--decoder_dropout_rate', type=float, default=0.08)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.04)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_poi_types', type=int, default=456)
    parser.add_argument('--embed_dim', type=int, default=5)
    parser.add_argument('--embed_dropout', type=float, default=0.05)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--patience_stop', type=int, default=7)
    parser.add_argument('--patience_lr', type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.5)

    parser.add_argument('--model_path', type=str, default="./model/final/pretraining/pre_la_ztp.pth")
    parser.add_argument('--num_frozen_layers', type=int, default=1)
    parser.add_argument('--shared_poi_index_file', type=str,
                        default="./POI_data/mapping/poi_type_mapping_la_to_sf.pkl")

    parser.add_argument('--w_1', type=float, default=0.05)
    parser.add_argument('--w_2', type=float, default=0.10)
    parser.add_argument('--tail_alpha', type=float, default=2.2)
    parser.add_argument('--w_max', type=float, default=4.5)
    parser.add_argument('--weight_mode', type=str, default='log', choices=['none', 'log', 'power'])
    parser.add_argument('--tail_min_count', type=int, default=3)
    parser.add_argument('--tail_max_count', type=int, default=10)

    parser.add_argument('--amp', type=str, default='bf16', choices=['auto', 'off', 'fp16', 'bf16'],
                        help="auto/off/fp16/bf16")
    parser.add_argument('--model_outputs_raw', type=bool, default=False,
                        help="If the model's forward pass returns raw logits (not soft-plus), set to True; otherwise, "
                             "set to False to indicate that λ is output directly.")

    parser.add_argument('--train_dirs', nargs='+', default=["./graph_data/SF/CS 9d/train"])
    parser.add_argument('--val_dirs', nargs='+', default=["./graph_data/SF/CS 9d/val"])

    parser.add_argument('--window_sizes', nargs='+', type=int, default=[12])
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--wandb_project', type=str, default="Paper1_cold_start")

    args = parser.parse_args()

    if args.amp == 'off':
        pre.AMP_ENABLED_GLOBAL = False
    elif args.amp == 'fp16':
        pre.AMP_ENABLED_GLOBAL = True
        pre.AMP_DTYPE_GLOBAL = torch.float16
    elif args.amp == 'bf16':
        pre.AMP_ENABLED_GLOBAL = True
        pre.AMP_DTYPE_GLOBAL = torch.bfloat16
    else:  # auto
        pre.AMP_ENABLED_GLOBAL = True
        pre.AMP_DTYPE_GLOBAL = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

    wandb.init(project=args.wandb_project)
    wandb.config.update(vars(args) | {
        "amp_enabled": pre.AMP_ENABLED_GLOBAL,
        "amp_dtype": str(pre.AMP_DTYPE_GLOBAL).replace("torch.", "")
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(
        gpickle_dir=args.train_dirs,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=args.window_sizes,
        num_workers=args.num_workers
    )

    val_dataloader = get_dataloader(
        gpickle_dir=args.val_dirs,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        window_sizes=args.window_sizes,
        num_workers=args.num_workers
    )

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
        emb_dropout=args.embed_dropout,
        attention_dropout_rate=args.attention_dropout_rate
    ).to(device)

    model = load_model(model, args.model_path)

    freeze_layers_cold_start(
        model,
        num_frozen_layers=args.num_frozen_layers,
        shared_poi_index_file=args.shared_poi_index_file
    )


    optimizer = build_adamw_like_pretraining(model, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor, patience=args.patience_lr, verbose=True
    )

    eval_losses = []
    best_val_loss = float('inf')
    best_val_loss_w = float('inf')

    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_loss = train(
            model, train_dataloader, optimizer, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max,
            weight_mode=args.weight_mode, tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            model_outputs_raw=args.model_outputs_raw,
        )

        val_loss, val_loss_w = evaluate(
            model, val_dataloader, device,
            tail_alpha=args.tail_alpha, w_max=args.w_max,
            weight_mode=args.weight_mode, tail_min_count=args.tail_min_count,
            tail_max_count=args.tail_max_count,
            model_outputs_raw=args.model_outputs_raw
        )
        eval_losses.append(val_loss_w)

        if val_loss_w < best_val_loss_w:
            best_val_loss_w = val_loss_w
            save_model(model, "final_model/cold_start_sf_9d.pth")

        torch.cuda.empty_cache()
        gc.collect()

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_loss_w": val_loss_w})
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f},"
              f" Weighted validation Loss: {val_loss_w:.8f}")

        scheduler.step(val_loss_w)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Learning Rate: {current_lr}")

        if early_stopping(args.patience_stop, eval_losses):
            torch.cuda.empty_cache()
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print("Training complete.")
