# src/train.py

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import add_args_to_parser, build_config_from_args
from src.data.dataset import make_dataloaders
from src.model.losses import plackett_luce_loss
from src.model.metrics import perm_from_scores as metrics_perm_from_scores, compute_all
from src.model.network import RankerNet
from src.utils import (
    set_all_seeds,
    get_device,
    ensure_dir,
    save_json,
    save_ckpt,
    stable_argsort_desc,  # optional alt to metrics' perm_from_scores
    log_epoch,
)


def perm_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Wrapper so you can switch between the two stable argsorts if desired.
    """
    return metrics_perm_from_scores(scores)
    # or: return stable_argsort_desc(scores)


def run_epoch(model, loader, optimizer, device, train: bool = True):
    model.train(train)
    total_loss = 0.0
    total_batches = 0

    all_pred_perm = []
    all_targ_perm = []

    for batch in loader:
        node = batch["node_features"].to(device)        # (B,4,F_n)
        edge = batch["edge_features"].to(device)        # (B,4,4,F_e)
        glob = batch["global_features"].to(device)      # (B,F_g)
        targ = batch["target_perm"].to(device)          # (B,4)

        if train:
            optimizer.zero_grad()

        scores = model(node, edge, glob)                # (B,4)
        loss = plackett_luce_loss(scores, targ)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        pred_perm = perm_from_scores(scores)
        all_pred_perm.append(pred_perm.detach().cpu())
        all_targ_perm.append(targ.detach().cpu())

        total_loss += float(loss.item())
        total_batches += 1

    all_pred_perm = torch.cat(all_pred_perm, dim=0)
    all_targ_perm = torch.cat(all_targ_perm, dim=0)

    metrics = {"loss": total_loss / max(1, total_batches)}
    metrics.update(compute_all(all_pred_perm, all_targ_perm, include_rank_corr=False))
    # If you have SciPy installed and want rank correlations:
    # metrics.update(compute_all(all_pred_perm, all_targ_perm, include_rank_corr=True))

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Score-then-Sort ranking model.")
    add_args_to_parser(parser)
    args = parser.parse_args()

    # Build structured config (file + CLI overrides)
    cfg = build_config_from_args(args)

    # I/O setup
    out_dir = ensure_dir(Path(cfg.train.out_dir))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    art_dir = ensure_dir(out_dir / "artifacts")
    logs_path = out_dir / "train_metrics.json"
    ckpt_best = ckpt_dir / "model_best.pt"
    ckpt_last = ckpt_dir / "model_last.pt"
    scaler_path = art_dir / "feature_scaler.json"
    cfg_path = art_dir / "config.json"

    # Determinism
    set_all_seeds(cfg.train.seed, deterministic=True)

    # Device
    device = get_device(None if cfg.model.device == "auto" else cfg.model.device)

    # Data
    if not cfg.data.data:
        raise ValueError("Config.data.data must point to the dataset JSON file.")
    train_loader, val_loader, scaler = make_dataloaders(
        cfg.data.data,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.train.seed,
    )

    # Persist scaler & config
    save_json({"mean": list(map(float, scaler["mean"])), "std": list(map(float, scaler["std"]))}, scaler_path)
    save_json(cfg.to_dict(), cfg_path)

    # Infer dims from a batch
    sample = next(iter(train_loader))
    node_dim = sample["node_features"].shape[-1]
    edge_dim = sample["edge_features"].shape[-1]
    global_dim = sample["global_features"].shape[-1]

    # Model
    model = RankerNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        node_hidden=cfg.model.node_hidden,
        edge_hidden=cfg.model.edge_hidden,
        score_hidden=cfg.model.score_hidden,
        use_edges=cfg.model.use_edges,
        gnn_layers=cfg.model.gnn_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    best_val = math.inf
    history = {"train": [], "val": []}

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        scheduler.step()

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        print(log_epoch(epoch, cfg.train.epochs, train_metrics, val_metrics))

        # Save "last" checkpoint every epoch
        save_ckpt(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler": scaler,
                "config": cfg.to_dict(),
                "val_metrics": val_metrics,
            },
            ckpt_last,
        )

        # Track best by validation loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ckpt(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler": scaler,
                    "config": cfg.to_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_best,
            )

        # Persist history each epoch
        save_json(history, logs_path)

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Artifacts: {ckpt_best} (best), {ckpt_last} (last), scaler → {scaler_path}, logs → {logs_path}")


if __name__ == "__main__":
    main()
