# src/infer.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data.dataset import RankDataset, collate_fn
from src.model.metrics import compute_all, perm_from_scores as metrics_perm_from_scores
from src.model.network import RankerNet
from src.utils import (
    get_device,
    ensure_dir,
    load_ckpt,
    save_json,
    break_ties_with_ids,
    format_perm,
)


def perm_from_scores(scores: torch.Tensor, ids_batch: List[List[str]] | None = None) -> torch.Tensor:
    """
    Stable permutation from utilities. If ids_batch is provided, uses IDs to
    deterministically break exact ties (FAA-friendly).
    """
    if ids_batch is None:
        return metrics_perm_from_scores(scores)
    return break_ties_with_ids(scores, ids_batch)


def build_model_from_ckpt(ckpt: Dict[str, Any], node_dim: int, edge_dim: int, global_dim: int, device: torch.device) -> RankerNet:
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = RankerNet(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        node_hidden=model_cfg.get("node_hidden", 64),
        edge_hidden=model_cfg.get("edge_hidden", 64),
        score_hidden=model_cfg.get("score_hidden", 64),
        use_edges=model_cfg.get("use_edges", True),
        gnn_layers=model_cfg.get("gnn_layers", 0),
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    state = ckpt["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_inference(
    data_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    batch_size: int = 256,
    device_spec: str | None = None,
    save_csv: bool = True,
) -> None:
    out_dir = ensure_dir(out_dir)
    preds_path = out_dir / "predictions.json"
    csv_path = out_dir / "predictions.csv"
    summary_path = out_dir / "summary.json"

    # Load checkpoint (contains scaler + config)
    ckpt = load_ckpt(ckpt_path, map_location="cpu")
    scaler = ckpt.get("scaler", None)
    cfg = ckpt.get("config", {})

    # Device
    device = get_device(None if cfg.get("model", {}).get("device", "auto") == "auto" else cfg["model"]["device"])
    if device_spec is not None and device_spec != "auto":
        device = get_device(device_spec)

    # Dataset / DataLoader (no split; full eval set)
    dataset = RankDataset(data_path, scaler=scaler)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer feature dims from one batch
    sample = next(iter(loader))
    node_dim = sample["node_features"].shape[-1]
    edge_dim = sample["edge_features"].shape[-1]
    global_dim = sample["global_features"].shape[-1]

    # Model
    model = build_model_from_ckpt(ckpt, node_dim, edge_dim, global_dim, device)

    # Inference loop
    all_pred_perm = []
    all_targ_perm = []
    per_sample_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        offset = 0
        for batch in loader:
            node = batch["node_features"].to(device)        # (B,4,F_n)
            edge = batch["edge_features"].to(device)        # (B,4,4,F_e)
            glob = batch["global_features"].to(device)      # (B,F_g)
            targ = batch["target_perm"].to(device)          # (B,4)
            ids_batch: List[List[str]] = batch["ids"]

            scores = model(node, edge, glob)                # (B,4)

            # Stable sort with ID-based tie-break
            pred_perm = perm_from_scores(scores, ids_batch)

            all_pred_perm.append(pred_perm.cpu())
            all_targ_perm.append(targ.cpu())

            # Per-sample rows for reporting
            for b in range(scores.size(0)):
                ids = ids_batch[b]
                scores_b = scores[b].cpu().tolist()
                pred_idx = pred_perm[b].cpu().tolist()
                targ_idx = targ[b].cpu().tolist()

                per_sample_records.append(
                    {
                        "index": offset + b,
                        "ids": ids,
                        "scores": scores_b,
                        "pred_perm_indices": pred_idx,
                        "pred_order_ids": [ids[i] for i in pred_idx],
                        "target_perm_indices": targ_idx,
                        "target_order_ids": [ids[i] for i in targ_idx],
                    }
                )
            offset += scores.size(0)

    all_pred_perm_t = torch.cat(all_pred_perm, dim=0)
    all_targ_perm_t = torch.cat(all_targ_perm, dim=0)

    # Metrics
    metrics = compute_all(all_pred_perm_t, all_targ_perm_t, include_rank_corr=False)
    # If SciPy is installed and you want rank correlations:
    # metrics = compute_all(all_pred_perm_t, all_targ_perm_t, include_rank_corr=True)

    # Save outputs
    save_json({"metrics": metrics, "count": len(dataset)}, summary_path)
    save_json(per_sample_records, preds_path)

    if save_csv:
        # Flat CSV that’s easy to scan
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "pred_order_ids",
                    "target_order_ids",
                    "scores_json",
                ]
            )
            for rec in per_sample_records:
                writer.writerow(
                    [
                        rec["index"],
                        " | ".join(rec["pred_order_ids"]),
                        " | ".join(rec["target_order_ids"]),
                        str(rec["scores"]),
                    ]
                )

    # Print summary and a few examples
    print("=== Inference Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\nSaved predictions to: {preds_path}")
    if save_csv:
        print(f"Saved CSV to:         {csv_path}")
    print(f"Saved summary to:     {summary_path}")

    # Pretty print first few scenarios
    print("\nExamples:")
    for rec in per_sample_records[:5]:
        pred_str = format_perm(rec["ids"], rec["pred_perm_indices"])
        targ_str = format_perm(rec["ids"], rec["target_perm_indices"])
        print(f"#{rec['index']:04d}  PRED  {pred_str}")
        print(f"           GOLD  {targ_str}")


def main():
    parser = argparse.ArgumentParser(description="Inference for Score-then-Sort ranking model.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset JSON (list of DataPoint).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--out_dir", type=str, default="outputs/infer", help="Directory to write predictions/metrics.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help='"cpu", "cuda", "cuda:N", or leave unset for config/auto.')
    parser.add_argument("--no_csv", action="store_true", help="Do not emit CSV file.")
    args = parser.parse_args()

    run_inference(
        data_path=Path(args.data),
        ckpt_path=Path(args.ckpt),
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        device_spec=args.device,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
