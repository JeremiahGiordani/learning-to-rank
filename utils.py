# src/utils.py
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


# ----------------------------
# Reproducibility & device
# ----------------------------

def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch; optionally enable deterministic kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(spec: Optional[str] = None) -> torch.device:
    """
    Resolve a torch.device from a string like "cuda", "cuda:1", or "cpu".
    If spec is None, prefer CUDA if available.
    """
    if spec is None:
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(spec)


# ----------------------------
# FS helpers
# ----------------------------

def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    """Save JSON, handling dataclasses gracefully."""
    if is_dataclass(obj):
        obj = asdict(obj)
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent))


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def save_ckpt(state: Dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_ckpt(path: Path | str, map_location: Optional[str | torch.device] = None) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


# ----------------------------
# Sorting, permutations, and pretty printing
# ----------------------------

def stable_argsort_desc(scores: torch.Tensor) -> torch.Tensor:
    """
    Stable argsort in descending order with a tiny deterministic tie-break
    by column index. Input shape (B, N); returns (B, N) permutation indices.
    """
    B, N = scores.shape
    eps = (torch.arange(N, device=scores.device, dtype=scores.dtype) * 1e-9)
    return torch.argsort(-(scores + eps), dim=1)


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    """
    Invert permutation rows: ranks[item] = position. Shapes: (B, N) -> (B, N).
    """
    B, N = perm.shape
    ranks = torch.empty_like(perm)
    arangeN = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, N)
    ranks.scatter_(1, perm, arangeN)
    return ranks


def format_perm(ids: Sequence[str], perm: Sequence[int]) -> str:
    """
    Human-friendly permutation string like: '1: AA260, 2: UA790, 3: DL123, 4: WN88'.
    """
    order = [ids[i] for i in perm]
    return ", ".join(f"{k+1}: {pid}" for k, pid in enumerate(order))


# ----------------------------
# Lightweight logging
# ----------------------------

class AverageMeter:
    """Tracks the average of a streaming scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class Timer:
    """Simple timing context manager."""
    def __init__(self, name: str = "", stream=None):
        self.name = name
        self.stream = stream

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if self.stream is not None:
            print(f"{self.name} took {dt:.3f}s", file=self.stream)
        else:
            print(f"{self.name} took {dt:.3f}s")


def log_epoch(epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> str:
    """
    Create a compact one-line epoch log string.
    """
    msg = (
        f"[{epoch:03d}/{total_epochs}] "
        f"train loss={train_metrics.get('loss', float('nan')):.4f} "
        f"val loss={val_metrics.get('loss', float('nan')):.4f} "
        f"val perm_acc={val_metrics.get('perm_acc', float('nan')):.3f} "
        f"val pairwise_acc={val_metrics.get('pairwise_acc', float('nan')):.3f}"
    )
    return msg


# ----------------------------
# Deterministic tie-breaking for IDs
# ----------------------------

def break_ties_with_ids(scores: torch.Tensor, ids_batch: List[List[str]]) -> torch.Tensor:
    """
    Given raw scores (B, N) and corresponding IDs per batch item, return a stable
    argsort that breaks exact ties using lexicographic ID order. This is useful
    for safety-critical determinism.

    Note: This converts IDs to a numeric tie-break tensor per batch; for N=4 this is trivial.
    """
    device = scores.device
    B, N = scores.shape
    # Build per-batch tie-break ranks from IDs: smaller rank => earlier in order
    tie_break = torch.zeros((B, N), device=device, dtype=scores.dtype)
    for b in range(B):
        # Map IDs to a consistent order
        id_order = sorted([(i, pid) for i, pid in enumerate(ids_batch[b])], key=lambda x: x[1])
        # rank_map[item_idx] = rank_by_lex
        rank_map = {i: r for r, (i, _) in enumerate(id_order)}
        tie_break[b] = torch.tensor([rank_map[i] for i in range(N)], device=device, dtype=scores.dtype)

    # We want descending by score; for ties, ascending by lex rank -> subtract a tiny epsilon * rank
    eps = 1e-9
    adjusted = scores - eps * tie_break
    return torch.argsort(-adjusted, dim=1)


# ----------------------------
# Misc
# ----------------------------

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_size(num_bytes: int) -> str:
    """Format byte counts in a human-friendly way."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
