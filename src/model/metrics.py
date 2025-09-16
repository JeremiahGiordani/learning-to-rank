# src/model/metrics.py
from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Optional SciPy for rank correlations (falls back gracefully if unavailable)
try:
    from scipy.stats import kendalltau, spearmanr  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------
# Helpers
# ----------------------------

def perm_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Stable argsort to turn utilities into a permutation (descending).
    scores: (B, N)
    returns: (B, N) indices, with column 0 = top-ranked item
    """
    B, N = scores.shape
    # Add tiny, deterministic tie-break by column index
    eps = (torch.arange(N, device=scores.device, dtype=scores.dtype) * 1e-9)
    return torch.argsort(-(scores + eps), dim=1)


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    """
    Invert permutations: for each row, returns ranks s.t. ranks[item] = position.
    perm: (B, N) with perm[:, 0] = best item
    returns ranks: (B, N) with 0 = best rank
    """
    B, N = perm.shape
    ranks = torch.empty_like(perm)
    arangeN = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, N)
    ranks.scatter_(1, perm, arangeN)
    return ranks


# ----------------------------
# Core metrics
# ----------------------------

def exact_perm_accuracy(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    Fraction of samples whose entire permutation matches exactly.
    """
    return (pred_perm == target_perm).all(dim=1).float().mean().item()


def top1_accuracy(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    Fraction with correct top-1 choice.
    """
    return (pred_perm[:, 0] == target_perm[:, 0]).float().mean().item()


def pairwise_accuracy(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    Fraction of correctly ordered pairs per sample, averaged over batch.
    For N items there are N*(N-1)/2 pairs.
    """
    B, N = pred_perm.shape
    pred_rank = invert_perm(pred_perm)   # (B, N)
    targ_rank = invert_perm(target_perm) # (B, N)

    correct = 0.0
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += B
            correct += (
                (pred_rank[:, i] < pred_rank[:, j]) ==
                (targ_rank[:, i] < targ_rank[:, j])
            ).float().sum().item()
    return correct / total if total > 0 else 0.0


def kendall_tau(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    Mean Kendall's tau over the batch. Uses SciPy if available; otherwise computes
    from pairwise discordance formula for small N.
    """
    B, N = pred_perm.shape
    if _HAVE_SCIPY:
        total = 0.0
        for b in range(B):
            pr = invert_perm(pred_perm[b:b+1]).squeeze(0).cpu().numpy()
            tr = invert_perm(target_perm[b:b+1]).squeeze(0).cpu().numpy()
            tau, _ = kendalltau(pr, tr)
            total += float(tau)
        return total / B if B > 0 else 0.0
    else:
        # Manual computation via concordant/discordant pairs
        pred_rank = invert_perm(pred_perm)
        targ_rank = invert_perm(target_perm)
        taus = []
        for b in range(B):
            c = d = 0
            for i in range(N):
                for j in range(i + 1, N):
                    a = (pred_rank[b, i] < pred_rank[b, j])
                    b_ = (targ_rank[b, i] < targ_rank[b, j])
                    if a == b_:
                        c += 1
                    else:
                        d += 1
            denom = c + d
            taus.append((c - d) / denom if denom > 0 else 0.0)
        return float(sum(taus) / len(taus)) if taus else 0.0


def spearman_rho(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    Mean Spearman's rho over the batch. Uses SciPy if available; otherwise exact
    formula on ranks.
    """
    B, N = pred_perm.shape
    if _HAVE_SCIPY:
        total = 0.0
        for b in range(B):
            pr = invert_perm(pred_perm[b:b+1]).squeeze(0).cpu().numpy()
            tr = invert_perm(target_perm[b:b+1]).squeeze(0).cpu().numpy()
            rho, _ = spearmanr(pr, tr)
            total += float(rho)
        return total / B if B > 0 else 0.0
    else:
        pred_rank = invert_perm(pred_perm).float()
        targ_rank = invert_perm(target_perm).float()
        # Spearman rho = 1 - 6 * sum(d_i^2) / (N*(N^2-1))
        d2 = (pred_rank - targ_rank).pow(2).sum(dim=1)  # (B,)
        denom = N * (N * N - 1)
        rho = 1.0 - 6.0 * d2 / denom
        return rho.mean().item()


def ndcg(pred_perm: torch.Tensor, target_perm: torch.Tensor) -> float:
    """
    NDCG with graded relevance derived from target ranks (4->1 gains).
    Since targets are strict permutations, assign gains: gain = N - rank_pos.
    """
    B, N = pred_perm.shape
    # Gains: higher for better (lower) ranks
    targ_rank = invert_perm(target_perm)  # 0..N-1
    gains = (N - targ_rank).float()      # N..1

    # DCG for predicted order
    # Position discounts: 1/log2(1 + position_index + 1)
    discounts = 1.0 / torch.log2(torch.arange(1, N + 1, device=pred_perm.device).float() + 1.0)  # (N,)
    discounts = discounts.unsqueeze(0).expand(B, N)  # (B,N)

    # Arrange gains by predicted order
    batch_idx = torch.arange(B, device=pred_perm.device).unsqueeze(-1).expand(B, N)
    gains_in_pred_order = gains[batch_idx, pred_perm]  # (B,N)
    dcg = (gains_in_pred_order * discounts).sum(dim=1)  # (B,)

    # Ideal DCG: gains sorted descending
    ideal_order = torch.argsort(targ_rank, dim=1)  # same as target_perm
    ideal_gains = gains[batch_idx, ideal_order]
    idcg = (ideal_gains * discounts).sum(dim=1).clamp(min=1e-8)

    ndcg_vals = (dcg / idcg).mean().item()
    return ndcg_vals


# ----------------------------
# Aggregated convenience API
# ----------------------------

def compute_all(
    pred_perm: torch.Tensor,
    target_perm: torch.Tensor,
    include_rank_corr: bool = True,
) -> Dict[str, float]:
    """
    Compute a set of standard ranking metrics and return as a dict of floats.
    """
    out = {
        "perm_acc": exact_perm_accuracy(pred_perm, target_perm),
        "top1_acc": top1_accuracy(pred_perm, target_perm),
        "pairwise_acc": pairwise_accuracy(pred_perm, target_perm),
        "ndcg": ndcg(pred_perm, target_perm),
    }
    if include_rank_corr:
        out["kendall_tau"] = kendall_tau(pred_perm, target_perm)
        out["spearman_rho"] = spearman_rho(pred_perm, target_perm)
    return out
