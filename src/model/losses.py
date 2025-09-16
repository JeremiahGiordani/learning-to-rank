# src/model/losses.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _sorted_by_perm(scores: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Reorder each row of `scores` according to the target permutation `perm`.

    Args:
        scores: (B, N) utilities for each item
        perm:   (B, N) indices giving the gold order (rank-1 first)

    Returns:
        scores_sorted: (B, N) where scores_sorted[:, k] = scores[:, perm[:, k]]
    """
    B, N = scores.shape
    gather_idx = perm
    return torch.gather(scores, dim=1, index=gather_idx)


def _tail_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """
    For each row, compute tail logsumexp cumulatively from the end.

    Given x (B, N), returns t (B, N) where:
      t[:, k] = logsumexp(x[:, k:])

    Numerically stable and vectorized.
    """
    B, N = x.shape
    # Work from right to left
    out = torch.empty_like(x)
    out[:, -1] = x[:, -1]
    for k in range(N - 2, -1, -1):
        # logsumexp([x_k, out_{k+1}]) = log( exp(x_k) + exp(out_{k+1}) )
        m = torch.maximum(x[:, k], out[:, k + 1])
        out[:, k] = m + torch.log(torch.exp(x[:, k] - m) + torch.exp(out[:, k + 1] - m))
    return out


def plackett_luce_loss(scores: torch.Tensor, perm: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Plackett–Luce (listwise) negative log-likelihood for permutations.

    For each sample, with utilities `scores` and gold permutation `perm`:
      L = - sum_{k=0}^{N-1} [ s_{π_k} - log sum_{m=k}^{N-1} exp(s_{π_m}) ]

    Args:
        scores: (B, N) real-valued utilities
        perm:   (B, N) integer indices; perm[:, 0] is the top-ranked item
        reduction: "mean" | "sum" | "none"

    Returns:
        loss: () if reduced, else (B,)
    """
    # Reorder scores by ground-truth permutation
    s_sorted = _sorted_by_perm(scores, perm)  # (B, N)
    # Compute logsumexp over tails efficiently
    tail_lse = _tail_logsumexp(s_sorted)      # (B, N)
    nll = -(s_sorted - tail_lse).sum(dim=1)   # (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


def listmle_loss(scores: torch.Tensor, perm: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    ListMLE loss (Xia et al., 2008). For deterministic total orders, this is
    equivalent to the Plackett–Luce NLL with scores as utilities.

    Kept as a separate function for clarity/experimentation.
    """
    return plackett_luce_loss(scores, perm, reduction=reduction)


def bradley_terry_pairwise_loss(
    scores: torch.Tensor,
    perm: torch.Tensor,
    reduction: str = "mean",
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Pairwise Bradley–Terry (logistic) loss over all ordered pairs implied by the permutation.

    For each ordered pair (i before j) in the gold ranking, penalize when s_i <= s_j:
      ℓ_ij = -log σ( (s_i - s_j) - margin )

    Args:
        scores: (B, N)
        perm:   (B, N) gold order (rank-1 first)
        reduction: "mean" | "sum" | "none"
        margin: optional safety margin to encourage larger separations

    Returns:
        loss: () if reduced, else (B,)
    """
    B, N = scores.shape
    # Invert permutation: rank_pos[b, item] = position in gold order (0 is best)
    rank_pos = torch.empty_like(perm)
    arangeN = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, N)
    rank_pos.scatter_(1, perm, arangeN)

    # We need all pairs (i, j) with rank_pos[i] < rank_pos[j]
    # Build pairwise matrices
    s_i = scores.unsqueeze(2).expand(B, N, N)
    s_j = scores.unsqueeze(1).expand(B, N, N)
    diff = s_i - s_j  # (B, N, N)

    rp_i = rank_pos.unsqueeze(2).expand(B, N, N)
    rp_j = rank_pos.unsqueeze(1).expand(B, N, N)
    mask = (rp_i < rp_j)  # True where i should be ahead of j

    # Exclude diagonal and lower triangle where not needed (mask already excludes eq)
    logits = diff - margin
    # Use only masked entries
    loss_mat = F.softplus(-logits)  # -log σ(logits) = softplus(-logits)
    # Sum over valid pairs for each sample
    loss_per_sample = (loss_mat * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)

    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    else:
        return loss_per_sample


def hinge_pairwise_loss(
    scores: torch.Tensor,
    perm: torch.Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pairwise hinge loss over all ordered pairs implied by the permutation:
      ℓ_ij = max(0, margin - (s_i - s_j))  for each i before j in gold order
    """
    B, N = scores.shape
    # Invert permutation: rank_pos[b, item] = position in gold order
    rank_pos = torch.empty_like(perm)
    arangeN = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, N)
    rank_pos.scatter_(1, perm, arangeN)

    s_i = scores.unsqueeze(2).expand(B, N, N)
    s_j = scores.unsqueeze(1).expand(B, N, N)
    diff = s_i - s_j

    rp_i = rank_pos.unsqueeze(2).expand(B, N, N)
    rp_j = rank_pos.unsqueeze(1).expand(B, N, N)
    mask = (rp_i < rp_j)

    loss_mat = torch.clamp(margin - diff, min=0.0)
    loss_per_sample = (loss_mat * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)

    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    else:
        return loss_per_sample


# Convenience alias for the default listwise loss
ListwiseLoss = plackett_luce_loss
