"""TADS scoring helpers (paper Eq.2, Eq.3, Eq.7, Eq.8).

Pure-function utilities for the deterministic TADS selection pipeline. No
PPO actor, no learned components — every step is a closed-form operation
over pool-level statistics.

Pipeline composition (one epoch):
    L_i, H_i           = per-sample loss / entropy            (compute_rewards in reward.py)
    R_i, w             = pool_reward(L_arr, H_arr)            (Eq.2 / Eq.3)
    R̃_i               = calibrated_utility(R_arr)             (Eq.8 inner: σ(z-score))
    align_i (raw)      = Σ_l <h̄_l(x_i), v_l> / L              (Eq.6, computed in selector)
    ã_i                = normalize_alignment(align_raw)        (Eq.7)
    s_i                = tads_score(R̃, ã, λ)                  (Eq.8 outer: multiplicative boost)
    selection          = top-B indices of s_i                  (Algorithm 1 step)

The functions accept 1-D tensors of shape (N,) where N is the candidate
pool size. All ops are vectorised and run on CPU after the per-sample
forwards complete; no GPU memory required.
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def pool_reward(
    loss_arr: torch.Tensor,
    entropy_arr: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """Compute composite reward (Eq.2) using the pool-level variance ratio (Eq.3).

    Args:
        loss_arr: per-sample mean CE loss over response tokens; shape (N,).
        entropy_arr: per-sample mean predictive entropy; shape (N,).
        eps: numerical stabiliser in the variance-ratio denominator.

    Returns:
        (R, w) where R has shape (N,) and w is a python float.
        w = Var(L) / (Var(L) + Var(H) + eps).  R = w·L + (1-w)·H.

    The variances are computed across the FULL pool (every sample of the
    epoch), not per mini-batch. A naive per-batch implementation degenerates
    to w ≡ 0 at batch_size=1 and silently collapses the difficulty signal,
    so callers must accumulate (L_i, H_i) over the whole pool first.
    """
    if loss_arr.shape != entropy_arr.shape:
        raise ValueError(
            f"pool_reward: shape mismatch L={tuple(loss_arr.shape)} "
            f"vs H={tuple(entropy_arr.shape)}"
        )
    if loss_arr.numel() < 2:
        w = 0.5  # variance undefined at N<2; falls back to equal weighting
    else:
        var_l = loss_arr.var().item()
        var_h = entropy_arr.var().item()
        w = var_l / (var_l + var_h + eps)
    R = w * loss_arr + (1.0 - w) * entropy_arr
    return R, float(w)


def calibrated_utility(
    R: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Paper Eq.8 inner: pool-level z-score.

        R̃_i = (R_i - R̄) / (σ_R + eps)

    Centres out the additive task-mix-dependent baseline of Eq.5/
    total-variance and scales by the pool std so cross-task drift in
    R-magnitude doesn't dominate the ranking.  Unlike the original paper
    formulation, the logistic sigmoid is intentionally NOT applied — R̃
    here is unbounded, so the multiplicative composition with the anchor
    factor `(1 + λ·ã_i)` in tads_score preserves R-magnitude information
    instead of squashing it. Sign is preserved (R below the pool mean →
    negative R̃ → smaller s_i after the anchor multiply).

    Args:
        R: (N,) composite reward.
        eps: numerical stabiliser added to the pool std (paper: 1e-6).

    Returns:
        R̃ of shape (N,), real-valued (typically ≈ [-3, +3]).
    """
    if R.numel() == 0:
        return R
    mean = R.mean()
    std = R.std(unbiased=False) if R.numel() > 1 else torch.tensor(0.0, device=R.device)
    # Audit-2 guard: a near-degenerate pool (all L_i, H_i identical → std≈0)
    # would blow R̃ up to ~1e6 magnitude via division by `eps=1e-6`, making
    # top-B effectively determined by floating-point rounding noise of
    # `(R - mean)`. NaN/Inf isn't triggered so `select_top_b` doesn't catch
    # it. Fall back to the raw `R - mean` (no scaling) and log a warning.
    _STD_FLOOR = 1e-4
    if float(std.item()) < _STD_FLOOR:
        logger.warning(
            "calibrated_utility: pool R-std %.2e below floor %.1e — "
            "skipping z-score scaling for this epoch to avoid 1e6-magnitude "
            "explosion. Selection will rank by raw (R - mean) instead.",
            float(std.item()), _STD_FLOOR,
        )
        return R - mean
    return (R - mean) / (std + eps)


def normalize_alignment(
    align_raw: torch.Tensor,
    *,
    collapse_eps: float = 1e-8,
) -> Tuple[torch.Tensor, bool]:
    """Paper Eq.7: pool-level min-max scaling to [0, 1].

        ã_i = (align_i - min_j align_j) / (max_j align_j - min_j align_j)

    Args:
        align_raw: (N,) raw alignment scores (Σ_l <h̄_l(x_i), v_l> / L).
        collapse_eps: if max-min < collapse_eps the pool is degenerate
            (anchor PCA hit a zero gap or layer dot products cancelled out).
            We return ã = 0.5 for every sample AND set the `collapsed` flag
            so the caller can log it — otherwise `boost = 1 + λ·0.5 = const`
            would silently turn TADS into Data Agent.

    Returns:
        (ã, collapsed) — tensor of shape (N,) in [0, 1] and a bool flag.
    """
    if align_raw.numel() == 0:
        return align_raw, False
    a_min = align_raw.min()
    a_max = align_raw.max()
    if (a_max - a_min) < collapse_eps:
        out = torch.full_like(align_raw, 0.5)
        return out, True
    return (align_raw - a_min) / (a_max - a_min), False


def tads_score(
    R_tilde: torch.Tensor,
    alignment_norm: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Paper Eq.8 outer: multiplicative composition of calibrated utility
    and the bounded anchor factor.

        s_i = R̃_i · (1 + λ · ã_i)

    At λ = 0 this reduces to s_i = R̃_i exactly — the clean-ablation
    property highlighted in the paper. The anchor factor lives in
    [1, 1+λ] (since ã ∈ [0,1]) so the score is bounded by R̃ · (1+λ).

    Args:
        R_tilde: (N,) calibrated utility, each in (0, 1).
        alignment_norm: (N,) min-max-normalised alignment, each in [0, 1].
        lam: anchor weighting λ ≥ 0. λ = 0 disables the anchor factor.

    Returns:
        s of shape (N,) — the final per-candidate ranking score.
    """
    if R_tilde.shape != alignment_norm.shape:
        raise ValueError(
            f"tads_score: shape mismatch R̃={tuple(R_tilde.shape)} "
            f"vs ã={tuple(alignment_norm.shape)}"
        )
    if lam < 0:
        raise ValueError(f"tads_score: lam must be >= 0, got {lam}")
    return R_tilde * (1.0 + lam * alignment_norm)


def select_top_b(scores: torch.Tensor, b: int) -> torch.Tensor:
    """Return the top-B indices of `scores` in descending order."""
    if b <= 0:
        raise ValueError(f"select_top_b: b must be >= 1, got {b}")
    if b > scores.numel():
        # Silent shrink would mask a caller bug (selection_ratio×N rounding
        # error, off-by-one in B calculation) and emit fewer indices than
        # downstream code expects. Loud error.
        raise ValueError(
            f"select_top_b: b={b} exceeds pool size N={scores.numel()}. "
            f"Caller computed an invalid top-B target."
        )
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise RuntimeError(
            "select_top_b: scores contain NaN/Inf — likely a degenerate "
            "reward (var=0 → calibrated utility blew up) or an alignment "
            "collapse upstream."
        )
    return scores.topk(b).indices
