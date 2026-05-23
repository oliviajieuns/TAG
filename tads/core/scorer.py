"""TADS scoring helpers (paper §3.3, Algorithm 1).

Pure-function utilities for the deterministic TADS selection pipeline. No
PPO actor, no learned components — paper §3.3 final paragraph: "No
learned transform, learned policy, z-score, or sigmoid is applied inside
the ranking rule."

Pipeline composition (one refresh step t):
    L_i, H_i              = per-sample loss / entropy         (compute_rewards in reward.py)
    R_i, w                = pool_reward(L_arr, H_arr)         (paper Eqs. 3-4)
    align_i (raw)         = (1/L) Σ_l <h̄_l(x_i), v_l>         (paper §3.3 anchor; in selector)
    widetilde-align_i     = normalize_alignment(align_raw)     (paper §3.3 anchor — min-max to [0,1])
    s_i                   = tads_score(R, widetilde-align, λ)  (paper Eq. 10)
    selection             = top-B indices of s_i              (Algorithm 1)

The functions accept 1-D tensors of shape (N,) where N is the candidate
pool size. All ops are vectorised and run on CPU after the per-sample
forwards complete; no GPU memory required.

Note: ``calibrated_utility`` (pool z-score of R) is kept in this module
for ablation experiments but is NOT part of the paper-faithful pipeline —
the main path in ``selector.collect_episode`` passes raw R directly to
``tads_score`` (paper Eq. 10).
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
    """Composite reward (paper Eq. 3) using the pool-level variance-ratio
    weight (paper Eq. 4).

    Args:
        loss_arr: per-sample mean CE loss over response tokens; shape (N,).
        entropy_arr: per-sample mean predictive entropy; shape (N,).
        eps: numerical stabiliser in the variance-ratio denominator (paper Eq. 4 ε).

    Returns:
        (R, w) where R has shape (N,) and w is a python float.
        w = Var_D(L) / (Var_D(L) + Var_D(H) + ε)   (paper Eq. 4)
        R_i = w · L_i + (1 - w) · H_i               (paper Eq. 3)

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
    """Pool-level z-score of the composite reward — **ablation-only**.

    NOT part of the paper-faithful TADS pipeline. Paper §3.3 final
    paragraph explicitly excludes this: "No learned transform, learned
    policy, z-score, or sigmoid is applied inside the ranking rule."
    The main pipeline in ``selector.collect_episode`` passes raw R
    directly to ``tads_score`` (paper Eq. 10).

    Kept here so ablation runs can opt-in to z-score variant by manually
    composing ``tads_score(calibrated_utility(R), widetilde_align, lam)``
    in lieu of the default raw-R path.

        R̃_i = (R_i - R̄) / (σ_R + ε)

    Args:
        R: (N,) composite reward (paper Eq. 3).
        eps: numerical stabiliser added to the pool std.

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
    """Min-max normalisation of raw alignment to [0, 1] (paper §3.3).

        widetilde-align_i = (align_i - min_j align_j)
                            / (max_j align_j - min_j align_j)

    Args:
        align_raw: (N,) raw alignment scores
            align_i = (1/L) Σ_l <h̄_l(x_i; θ_{t-1}), v_l^{(t)}>  (paper §3.3 anchor).
        collapse_eps: if max-min < collapse_eps the pool is degenerate
            (anchor PCA hit a zero gap or layer dot products cancelled out).
            We return widetilde-align = 0.5 for every sample AND set the
            ``collapsed`` flag so the caller can log it — otherwise
            ``1 + λ·0.5 = const`` would silently turn TADS into the
            composite-reward base ranking.

    Returns:
        (widetilde-align, collapsed) — tensor of shape (N,) in [0, 1] and a bool flag.
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
    R: torch.Tensor,
    alignment_norm: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Paper Eq. 10: multiplicative composition of the composite reward
    and the bounded anchor factor.

        s_i = R_i · (1 + λ · widetilde-align_i)

    At λ = 0 this recovers the composite-reward base ranking exactly
    (paper §3.3): s_i = R_i. The anchor factor lives in [1, 1+λ] (since
    widetilde-align ∈ [0,1]) so the score is bounded by R · (1+λ).

    Args:
        R: (N,) composite reward (paper Eq. 3). Raw — NOT z-scored.
            Paper §3.3 final paragraph: "No learned transform, learned
            policy, z-score, or sigmoid is applied inside the ranking
            rule." For an ablation variant, callers may compose
            ``tads_score(calibrated_utility(R), widetilde_align, lam)``
            but the main pipeline passes raw R here.
        alignment_norm: (N,) min-max-normalised alignment widetilde-align,
            each in [0, 1] (paper §3.3 anchor).
        lam: anchor weighting λ ≥ 0. λ = 0 recovers the composite-reward
            base ranking exactly (paper §3.3).

    Returns:
        s of shape (N,) — the final per-candidate ranking score (paper Eq. 10).
    """
    if R.shape != alignment_norm.shape:
        raise ValueError(
            f"tads_score: shape mismatch R={tuple(R.shape)} "
            f"vs widetilde-align={tuple(alignment_norm.shape)}"
        )
    if lam < 0:
        raise ValueError(f"tads_score: lam must be >= 0, got {lam}")
    return R * (1.0 + lam * alignment_norm)


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
            "reward pool or an alignment collapse upstream."
        )
    return scores.topk(b).indices
