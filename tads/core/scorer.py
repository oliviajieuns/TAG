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
from typing import Optional, Tuple

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


# ---------------------------------------------------------------------------
# MVF scoring path (reliability × learnability × alignment)
# ---------------------------------------------------------------------------
# The multi-view-fusion score replaces the uncertainty-as-quality composite
# with three disentangled views (docs/plan_low_quality_multiview.md §1):
#
#     Q_i   reliability   — counterfactual instruction fidelity (static,
#                           base checkpoint; tads/core/reliability.py)
#     c_i   completeness  — EOS/truncation gate, data-level
#     D_i^t learnability  — rank(L^t) modulated by epoch-to-epoch progress
#     A_i^t alignment     — hidden-state anchor alignment (unchanged)
#
#     S_i^t = (Q_i · c_i + ε)^γ · (D_i^t + ε) · (1 + λ · A_i^t)
#
# Q·c enters as a multiplicative GATE, not an additive term: a noisy
# sample's high loss must not be able to offset its low reliability.
# Entropy is intentionally absent — it is uncertainty, not quality, and is
# kept only as a logged diagnostic / ablation arm.


def rank01(x: torch.Tensor) -> torch.Tensor:
    """Normalised rank transform to [0, 1]: smallest value → 0, largest → 1.

    Ties are broken by position (stable argsort), which keeps the transform
    deterministic. For N == 1 the single element maps to 0.5.
    """
    n = x.numel()
    if n == 0:
        return x.float()
    if n == 1:
        return torch.full_like(x.float(), 0.5)
    order = torch.argsort(x, stable=True)
    ranks = torch.empty(n, dtype=torch.float32, device=x.device)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=x.device)
    return ranks / float(n - 1)


def learnable_difficulty(
    loss_t: torch.Tensor,
    loss_prev: Optional[torch.Tensor] = None,
    eta: float = 0.5,
) -> torch.Tensor:
    """Learnable-difficulty view D_i^t (plan §1.2).

        t = 1 (no history):  D_i = rank01(L_i^t)
        t ≥ 2:               P_i = rank01([L_i^{t-1} - L_i^t]_+)
                             D_i = rank01(L_i^t) · (η + (1-η) · P_i)

    High current loss alone is ambiguous between "hard but clean" and
    "noise". Progress P disambiguates: a sample whose loss is falling
    across refreshes is hard-but-learnable and keeps its difficulty
    weight; a persistently-stuck sample is discounted toward η·rank(L).

    Args:
        loss_t: (N,) per-sample response CE loss at the current refresh.
        loss_prev: (N,) loss at the previous refresh, or None at t=1.
        eta: floor in [0, 1] on the progress modulation (η=1 disables it).
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"learnable_difficulty: eta must be in [0,1], got {eta}")
    base = rank01(loss_t)
    if loss_prev is None:
        return base
    if loss_prev.shape != loss_t.shape:
        raise ValueError(
            f"learnable_difficulty: shape mismatch loss_t={tuple(loss_t.shape)} "
            f"vs loss_prev={tuple(loss_prev.shape)}"
        )
    progress = rank01(torch.clamp(loss_prev - loss_t, min=0.0))
    return base * (eta + (1.0 - eta) * progress)


def mvf_score(
    reliability: torch.Tensor,
    completeness: torch.Tensor,
    difficulty: torch.Tensor,
    alignment_norm: Optional[torch.Tensor],
    *,
    lam: float = 1.0,
    gamma: float = 1.0,
    eps: float = 0.01,
) -> torch.Tensor:
    """Quality-gated multi-view fusion score (plan §1.4):

        S_i = (Q_i · c_i + ε)^γ · (D_i + ε) · (1 + λ · Ã_i)

    Args:
        reliability: (N,) Q_i in [0, 1] (rank-normalised counterfactual
            fidelity; see ``tads.core.reliability``).
        completeness: (N,) c_i in (0, 1] (1 = complete response;
            ``c_trunc`` for truncated ones).
        difficulty: (N,) D_i in [0, 1] from :func:`learnable_difficulty`.
        alignment_norm: (N,) min-max-normalised anchor alignment in [0, 1],
            or None when the anchor is disabled (factor becomes 1).
        lam: anchor weight λ ≥ 0 (same role as in :func:`tads_score`).
        gamma: gate sharpness γ ≥ 0. γ=0 disables the reliability gate
            (ablation arm); larger γ makes the gate harder.
        eps: gate floor — keeps S non-zero so ranking below the gate stays
            defined and γ-exponentiation is stable at Q·c = 0.
    """
    for name, t in (
        ("reliability", reliability),
        ("completeness", completeness),
        ("difficulty", difficulty),
    ):
        if t.shape != reliability.shape:
            raise ValueError(
                f"mvf_score: shape mismatch {name}={tuple(t.shape)} "
                f"vs reliability={tuple(reliability.shape)}"
            )
    if lam < 0:
        raise ValueError(f"mvf_score: lam must be >= 0, got {lam}")
    if gamma < 0:
        raise ValueError(f"mvf_score: gamma must be >= 0, got {gamma}")
    if eps <= 0:
        raise ValueError(f"mvf_score: eps must be > 0, got {eps}")
    gate = torch.pow(reliability * completeness + eps, gamma)
    score = gate * (difficulty + eps)
    if alignment_norm is not None:
        if alignment_norm.shape != reliability.shape:
            raise ValueError(
                f"mvf_score: shape mismatch alignment={tuple(alignment_norm.shape)} "
                f"vs reliability={tuple(reliability.shape)}"
            )
        score = score * (1.0 + lam * alignment_norm)
    return score


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
