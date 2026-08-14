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


def rank01(x: torch.Tensor, *, ties: str = "midrank") -> torch.Tensor:
    """Normalised rank transform to [0, 1]: smallest value → 0, largest → 1.

    ``ties="midrank"`` (default): every member of a tie group receives the
    GROUP-AVERAGE rank, so equal inputs get equal outputs. This matters
    because the progress statistic ``[L^{t-1} - L^t]_+`` has a large mass
    at exactly 0 (every non-improving sample); the previous positional
    tie-breaking assigned those tied samples DISTINCT ranks spanning
    [0, m/(n-1)] purely by their index in the pool file — dataset ORDER
    leaked into D, into S, and into top-B membership (adversarial review
    2026-08, "rank01 tie-breaking converts dataset order into score
    signal"). Midranks also make the transform invariant to pool
    reordering, which the golden regression tests rely on.

    ``ties="positional"``: the legacy v1 behaviour (stable argsort order),
    kept only for reproducing v1 diagnostics.

    For N == 1 the single element maps to 0.5; a constant vector maps to
    0.5 everywhere under midranks.
    """
    n = x.numel()
    if n == 0:
        return x.float()
    if n == 1:
        return torch.full_like(x.float(), 0.5)
    xf = x.float()
    order = torch.argsort(xf, stable=True)
    positional = torch.arange(n, dtype=torch.float32, device=x.device)
    if ties == "positional":
        ranks = torch.empty(n, dtype=torch.float32, device=x.device)
        ranks[order] = positional
        return ranks / float(n - 1)
    if ties != "midrank":
        raise ValueError(f"rank01: ties must be 'midrank' or 'positional', got {ties!r}")
    sorted_x = xf[order]
    # Group equal values, then give each group the mean of its positions.
    new_group = torch.ones(n, dtype=torch.bool, device=x.device)
    new_group[1:] = sorted_x[1:] != sorted_x[:-1]
    group_id = torch.cumsum(new_group.to(torch.long), dim=0) - 1
    n_groups = int(group_id[-1].item()) + 1
    sums = torch.zeros(n_groups, device=x.device).scatter_add_(0, group_id, positional)
    counts = torch.zeros(n_groups, device=x.device).scatter_add_(
        0, group_id, torch.ones(n, device=x.device),
    )
    midranks = sums / counts
    ranks = torch.empty(n, dtype=torch.float32, device=x.device)
    ranks[order] = midranks[group_id]
    return ranks / float(n - 1)


def learnable_difficulty(
    loss_t: torch.Tensor,
    loss_prev: Optional[torch.Tensor] = None,
    eta: float = 0.5,
    *,
    selected_prev: Optional[torch.Tensor] = None,
    progress_mode: str = "split",
    neutral: float = 0.5,
) -> torch.Tensor:
    """Learnable-difficulty view D_i^t (plan §2.2, v3).

        t = 1 (no history):  D_i = rank01(L_i^t)
        t ≥ 2 ("split"):     P̂_i = rank01_{Selected(t-1)}([L^{t-1}-L^t]_+)   i ∈ Selected(t-1)
                             P̂_i = neutral (0.5)                             otherwise
                             D_i = rank01(L_i^t) · (η + (1-η) · P̂_i)

    Why the split (v3): ranking progress over the WHOLE pool lets the
    previous epoch's selection leak into D — samples that received direct
    gradient updates structurally dominate the progress ranks, so P
    measures "was I picked last refresh", not learnability, and selection
    collapses into a rich-get-richer loop. Under "split", only samples
    with gradient EVIDENCE (trained on at t-1) are judged by progress:
    a sample that was trained on and still did not improve is demoted —
    the real dynamic signal for noisy/wrong-answer data. Samples without
    evidence get the neutral value; their churn-in is carried by the
    rank01(L^t) base term (as selected samples' losses fall, unselected
    samples rise in relative rank).

    progress_mode:
        "split"   v3 default (requires ``selected_prev``; falls back to the
                  base ranking with a loud warning when it is missing —
                  neutral-everywhere modulation is a constant factor and
                  would silently pretend progress information existed).
        "global"  v1 behaviour (whole-pool rank) — ablation arm only.
        "off"     D = rank01(L^t), no progress modulation.

    Args:
        loss_t: (N,) per-sample response CE loss at the current refresh.
        loss_prev: (N,) loss at the previous refresh, or None at t=1.
        eta: floor in [0, 1] on the progress modulation (η=1 disables it).
        selected_prev: indices (LongTensor / list) or bool mask of the
            samples selected — hence trained on — at refresh t-1.
        neutral: P̂ value for samples without gradient evidence.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"learnable_difficulty: eta must be in [0,1], got {eta}")
    if progress_mode not in ("split", "global", "off"):
        raise ValueError(
            f"learnable_difficulty: progress_mode must be split/global/off, "
            f"got {progress_mode!r}"
        )
    base = rank01(loss_t)
    if loss_prev is None or progress_mode == "off":
        return base
    if loss_prev.shape != loss_t.shape:
        raise ValueError(
            f"learnable_difficulty: shape mismatch loss_t={tuple(loss_t.shape)} "
            f"vs loss_prev={tuple(loss_prev.shape)}"
        )
    raw_progress = torch.clamp(loss_prev - loss_t, min=0.0)
    if progress_mode == "global":
        progress = rank01(raw_progress)
        return base * (eta + (1.0 - eta) * progress)
    # --- split ---
    if selected_prev is None:
        logger.warning(
            "learnable_difficulty(progress_mode='split'): selected_prev is "
            "missing — progress cannot be attributed to gradient evidence. "
            "Falling back to D = rank01(L^t) for this refresh.",
        )
        return base
    n = loss_t.numel()
    if not torch.is_tensor(selected_prev):
        selected_prev = torch.as_tensor(selected_prev)
    if selected_prev.dtype == torch.bool:
        if selected_prev.numel() != n:
            raise ValueError(
                f"learnable_difficulty: selected_prev mask length "
                f"{selected_prev.numel()} != pool size {n}"
            )
        mask = selected_prev
    else:
        idx = selected_prev.view(-1).long()
        if idx.numel() > 0 and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise ValueError(
                f"learnable_difficulty: selected_prev indices out of range "
                f"[0, {n}) — got min={int(idx.min())}, max={int(idx.max())}"
            )
        mask = torch.zeros(n, dtype=torch.bool, device=loss_t.device)
        mask[idx] = True
    p_hat = torch.full((n,), float(neutral), dtype=torch.float32, device=loss_t.device)
    n_sel = int(mask.sum().item())
    if n_sel > 0:
        p_hat[mask] = rank01(raw_progress[mask])
    return base * (eta + (1.0 - eta) * p_hat)


def mvf_score(
    reliability: torch.Tensor,
    completeness: torch.Tensor,
    difficulty: torch.Tensor,
    alignment_norm: Optional[torch.Tensor],
    *,
    lam: float = 1.0,
    gamma: float = 1.0,
    eps: float = 0.01,
    d_floor: float = 0.5,
) -> torch.Tensor:
    """Quality-gated multi-view fusion score (plan §2.4, v3):

        D'_i = d_floor + (1 - d_floor) · D_i                 (range compression)
        S_i  = (Q_i · c_i + ε)^γ · (D'_i + ε) · (1 + λ · Ã_i)

    Why d_floor (v3): with the raw D ∈ [0, 1], the learnability factor's
    dynamic range is (1+ε)/ε ≈ 101× at ε = 0.01 while the calibrated gate
    separates clean from corrupted by only a small ratio — so a corrupted
    high-loss sample (Q suppressed, D ≈ 1) could outscore a clean easy
    sample (Q high, D ≈ 0) by orders of magnitude, silently REVERSING the
    non-compensation property (adversarial review 2026-08: "the view that
    cannot be overridden is overridden by two orders of magnitude").
    Compressing D to [d_floor, 1] caps the learnability factor's ratio at
    (1+ε)/(d_floor+ε) ≈ 2× at the default d_floor = 0.5: D modulates the
    ranking among reliable samples instead of dominating the gate. The
    explicit non-compensation condition (γ > γ*) is stated in the paper's
    parametric theorem; d_floor = 0 recovers the v2 behaviour as an
    ablation arm.

    Args:
        reliability: (N,) Q_i in [0, 1] (calibrated counterfactual
            fidelity; see ``tads.core.reliability``).
        completeness: (N,) c_i in (0, 1] (1 = complete response;
            ``c_trunc`` for truncated ones).
        difficulty: (N,) D_i in [0, 1] from :func:`learnable_difficulty`.
        alignment_norm: (N,) pool-CDF-normalised anchor alignment in [0, 1],
            or None when the anchor is disabled (factor becomes 1).
        lam: anchor weight λ ≥ 0 (same role as in :func:`tads_score`).
        gamma: gate sharpness γ ≥ 0. γ=0 disables the reliability gate
            (ablation arm); larger γ makes the gate harder.
        eps: gate floor — keeps S non-zero so ranking below the gate stays
            defined and γ-exponentiation is stable at Q·c = 0.
        d_floor: lower bound of the compressed learnability factor in
            [0, 1). 0 disables compression (v2 ablation).
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
    if not (0.0 <= d_floor < 1.0):
        raise ValueError(f"mvf_score: d_floor must be in [0, 1), got {d_floor}")
    gate = torch.pow(reliability * completeness + eps, gamma)
    difficulty_eff = d_floor + (1.0 - d_floor) * difficulty
    score = gate * (difficulty_eff + eps)
    if alignment_norm is not None:
        if alignment_norm.shape != reliability.shape:
            raise ValueError(
                f"mvf_score: shape mismatch alignment={tuple(alignment_norm.shape)} "
                f"vs reliability={tuple(reliability.shape)}"
            )
        score = score * (1.0 + lam * alignment_norm)
    return score


# ---------------------------------------------------------------------------
# TAG scoring path (reliability gate × legacy trajectory-anchored score)
# ---------------------------------------------------------------------------
# Paper Eq. 1:
#
#     s_i^(t) = G_i · R_i^(t) · (1 + λ · widetilde-align_i^(t))
#
# Unlike the MVF path, TAG does NOT replace the composite reward: it keeps
# the trajectory-anchored selector intact and multiplies one static factor in
# front. R and the alignment factor are computed exactly as in the legacy
# path (min-max alignment, raw R, no rank transform), so λ=0 + G≡1 recovers
# the legacy ranking bit-for-bit.


def tag_score(
    gate: torch.Tensor,
    R: torch.Tensor,
    alignment_norm: Optional[torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """Paper Eq. 1: ``s_i = G_i · R_i · (1 + λ · widetilde-align_i)``.

    ``alignment_norm=None`` drops the anchor factor (λ=0 / use_anchor=False),
    giving ``s_i = G_i · R_i``.

    Both dynamic factors are bounded above — R by the pool's own reward range
    and the anchor factor by ``1+λ`` — which is what makes ``G_i = 0`` an
    an exact zero rather than a large penalty.
    """
    if gate.shape != R.shape:
        raise ValueError(
            f"tag_score: shape mismatch gate={tuple(gate.shape)} vs R={tuple(R.shape)}"
        )
    if lam < 0:
        raise ValueError(f"tag_score: lam must be >= 0, got {lam}")
    base = R if alignment_norm is None else tads_score(R, alignment_norm, lam)
    return gate.float() * base


def gated_selection_key(
    score: torch.Tensor,
    fallback_score: torch.Tensor,
    gate: torch.Tensor,
    *,
    validate: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Total order that keeps the exact zero intact and still breaks zero-ties.

    A zero-weight sample scores EXACTLY 0 (Eq. 6 clamps at zero gain), so when the
    budget B exceeds the number of admissible samples, ``topk`` would have to
    choose among a large block of exact ties — and ``torch.topk`` resolves
    ties by index, which promotes the candidate pool's FILE ORDER into the
    selection. That is the same class of bug the ``rank01`` midrank fix
    removed from the progress statistic (plan §1-4).

    The key is a two-level order:

        admissible (G > 0)  ->  2 + rank01(gated score)      in [2, 3]
        zero-weight (G == 0) ->     rank01(fallback score)   in [0, 1]

    so every admissible sample outranks every zero-weight one, ties inside each
    block are broken by the pool-relative rank of a meaningful statistic, and
    the caller can hand the key straight to ``select_top_b`` or
    ``constrained_topk`` — the dedup constraint composes unchanged.

    ``fallback_score`` orders the ZERO-WEIGHT block. Prefer the gate's own
    evidence (``Delta_hat``) so a forced backfill takes the LEAST unreliable
    rejects. Ordering that block by the ungated reward instead would be
    actively perverse: ``R = w·L + (1−w)·H`` increases with response loss,
    and the corruptions the gate exists to reject are high-loss, so the
    backfill would preferentially pull in the most corrupted rejects first.

    ``validate`` checks ``score``/``fallback_score`` for NaN/Inf. This has to
    happen HERE: the caller hands the KEY to ``select_top_b``, whose own
    NaN guard would then inspect the key rather than the score — and
    ``rank01`` maps NaN to the LARGEST rank (``torch.argsort`` puts NaN
    last), so a NaN score would silently become key ``3.0`` and be selected
    FIRST.

    Returns ``(key, n_admissible)``.
    """
    if not (score.shape == fallback_score.shape == gate.shape):
        raise ValueError(
            f"gated_selection_key: shape mismatch score={tuple(score.shape)}, "
            f"fallback={tuple(fallback_score.shape)}, gate={tuple(gate.shape)}"
        )
    if validate:
        for name, t in (("score", score), ("fallback_score", fallback_score)):
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise RuntimeError(
                    f"gated_selection_key: {name} contains NaN/Inf — likely a "
                    f"degenerate reward pool or an alignment collapse upstream. "
                    f"Caught here because rank01 would launder NaN into the TOP "
                    f"of the ranking, bypassing select_top_b's guard."
                )
    admissible = gate.float() > 0
    key = torch.where(
        admissible,
        2.0 + rank01(score),
        rank01(fallback_score),
    )
    return key, int(admissible.sum().item())


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
