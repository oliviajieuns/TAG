"""Unit tests for tads.core.scorer — paper §3.3 (Eq. 3-4, anchor min-max, Eq. 10).

`calibrated_utility` is kept in the module as an ablation helper; the
main paper-faithful path passes raw R into `tads_score` (paper §3.3
final paragraph: "No learned transform, learned policy, z-score, or
sigmoid is applied inside the ranking rule.")."""
from __future__ import annotations

import math

import torch

from tads.core.scorer import (
    calibrated_utility,
    normalize_alignment,
    pool_reward,
    select_top_b,
    tads_score,
)


def test_pool_reward_variance_ratio():
    """w should weight whichever signal has the larger pool variance."""
    # Loss has variance 1, entropy has variance 0 → w ≈ 1.
    loss = torch.tensor([0.0, 1.0, 2.0])
    entropy = torch.tensor([0.5, 0.5, 0.5])
    R, w = pool_reward(loss, entropy)
    assert w > 0.99
    # R is essentially equal to loss when w ≈ 1.
    assert torch.allclose(R, loss, atol=1e-2)


def test_pool_reward_equal_variance():
    """Equal variances → w ≈ 0.5 → R is the simple average."""
    loss = torch.tensor([0.0, 1.0])
    entropy = torch.tensor([0.0, 1.0])
    R, w = pool_reward(loss, entropy)
    assert abs(w - 0.5) < 1e-3
    assert torch.allclose(R, torch.tensor([0.0, 1.0]), atol=1e-2)


def test_calibrated_utility_zero_at_pool_mean():
    """Ablation helper: R̃ = (R - R̄) / (σ_R + ε) — z-score of composite R.

    NOT used by the main paper-faithful pipeline (which feeds raw R to
    `tads_score`). Kept here as an opt-in ablation transform; this test
    just verifies the function's z-score semantics.
    """
    R = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    R_tilde = calibrated_utility(R)
    # Pool mean is 3 → R_tilde[2] ≈ 0.
    assert abs(R_tilde[2].item()) < 1e-3
    # Monotone — larger R yields larger R̃.
    diffs = R_tilde[1:] - R_tilde[:-1]
    assert torch.all(diffs > 0)
    # Sign-preserving — entries below pool mean are negative.
    assert R_tilde[0].item() < 0 and R_tilde[-1].item() > 0


def test_normalize_alignment_minmax():
    """Min-max into [0, 1]; collapse flag triggers on zero spread."""
    a, collapsed = normalize_alignment(torch.tensor([0.1, 0.5, 0.9]))
    assert not collapsed
    assert math.isclose(a.min().item(), 0.0, abs_tol=1e-6)
    assert math.isclose(a.max().item(), 1.0, abs_tol=1e-6)

    a2, collapsed2 = normalize_alignment(torch.tensor([0.3, 0.3, 0.3]))
    assert collapsed2
    assert torch.all(a2 == 0.5).item()


def test_tads_score_lam_zero_recovers_base_reward():
    """At λ=0 the anchor factor is 1, so s_i == R_i exactly.

    Paper §3.3: "Setting λ = 0 recovers the composite-reward base ranking
    exactly." `tads_score` is a monotonic transform of its first argument,
    so the identity holds for any input scale.
    """
    R = torch.tensor([0.2, 0.5, 0.8])
    a = torch.tensor([0.1, 0.5, 0.9])
    s = tads_score(R, a, lam=0.0)
    assert torch.allclose(s, R)


def test_tads_score_multiplicative_boost():
    """At λ=1 a top-alignment sample gets the full 2× boost (paper Eq. 10)."""
    R = torch.tensor([1.2, 1.2])
    a = torch.tensor([0.0, 1.0])
    s = tads_score(R, a, lam=1.0)
    assert math.isclose(s[0].item(), 1.2, abs_tol=1e-6)
    assert math.isclose(s[1].item(), 2.4, abs_tol=1e-6)


def test_select_top_b_descending():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7])
    idx = select_top_b(scores, b=2).cpu().tolist()
    assert idx == [1, 3]


def test_select_top_b_rejects_nan():
    scores = torch.tensor([0.5, float("nan"), 0.3])
    try:
        select_top_b(scores, b=2)
    except RuntimeError:
        return
    raise AssertionError("select_top_b should refuse NaN scores")
