"""Unit tests for the MVF scoring path (plan §1): rank01, learnable
difficulty, the quality-gated fusion score, and reliability from
counterfactual losses. Also guards that the legacy scorer functions are
untouched by the MVF additions."""
from __future__ import annotations

import torch

from tads.core.reliability import reliability_from_losses
from tads.core.scorer import (
    learnable_difficulty,
    mvf_score,
    rank01,
    tads_score,
)


# ---------------------------------------------------------------------------
# rank01
# ---------------------------------------------------------------------------

def test_rank01_bounds_and_order():
    x = torch.tensor([3.0, 1.0, 2.0, 10.0])
    r = rank01(x)
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0
    assert torch.equal(torch.argsort(r), torch.argsort(x, stable=True))
    assert rank01(torch.tensor([5.0])).item() == 0.5


def test_rank01_is_scale_invariant():
    x = torch.randn(100)
    assert torch.allclose(rank01(x), rank01(x * 1000 + 5))


# ---------------------------------------------------------------------------
# learnable difficulty D
# ---------------------------------------------------------------------------

def test_difficulty_no_history_is_loss_rank():
    loss = torch.tensor([0.5, 2.0, 1.0])
    assert torch.allclose(learnable_difficulty(loss, None), rank01(loss))


def test_progress_disambiguates_stuck_vs_learnable():
    """Two samples with identical (high) current loss: the one whose loss
    is FALLING must outrank the persistently-stuck one."""
    loss_prev = torch.tensor([1.0, 3.0, 3.0, 1.0])
    loss_t = torch.tensor([0.9, 3.0, 2.0, 0.5])  # idx1 stuck, idx2 improving
    d = learnable_difficulty(loss_t, loss_prev, eta=0.5)
    assert d[2] > d[1]


def test_eta_one_disables_progress():
    loss_prev = torch.rand(20) + 1.0
    loss_t = torch.rand(20)
    d = learnable_difficulty(loss_t, loss_prev, eta=1.0)
    assert torch.allclose(d, rank01(loss_t))


# ---------------------------------------------------------------------------
# reliability Q
# ---------------------------------------------------------------------------

def test_reliability_ranks_mismatch_low():
    """A mismatched sample gains nothing from its true instruction
    (loss_cf ≈ loss_orig) → lowest Q. A clean-but-hard sample keeps a large
    counterfactual delta → high Q even though its loss is the highest."""
    #                    clean-easy  clean-HARD  mismatch
    loss_orig = torch.tensor([1.0,      3.0,       2.5])
    loss_cf = torch.tensor([3.0,       6.0,       2.6])
    q = reliability_from_losses(loss_orig, loss_cf)
    assert q[2] == q.min()
    assert q[1] == q.max()


# ---------------------------------------------------------------------------
# fused score S
# ---------------------------------------------------------------------------

def test_gate_blocks_noisy_high_loss_sample():
    """The core failure mode of uncertainty-as-quality: a noisy sample with
    the HIGHEST difficulty must not outrank a clean hard sample, because Q
    gates multiplicatively (plan §1.4)."""
    #                       clean-hard  noisy
    q = torch.tensor([0.9,       0.05])
    c = torch.tensor([1.0,       1.0])
    d = torch.tensor([0.7,       1.0])   # noisy sample has max difficulty
    s = mvf_score(q, c, d, None, gamma=1.0, eps=0.01)
    assert s[0] > s[1]


def test_completeness_gates_truncated():
    q = torch.tensor([0.8, 0.8])
    c = torch.tensor([1.0, 0.2])  # second sample truncated
    d = torch.tensor([0.5, 0.5])
    s = mvf_score(q, c, d, None)
    assert s[0] > s[1]


def test_gamma_zero_disables_gate():
    q = torch.rand(10)
    c = torch.rand(10) * 0.5 + 0.5
    d = torch.rand(10)
    s = mvf_score(q, c, d, None, gamma=0.0, eps=0.01)
    assert torch.allclose(s, d + 0.01)


def test_alignment_factor_matches_legacy_boost():
    """With the gate wide open (Q=c=1, γ=1) the alignment factor must act
    exactly like the legacy (1 + λ·align) boost."""
    n = 16
    ones = torch.ones(n)
    d = torch.rand(n)
    align = torch.rand(n)
    lam = 1.3
    s = mvf_score(ones, ones, d, align, lam=lam, gamma=1.0, eps=0.01)
    expected = (1.0 + 0.01) * (d + 0.01) * (1.0 + lam * align)
    assert torch.allclose(s, expected)


def test_mvf_score_validation():
    q = torch.rand(4)
    c = torch.rand(4)
    d = torch.rand(4)
    for bad_kwargs in ({"lam": -1.0}, {"gamma": -0.1}, {"eps": 0.0}):
        try:
            mvf_score(q, c, d, None, **bad_kwargs)
            assert False, f"expected ValueError for {bad_kwargs}"
        except ValueError:
            pass


def test_legacy_tads_score_unchanged():
    """The MVF additions must not perturb the legacy Eq. 10 path."""
    R = torch.tensor([1.0, 2.0, 3.0])
    align = torch.tensor([0.0, 0.5, 1.0])
    s = tads_score(R, align, lam=1.0)
    assert torch.allclose(s, torch.tensor([1.0, 3.0, 6.0]))


# ---------------------------------------------------------------------------
# end-to-end sanity on a synthetic contaminated pool
# ---------------------------------------------------------------------------

def test_mvf_beats_uncertainty_on_synthetic_dirty_pool():
    """Synthetic pool: 20 % dirty samples get high loss/entropy but near-zero
    counterfactual delta. Entropy top-K should over-select them; the MVF
    score should reject them."""
    g = torch.Generator().manual_seed(0)
    n, n_dirty = 500, 100
    dirty = torch.zeros(n, dtype=torch.bool)
    dirty[torch.randperm(n, generator=g)[:n_dirty]] = True

    loss = torch.rand(n, generator=g) * 2 + 1.0
    loss[dirty] += 1.5                       # dirt looks "hard"
    entropy = torch.rand(n, generator=g) + 0.5
    entropy[dirty] += 1.0                    # ...and "uncertain"
    delta = torch.rand(n, generator=g) * 2 + 1.0
    delta[dirty] = torch.rand(n_dirty, generator=g) * 0.2  # but no fidelity
    loss_cf = loss + delta

    q = reliability_from_losses(loss, loss_cf)
    c = torch.ones(n)
    d = learnable_difficulty(loss, None)
    s = mvf_score(q, c, d, None)

    k = n // 10
    dirty_entropy = dirty[entropy.topk(k).indices].float().mean()
    dirty_mvf = dirty[s.topk(k).indices].float().mean()
    assert dirty_entropy > 0.4     # uncertainty view is contaminated
    assert dirty_mvf < 0.05        # gated fusion rejects the dirt
