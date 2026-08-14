"""Unit tests for the MVF scoring path (plan §1): rank01, learnable
difficulty, the quality-gated fusion score, and reliability from
counterfactual losses. Also guards that the legacy scorer functions are
untouched by the MVF additions."""
from __future__ import annotations

import torch

from tag.core.reliability import reliability_from_losses
from tag.core.scorer import (
    learnable_difficulty,
    mvf_score,
    rank01,
    legacy_score,
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


def test_rank01_midranks_ties():
    """v3: tied inputs must receive IDENTICAL outputs — positional
    tie-breaking leaked dataset order into the score wherever the input has
    mass at one value (the [L_prev - L]_+ progress statistic is exactly 0
    for every non-improving sample)."""
    # Constant vector → 0.5 everywhere, regardless of length.
    assert torch.allclose(rank01(torch.zeros(7)), torch.full((7,), 0.5))
    # Two tie groups: [0,0,0,1,1] → midranks [1,1,1,3.5,3.5]/4.
    r = rank01(torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
    expected = torch.tensor([0.25, 0.875, 0.25, 0.875, 0.25])
    assert torch.allclose(r, expected)
    # Permutation invariance: output is a function of the VALUE, not the
    # position.
    x = torch.tensor([2.0, 2.0, 5.0, 1.0, 2.0])
    perm = torch.tensor([4, 2, 0, 3, 1])
    assert torch.allclose(rank01(x)[perm], rank01(x[perm]))


def test_rank01_positional_mode_is_legacy():
    x = torch.tensor([1.0, 1.0, 0.0])
    r = rank01(x, ties="positional")
    # Legacy: ties broken by pool position → distinct ranks.
    assert torch.allclose(r, torch.tensor([0.5, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# learnable difficulty D
# ---------------------------------------------------------------------------

def test_difficulty_no_history_is_loss_rank():
    loss = torch.tensor([0.5, 2.0, 1.0])
    assert torch.allclose(learnable_difficulty(loss, None), rank01(loss))


def test_progress_disambiguates_stuck_vs_learnable():
    """Two samples with identical (high) current loss, BOTH trained on at
    t-1: the one whose loss is FALLING must outrank the persistently-stuck
    one. (v3: progress judgment requires gradient evidence, so both must be
    in selected_prev for the comparison to be attributable.)"""
    loss_prev = torch.tensor([1.0, 3.0, 3.0, 1.0])
    loss_t = torch.tensor([0.9, 3.0, 2.0, 0.5])  # idx1 stuck, idx2 improving
    d = learnable_difficulty(
        loss_t, loss_prev, eta=0.5, selected_prev=torch.tensor([1, 2]),
    )
    assert d[2] > d[1]


def test_eta_one_disables_progress():
    loss_prev = torch.rand(20) + 1.0
    loss_t = torch.rand(20)
    sel = torch.arange(10)
    d = learnable_difficulty(loss_t, loss_prev, eta=1.0, selected_prev=sel)
    assert torch.allclose(d, rank01(loss_t))


# ---------------------------------------------------------------------------
# v3 split-progress D (selection-feedback-loop fix)
# ---------------------------------------------------------------------------

def test_split_progress_neutralizes_unselected():
    """Unselected samples have no gradient evidence: their P̂ must be the
    neutral 0.5, NOT a rank of their (generalisation-noise) loss deltas.
    Under the v1 global rank, being selected at t-1 was itself the dominant
    progress signal — the rich-get-richer loop."""
    n = 10
    loss_prev = torch.ones(n) * 2.0
    loss_t = torch.ones(n) * 1.5           # identical current loss for all
    # Give unselected samples LARGER raw deltas than selected ones — under
    # the global rank they would dominate P; under split they must all sit
    # at neutral.
    loss_t[:5] = 1.9                        # selected: small progress (0.1)
    loss_t[5:] = 1.0                        # unselected: big progress (1.0)
    sel = torch.arange(5)
    d = learnable_difficulty(
        loss_t, loss_prev, eta=0.0, selected_prev=sel, progress_mode="split",
    )
    base = rank01(loss_t)
    # Unselected: D = base · neutral(0.5) exactly.
    assert torch.allclose(d[5:], base[5:] * 0.5)
    # Selected group: ranked within the group only (identical deltas → all
    # midrank 0.5 too, but via the group path).
    assert torch.allclose(d[:5], base[:5] * 0.5)


def test_split_progress_demotes_trained_but_stuck():
    """The real dynamic noise signal: a sample TRAINED ON at t-1 whose loss
    did not fall must rank below a trained-on sample that improved."""
    loss_prev = torch.tensor([3.0, 3.0, 1.0, 1.0])
    loss_t = torch.tensor([3.0, 2.0, 1.0, 1.0])   # idx0 stuck, idx1 improved
    sel = torch.tensor([0, 1])
    d = learnable_difficulty(
        loss_t, loss_prev, eta=0.0, selected_prev=sel, progress_mode="split",
    )
    # Same-loss comparison isn't available (losses differ), so compare the
    # progress factors directly through D / rank01(L).
    base = rank01(loss_t)
    p0 = d[0] / base[0]
    p1 = d[1] / base[1]
    assert p1 > p0
    assert torch.isclose(d[2] / base[2], torch.tensor(0.5))  # unselected neutral


def test_split_progress_accepts_bool_mask_and_list():
    loss_prev = torch.rand(8) + 1.0
    loss_t = torch.rand(8)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:4] = True
    d_mask = learnable_difficulty(loss_t, loss_prev, selected_prev=mask)
    d_list = learnable_difficulty(loss_t, loss_prev, selected_prev=[0, 1, 2, 3])
    assert torch.allclose(d_mask, d_list)


def test_split_progress_out_of_range_raises():
    try:
        learnable_difficulty(
            torch.rand(4), torch.rand(4), selected_prev=torch.tensor([3, 4]),
        )
        assert False
    except ValueError:
        pass


def test_split_without_evidence_falls_back_to_base():
    """progress_mode='split' with no selected_prev cannot attribute progress
    — it must fall back to the base ranking (with a warning), not silently
    pretend evidence existed."""
    loss_prev = torch.rand(6) + 1.0
    loss_t = torch.rand(6)
    d = learnable_difficulty(loss_t, loss_prev, progress_mode="split")
    assert torch.allclose(d, rank01(loss_t))


def test_global_mode_is_v1_ablation():
    loss_prev = torch.rand(12) + 1.0
    loss_t = torch.rand(12)
    d = learnable_difficulty(loss_t, loss_prev, eta=0.5, progress_mode="global")
    expected = rank01(loss_t) * (
        0.5 + 0.5 * rank01(torch.clamp(loss_prev - loss_t, min=0.0))
    )
    assert torch.allclose(d, expected)


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
    # d_floor=0 isolates the gate factor exactly (v2 semantics).
    s = mvf_score(q, c, d, None, gamma=0.0, eps=0.01, d_floor=0.0)
    assert torch.allclose(s, d + 0.01)
    # v3 default d_floor=0.5 compresses D to [0.5, 1].
    s3 = mvf_score(q, c, d, None, gamma=0.0, eps=0.01)
    assert torch.allclose(s3, 0.5 + 0.5 * d + 0.01)


def test_alignment_factor_matches_legacy_boost():
    """With the gate wide open (Q=c=1, γ=1, d_floor=0) the alignment factor
    must act exactly like the legacy (1 + λ·align) boost."""
    n = 16
    ones = torch.ones(n)
    d = torch.rand(n)
    align = torch.rand(n)
    lam = 1.3
    s = mvf_score(ones, ones, d, align, lam=lam, gamma=1.0, eps=0.01, d_floor=0.0)
    expected = (1.0 + 0.01) * (d + 0.01) * (1.0 + lam * align)
    assert torch.allclose(s, expected)


def test_d_floor_caps_learnability_ratio():
    """The v3 fix for gate-vs-D magnitude reversal: with d_floor=0.5 the
    learnability factor's dynamic range is at most (1+ε)/(0.5+ε) ≈ 2, so a
    corrupted sample with maximal D cannot override an ε-floored gate the
    way the uncompressed v2 D (ratio ≈ 101 at ε=0.01) provably could."""
    eps = 0.01
    # Worst case: dirty sample maximal D, clean sample minimal D.
    q = torch.tensor([0.0, 0.8])   # dirty gated to the floor, clean high
    c = torch.ones(2)
    d = torch.tensor([1.0, 0.0])
    s = mvf_score(q, c, d, None, gamma=1.0, eps=eps, d_floor=0.5)
    assert s[1] > s[0], "clean-easy must survive against dirty-max-D"
    # And the explicit v2 regression: d_floor=0 lets the dirt win.
    s_v2 = mvf_score(q, c, d, None, gamma=1.0, eps=eps, d_floor=0.0)
    assert s_v2[0] > s_v2[1], (
        "expected the v2 reversal — if this stops failing, the theorem's "
        "motivating counterexample needs rewriting"
    )


def test_mvf_score_validation():
    q = torch.rand(4)
    c = torch.rand(4)
    d = torch.rand(4)
    for bad_kwargs in (
        {"lam": -1.0}, {"gamma": -0.1}, {"eps": 0.0}, {"d_floor": 1.0},
        {"d_floor": -0.1},
    ):
        try:
            mvf_score(q, c, d, None, **bad_kwargs)
            assert False, f"expected ValueError for {bad_kwargs}"
        except ValueError:
            pass


def test_legacy_score_unchanged():
    """The MVF additions must not perturb the legacy Eq. 10 path."""
    R = torch.tensor([1.0, 2.0, 3.0])
    align = torch.tensor([0.0, 0.5, 1.0])
    s = legacy_score(R, align, lam=1.0)
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
