"""Tests for the TAG reliability gate (paper Eqs. 2-6).

The properties pinned here are the ones the paper's claims rest on:

  * Eq. 3 is the ratio, not the raw difference — invariant to a global
    rescaling of the response's difficulty.
  * Eq. 5 catches LOCALIZED corruption that Eq. 3 dilutes. This is the whole
    reason spans exist; if it regresses, the method reduces to IFD.
  * Eq. 6 zeroes EXACTLY at zero gain, so the fusion is non-compensatory.
  * The common-prefix trim, without which the two pools' per-token vectors
    are silently misaligned whenever prompt lengths differ.
"""
from __future__ import annotations

import math

import pytest
import torch

from tads.core.gate import (
    GateConfig,
    calibrate_gate_scale,
    compute_gate,
    gate_components,
    gate_from_delta_hat,
    overall_gain,
    resolve_scale,
    span_gains,
    spans_from_token_losses,
    tail_gain,
    valid_span_mask,
)


def _cfg(**kw):
    base = dict(span_tokens=4, tau=0.5, scale=0.2, min_span_tokens=2,
                min_common_tokens=4)
    base.update(kw)
    return GateConfig(**base)


# ---------------------------------------------------------------------------
# Eq. 3 — overall relative gain
# ---------------------------------------------------------------------------

def test_overall_gain_matches_closed_form():
    total_true = torch.tensor([2.0, 5.0])
    total_cf = torch.tensor([10.0, 5.0])
    got = overall_gain(total_true, total_cf)
    assert got.tolist() == pytest.approx([1.0 - 0.2, 0.0])


def test_overall_gain_is_scale_free():
    """The defining property of the ratio form: an intrinsically harder
    response with the SAME instruction dependency gets the same gain. The
    raw difference ΔL used by the MVF gate does not have this property."""
    easy = overall_gain(torch.tensor([1.0]), torch.tensor([4.0]))
    hard = overall_gain(torch.tensor([10.0]), torch.tensor([40.0]))
    assert float(easy.item()) == pytest.approx(float(hard.item()))
    # ...whereas the raw difference differs by an order of magnitude.
    assert (40.0 - 10.0) != pytest.approx(4.0 - 1.0)


def test_overall_gain_denominator_guard():
    got = overall_gain(torch.tensor([0.0]), torch.tensor([0.0]), eps_den=1e-3)
    assert torch.isfinite(got).all()


# ---------------------------------------------------------------------------
# Eqs. 4-5 — spans, C_i, tail
# ---------------------------------------------------------------------------

def test_span_partition_sums_and_counts():
    tok = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    n = torch.tensor([6])
    sp = spans_from_token_losses(tok, n, tok, n, span_tokens=4)
    assert sp["n_spans"].tolist() == [2]
    assert sp["span_true"][0, :2].tolist() == pytest.approx([10.0, 11.0])
    assert sp["span_len"][0, :2].tolist() == [4, 2]
    assert float(sp["total_true"][0]) == pytest.approx(21.0)


def test_common_prefix_trim_when_pools_have_different_lengths():
    """Budget truncation differs between the true and counterfactual prompt,
    so only the shared prefix is comparable. Tokens past it must be dropped
    from BOTH sides, not silently compared against padding."""
    tok_true = torch.tensor([[1.0, 1.0, 1.0, 1.0, 9.9, 9.9]])
    tok_cf = torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 0.0]])
    n_true = torch.tensor([6])
    n_cf = torch.tensor([4])
    sp = spans_from_token_losses(tok_true, n_true, tok_cf, n_cf, span_tokens=4)
    assert sp["n_common"].tolist() == [4]
    # The 9.9 tokens past the common prefix are excluded entirely.
    assert float(sp["total_true"][0]) == pytest.approx(4.0)
    assert float(sp["total_cf"][0]) == pytest.approx(8.0)


def test_localized_corruption_is_caught_by_the_tail_but_not_the_mean():
    """The headline property (paper: 'the mean alone can be diluted by
    localized corruption'). A response that is clean for 8 tokens and wrong
    for 4 keeps a healthy Delta_bar but must fail on Delta_min."""
    tok_true = torch.tensor([[0.5] * 8 + [3.0] * 4])
    tok_cf = torch.tensor([[3.0] * 12])
    n = torch.tensor([12])
    comp = gate_components(tok_true, n, tok_cf, n, cfg=_cfg())
    assert float(comp["delta_bar"][0]) > 0.5      # mean looks fine
    assert float(comp["delta_min"][0]) == pytest.approx(0.0)  # tail does not
    assert float(comp["delta_hat"][0]) == pytest.approx(0.0)


def test_low_information_spans_are_excluded_from_C():
    """Boilerplate carries no instruction dependency by nature and must not
    be allowed to trigger the gate."""
    span_cf = torch.tensor([[8.0, 0.4]])   # second span is boilerplate
    span_len = torch.tensor([[4, 4]])
    mask = valid_span_mask(span_cf, span_len, tau=0.5, min_span_tokens=2)
    assert mask.tolist() == [[True, False]]


def test_tau_per_token_is_length_independent():
    """A trailing partial span with the same per-token content as a full one
    must survive C_i. Under the literal 'absolute' reading it does not —
    that is docs/tag-paper-deltas.md item P1."""
    span_cf = torch.tensor([[8.0, 2.0]])   # 4 tokens @2.0, then 1 token @2.0
    span_len = torch.tensor([[4, 1]])
    per_token = valid_span_mask(
        span_cf, span_len, tau=1.0, tau_mode="per_token", min_span_tokens=1,
    )
    absolute = valid_span_mask(
        span_cf, span_len, tau=4.0, tau_mode="absolute", min_span_tokens=1,
    )
    assert per_token.tolist() == [[True, True]]
    assert absolute.tolist() == [[True, False]]


def test_empty_C_falls_back_to_delta_bar():
    """Every span excluded => Eq. 5 is undefined. The tail abstains rather
    than vetoing (docs item P3)."""
    gains = torch.tensor([[0.9, 0.9]])
    mask = torch.tensor([[False, False]])
    fallback = torch.tensor([0.42])
    tail, used = tail_gain(gains, mask, fallback)
    assert float(tail[0]) == pytest.approx(0.42)
    assert bool(used[0])


def test_tail_quantile_mode_is_less_extreme_than_min():
    gains = torch.tensor([[0.1, 0.5, 0.9]])
    mask = torch.ones_like(gains, dtype=torch.bool)
    fb = torch.tensor([0.0])
    t_min, _ = tail_gain(gains, mask, fb, mode="min")
    t_q, _ = tail_gain(gains, mask, fb, mode="quantile", quantile=0.5)
    assert float(t_min[0]) == pytest.approx(0.1)
    assert float(t_q[0]) == pytest.approx(0.5)


def test_span_gains_zero_on_padding():
    g = span_gains(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]]),
        torch.tensor([[4, 0]]),
    )
    assert float(g[0, 1]) == 0.0


# ---------------------------------------------------------------------------
# Eq. 6 — the gate
# ---------------------------------------------------------------------------

def test_gate_zeroes_exactly_at_or_below_zero_gain():
    """Non-compensation depends on EXACT zero, not a small value: the fused
    product must be identically 0 regardless of the dynamic factors."""
    d = torch.tensor([-1.0, 0.0, 0.5])
    c = torch.ones(3)
    g = gate_from_delta_hat(d, c, scale=0.2)
    assert float(g[0]) == 0.0
    assert float(g[1]) == 0.0
    assert float(g[2]) > 0.0
    # And the product with an arbitrarily large dynamic score stays zero.
    assert float(g[0] * 1e12) == 0.0


def test_gate_is_monotone_and_bounded():
    d = torch.tensor([0.1, 0.3, 0.6, 0.95])
    g = gate_from_delta_hat(d, torch.ones(4), scale=0.2)
    assert torch.all(g[1:] >= g[:-1])
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_completeness_scales_the_gate_inside_eq6():
    d = torch.tensor([0.8, 0.8])
    c = torch.tensor([1.0, 0.2])
    g = gate_from_delta_hat(d, c, scale=0.2)
    assert float(g[1]) == pytest.approx(0.2 * float(g[0]))


def test_calibration_hits_the_target_quantile():
    """By construction: sigma(P10(Delta_hat_clean)/s) == target_q."""
    ref = torch.linspace(0.05, 0.95, steps=1000)
    s = calibrate_gate_scale(ref, target_pct=0.10, target_q=0.8)
    p10 = float(torch.quantile(ref, 0.10).item())
    assert 1.0 / (1.0 + math.exp(-p10 / s)) == pytest.approx(0.8, abs=1e-6)
    # ...and the resulting gate for that sample is 2*0.8 - 1 = 0.6.
    g = gate_from_delta_hat(torch.tensor([p10]), torch.ones(1), scale=s)
    assert float(g[0]) == pytest.approx(0.6, abs=1e-5)


def test_calibration_warns_and_falls_back_on_contaminated_reference():
    ref = torch.full((200,), -1.0)
    s = calibrate_gate_scale(ref, target_pct=0.10, target_q=0.8)
    assert s == 1.0  # documented diagnostic-only fallback


def test_resolve_scale_prefers_explicit_over_in_pool():
    cfg = _cfg(scale=0.33)
    assert resolve_scale(cfg, torch.tensor([0.5])) == pytest.approx(0.33)
    cfg_none = _cfg(scale=None)
    got = resolve_scale(cfg_none, torch.tensor([0.2, 0.4, 0.6]))
    assert got > 0


# ---------------------------------------------------------------------------
# compute_gate: assembly, undefined policy, K > 1
# ---------------------------------------------------------------------------

def _three_samples():
    """clean / mismatched / locally-wrong, 12 tokens each."""
    tok_true = torch.tensor([
        [0.5] * 12,
        [3.0] * 12,
        [0.5] * 8 + [3.0] * 4,
    ])
    tok_cf = torch.full((3, 12), 3.0)
    n = torch.tensor([12, 12, 12])
    return tok_true, n, tok_cf, n


def test_compute_gate_separates_clean_from_both_corruption_types():
    tok_true, n, tok_cf, n_cf = _three_samples()
    res = compute_gate(tok_true, n, [tok_cf], [n_cf], torch.ones(3), cfg=_cfg())
    g = res["gate"]
    assert float(g[0]) > 0.5      # clean passes
    assert float(g[1]) == 0.0     # mismatch vetoed
    assert float(g[2]) == 0.0     # localized wrong answer vetoed


def test_undefined_policy_pass_does_not_veto_on_short_evidence():
    """A sample whose common prefix is too short has NO evidence; vetoing it
    would punish a tokenisation artifact."""
    tok_true = torch.tensor([[0.5, 0.5, 9.0, 9.0]])
    tok_cf = torch.tensor([[3.0, 3.0, 0.0, 0.0]])
    n_true = torch.tensor([4])
    n_cf = torch.tensor([2])      # common prefix = 2 < min_common_tokens = 4
    c = torch.tensor([1.0])
    res = compute_gate(
        tok_true, n_true, [tok_cf], [n_cf], c, cfg=_cfg(undefined_policy="pass"),
    )
    assert bool(res["undefined"][0])
    assert float(res["gate"][0]) == pytest.approx(1.0)
    res_veto = compute_gate(
        tok_true, n_true, [tok_cf], [n_cf], c, cfg=_cfg(undefined_policy="veto"),
    )
    assert float(res_veto["gate"][0]) == 0.0


def test_k_gt_1_averages_gates_and_discounts_dispersion():
    """Gate-then-average (not average-then-gate): evidence that straddles
    zero must not collapse to an exact veto by Jensen."""
    tok_true = torch.tensor([[0.5] * 8])
    n = torch.tensor([8])
    cf_agree = torch.tensor([[3.0] * 8])
    cf_disagree = torch.tensor([[0.5] * 8])   # no gain under this pairing
    cfg = _cfg(dispersion_discount=False)
    res = compute_gate(
        tok_true, n, [cf_agree, cf_disagree], [n, n], torch.ones(1), cfg=cfg,
    )
    per_cf = res["gate_per_cf"]
    assert float(per_cf[1, 0]) == 0.0        # the disagreeing pairing vetoes
    assert float(per_cf[0, 0]) > 0.0
    # Mean of gates is strictly positive; gate-of-mean would have been 0.
    assert float(res["gate"][0]) == pytest.approx(float(per_cf[:, 0].mean()))
    assert float(res["gate"][0]) > 0.0

    discounted = compute_gate(
        tok_true, n, [cf_agree, cf_disagree], [n, n], torch.ones(1),
        cfg=_cfg(dispersion_discount=True),
    )
    assert float(discounted["gate"][0]) < float(res["gate"][0])


def test_compute_gate_requires_a_calibrated_scale():
    tok_true, n, tok_cf, n_cf = _three_samples()
    with pytest.raises(ValueError, match="scale is None"):
        compute_gate(
            tok_true, n, [tok_cf], [n_cf], torch.ones(3), cfg=_cfg(scale=None),
        )


def test_compute_gate_rejects_mismatched_completeness_length():
    tok_true, n, tok_cf, n_cf = _three_samples()
    with pytest.raises(ValueError, match="completeness length"):
        compute_gate(tok_true, n, [tok_cf], [n_cf], torch.ones(2), cfg=_cfg())


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"span_tokens": 0},
    {"tau": -1.0},
    {"tau_mode": "bogus"},
    {"tail_mode": "bogus"},
    {"c_trunc": 0.0},
    {"c_trunc": 1.5},
    {"undefined_policy": "maybe"},
    {"scale": 0.0},
    {"min_common_tokens": 0},
])
def test_gate_config_rejects_invalid_values(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)
