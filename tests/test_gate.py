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
    # null_correction is OFF by default here so that each test below pins the
    # property it is actually about (Eqs. 3-6 on the raw statistic) rather
    # than the interaction of that property with Eq. 5' centring. The
    # correction has its own section at the bottom of this file, and the
    # PRODUCTION default is on — see test_gate_config_defaults_to_corrected.
    base = dict(span_tokens=4, tau=0.5, scale=0.2, min_span_tokens=2,
                min_common_tokens=4, null_correction=False)
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


# ---------------------------------------------------------------------------
# Regressions from the adversarial review (2026-08-13)
# ---------------------------------------------------------------------------

def test_undefined_neutral_does_not_outrank_evidenced_samples():
    """G = c_i is the SUPREMUM no evidenced sample can reach (2*sigma-1 < 1),
    so handing it to zero-evidence samples would rank them above every sample
    the gate actually examined — short responses promoted for being short."""
    tok_true = torch.tensor([[0.01] * 12])   # about as clean as it gets
    tok_cf = torch.tensor([[9.0] * 12])
    n = torch.tensor([12])
    best = compute_gate(
        tok_true, n, [tok_cf], [n], torch.ones(1), cfg=_cfg(),
    )["gate"]

    short_true = torch.tensor([[0.5, 0.5]])
    short_cf = torch.tensor([[3.0, 3.0]])
    ns = torch.tensor([2])
    neutral = compute_gate(
        short_true, ns, [short_cf], [ns], torch.ones(1),
        cfg=_cfg(undefined_policy="neutral"),
    )
    assert bool(neutral["undefined"][0])
    assert float(neutral["gate"][0]) < float(best[0]), (
        "an unjudgeable sample must not outrank the best evidenced sample"
    )
    assert float(neutral["gate"][0]) == pytest.approx(0.6)

    # The 'pass' ablation is exactly the pathology, kept for the ablation row.
    passed = compute_gate(
        short_true, ns, [short_cf], [ns], torch.ones(1),
        cfg=_cfg(undefined_policy="pass"),
    )["gate"]
    assert float(passed[0]) == pytest.approx(1.0)
    assert float(passed[0]) > float(best[0])


def test_recompute_from_cache_refuses_forward_bound_changes():
    """include_eos and c_trunc are baked into the cached token losses and
    completeness vector. Re-deriving under a new value would apply the OLD
    one while stamping the NEW identity, so every later run would then get a
    'cache hit' on a silently wrong gate."""
    from tads.core.gate import recompute_gate_from_cache

    tok_true, n, tok_cf, n_cf = _three_samples()
    cfg = _cfg()
    res = compute_gate(tok_true, n, [tok_cf], [n_cf], torch.ones(3), cfg=cfg)
    cache = {
        "token_true": tok_true, "n_true": n,
        "token_cf": [tok_cf], "n_cf": [n_cf],
        "completeness": torch.ones(3),
        "gate": res["gate"], "config": cfg.identity(),
    }
    # A pure span-parameter change IS re-derivable without a forward.
    assert recompute_gate_from_cache(cache, _cfg(tau=0.9)) is not None
    # These two are not.
    assert recompute_gate_from_cache(cache, _cfg(include_eos=True)) is None
    assert recompute_gate_from_cache(cache, _cfg(c_trunc=0.5)) is None


def test_cache_identity_catches_a_wrong_pool_or_backbone():
    """A SHARED gate cache is reachable by runs it was never computed for.
    Shape alone would not catch a cache from a different backbone, and G is
    only meaningful for the (pool, base checkpoint) it was measured on."""
    from tads.core.gate import cache_identity, check_cache_identity

    want = cache_identity(model_path="/m/qwen-7b", pool_files="/p/pool.json", n_pool=100)
    assert check_cache_identity({"identity": dict(want)}, want) is None
    assert "pool_files" in check_cache_identity(
        {"identity": dict(want, pool_files="/p/other.json")}, want
    )
    assert "model_path" in check_cache_identity(
        {"identity": dict(want, model_path="/m/qwen-05b")}, want
    )
    # Pre-identity caches were per-run and never cross-used: accept + warn.
    assert check_cache_identity({}, want) is None


def test_shared_cache_round_trips_through_an_explicit_path(tmp_path):
    from tads.core.gate import (
        cache_identity, load_gate_cache, save_gate_cache,
    )

    tok_true, n, tok_cf, n_cf = _three_samples()
    cfg = _cfg()
    res = compute_gate(tok_true, n, [tok_cf], [n_cf], torch.ones(3), cfg=cfg)
    ident = cache_identity(model_path="/m", pool_files="/p", n_pool=3)
    shared = tmp_path / "nested" / "shared_gate.pt"
    save_gate_cache(None, result=res, cfg=cfg, epoch=1, identity=ident, path=shared)
    assert shared.exists()
    # The per-run location must NOT have been written.
    assert not (tmp_path / "tag_gate_cache.pt").exists()
    back = load_gate_cache(None, path=shared)
    assert back is not None
    assert back["identity"] == ident
    assert torch.allclose(back["gate"], res["gate"])


# ---------------------------------------------------------------------------
# Eq. 5' — length-conditional null correction
# ---------------------------------------------------------------------------

def _length_varying_pool(n=6000, seed=0):
    """A CLEAN pool whose only structure is that responses vary in length.

    Every sample has the same per-token instruction dependency (the
    counterfactual costs 12.5% more per token) and the same per-token noise.
    Any length dependence in the veto rate is therefore an artifact of the
    statistic, not of the data — which is exactly what Eq. 5' is about.
    """
    g = torch.Generator().manual_seed(seed)
    lens = torch.clamp(
        torch.distributions.LogNormal(4.3, 0.9).sample((n,)).long(), 16, 400,
    )
    t_max = int(lens.max())
    mask = torch.arange(t_max).unsqueeze(0) < lens.unsqueeze(1)
    base = torch.rand(n, t_max, generator=g) * 1.5 + 1.0
    noise_a = 0.35 * torch.randn(n, t_max, generator=g)
    noise_b = 0.35 * torch.randn(n, t_max, generator=g)
    true = torch.where(mask, base + noise_a, torch.zeros(1)).clamp(min=0.01)
    cf = torch.where(mask, base * 1.125 + noise_b, torch.zeros(1)).clamp(min=0.01)
    return true, lens, cf


def test_raw_tail_min_veto_rate_drifts_with_response_length():
    """The pathology, pinned. Without this the correction has no motivation.

    Same per-token dependency at every length, yet the uncorrected Eq. 5
    vetoes the longest quintile several times more often than the shortest —
    because Delta^min is a minimum over M = ceil(n/W) spans and M grows.
    """
    from tads.core.gate import GateConfig, gate_components

    true, lens, cf = _length_varying_pool()
    cfg = GateConfig(span_tokens=16, null_correction=False, scale=1.0)
    raw = gate_components(true, lens, cf, lens, cfg=cfg)["delta_hat"]
    quint = torch.tensor_split(torch.argsort(lens.float()), 5)
    short = float((raw[quint[0]] <= 0).float().mean())
    long_ = float((raw[quint[-1]] <= 0).float().mean())
    assert long_ > 2.5 * short, (short, long_)


def test_null_correction_pins_the_clean_veto_rate_in_every_length_bin():
    from tads.core.gate import GateConfig, fit_calibration, gate_components

    true, lens, cf = _length_varying_pool()
    cfg = GateConfig(span_tokens=16, null_correction=False, scale=1.0)
    comp = gate_components(true, lens, cf, lens, cfg=cfg)
    fit = fit_calibration(
        comp["delta_hat"], comp["n_spans"], span_tokens=16, target_veto=0.05,
    )
    centered = fit["delta_hat"]
    assert fit["veto_rate"] == pytest.approx(0.05, abs=0.01)
    quint = torch.tensor_split(torch.argsort(lens.float()), 5)
    rates = [float((centered[q] <= 0).float().mean()) for q in quint]
    # Uniform in length is the whole claim.
    assert max(rates) - min(rates) < 0.03, rates
    # ...and s is now derivable, where the uncorrected statistic fell back to
    # the diagnostic-only s = 1.0.
    assert fit["scale"] > 0


def test_null_correction_preserves_detection_of_localized_corruption():
    """Centering must not launder corruption: mu is fit on CLEAN data only.

    A sample with one span the instruction no longer explains still lands
    below the clean null at its own length, so it is still vetoed.
    """
    from tads.core.gate import GateConfig, fit_calibration, gate_components

    true, lens, cf = _length_varying_pool()
    cfg = GateConfig(span_tokens=16, null_correction=False, scale=1.0)
    comp = gate_components(true, lens, cf, lens, cfg=cfg)
    fit = fit_calibration(
        comp["delta_hat"], comp["n_spans"], span_tokens=16, target_veto=0.05,
    )
    null = fit["null"]

    dirty = true.clone()
    hit = torch.arange(1000)
    for i in hit.tolist():
        L = int(lens[i])
        a = L // 2
        dirty[i, a: min(a + 16, L)] = cf[i, a: min(a + 16, L)] * 1.6
    comp_d = gate_components(dirty, lens, cf, lens, cfg=cfg)
    veto_dirty = float(
        (null.apply(comp_d["delta_hat"], comp_d["n_spans"])[hit] <= 0).float().mean()
    )
    veto_clean = float((fit["delta_hat"][hit] <= 0).float().mean())
    assert veto_dirty > 0.9, veto_dirty
    assert veto_clean < 0.1, veto_clean


def test_null_curve_is_nonincreasing_in_span_count():
    """mu(M) is projected onto the monotone cone; a rise would be noise."""
    from tads.core.gate import GateConfig, fit_null_calibration, gate_components

    true, lens, cf = _length_varying_pool()
    cfg = GateConfig(span_tokens=16, null_correction=False, scale=1.0)
    comp = gate_components(true, lens, cf, lens, cfg=cfg)
    cal = fit_null_calibration(
        comp["delta_hat"], comp["n_spans"], target_veto=0.05, span_tokens=16,
    )
    assert len(cal.bin_edges) > 1
    assert all(a >= b - 1e-9 for a, b in zip(cal.mu, cal.mu[1:])), cal.mu


def test_null_curve_cannot_be_reused_at_a_different_span_width():
    """M is a span COUNT; mu(M) means something else at another W."""
    from tads.core.gate import GateConfig, NullCalibration

    cal = NullCalibration(
        bin_edges=(4, 100), mu=(0.1, -0.2), counts=(500, 500),
        target_veto=0.05, span_tokens=16, n_ref=1000,
    )
    GateConfig(span_tokens=16, target_veto=0.05, null=cal, scale=1.0)
    with pytest.raises(ValueError, match="span_tokens"):
        GateConfig(span_tokens=32, target_veto=0.05, null=cal, scale=1.0)
    with pytest.raises(ValueError, match="target_veto"):
        GateConfig(span_tokens=16, target_veto=0.10, null=cal, scale=1.0)


def test_gate_config_defaults_to_corrected_and_says_so_when_uncalibrated():
    """The production default is ON, and a missing curve is a loud error."""
    from tads.core.gate import GateConfig, gate_components

    cfg = GateConfig(span_tokens=4, scale=0.2)
    assert cfg.null_correction is True
    assert cfg.target_veto == 0.05
    tok_true, n, tok_cf, n_cf = _three_samples()
    with pytest.raises(ValueError, match="null_correction"):
        gate_components(tok_true, n, tok_cf, n_cf, cfg=cfg)


def test_scale_calibration_rejects_a_target_pct_below_the_veto_target():
    """Centering puts the target_veto quantile at exactly 0, so a target_pct
    at or below it derives s from a non-positive quantile — a config error
    with an exact fix, not something to paper over with the median."""
    from tads.core.gate import calibrate_gate_scale, fit_calibration

    ref = torch.linspace(-1.0, 1.0, 500)
    with pytest.raises(ValueError, match="target_pct"):
        fit_calibration(
            ref, torch.ones(500, dtype=torch.long), span_tokens=16,
            target_veto=0.10, target_pct=0.10,
        )
    with pytest.raises(ValueError, match="NULL-CORRECTED"):
        calibrate_gate_scale(
            torch.linspace(-1.0, 0.0, 500), target_pct=0.10, null_corrected=True,
        )


def test_null_calibration_survives_a_cache_round_trip(tmp_path):
    from tads.core.gate import (
        GateConfig, NullCalibration, cache_identity, load_gate_cache,
        save_gate_cache,
    )

    cal = NullCalibration(
        bin_edges=(4, 100), mu=(0.1, -0.2), counts=(500, 500),
        target_veto=0.05, span_tokens=4, n_ref=1000,
    )
    cfg = _cfg(null_correction=True, null=cal)
    tok_true, n, tok_cf, n_cf = _three_samples()
    res = compute_gate(tok_true, n, [tok_cf], [n_cf], torch.ones(3), cfg=cfg)
    path = tmp_path / "gate.pt"
    save_gate_cache(
        None, result=res, cfg=cfg, epoch=1, path=path,
        identity=cache_identity(model_path="/m", pool_files="/p", n_pool=3),
    )
    back = load_gate_cache(None, path=path)
    assert NullCalibration.from_dict(back["null"]) == cal
    # identity() carries a digest, not the raw curve — but it still changes
    # when the curve does, so a cache from another calibration cannot hit.
    other = NullCalibration(
        bin_edges=(4, 100), mu=(0.1, -0.3), counts=(500, 500),
        target_veto=0.05, span_tokens=4, n_ref=1000,
    )
    assert back["config"] == cfg.identity()
    assert back["config"] != _cfg(null_correction=True, null=other).identity()
