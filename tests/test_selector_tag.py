"""Tests for the TAG score path in ``collect_episode`` (paper Eq. 1).

    s_i^(t) = G_i · R_i^(t) · (1 + lam · widetilde-align_i^(t))

The contract under test: TAG must leave the trajectory-anchored selector
untouched and add exactly one multiplicative factor, so that G == 1
reproduces the legacy ranking bit-for-bit, and G == 0 is an exact zero no dynamic
evidence can overturn.
"""
from __future__ import annotations

import pytest
import torch
import transformers

from tag.core.scorer import (
    gated_selection_key,
    legacy_score,
    tag_score,
    transform_gate,
)
from tag.core.selector import collect_episode


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=12, T=16, vocab=96, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab, (n, T), generator=g)
        self.attention_mask = torch.ones(n, T, dtype=torch.long)
        self.labels = self.input_ids.clone()
        self.labels[:, : T // 2] = -100

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(1234)
    cfg = transformers.GPT2Config(
        vocab_size=96, n_positions=32, n_embd=32, n_layer=2, n_head=2,
    )
    model = transformers.GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _run(model, dataset, *, tag=None, mvf=None, ratio=0.5, lam=0.0):
    return collect_episode(
        model=model,
        dataset=dataset,
        selection_ratio=ratio,
        trajectory_anchor=None,
        lam=lam,
        use_anchor=False,
        batch_size=4,
        device="cpu",
        seed=0,
        epoch=1,
        mvf=mvf,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Equivalence with the legacy path
# ---------------------------------------------------------------------------

def test_unit_gate_reproduces_legacy_selection_exactly(tiny_model):
    """G ≡ 1 must be a no-op: same score, same top-B, same order. This is
    what makes the gate a clean ablation of the legacy arm."""
    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    gated = _run(tiny_model, ds, tag={"gate": torch.ones(len(ds))})
    assert gated["score_mode"] == "tag"
    assert torch.allclose(gated["score"], legacy["score"])
    assert gated["selected_indices"] == legacy["selected_indices"]


def test_tag_score_matches_the_equation(tiny_model):
    ds = _TinyDataset()
    g = torch.rand(len(ds)) * 0.5 + 0.5   # all strictly positive
    ep = _run(tiny_model, ds, tag={"gate": g})
    assert torch.allclose(ep["score"], g * ep["rewards"])
    assert torch.allclose(ep["ungated_score"], ep["rewards"])


def test_tag_episode_exposes_the_gate(tiny_model):
    ds = _TinyDataset()
    g = torch.ones(len(ds))
    ep = _run(tiny_model, ds, tag={"gate": g})
    assert ep["gate"] is not None and ep["gate"].shape == (len(ds),)


# ---------------------------------------------------------------------------
# Non-compensation: the attainable zero
# ---------------------------------------------------------------------------

def test_zero_weight_sample_is_never_selected_while_budget_allows(tiny_model):
    """The paper's central claim, end-to-end: zero the gate on whichever
    sample has the LARGEST reward and it must drop out of the selection
    entirely — no amount of difficulty evidence buys it back."""
    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    best = int(legacy["rewards"].argmax().item())
    assert best in legacy["selected_indices"]

    g = torch.ones(len(ds))
    g[best] = 0.0
    gated = _run(tiny_model, ds, tag={"gate": g}, ratio=0.5)
    assert best not in gated["selected_indices"]
    assert float(gated["score"][best]) == 0.0


def test_zero_weight_survives_an_arbitrarily_large_reward(tiny_model):
    ds = _TinyDataset()
    g = torch.ones(len(ds))
    g[0] = 0.0
    ep = _run(tiny_model, ds, tag={"gate": g})
    # Even scaled by 1e12 the zeroed score stays exactly zero.
    assert float(ep["score"][0]) == 0.0
    assert 0 not in ep["selected_indices"]


# ---------------------------------------------------------------------------
# Budget shortfall: the documented fallback
# ---------------------------------------------------------------------------

def test_budget_shortfall_fills_with_ungated_ranking_not_file_order(tiny_model):
    """When fewer samples pass the gate than the budget needs, the leftover
    slots must go to the best ZERO-WEIGHT samples by the ungated score. Without
    the composite key, torch.topk would break the exact-zero tie by index
    and promote pool file order into the selection."""
    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    n = len(ds)
    k = n // 2                       # ratio 0.5

    admissible = [3]                 # only ONE sample passes the gate
    g = torch.zeros(n)
    g[admissible[0]] = 1.0
    ep = _run(tiny_model, ds, tag={"gate": g}, ratio=0.5)
    sel = ep["selected_indices"]

    assert len(sel) == k
    assert sel[0] == admissible[0]   # the only admissible sample ranks first
    # The remaining slots are the top zero-weight samples by ungated score.
    ungated = legacy["score"].clone()
    ungated[admissible[0]] = float("-inf")
    expected_fill = ungated.topk(k - 1).indices.tolist()
    assert sel[1:] == expected_fill


def test_all_zero_weight_pool_still_ranks_by_ungated_score(tiny_model):
    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    ep = _run(tiny_model, ds, tag={"gate": torch.zeros(len(ds))}, ratio=0.5)
    assert ep["selected_indices"] == legacy["score"].topk(len(ds) // 2).indices.tolist()


def test_gated_selection_key_orders_positive_weight_above_zero():
    score = torch.tensor([0.0, 5.0, 0.0, 1.0])
    fallback = torch.tensor([9.0, 5.0, 8.0, 1.0])
    gate = torch.tensor([0.0, 1.0, 0.0, 1.0])
    key, n_adm = gated_selection_key(score, fallback, gate)
    assert n_adm == 2
    assert float(key[1]) > float(key[3]) > float(key[0]) > float(key[2])


# ---------------------------------------------------------------------------
# Dedup, validation, mode exclusivity
# ---------------------------------------------------------------------------

def test_dedup_constraint_applies_in_tag_mode(tiny_model):
    """The legacy path never deduplicated; TAG must thread cluster_ids."""
    ds = _TinyDataset()
    clusters = [0] * len(ds)          # every sample is the same duplicate group
    ep = _run(
        tiny_model, ds,
        tag={"gate": torch.ones(len(ds)), "cluster_ids": clusters},
        ratio=0.5,
    )
    sel = ep["selected_indices"]
    # One pick from the cluster, then the constraint is exhausted and the
    # remaining slots are backfilled — the first pick is still the best.
    legacy = _run(tiny_model, ds)
    assert sel[0] == legacy["selected_indices"][0]


def test_mvf_and_tag_are_mutually_exclusive(tiny_model):
    ds = _TinyDataset()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(
            tiny_model, ds,
            tag={"gate": torch.ones(len(ds))},
            mvf={"completeness": torch.ones(len(ds))},
        )


def test_missing_gate_raises(tiny_model):
    ds = _TinyDataset()
    with pytest.raises(ValueError, match="'gate' is required"):
        _run(tiny_model, ds, tag={})


def test_stale_gate_size_raises(tiny_model):
    ds = _TinyDataset()
    with pytest.raises(ValueError, match="gate length"):
        _run(tiny_model, ds, tag={"gate": torch.ones(len(ds) + 3)})


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_out_of_range_gate_raises(tiny_model, bad):
    ds = _TinyDataset()
    g = torch.ones(len(ds))
    g[0] = bad
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _run(tiny_model, ds, tag={"gate": g})


# ---------------------------------------------------------------------------
# Pure-function equivalence
# ---------------------------------------------------------------------------

def test_tag_score_equals_gate_times_legacy_score():
    R = torch.tensor([1.0, 2.0, 3.0])
    a = torch.tensor([0.0, 0.5, 1.0])
    g = torch.tensor([1.0, 0.5, 0.0])
    assert torch.allclose(tag_score(g, R, a, 1.0), g * legacy_score(R, a, 1.0))


def test_tag_score_without_anchor_is_gate_times_reward():
    R = torch.tensor([1.0, 2.0])
    g = torch.tensor([0.5, 1.0])
    assert torch.allclose(tag_score(g, R, None, 1.0), g * R)


def test_zero_preserving_weak_gate_lifts_only_positive_weights():
    raw = torch.tensor([0.0, 0.04, 0.25, 0.81, 1.0])
    weak = transform_gate(raw, power=0.5)
    assert torch.allclose(weak, raw.sqrt())
    assert weak[0].item() == 0.0 and weak[-1].item() == 1.0
    assert torch.all(weak[1:-1] > raw[1:-1])
    assert torch.equal(torch.argsort(weak), torch.argsort(raw))


def test_soft_gate_strength_endpoints_and_legacy_equivalence(tiny_model):
    raw = torch.tensor([0.0, 0.2, 0.7, 1.0])
    assert torch.equal(transform_gate(raw, strength=1.0), raw)
    assert torch.equal(transform_gate(raw, strength=0.0), torch.ones_like(raw))
    assert torch.allclose(
        transform_gate(raw, strength=0.5),
        0.5 + 0.5 * raw,
    )

    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    unit = transform_gate(torch.rand(len(ds)), strength=0.0)
    soft_off = _run(tiny_model, ds, tag={"gate": unit})
    assert torch.equal(soft_off["score"], legacy["score"])
    assert soft_off["selected_indices"] == legacy["selected_indices"]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"power": 0.0}, "power"),
        ({"power": float("nan")}, "power"),
        ({"strength": -0.1}, "strength"),
        ({"strength": 1.1}, "strength"),
    ],
)
def test_gate_transform_rejects_invalid_knobs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        transform_gate(torch.tensor([0.0, 1.0]), **kwargs)


# ---------------------------------------------------------------------------
# Regressions from the adversarial review (2026-08-13)
# ---------------------------------------------------------------------------

def test_nan_score_is_rejected_not_promoted_to_the_top(tiny_model):
    """rank01 sorts NaN LAST (torch.argsort), so a NaN score would become
    key = 2 + 1 = 3.0 — the single highest key — and be selected FIRST. The
    guard in select_top_b cannot catch it because in TAG mode it receives the
    KEY, which is never NaN. Validation therefore has to happen on the score."""
    n = 5
    score = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    fallback = torch.ones(n)
    gate = torch.ones(n)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        gated_selection_key(score, fallback, gate)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        gated_selection_key(torch.ones(n), torch.full((n,), float("inf")), gate)


def test_backfill_takes_the_least_unreliable_rejects_not_the_highest_loss(tiny_model):
    """When a backfill is forced, ordering the zero-weight block by the ungated
    reward would pull in the HIGHEST-loss rejects — and the corruptions the
    gate exists to reject are exactly the high-loss ones. Delta_hat orders
    them by how close each came to passing instead."""
    ds = _TinyDataset()
    n = len(ds)
    gate = torch.zeros(n)
    gate[0] = 1.0                      # one admissible sample, budget is 6
    # Sample 7 is the closest to passing; sample 3 is the furthest.
    delta_hat = torch.full((n,), -1.0)
    delta_hat[7] = -0.01
    delta_hat[3] = -5.0
    ep = _run(tiny_model, ds, tag={"gate": gate, "delta_hat": delta_hat}, ratio=0.5)
    sel = ep["selected_indices"]
    assert sel[0] == 0                 # the admissible sample still ranks first
    assert sel[1] == 7                 # then the least unreliable reject
    # ...and the WORST reject is left out of the budget entirely.
    assert 3 not in sel


def test_episode_reports_realised_zero_weight_accounting(tiny_model):
    """n_admissible is a pool-wide prediction; what supports the paper's claim
    is how many SELECTED samples carry G == 0."""
    ds = _TinyDataset()
    n = len(ds)
    ep_ok = _run(tiny_model, ds, tag={"gate": torch.ones(n)}, ratio=0.5)
    assert ep_ok["n_admissible"] == n
    assert ep_ok["n_zero_weight_selected"] == 0
    assert ep_ok["selection_budget"] == n // 2

    gate = torch.zeros(n)
    gate[0] = 1.0
    ep_short = _run(tiny_model, ds, tag={"gate": gate}, ratio=0.5)
    assert ep_short["n_admissible"] == 1
    assert ep_short["n_zero_weight_selected"] == n // 2 - 1


def test_dedup_exhaustion_is_counted_even_when_the_budget_would_fit(tiny_model):
    """n_admissible >= B does NOT imply non-compensation held: constrained_topk takes
    at most one sample per near-duplicate cluster, so the admissible set can
    be exhausted by the dedup constraint before B is reached."""
    ds = _TinyDataset()
    n = len(ds)
    gate = torch.zeros(n)
    gate[:8] = 1.0                     # 8 admissible, budget 6 -> "fits"
    clusters = [0] * 8 + [-1] * (n - 8)   # ...but all 8 are ONE cluster
    ep = _run(
        tiny_model, ds,
        tag={"gate": gate, "cluster_ids": clusters}, ratio=0.5,
    )
    assert ep["n_admissible"] == 8 >= ep["selection_budget"]
    assert ep["n_zero_weight_selected"] > 0, (
        "dedup exhausted the admissible clusters, so zero-weight samples entered "
        "the selection even though n_admissible >= B"
    )
