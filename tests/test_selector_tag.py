"""Tests for the TAG score path in ``collect_episode`` (paper Eq. 1).

    s_i^(t) = G_i · R_i^(t) · (1 + lam · widetilde-align_i^(t))

The contract under test: TAG must leave the trajectory-anchored selector
untouched and add exactly one multiplicative factor, so that G == 1
reproduces the legacy ranking bit-for-bit, and G == 0 is a veto no dynamic
evidence can overturn.
"""
from __future__ import annotations

import pytest
import torch
import transformers

from tads.core.scorer import gated_selection_key, tads_score, tag_score
from tads.core.selector import collect_episode


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
# Non-compensation: the veto
# ---------------------------------------------------------------------------

def test_vetoed_sample_is_never_selected_while_budget_allows(tiny_model):
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


def test_veto_survives_an_arbitrarily_large_reward(tiny_model):
    ds = _TinyDataset()
    g = torch.ones(len(ds))
    g[0] = 0.0
    ep = _run(tiny_model, ds, tag={"gate": g})
    # Even scaled by 1e12 the vetoed score stays exactly zero.
    assert float(ep["score"][0]) == 0.0
    assert 0 not in ep["selected_indices"]


# ---------------------------------------------------------------------------
# Budget shortfall: the documented fallback
# ---------------------------------------------------------------------------

def test_budget_shortfall_fills_with_ungated_ranking_not_file_order(tiny_model):
    """When fewer samples pass the gate than the budget needs, the leftover
    slots must go to the best VETOED samples by the ungated score. Without
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
    # The remaining slots are the top vetoed samples by ungated score.
    ungated = legacy["score"].clone()
    ungated[admissible[0]] = float("-inf")
    expected_fill = ungated.topk(k - 1).indices.tolist()
    assert sel[1:] == expected_fill


def test_all_vetoed_pool_still_ranks_by_ungated_score(tiny_model):
    ds = _TinyDataset()
    legacy = _run(tiny_model, ds)
    ep = _run(tiny_model, ds, tag={"gate": torch.zeros(len(ds))}, ratio=0.5)
    assert ep["selected_indices"] == legacy["score"].topk(len(ds) // 2).indices.tolist()


def test_gated_selection_key_orders_admissible_above_vetoed():
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
    assert torch.allclose(tag_score(g, R, a, 1.0), g * tads_score(R, a, 1.0))


def test_tag_score_without_anchor_is_gate_times_reward():
    R = torch.tensor([1.0, 2.0])
    g = torch.tensor([0.5, 1.0])
    assert torch.allclose(tag_score(g, R, None, 1.0), g * R)
