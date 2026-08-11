"""Integration smoke test: collect_episode's MVF path end-to-end on CPU
with a tiny randomly-initialised GPT-2. Covers the legacy-path regression
(mvf=None unchanged), Q derivation from a real counterfactual forward,
loss-history-driven difficulty, and the dedup-constrained top-K."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tads.core.reliability import compute_pool_loss  # noqa: E402
from tads.core.selector import collect_episode  # noqa: E402


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=12, T=16, vocab=96, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab, (n, T), generator=g)
        self.attention_mask = torch.ones(n, T, dtype=torch.long)
        # Prompt (first half) masked out; response = second half.
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
    # Weight init draws from the GLOBAL torch RNG — pin it so results do
    # not depend on which tests ran earlier in the session.
    torch.manual_seed(1234)
    cfg = transformers.GPT2Config(
        vocab_size=96, n_positions=32, n_embd=32, n_layer=2, n_head=2,
    )
    model = transformers.GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _run(model, dataset, mvf=None):
    return collect_episode(
        model=model,
        dataset=dataset,
        selection_ratio=0.5,
        trajectory_anchor=None,
        lam=0.0,
        use_anchor=False,
        batch_size=4,
        device="cpu",
        seed=0,
        epoch=1,
        mvf=mvf,
    )


def test_legacy_path_unchanged_and_vectors_exposed(tiny_model):
    ds = _TinyDataset()
    ep = _run(tiny_model, ds)
    assert ep["score_mode"] == "tads"
    assert ep["r_loss"].shape == (len(ds),)
    assert ep["r_entropy"].shape == (len(ds),)
    # Legacy score at lam=0 is exactly the composite reward R.
    assert torch.allclose(ep["score"], ep["rewards"])
    assert ep["selected_indices"] == ep["rewards"].topk(6).indices.tolist()


def test_mvf_path_with_real_counterfactual_forward(tiny_model):
    ds = _TinyDataset(seed=0)
    cf_ds = _TinyDataset(seed=1)  # different "instructions", same shape
    loss_cf = compute_pool_loss(tiny_model, cf_ds, batch_size=4, device="cpu")
    assert loss_cf.shape == (len(ds),)

    completeness = torch.ones(len(ds))
    ep = _run(
        tiny_model, ds,
        mvf={
            "reliability": None,
            "loss_cf": loss_cf,
            "completeness": completeness,
            "loss_prev": None,
            "cluster_ids": None,
            "eta": 0.5, "gamma": 1.0, "eps": 0.01,
        },
    )
    assert ep["score_mode"] == "mvf"
    q = ep["reliability"]
    assert q is not None and q.shape == (len(ds),)
    assert float(q.min()) >= 0.0 and float(q.max()) <= 1.0
    assert ep["difficulty"] is not None
    assert len(ep["selected_indices"]) == 6
    # Determinism: same inputs → same selection.
    ep2 = _run(
        tiny_model, ds,
        mvf={
            "reliability": None,
            "loss_cf": loss_cf,
            "completeness": completeness,
            "loss_prev": None,
            "cluster_ids": None,
            "eta": 0.5, "gamma": 1.0, "eps": 0.01,
        },
    )
    assert ep2["selected_indices"] == ep["selected_indices"]


def test_mvf_loss_history_changes_difficulty(tiny_model):
    ds = _TinyDataset()
    n = len(ds)
    base = {
        "reliability": torch.linspace(0.1, 1.0, n),
        "completeness": torch.ones(n),
        "cluster_ids": None,
        "eta": 0.0, "gamma": 1.0, "eps": 0.01,
    }
    ep_no_hist = _run(tiny_model, ds, mvf={**base, "loss_prev": None})
    # Fabricate a history where every sample REGRESSED except sample 0,
    # which improved a lot → with eta=0 progress dominates D.
    loss_now = ep_no_hist["r_loss"]
    loss_prev = loss_now - 1.0
    loss_prev[0] = loss_now[0] + 5.0
    ep_hist = _run(tiny_model, ds, mvf={**base, "loss_prev": loss_prev})
    assert ep_hist["difficulty"][0] > 0.0
    assert not torch.allclose(ep_hist["difficulty"], ep_no_hist["difficulty"])


def test_mvf_dedup_constraint_applies(tiny_model):
    ds = _TinyDataset()
    n = len(ds)
    # All samples share one duplicate cluster except the last two → the
    # constrained top-6 can pick at most 1 from the big cluster and must
    # fill the rest from the singletons + skipped ones.
    cluster_ids = [0] * (n - 2) + [-1, -1]
    ep = _run(
        tiny_model, ds,
        mvf={
            "reliability": torch.ones(n),
            "completeness": torch.ones(n),
            "loss_prev": None,
            "cluster_ids": cluster_ids,
            "eta": 0.5, "gamma": 1.0, "eps": 0.01,
        },
    )
    sel = ep["selected_indices"]
    # The two singletons must both be selected; constrained pick admits
    # exactly one cluster-0 member before the fill stage.
    assert (n - 2) in sel and (n - 1) in sel


def test_mvf_stale_cache_size_raises(tiny_model):
    ds = _TinyDataset()
    with pytest.raises(ValueError, match="stale cache|length"):
        _run(
            tiny_model, ds,
            mvf={
                "reliability": torch.ones(5),  # wrong length
                "completeness": torch.ones(len(ds)),
                "loss_prev": None,
                "cluster_ids": None,
                "eta": 0.5, "gamma": 1.0, "eps": 0.01,
            },
        )
