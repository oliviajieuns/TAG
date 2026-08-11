"""Unit tests for tads.core.reliability — completeness gate and cache
round-trip. The counterfactual forward pass itself is exercised end-to-end
on GPU runs; here we test the pure logic around it."""
from __future__ import annotations

import torch

from tads.core.reliability import (
    completeness_from_dataset,
    load_reliability_cache,
    reliability_from_losses,
    save_reliability_cache,
)

EOS = 2


class _FakeDataset:
    """Minimal stand-in for the tokenised HF dataset (labels only)."""

    def __init__(self, label_rows):
        self.rows = [torch.tensor(r, dtype=torch.long) for r in label_rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"labels": self.rows[i]}


def test_completeness_eos_gate():
    ds = _FakeDataset([
        [-100, -100, 5, 6, EOS],        # complete: ends with EOS
        [-100, 5, 6, 7, 8],             # truncated: no EOS
        [-100, -100, -100, -100, -100],  # no response tokens at all
        [-100, 5, EOS, -100, -100],     # EOS then padding-masked tail
    ])
    c = completeness_from_dataset(ds, eos_token_id=EOS, c_trunc=0.2)
    assert torch.allclose(c, torch.tensor([1.0, 0.2, 0.2, 1.0]))


def test_completeness_validates_c_trunc():
    ds = _FakeDataset([[5, EOS]])
    for bad in (0.0, -1.0, 1.5):
        try:
            completeness_from_dataset(ds, EOS, c_trunc=bad)
            assert False
        except ValueError:
            pass


def test_reliability_shape_mismatch_raises():
    try:
        reliability_from_losses(torch.rand(3), torch.rand(4))
        assert False
    except ValueError:
        pass


def test_cache_roundtrip(tmp_path):
    q = torch.rand(10)
    c = torch.ones(10)
    lo = torch.rand(10)
    lc = lo + torch.rand(10)
    save_reliability_cache(
        tmp_path, q=q, completeness=c, loss_orig=lo, loss_cf=lc, epoch=1,
    )
    cache = load_reliability_cache(tmp_path)
    assert cache is not None
    assert torch.allclose(cache["q"], q)
    assert torch.allclose(cache["completeness"], c)
    assert cache["epoch"] == 1


def test_cache_missing_returns_none(tmp_path):
    assert load_reliability_cache(tmp_path) is None
