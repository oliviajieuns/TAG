"""Sanity tests for compute_rewards (paper Eq. 1, 3, 5)."""
from __future__ import annotations

import pytest
import torch

from tads.core.reward import composite_reward, compute_rewards


@pytest.mark.parametrize("B,T,V", [(4, 16, 32)])
def test_compute_rewards_shape(B, T, V):
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # Mask the first half as prompt.
    labels[:, : T // 2] = -100
    r_loss, r_entropy, r_weight = compute_rewards(logits, labels)
    assert r_loss.shape == (B,)
    assert r_entropy.shape == (B,)
    assert r_weight.dim() == 0
    assert r_loss.isfinite().all()
    assert r_entropy.isfinite().all()


def test_compute_rewards_b1_degenerate():
    """Batch-of-one: variance is zero so r_weight should be exactly 0 (no NaN)."""
    logits = torch.randn(1, 8, 16)
    labels = torch.randint(0, 16, (1, 8))
    labels[:, :4] = -100
    r_loss, r_entropy, r_weight = compute_rewards(logits, labels)
    assert r_loss.shape == (1,)
    assert float(r_weight.item()) == 0.0


def test_composite_reward_matches_eq6():
    r_loss = torch.tensor([1.0, 2.0])
    r_entropy = torch.tensor([0.5, 1.5])
    r_weight = torch.tensor(0.7)
    out = composite_reward(r_loss, r_entropy, r_weight)
    expected = 0.7 * r_loss + 0.3 * r_entropy
    assert torch.allclose(out, expected)
