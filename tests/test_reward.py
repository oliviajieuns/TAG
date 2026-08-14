"""Sanity tests for compute_rewards (paper §3.2 preliminaries → Eq. 3-4).

compute_rewards computes per-sample L_i (mean CE loss over response
tokens) and H_i (mean predictive entropy over response tokens) — the
inputs to the composite reward R_i (paper Eq. 3) with variance-ratio
weight w (paper Eq. 4).
"""
from __future__ import annotations

import pytest
import torch

from tag.core.reward import composite_reward, compute_rewards


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


def _reference_rewards(logits, labels):
    """The pre-optimisation formulation: cross_entropy and log_softmax
    computed independently at EVERY position, then multiplied by the
    response mask. Restricting to the masked positions and reading the CE
    off the same log_softmax must not change the answer."""
    import torch.nn.functional as F

    B = logits.size(0)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    resp_mask = (shift_labels != -100).float()
    n_resp = resp_mask.sum(dim=-1).clamp(min=1)
    r_loss = torch.zeros(B, dtype=torch.float32)
    r_entropy = torch.zeros(B, dtype=torch.float32)
    for i in range(B):
        sl = shift_logits[i].float()
        ce = F.cross_entropy(sl, shift_labels[i].clamp(min=0), reduction="none")
        r_loss[i] = (ce * resp_mask[i]).sum() / n_resp[i]
        lp = F.log_softmax(sl, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1)
        r_entropy[i] = (ent * resp_mask[i]).sum() / n_resp[i]
    return r_loss, r_entropy


def test_compute_rewards_matches_the_all_positions_formulation():
    torch.manual_seed(5)
    B, T, V = 4, 24, 97
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :9] = -100                       # prompt
    labels[2, 18:] = -100                      # ragged response lengths
    labels[3, 11:] = -100
    got_loss, got_ent, _ = compute_rewards(logits, labels)
    ref_loss, ref_ent = _reference_rewards(logits, labels)
    assert torch.allclose(got_loss, ref_loss, atol=1e-5, rtol=0)
    assert torch.allclose(got_ent, ref_ent, atol=1e-5, rtol=0)


def test_compute_rewards_row_with_no_response_tokens_is_zero_not_nan():
    """clamp(min=1) on the divisor used to hide this; with the response-only
    gather the row is skipped outright, so pin that it stays finite."""
    torch.manual_seed(6)
    logits = torch.randn(2, 12, 40)
    labels = torch.randint(0, 40, (2, 12))
    labels[0] = -100                            # no supervised positions at all
    labels[1, :4] = -100
    r_loss, r_entropy, _ = compute_rewards(logits, labels)
    assert torch.isfinite(r_loss).all() and torch.isfinite(r_entropy).all()
    assert float(r_loss[0]) == 0.0 and float(r_entropy[0]) == 0.0
    assert float(r_loss[1]) > 0.0
