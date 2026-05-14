"""Reward computation for instruction-tuning data selection.

Implements paper Eq. 1, 3, 5, 6:
    r_loss_i    = mean CE loss over response tokens of sample i      (Eq. 1)
    r_entropy_i = mean predictive entropy over response tokens       (Eq. 3)
    r_weight    = Var(r_loss) / (Var(r_loss) + Var(r_entropy) + eps) (Eq. 5)
    R_i         = r_weight * r_loss_i + (1 - r_weight) * r_entropy_i (Eq. 6)

Naming
------
r_loss     — optimization-impact signal (mean CE loss over response tokens).
             Originally referred to as ``rdiff`` in the Data Agent paper.
r_entropy  — predictive-uncertainty signal (mean predictive entropy).
             Originally referred to as ``rconf`` in the Data Agent paper.
r_weight   — variance-ratio weight that auto-balances the two signals.
             Originally referred to as ``r`` in the Data Agent paper.

For metric compatibility with older Data Agent analysis scripts, callers may
log both names (e.g. ``rdiff_mean = r_loss_mean``) in their metrics.json.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_rewards(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample (r_loss, r_entropy) plus the adaptive weight r_weight.

    Note on ``r_weight`` scope: when this function is called with a single
    mini-batch, the variance is computed *within* that batch and is degenerate
    at batch_size=1. For the dataset-level weight used by ``collect_episode``,
    the selector recomputes ``r_weight`` once at the end across all samples
    (see ``tads.core.selector.collect_episode``).
    """
    B, T, V = logits.shape
    device = logits.device

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    resp_mask = (shift_labels != -100).float()
    n_resp = resp_mask.sum(dim=-1).clamp(min=1)

    flat_logits = shift_logits.reshape(-1, V)
    flat_labels = shift_labels.clamp(min=0).reshape(-1)
    flat_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    flat_loss = flat_loss * resp_mask.reshape(-1)
    r_loss = flat_loss.reshape(B, -1).sum(dim=-1) / n_resp

    probs = F.softmax(shift_logits, dim=-1)
    log_probs = probs.clamp(min=eps).log()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    r_entropy = (token_entropy * resp_mask).sum(dim=-1) / n_resp

    if r_loss.numel() > 1:
        var_loss = r_loss.var()
    else:
        var_loss = torch.tensor(0.0, device=device)
    if r_entropy.numel() > 1:
        var_entropy = r_entropy.var()
    else:
        var_entropy = torch.tensor(0.0, device=device)

    r_weight = var_loss / (var_loss + var_entropy + eps)
    return r_loss.detach(), r_entropy.detach(), r_weight.detach()


def composite_reward(
    r_loss: torch.Tensor,
    r_entropy: torch.Tensor,
    r_weight: torch.Tensor,
) -> torch.Tensor:
    """R = r_weight * r_loss + (1 - r_weight) * r_entropy  (paper Eq. 6)."""
    return r_weight * r_loss + (1.0 - r_weight) * r_entropy
