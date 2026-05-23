"""Per-sample (L_i, H_i) inputs to the TADS composite reward (paper Eq. 3).

This module provides ONLY the per-sample forward-pass derivations:
    L_i = mean cross-entropy loss over response tokens of sample i
    H_i = mean predictive entropy over response tokens of sample i

The pool-level composite reward R_i = w·L_i + (1-w)·H_i (paper Eq. 3) and
the variance-ratio weight w (paper Eq. 4) are computed in
``tads.core.scorer.pool_reward`` after the per-sample arrays are
accumulated across the whole epoch.

Naming
------
r_loss / L_i     — optimization-impact signal (mean CE loss over response
                   tokens). Historically referred to as ``rdiff`` in the
                   Data Agent paper.
r_entropy / H_i  — predictive-uncertainty signal (mean predictive entropy).
                   Historically referred to as ``rconf`` in the Data Agent paper.

For metric compatibility with older Data Agent analysis scripts, callers
may log both names (e.g. ``rdiff_mean = r_loss_mean``) in their
metrics.json. None of these carry RL semantics in TADS — every step is a
closed-form deterministic operation over pool-level statistics.
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

    Memory: processes the batch one sample at a time so peak GPU memory holds
    only a single ``(T, V)`` fp32 softmax + log-prob pair (≈ ``2·T·V·4`` bytes)
    instead of three full ``(B, T, V)`` fp32 tensors at once. For Qwen2.5
    (V=151k) at episode_batch_size=16 this drops the entropy peak from ~15 GB
    to ~1 GB; for Llama2 (V=32k) from ~3 GB to ~256 MB. The slowdown is
    negligible because the inner ops are still vectorised over (T, V).

    Note on ``r_weight`` scope: when this function is called with a single
    mini-batch, the variance is computed *within* that batch and is degenerate
    at batch_size=1. For the dataset-level weight used by ``collect_episode``,
    the selector recomputes ``r_weight`` once at the end across all samples
    (see ``tads.core.selector.collect_episode``).
    """
    B, T, V = logits.shape
    device = logits.device

    shift_logits = logits[:, :-1, :]                          # (B, T-1, V), bf16/fp16
    shift_labels = labels[:, 1:]                              # (B, T-1)
    resp_mask = (shift_labels != -100).float()                # (B, T-1)
    n_resp = resp_mask.sum(dim=-1).clamp(min=1)               # (B,)

    r_loss = torch.empty(B, device=device, dtype=torch.float32)
    r_entropy = torch.empty(B, device=device, dtype=torch.float32)

    for i in range(B):
        sl = shift_logits[i]                                  # (T-1, V)
        ll = shift_labels[i].clamp(min=0)                     # (T-1,)
        rm = resp_mask[i]                                     # (T-1,)
        nr = n_resp[i]

        # CE over response tokens only.
        ce_i = F.cross_entropy(sl, ll, reduction="none")      # (T-1,) fp32
        r_loss[i] = (ce_i * rm).sum() / nr

        # Entropy via log_softmax: H = -Σ p log p = -Σ exp(lp) * lp.
        # log_softmax allocates a single (T-1, V) fp32 tensor; ent_tok is (T-1,).
        lp = F.log_softmax(sl.float(), dim=-1)                # (T-1, V) fp32
        ent_tok = -(lp.exp() * lp).sum(dim=-1)                # (T-1,) fp32
        r_entropy[i] = (ent_tok * rm).sum() / nr
        del sl, ce_i, lp, ent_tok

    if r_loss.numel() > 1:
        var_loss = r_loss.var()
    else:
        var_loss = torch.tensor(0.0, device=device)
    if r_entropy.numel() > 1:
        var_entropy = r_entropy.var()
    else:
        var_entropy = torch.tensor(0.0, device=device)

    # The r_weight returned here is the WITHIN-BATCH weight. Callers using
    # batch_size=1 (e.g. collect_episode with episode_batch_size=1) will see
    # r_weight=0 because both variances collapse — selector.collect_episode
    # recomputes a DATASET-LEVEL r_weight after the loop and discards this
    # one. Document that here so a direct caller of compute_rewards doesn't
    # accidentally use the per-batch weight as if it were paper Eq. 4.
    r_weight = var_loss / (var_loss + var_entropy + eps)
    return r_loss.detach(), r_entropy.detach(), r_weight.detach()


def composite_reward(
    r_loss: torch.Tensor,
    r_entropy: torch.Tensor,
    r_weight: torch.Tensor,
) -> torch.Tensor:
    """R = r_weight * r_loss + (1 - r_weight) * r_entropy  (paper Eq. 3)."""
    return r_weight * r_loss + (1.0 - r_weight) * r_entropy
