"""Per-sample (L_i, H_i) inputs to the TAG composite reward (paper Eq. 3).

This module provides ONLY the per-sample forward-pass derivations:
    L_i = mean cross-entropy loss over response tokens of sample i
    H_i = mean predictive entropy over response tokens of sample i

The pool-level composite reward R_i = w·L_i + (1-w)·H_i (paper Eq. 3) and
the variance-ratio weight w (paper Eq. 4) are computed in
``tag.core.scorer.pool_reward`` after the per-sample arrays are
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
metrics.json. None of these carry RL semantics in TAG — every step is a
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

    Memory: processes the batch one sample at a time, and within a sample
    only the RESPONSE positions, so peak GPU memory holds three fp32
    ``(k, V)`` tensors for that sample's ``k`` response tokens rather than
    three full ``(B, T, V)`` ones. For Qwen2.5 (V=151k) at
    episode_batch_size=16 that is a few hundred MB instead of ~15 GB.

    Both quantities are read off ONE ``log_softmax``. The earlier version
    called ``cross_entropy`` and ``log_softmax`` separately — two softmaxes
    over the same ``(T-1, V)`` block — and computed both at every position,
    including the ~40 % of the sequence that is prompt and the padding
    beyond it, only to multiply the result by a 0/1 mask afterwards. On the
    7B pool pass that was most of the wall clock. Restricting to the
    positions the mask keeps is exact: the discarded terms were multiplied
    by zero.

    Note on ``r_weight`` scope: when this function is called with a single
    mini-batch, the variance is computed *within* that batch and is degenerate
    at batch_size=1. For the dataset-level weight used by ``collect_episode``,
    the selector recomputes ``r_weight`` once at the end across all samples
    (see ``tag.core.selector.collect_episode``).
    """
    B, T, V = logits.shape
    device = logits.device

    shift_logits = logits[:, :-1, :]                          # (B, T-1, V), bf16/fp16
    shift_labels = labels[:, 1:]                              # (B, T-1)
    resp_mask = shift_labels != -100                          # (B, T-1) bool
    n_resp = resp_mask.sum(dim=-1).clamp(min=1).float()       # (B,)

    r_loss = torch.zeros(B, device=device, dtype=torch.float32)
    r_entropy = torch.zeros(B, device=device, dtype=torch.float32)

    for i in range(B):
        rm = resp_mask[i]
        if not bool(rm.any()):
            continue                                          # stays 0.0
        # Cast to fp32 BEFORE the softmax — an earlier version's comment
        # claimed fp32 but no cast happened, so loss carried bf16 rounding
        # (~0.4% rel.) while the counterfactual pass
        # (reliability.compute_pool_loss) is fp32. ΔL = loss_cf − loss_orig
        # is zero-ANCHORED at the rezero kink, exactly where that dtype
        # asymmetry moves samples across Q = 0.
        sl = shift_logits[i][rm].float()                      # (k, V) fp32
        tgt = shift_labels[i][rm]                             # (k,)
        lp = F.log_softmax(sl, dim=-1)                        # (k, V) fp32
        del sl
        # CE = -log p(target): the same value cross_entropy returns, read
        # off the log-probabilities already computed for the entropy.
        ce_i = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)      # (k,) fp32
        # H = -Σ p log p = -Σ exp(lp) * lp
        ent_i = -(lp.exp() * lp).sum(dim=-1)                   # (k,) fp32
        r_loss[i] = ce_i.sum() / n_resp[i]
        r_entropy[i] = ent_i.sum() / n_resp[i]
        del lp, ce_i, ent_i

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
