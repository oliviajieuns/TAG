"""Reliability view: counterfactual instruction fidelity + completeness.

Implements the static reliability gate of the MVF score
(docs/plan_low_quality_multiview.md §1.1):

    ΔL_i = L(y_i | x_i^-) - L(y_i | x_i)        counterfactual delta
    Q_i  = rank01(ΔL_i) ∈ [0, 1]                reliability
    c_i  ∈ {1, c_trunc}                          completeness (EOS gate)

``x_i^-`` is a semantically unrelated instruction drawn from the same
response-length bucket (see ``tads.data.corruption.make_counterfactual``;
the counterfactual pool is materialised offline by
``scripts/make_corrupted_pool.py --emit-counterfactual`` and loaded through
the standard Alpaca tokenisation path so both pools stay index-aligned).

Interpretation: if conditioning on the TRUE instruction does not improve
response prediction over a random instruction (ΔL ≈ 0), the response is
unreliable — mismatched, noisy, or generic. Clean-but-hard samples keep a
large ΔL, which is exactly the separation entropy/loss alone cannot make.

Q is computed ONCE at the base checkpoint and cached
(``reliability_cache.pt`` in the run's output dir): the reliability of a
(instruction, response) pair is a property of the data, not of the current
checkpoint. The dynamic part of the score comes from learnability and
alignment (plan §6, "static reliability gate + dynamic learnability/
alignment fusion").
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .scorer import rank01
from .utils import cuda_mem_str

logger = logging.getLogger(__name__)

CACHE_FILENAME = "reliability_cache.pt"


# ---------------------------------------------------------------------------
# Completeness (data-level, no forward pass)
# ---------------------------------------------------------------------------

def completeness_from_dataset(
    dataset,
    eos_token_id: int,
    c_trunc: float = 0.2,
) -> torch.Tensor:
    """Per-sample completeness gate c_i from tokenised labels.

    A response is complete when its label sequence ends with the EOS token:
    responses cut by ``max_seq_len`` truncation, or corrupted mid-sentence
    truncations whose text never reached EOS, fail the check and receive
    ``c_trunc``. Rows with no response tokens at all also get ``c_trunc``.
    """
    if not (0.0 < c_trunc <= 1.0):
        raise ValueError(f"completeness_from_dataset: c_trunc must be in (0,1], got {c_trunc}")
    out = torch.empty(len(dataset), dtype=torch.float32)
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        resp = labels[labels != -100]
        if resp.numel() == 0:
            out[i] = c_trunc
        else:
            out[i] = 1.0 if int(resp[-1].item()) == int(eos_token_id) else c_trunc
    n_trunc = int((out < 1.0).sum().item())
    logger.info(
        "completeness_from_dataset | n=%d | flagged_incomplete=%d (%.1f%%) | "
        "c_trunc=%.2f",
        len(dataset), n_trunc, 100.0 * n_trunc / max(1, len(dataset)), c_trunc,
    )
    return out


# ---------------------------------------------------------------------------
# Loss-only pool forward (cheaper than compute_rewards: no entropy softmax)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pool_loss(
    model,
    dataset,
    *,
    batch_size: int = 1,
    device: str = "cuda",
    progress_interval: int = 200,
    empty_cache_interval: int = 10,
    tag: str = "",
) -> torch.Tensor:
    """Per-sample mean CE loss over response tokens for the whole pool.

    One forward pass, loss only — used for the counterfactual pool, where
    entropy/hidden states are not needed. Roughly half the memory of
    ``collect_episode`` per batch (no entropy log-softmax, no
    output_hidden_states).
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    losses = []
    t0 = time.time()
    total_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        resp_mask = (shift_labels != -100).float()
        n_resp = resp_mask.sum(dim=-1).clamp(min=1)
        B = shift_logits.size(0)
        batch_loss = torch.empty(B, dtype=torch.float32, device=input_ids.device)
        for i in range(B):
            ce = F.cross_entropy(
                shift_logits[i].float(),
                shift_labels[i].clamp(min=0),
                reduction="none",
            )
            batch_loss[i] = (ce * resp_mask[i]).sum() / n_resp[i]
        losses.append(batch_loss.detach().cpu())
        del out, input_ids, attention_mask, labels, shift_logits, shift_labels
        if (
            torch.cuda.is_available()
            and empty_cache_interval > 0
            and step % empty_cache_interval == 0
        ):
            torch.cuda.empty_cache()
        if step == 1 or step % progress_interval == 0 or step == total_batches:
            logger.info(
                "compute_pool_loss%s | batch=%d/%d | elapsed=%.1fmin | %s",
                f" [{tag}]" if tag else "", step, total_batches,
                (time.time() - t0) / 60, cuda_mem_str(),
            )
    if was_training:
        model.train()
    return torch.cat(losses, dim=0)


# ---------------------------------------------------------------------------
# Reliability score + cache
# ---------------------------------------------------------------------------

def reliability_from_losses(
    loss_orig: torch.Tensor,
    loss_cf: torch.Tensor,
) -> torch.Tensor:
    """Q = rank01(L(y|x^-) - L(y|x)). Both tensors shape (N,), same order."""
    if loss_orig.shape != loss_cf.shape:
        raise ValueError(
            f"reliability_from_losses: shape mismatch orig={tuple(loss_orig.shape)} "
            f"vs cf={tuple(loss_cf.shape)}"
        )
    return rank01(loss_cf - loss_orig)


def cache_path_for(output_dir) -> Path:
    return Path(output_dir) / CACHE_FILENAME


def load_reliability_cache(output_dir) -> Optional[Dict[str, Any]]:
    p = cache_path_for(output_dir)
    if not p.exists():
        return None
    try:
        cache = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # corrupted cache: recompute rather than crash
        logger.warning("Could not load reliability cache at %s (%s)", p, e)
        return None
    for key in ("q", "completeness"):
        if key not in cache:
            logger.warning("Reliability cache at %s missing key %r; ignoring", p, key)
            return None
    logger.info(
        "Loaded reliability cache from %s | n=%d | computed_at_epoch=%s",
        p, cache["q"].numel(), cache.get("epoch"),
    )
    return cache


def save_reliability_cache(
    output_dir,
    *,
    q: torch.Tensor,
    completeness: torch.Tensor,
    loss_orig: torch.Tensor,
    loss_cf: torch.Tensor,
    epoch: int,
) -> None:
    p = cache_path_for(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save(
        {
            "q": q.cpu(),
            "completeness": completeness.cpu(),
            "loss_orig": loss_orig.cpu(),
            "loss_cf": loss_cf.cpu(),
            "epoch": epoch,
        },
        tmp,
    )
    tmp.replace(p)
    logger.info("Saved reliability cache to %s (n=%d)", p, q.numel())
