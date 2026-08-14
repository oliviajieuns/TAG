"""Reliability (Consistency) view: counterfactual instruction fidelity +
completeness.

Implements the static reliability gate of the MVF score
(docs/plan_low_quality_multiview.md §2.1, v3):

    ΔL_i   = L(y_i | x_i^-) - L(y_i | x_i)       counterfactual delta
    Q_i    = clip(2·(σ(ΔL_i / s) - 0.5), 0, 1)   zero-anchored calibrated gate
    c_i    ∈ {1, c_trunc}                         completeness (text + EOS gate)

``x_i^-`` is a semantically unrelated instruction drawn from the same
response-length bucket (see ``tag.data.corruption.make_counterfactual``;
the counterfactual pool is materialised offline by
``scripts/make_corrupted_pool.py --emit-counterfactual`` and loaded through
the standard Alpaca tokenisation path so both pools stay index-aligned).

Why zero-anchored sigmoid instead of rank01 (v3): rank01 makes Q's
distribution uniform BY CONSTRUCTION — half of a perfectly clean pool is
gated below 0.5, and the top of an 80%-corrupted pool still gets Q ≈ 1;
gate strength cannot respond to pool contamination. ΔL = 0 is a physical
reference point independent of pool composition ("the true instruction
does not help predict the response at all"), so the sigmoid is anchored
there and its scale ``s`` is calibrated ONCE per backbone on a clean
reference pool (:func:`calibrate_reliability_scale`). The re-zeroing
``clip(2·(σ-0.5), 0, 1)`` maps ΔL ≤ 0 to 0 (the ε floor of the fused
gate) rather than to σ(0) = 0.5 — without it the gate suppresses
corrupted samples by barely ~2× while the learnability factor spans a
far larger range, and non-compensation quietly fails (adversarial review
2026-08). ``mode="rank"`` keeps the v1 transform as an ablation arm.

K > 1 counterfactuals (optional, "evidential-lite"): pass ``loss_cf`` of
shape (K, N) and Q is the mean per-counterfactual gate discounted by the
cross-counterfactual dispersion, Q ← Q · (1 - 2·std_k). High disagreement
between counterfactual pairings means the ΔL evidence is unstable for
that sample, so the gate trusts it less. Heuristic, monotone, documented
as experimental in the paper.

Q is computed ONCE at the base checkpoint and cached
(``reliability_cache.pt`` in the run's output dir): the reliability of a
(instruction, response) pair is a property of the data, not of the current
checkpoint. Recomputing it at a later checkpoint silently changes the
view's meaning, so the pipeline hard-errors when the cache is missing at
epoch > 1 (selector.collect_episode). The dynamic part of the score comes
from learnability and alignment.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import forward as fwd
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
    """Per-sample completeness gate c_i (v3: text-level AND token-level).

    Two independent checks must BOTH pass for c_i = 1.0:

    1. ``text_complete`` — raw-text heuristic computed at tokenisation time
       (``tag.data.sft_prompts.text_is_complete``): terminal punctuation /
       closed code fence / numeric answer ending. This is the check that
       catches T3-style textual truncation. The token-level EOS check alone
       CANNOT catch it, because ``tokenize_alpaca`` unconditionally appends
       EOS to every response — including truncated ones — so before v3
       every T3-corrupted sample silently received c_i = 1.0 (plan §1.2,
       "the view assigned to catch truncation could not catch it").
    2. Label sequence ends with EOS — catches responses cut by
       ``max_seq_len`` budget truncation (there the appended EOS is
       dropped, so the token check IS informative).

    Datasets tokenised before v3 lack the ``text_complete`` column; the
    text check is then skipped with a single loud warning (token-only
    behaviour, i.e. the pre-v3 semantics).
    """
    if not (0.0 < c_trunc <= 1.0):
        raise ValueError(f"completeness_from_dataset: c_trunc must be in (0,1], got {c_trunc}")
    out = torch.empty(len(dataset), dtype=torch.float32)
    warned_missing_text = False
    for i in range(len(dataset)):
        row = dataset[i]
        labels = row["labels"]
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        resp = labels[labels != -100]
        if resp.numel() == 0:
            out[i] = c_trunc
            continue
        token_ok = int(resp[-1].item()) == int(eos_token_id)
        text_flag = row.get("text_complete") if hasattr(row, "get") else None
        if text_flag is None:
            if not warned_missing_text:
                logger.warning(
                    "completeness_from_dataset: dataset has no 'text_complete' "
                    "column (tokenised before v3, or stale HF map cache — set "
                    "TAG_FRESH_DATA_CACHE=1). Falling back to the token-level "
                    "EOS check only; T3-style textual truncation will NOT be "
                    "detected.",
                )
                warned_missing_text = True
            text_ok = True
        else:
            text_ok = bool(int(text_flag))
        out[i] = 1.0 if (token_ok and text_ok) else c_trunc
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
    sort_by_length: bool = True,
    split_lm_head: bool = True,
    head_chunk_tokens: int = fwd.DEFAULT_HEAD_CHUNK,
) -> torch.Tensor:
    """Per-sample mean CE loss over response tokens for the whole pool.

    One forward pass, loss only — used for the counterfactual pool, where
    entropy/hidden states are not needed.

    The pass is scheduled by :mod:`tag.core.forward`: padding is cropped per
    batch, records of similar length share a batch, and only response
    positions are projected to the vocabulary. All three are arithmetically
    neutral for a right-padded causal LM — see that module — and the result
    is in dataset order regardless of the order the batches ran in. Set
    ``sort_by_length=False`` / ``split_lm_head=False`` to run the plain path.
    """
    was_training = model.training
    model.eval()

    n_records = len(dataset)
    plan = (
        fwd.length_sorted_batches(dataset, batch_size)
        if sort_by_length and n_records > batch_size
        else None
    )
    if plan is None:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False,
        )
        emit_order = list(range(n_records))
    else:
        batches, emit_order = plan
        loader = DataLoader(
            dataset, batch_sampler=batches, num_workers=0, pin_memory=True,
        )

    response_ce = fwd.ResponseCE(
        model, split_head=split_lm_head, chunk=head_chunk_tokens, tag=tag,
    )

    out_loss = torch.zeros(n_records, dtype=torch.float32)
    t0 = time.time()
    total_batches = len(loader)
    cursor = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        B = input_ids.size(0)
        input_ids, attention_mask, labels, t_eff = fwd.crop_padded_batch(
            input_ids, attention_mask, labels,
        )
        shift_labels = labels[:, 1:]
        sel = shift_labels != -100
        tgt = shift_labels[sel]
        counts = sel.sum(dim=1)

        ce_all = response_ce(input_ids, attention_mask, sel, tgt)

        # Per-row mean over that row's response positions. segment_reduce
        # would need a fixed layout; the flattened CE is already row-major,
        # so a padded scatter-add is both exact and one kernel.
        row_of = torch.repeat_interleave(
            torch.arange(B, device=ce_all.device), counts,
        )
        sums = torch.zeros(B, dtype=torch.float32, device=ce_all.device)
        sums.scatter_add_(0, row_of, ce_all)
        means = (sums / counts.clamp(min=1).float()).cpu()
        for i in range(B):
            out_loss[emit_order[cursor]] = means[i]
            cursor += 1

        del input_ids, attention_mask, labels, shift_labels, sel, tgt, counts, ce_all
        if (
            torch.cuda.is_available()
            and empty_cache_interval > 0
            and step % empty_cache_interval == 0
        ):
            torch.cuda.empty_cache()
        if step == 1 or step % progress_interval == 0 or step == total_batches:
            logger.info(
                "compute_pool_loss%s | batch=%d/%d | bs=%d | T=%d | "
                "elapsed=%.1fmin | %s",
                f" [{tag}]" if tag else "", step, total_batches, B, t_eff,
                (time.time() - t0) / 60, cuda_mem_str(),
            )
    if was_training:
        model.train()
    return out_loss


# ---------------------------------------------------------------------------
# Reliability score + cache
# ---------------------------------------------------------------------------

def calibrate_reliability_scale(
    ref_delta: torch.Tensor,
    *,
    target_pct: float = 0.10,
    target_q: float = 0.8,
) -> float:
    """Calibrate the sigmoid scale ``s`` on a CLEAN reference pool's ΔL.

        s = quantile_{target_pct}(ΔL_clean) / logit(target_q)

    i.e. "(1 - target_pct) of clean reference samples receive raw
    σ(ΔL/s) ≥ target_q". Note the raw σ is the CALIBRATION TARGET; the
    value entering the fused score is the re-zeroed gate, so with the
    defaults 90% of clean data receives Q ≥ 2·(0.8 − 0.5) = 0.6 (gate
    factor ≥ 0.61 at c = 1 → suppression ≈ 61× over the ε floor). The
    gate is near-inert on clean pools (clean-equivalence built in by
    construction) while corrupted samples with ΔL ≈ 0 stay pinned at
    the zero anchor.

    Guards: if the target quantile of the reference is not positive the
    clean reference itself is suspect (a clean pool should overwhelmingly
    have ΔL > 0); fall back to the median, then to 1.0, with loud
    warnings — never a silent default.
    """
    if not (0.0 < target_pct < 1.0):
        raise ValueError(f"calibrate_reliability_scale: target_pct in (0,1), got {target_pct}")
    if not (0.5 < target_q < 1.0):
        raise ValueError(f"calibrate_reliability_scale: target_q in (0.5,1), got {target_q}")
    ref = ref_delta.detach().float().view(-1)
    if ref.numel() < 100:
        logger.warning(
            "calibrate_reliability_scale: reference has only %d samples — "
            "quantile estimate will be unstable.", ref.numel(),
        )
    logit = math.log(target_q / (1.0 - target_q))
    q = float(torch.quantile(ref, target_pct).item())
    if q <= 0:
        med = float(ref.median().item())
        logger.warning(
            "calibrate_reliability_scale: P%d(ΔL_clean)=%.4f is not positive — "
            "the clean reference pool looks contaminated or the counterfactuals "
            "are not truly unrelated. Falling back to the median (%.4f).",
            int(100 * target_pct), q, med,
        )
        q = med
    if q <= 0:
        logger.warning(
            "calibrate_reliability_scale: median ΔL_clean is also non-positive "
            "(%.4f). Using s=1.0 — treat every Q from this calibration as "
            "diagnostic-only.", q,
        )
        return 1.0
    s = q / logit
    logger.info(
        "calibrate_reliability_scale | n_ref=%d | P%d=%.4f | target_q=%.2f | s=%.5f",
        ref.numel(), int(100 * target_pct), q, target_q, s,
    )
    return s


def reliability_from_losses(
    loss_orig: torch.Tensor,
    loss_cf: torch.Tensor,
    *,
    mode: str = "sigmoid",
    scale: Optional[float] = None,
    rezero: bool = True,
) -> torch.Tensor:
    """Reliability Q from counterfactual deltas ΔL = L(y|x^-) - L(y|x).

    Args:
        loss_orig: (N,) pool loss under the TRUE instructions.
        loss_cf: (N,) counterfactual pool loss, or (K, N) for K
            counterfactual pairings per sample (evidential-lite).
        mode: "sigmoid" (v3 default, zero-anchored calibrated gate) or
            "rank" (v1 rank01 transform — ablation arm only; uses the
            MEAN ΔL when K > 1).
        scale: calibrated sigmoid scale s from
            :func:`calibrate_reliability_scale`. When None in sigmoid
            mode, an IN-POOL fallback (median of the positive ΔL mass /
            logit(0.8)) is used with a loud warning — self-calibration
            reintroduces exactly the pool-dependence the sigmoid design
            removes, so it is acceptable for forward-only diagnostics but
            NOT for reported selection runs.
        rezero: map ΔL ≤ 0 to 0 via clip(2·(σ-0.5), 0, 1) (v3 default).
            False keeps raw σ (the v2 ablation whose suppression margin
            is provably too weak against the D factor).

    Returns:
        Q of shape (N,) in [0, 1].
    """
    if mode not in ("sigmoid", "rank"):
        raise ValueError(f"reliability_from_losses: mode must be sigmoid/rank, got {mode!r}")
    orig = loss_orig.view(-1).float()
    cf = loss_cf.float()
    if cf.dim() == 1:
        cf = cf.unsqueeze(0)
    if cf.dim() != 2 or cf.size(1) != orig.numel():
        raise ValueError(
            f"reliability_from_losses: loss_cf shape {tuple(loss_cf.shape)} "
            f"incompatible with loss_orig {tuple(loss_orig.shape)} — expected "
            f"(N,) or (K, N)."
        )
    delta = cf - orig.unsqueeze(0)  # (K, N)
    if mode == "rank":
        return rank01(delta.mean(dim=0))
    if scale is None:
        pos = delta.mean(dim=0)
        pos = pos[pos > 0]
        if pos.numel() == 0:
            logger.warning(
                "reliability_from_losses: no positive ΔL mass for in-pool "
                "fallback calibration — using s=1.0 (diagnostic-only).",
            )
            scale = 1.0
        else:
            scale = float(pos.median().item()) / math.log(0.8 / 0.2)
            logger.warning(
                "reliability_from_losses: no calibrated scale provided — "
                "using IN-POOL fallback s=%.5f. This reintroduces pool "
                "dependence; pass a clean-reference scale for reported runs "
                "(selection.mvf.reliability_scale / reliability_ref_file).", scale,
            )
    if scale <= 0:
        raise ValueError(f"reliability_from_losses: scale must be > 0, got {scale}")
    q_raw = torch.sigmoid(delta / scale)  # (K, N)
    # Per-counterfactual gate FIRST, then reduce (mean of gates, plan
    # §2.1). Gate-of-mean is not the same estimator: the rezero clip is
    # convex, so gate(mean σ) ≤ mean(gate σ) by Jensen — evidence
    # straddling σ = 0.5 would collapse to exactly 0 under gate-of-mean
    # even when some counterfactuals give positive evidence (adversarial
    # review 2026-08). The dispersion is likewise taken over the GATES,
    # matching the documented estimator.
    if rezero:
        gates = torch.clamp(2.0 * (q_raw - 0.5), min=0.0, max=1.0)
    else:
        gates = q_raw
    q = gates.mean(dim=0)
    if gates.size(0) > 1:
        # Dispersion discount: std of values in [0,1] is at most 0.5, so
        # 2·std ∈ [0,1] gives a full discount only under maximal
        # cross-counterfactual disagreement.
        dispersion = gates.std(dim=0, unbiased=False)
        q = q * torch.clamp(1.0 - 2.0 * dispersion, min=0.0)
    return q


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
    mode: str = "sigmoid",
    scale: Optional[float] = None,
    rezero: bool = True,
) -> None:
    """Persist Q plus the raw loss vectors AND the gate configuration.

    Storing (loss_orig, loss_cf, mode, scale, rezero) lets a later run with
    a different gate configuration recompute Q from the cached losses —
    no counterfactual forward pass needed — instead of either silently
    serving a stale-config Q or recomputing at the wrong checkpoint.
    """
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
            "mode": mode,
            "scale": scale,
            "rezero": rezero,
        },
        tmp,
    )
    tmp.replace(p)
    logger.info(
        "Saved reliability cache to %s (n=%d, mode=%s, scale=%s, rezero=%s)",
        p, q.numel(), mode, scale, rezero,
    )
