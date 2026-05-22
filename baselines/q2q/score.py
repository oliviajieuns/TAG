"""IFD (Instruction-Following Difficulty) scoring — Q2Q / Cherry_LLM Eq.3.

    IFD(x, y) = PPL(y | x) / PPL(y)
            = exp( mean_nll(y | x) - mean_nll(y) )

Both numerator and denominator average NLL over the SAME response tokens,
so the ratio is well-defined regardless of response length. High IFD
means the instruction *doesn't* help the model predict the response,
i.e., the example is informative for SFT (paper §3.2).
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def _batched_mean_nll_of_response(
    model,
    tokenizer,
    device,
    prefix_texts: Sequence[str],
    response_texts: Sequence[str],
    max_length: int,
    batch_size: int,
    pad_token_id: int,
) -> List[float]:
    """Batched mean-NLL-over-response. One scalar per (prefix, response) pair.

    Replicates ``_mean_nll_of_response`` numerics exactly under right-side
    padding — same prefix/response tokenization, same start = max(0,
    n_prefix - 1) mask, same average over the scored positions. Padding
    rows contribute 0 to both numerator and denominator because
    ``target_mask`` is 0 over the padded slots.

    The per-sample boundary (where prefix ends and response begins) is
    preserved by tokenising prefix and response separately, then
    concatenating before padding to the batch's max length. This is the
    pattern the un-batched version uses; batching just stacks them.
    """
    n = len(prefix_texts)
    if len(response_texts) != n:
        raise ValueError(
            f"prefix/response length mismatch: {len(prefix_texts)} vs {n}"
        )
    if n == 0:
        return []

    # Pre-tokenize prefix and response per sample, with the same truncation
    # bookkeeping the un-batched function uses.
    samples: List[tuple] = []
    for ptext, rtext in zip(prefix_texts, response_texts):
        if ptext:
            pids = tokenizer(
                ptext, add_special_tokens=True,
                truncation=True, max_length=max_length - 1,
            ).input_ids
        else:
            pids = tokenizer(
                "", add_special_tokens=True,
                truncation=True, max_length=max_length,
            ).input_ids
        rids = tokenizer(
            rtext, add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_length - max(1, len(pids))),
        ).input_ids
        samples.append((pids, rids))

    results: List[float] = [float("nan")] * n

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = samples[batch_start:batch_end]
        B = len(batch)

        # Drop samples with empty response (NaN as in the un-batched fallback).
        valid_idx = [j for j, (_, r) in enumerate(batch) if len(r) > 0]
        if not valid_idx:
            continue

        # Build right-padded input_ids + attention_mask + target_mask.
        valid_batch = [batch[j] for j in valid_idx]
        Bv = len(valid_batch)
        seqs = [pids + rids for pids, rids in valid_batch]
        max_seq_len = max(len(s) for s in seqs)
        input_ids = torch.full(
            (Bv, max_seq_len), pad_token_id, dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(
            Bv, max_seq_len, dtype=torch.long, device=device,
        )
        target_mask = torch.zeros(
            Bv, max_seq_len - 1, dtype=torch.float32, device=device,
        )
        for i, (pids, rids) in enumerate(valid_batch):
            seq = pids + rids
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :L] = 1
            n_prefix = len(pids)
            # Same masking convention as the un-batched function:
            #   target index i predicts input position i+1
            #   response positions in target start at max(0, n_prefix - 1)
            #   and end at L - 2 inclusive (target has length L - 1)
            start = max(0, n_prefix - 1)
            end = L - 1  # slice end-exclusive → covers target[start:L-1]
            if end > start:
                target_mask[i, start:end] = 1.0

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]           # (Bv, max-1, V)
        target = input_ids[:, 1:]                # (Bv, max-1)
        Bv_, T_, V = logits.shape
        nll = F.cross_entropy(
            logits.reshape(-1, V).float(),
            target.reshape(-1),
            reduction="none",
        ).reshape(Bv_, T_)
        nll = nll * target_mask
        denom = target_mask.sum(dim=1).clamp_min(1)
        per_sample = (nll.sum(dim=1) / denom).cpu().tolist()

        for k, j in enumerate(valid_idx):
            results[batch_start + j] = float(per_sample[k])

    return results


@torch.no_grad()
def compute_ifd_scores(
    model,
    tokenizer,
    device,
    *,
    instructions: Sequence[str],
    responses: Sequence[str],
    prompt_format,
    max_length: int = 2048,
    batch_size: int = 8,
    log_every: int = 200,
) -> List[float]:
    """Return one IFD score per (instruction, response) pair (batched).

    Args:
        prompt_format: callable (instruction) -> prefix_text. Should produce the
            same prefix the SFT pipeline would feed to the model BEFORE the
            response (e.g. Alpaca's "### Instruction:\n{ins}\n\n### Response:\n").
        batch_size: forwards per model call. The original loop called the
            model TWICE per sample (cond + uncond), one sample at a time
            (~2 × 50K = 100K sequential forwards). Batched: total forwards
            drop to 2 × ⌈N/batch_size⌉ → on 7B that's 1-2 hours instead of
            ~5-6 hours. Numerics match the un-batched version under right
            padding (pad tokens are masked out of both prefix and response
            mean-NLL).
    """
    if len(instructions) != len(responses):
        raise ValueError(
            f"instructions/responses length mismatch: {len(instructions)} vs {len(responses)}"
        )
    n = len(instructions)
    prefix_texts = [prompt_format(ins) for ins in instructions]

    # Resolve pad token; restore tokenizer state on exit.
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    orig_pad_token = tokenizer.pad_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    was_training = model.training
    model.eval()
    try:
        logger.info(
            "Q2Q IFD batched | n=%d | batch_size=%d | est forwards=%d (cond+uncond)",
            n, batch_size, 2 * ((n + batch_size - 1) // batch_size),
        )
        nll_cond = _batched_mean_nll_of_response(
            model, tokenizer, device,
            prefix_texts=prefix_texts,
            response_texts=responses,
            max_length=max_length,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
        )
        nll_uncond = _batched_mean_nll_of_response(
            model, tokenizer, device,
            prefix_texts=[""] * n,
            response_texts=responses,
            max_length=max_length,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
        )

        scores: List[float] = []
        for i, (c, u) in enumerate(zip(nll_cond, nll_uncond)):
            if math.isnan(c) or math.isnan(u):
                scores.append(float("nan"))
                continue
            # IFD = exp(nll_cond - nll_uncond). Clamp diff to [-15, 15] for
            # the same outlier-control as the un-batched version.
            diff = max(-15.0, min(15.0, c - u))
            scores.append(math.exp(diff))
            if (i + 1) % log_every == 0:
                logger.info("IFD %d / %d (last=%.4f)", i + 1, n, scores[-1])
        return scores
    finally:
        if was_training:
            model.train()
        tokenizer.padding_side = orig_padding_side
        tokenizer.pad_token = orig_pad_token


def select_top_proportion_by_ifd(
    scores: Sequence[float],
    proportion: float,
    *,
    ifd_low: float = 0.5,
    ifd_high: float = 1.0,
) -> List[int]:
    """Cherry_LLM-style filter + top-K.

    Paper §3.3: discard samples with IFD outside [low, high] (very-easy and
    pathological-hard), then keep top `proportion` by IFD score (descending).
    If after filtering the candidate pool is smaller than the target K, fall
    back to top-K over the whole score array (with NaN excluded).
    """
    scores_np = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores_np)
    n_total = len(scores_np)
    k = max(1, int(n_total * proportion))

    in_range = valid & (scores_np >= ifd_low) & (scores_np <= ifd_high)
    pool = np.where(in_range)[0]
    if len(pool) < k:
        logger.warning(
            "Q2Q: in-range pool (%d) smaller than target K (%d); falling back "
            "to top-K over the full valid score range.",
            len(pool), k,
        )
        pool = np.where(valid)[0]
    order = pool[np.argsort(-scores_np[pool])]
    return order[:k].tolist()
