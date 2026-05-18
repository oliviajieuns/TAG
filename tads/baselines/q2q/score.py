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
def _mean_nll_of_response(
    model,
    tokenizer,
    device,
    prefix_text: str,
    response_text: str,
    max_length: int,
) -> float:
    """Return the mean NLL (natural log) over the response tokens, given
    the prefix as context. Set prefix_text=""  to score response in isolation.
    """
    # Encode prefix + response separately so we know exactly which token
    # positions belong to the response (and thus get scored).
    if prefix_text:
        prefix_ids = tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=max_length - 1,
        ).input_ids[0]
    else:
        prefix_ids = tokenizer(
            "", return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=max_length,
        ).input_ids[0]
    response_ids = tokenizer(
        response_text, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max(1, max_length - len(prefix_ids)),
    ).input_ids[0]
    if response_ids.numel() == 0:
        return float("nan")

    full_ids = torch.cat([prefix_ids, response_ids], dim=0).unsqueeze(0).to(device)
    out = model(full_ids)
    logits = out.logits[0, :-1, :]
    target = full_ids[0, 1:]

    # Mask: 1 over response positions, 0 over prefix.
    n_prefix = prefix_ids.numel()
    mask = torch.zeros_like(target, dtype=torch.float32)
    # Position i in `target` predicts token at position i+1 in full_ids.
    # So response positions in `target` are indices >= n_prefix - 1 (the
    # last prefix token transitions to the first response token).
    start = max(0, n_prefix - 1)
    mask[start:] = 1.0

    nll = F.cross_entropy(logits.float(), target, reduction="none")
    denom = mask.sum().clamp_min(1)
    return float((nll * mask).sum().item() / denom.item())


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
    log_every: int = 200,
) -> List[float]:
    """Return one IFD score per (instruction, response) pair.

    Args:
        prompt_format: callable (instruction) -> prefix_text. Should produce the
            same prefix the SFT pipeline would feed to the model BEFORE the
            response (e.g. Alpaca's "### Instruction:\n{ins}\n\n### Response:\n").
    """
    if len(instructions) != len(responses):
        raise ValueError(
            f"instructions/responses length mismatch: {len(instructions)} vs {len(responses)}"
        )
    was_training = model.training
    model.eval()
    try:
        scores: List[float] = []
        for i, (ins, res) in enumerate(zip(instructions, responses)):
            try:
                nll_cond = _mean_nll_of_response(
                    model, tokenizer, device,
                    prefix_text=prompt_format(ins),
                    response_text=res,
                    max_length=max_length,
                )
                nll_uncond = _mean_nll_of_response(
                    model, tokenizer, device,
                    prefix_text="",
                    response_text=res,
                    max_length=max_length,
                )
                # IFD = exp(nll_cond - nll_uncond). Clamp to avoid inf.
                diff = max(-50.0, min(50.0, nll_cond - nll_uncond))
                ifd = math.exp(diff)
            except Exception as ex:
                logger.warning("IFD scoring failed on sample %d: %s — score=nan", i, ex)
                ifd = float("nan")
            scores.append(ifd)
            if (i + 1) % log_every == 0:
                logger.info("IFD %d / %d (last=%.4f)", i + 1, len(instructions), scores[-1])
        return scores
    finally:
        if was_training:
            model.train()


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
