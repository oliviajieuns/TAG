"""Batched greedy generation for the benchmark evaluators.

Every evaluator except HumanEval used to call ``model.generate`` once per
example. At 7B that is the whole eval budget: decoding is memory-bandwidth
bound, so a batch of one moves the entire weight matrix through HBM to
produce a single token. Measured on BBH with LLaMA-class weights it came to
~16 examples/min, i.e. ~7 h for BBH alone and >20 h for the eight Table 2
benchmarks — per arm.

Batching does not change what is computed. Greedy decoding of a right-hand
continuation is independent of what sits to the LEFT of the prompt as long
as the padding is masked, which is why the padding side matters and is
forced here rather than trusted from the caller's tokenizer.

Two properties this module keeps, because both are load-bearing for a table
row:

  * **Order.** Prompts are reordered internally (longest first, so a batch
    that will not fit OOMs on the first batch rather than an hour in, and so
    padding waste is minimised). The returned list is in the caller's order.
  * **Agreement.** The first batch of the process is also decoded one
    example at a time and the two are compared. A silent divergence between
    the batched and unbatched paths would move a published number, so it is
    checked once, cheaply, and logged either way.

``TAG_EVAL_GEN_BS`` overrides the batch size (default 16); ``1`` restores
the old one-at-a-time behaviour exactly. ``TAG_EVAL_GEN_VERIFY=0`` skips the
first-batch agreement check.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16

# Has the batched-vs-sequential agreement check already run in this process?
# Once per process is enough: it is a property of the model/tokenizer pair,
# not of the benchmark.
_VERIFIED = False


def gen_batch_size(default: int = DEFAULT_BATCH_SIZE) -> int:
    raw = os.environ.get("TAG_EVAL_GEN_BS", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("TAG_EVAL_GEN_BS=%r is not an integer; using %d", raw, default)
        return default


def _encode(tokenizer, prompts: Sequence[str], max_input_tokens: int, device):
    """Left-padded batch encoding. Restores the tokenizer's own settings."""
    old_side = tokenizer.padding_side
    old_pad = tokenizer.pad_token
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        enc = tokenizer(
            list(prompts), return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_tokens,
        )
    finally:
        tokenizer.padding_side = old_side
        if tokenizer.pad_token != old_pad and old_pad is not None:
            tokenizer.pad_token = old_pad
    return {k: v.to(device) for k, v in enc.items()}


def _generate_once(model, tokenizer, prompts, *, device, max_new_tokens,
                   max_input_tokens, gen_kwargs) -> List[str]:
    enc = _encode(tokenizer, prompts, max_input_tokens, device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kwargs)
    # Left padding means every row's prompt ends at the same column, so one
    # slice serves the whole batch. Token-id slicing rather than a
    # `text[len(prompt):]` char offset — see the tydiqa.py note on the BOS /
    # whitespace decode round-trip that offset slicing cannot survive.
    texts = [
        tokenizer.decode(out[j, prompt_len:], skip_special_tokens=True)
        for j in range(out.shape[0])
    ]
    del enc, out
    return texts


def _verify_first_batch(model, tokenizer, prompts, batched, *, device,
                        max_new_tokens, max_input_tokens, gen_kwargs) -> None:
    """Decode the same prompts one at a time and compare."""
    global _VERIFIED
    _VERIFIED = True
    n = min(len(prompts), 8)
    agree = 0
    for j in range(n):
        one = _generate_once(
            model, tokenizer, [prompts[j]], device=device,
            max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens,
            gen_kwargs=gen_kwargs,
        )[0]
        if one.strip() == batched[j].strip():
            agree += 1
        elif agree + (n - j - 1) < n:  # only log the first few disagreements
            logger.warning(
                "  batched-vs-single mismatch on example %d:\n"
                "    batched: %r\n    single : %r",
                j, batched[j][:200], one[:200],
            )
    if agree == n:
        logger.info("  batched generation verified: %d/%d identical to "
                    "one-at-a-time decoding", agree, n)
    else:
        logger.warning(
            "  batched generation DIFFERS from one-at-a-time decoding on "
            "%d/%d prompts. Greedy decoding is padding-invariant in exact "
            "arithmetic, so a few late-token drifts on near-ties are "
            "expected; a large fraction is a bug. Re-run with "
            "TAG_EVAL_GEN_BS=1 to reproduce the unbatched numbers.",
            n - agree, n,
        )


def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device,
    max_new_tokens: int,
    max_input_tokens: int,
    batch_size: Optional[int] = None,
    stop_strings: Optional[Sequence[str]] = None,
    extra_gen_kwargs: Optional[Dict[str, Any]] = None,
    progress_every: int = 0,
    progress_label: str = "",
) -> List[str]:
    """Greedy-decode ``prompts``; return the continuations in the same order."""
    if not prompts:
        return []

    # No temperature / top_p: with do_sample=False they are unused, and
    # newer transformers warns (or errors) when a sampling knob is set on a
    # greedy config.
    gen_kwargs: Dict[str, Any] = dict(
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if stop_strings:
        gen_kwargs["stop_strings"] = list(stop_strings)
        gen_kwargs["tokenizer"] = tokenizer
    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)

    bs = gen_batch_size() if batch_size is None else max(1, int(batch_size))

    # Longest first: padding waste is minimised within a batch, and a batch
    # size that cannot fit fails on batch 0 instead of on task 19.
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"])
               for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])

    out: List[Optional[str]] = [None] * len(prompts)
    done = 0
    pos = 0
    cur_bs = bs
    while pos < len(order):
        idx = order[pos: pos + cur_bs]
        chunk = [prompts[i] for i in idx]
        try:
            texts = _generate_once(
                model, tokenizer, chunk, device=device,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens, gen_kwargs=gen_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            if cur_bs == 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            logger.warning("  OOM during generation — halving batch size to %d",
                           cur_bs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if not _VERIFIED and os.environ.get("TAG_EVAL_GEN_VERIFY", "1") != "0" \
                and len(chunk) > 1:
            _verify_first_batch(
                model, tokenizer, chunk, texts, device=device,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens, gen_kwargs=gen_kwargs,
            )

        for i, t in zip(idx, texts):
            out[i] = t
        pos += len(idx)
        done += len(idx)
        # The allocator fragments across thousands of long prompts; the same
        # hygiene the per-example loops used, at batch granularity.
        if torch.cuda.is_available() and (pos // max(cur_bs, 1)) % 8 == 0:
            torch.cuda.empty_cache()
        if progress_every and done % progress_every < len(idx):
            logger.info("    %s%d/%d", progress_label and progress_label + " ",
                        done, len(prompts))

    return [t if t is not None else "" for t in out]
