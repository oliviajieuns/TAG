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

Batching is **not bit-exact**, and the size is therefore part of the number:
measured on BBH, batch 16 vs batch 1 flips the correctness of 2.31% of
examples (unbiased in direction), which propagates to a ~0.2pp SD on the
27-task macro average at the full 250 examples per task. Hold the batch size
fixed across every arm and seed; it is stamped into each summary JSON as
``generation_batch_size``. docs/tag-paper-deltas.md D1 has the measurement
and what the paper has to say about it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _first_divergence(a: str, b: str) -> int:
    """Index of the first differing character, or -1 if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def _check_padding_is_masked(model, tokenizer, prompts, *, device,
                             max_input_tokens) -> None:
    """Does left padding change the FIRST predicted token? It must not.

    This is the check that separates the two reasons a batched continuation
    can differ from an unbatched one:

      * the padding is not masked — the model is attending to pad tokens, the
        whole benchmark is wrong, and no amount of eyeballing the text will
        show it; or
      * the arithmetic reassociated — cuBLAS picks a different reduction
        order for a (16, T, H) matmul than for a (1, T, H) one, logits move
        by ~1e-3, and a long greedy chain that passes near a tie forks.

    The first is a bug and aborts. The second is the same nondeterminism a
    different GPU or a different transformers build already gives you.

    One forward per prompt, no generation — cheap enough to always run.
    """
    n = min(len(prompts), 6)
    sub = list(prompts[:n])
    enc_b = _encode(tokenizer, sub, max_input_tokens, device)
    with torch.inference_mode():
        # Left padding means column -1 is a real token on every row.
        lb = model(**enc_b).logits[:, -1, :].float()
    bad, max_abs = [], 0.0
    for j in range(n):
        enc_s = _encode(tokenizer, [sub[j]], max_input_tokens, device)
        with torch.inference_mode():
            ls = model(**enc_s).logits[0, -1, :].float()
        max_abs = max(max_abs, (lb[j] - ls).abs().max().item())
        if int(lb[j].argmax()) != int(ls.argmax()):
            bad.append(j)
        del enc_s, ls
    del enc_b, lb
    if bad:
        raise RuntimeError(
            f"Left padding is changing the model's prediction: the first "
            f"generated token differs between a padded batch and the same "
            f"prompt alone on {len(bad)}/{n} prompts (max |logit diff| "
            f"{max_abs:.3g}). That is a masking bug, not float noise — every "
            f"number this eval produces would be wrong. Re-run with "
            f"TAG_EVAL_GEN_BS=1 to fall back to one prompt per generate()."
        )
    logger.info(
        "  padding check: first-token argmax identical on %d/%d prompts "
        "(max |logit diff| %.3g) — the mask is right", n, n, max_abs,
    )


def _verify_first_batch(model, tokenizer, prompts, batched, *, device,
                        max_new_tokens, max_input_tokens, gen_kwargs,
                        score_key=None) -> None:
    """Compare the batched decode against one-at-a-time decoding.

    Reports on two levels, because they mean different things:

      raw text   how many continuations are character-identical, and where
                 the first difference is. A divergence 200 characters into a
                 chain-of-thought is a coin landing the other way on a
                 near-tie; a divergence at character 0 is a bug.
      graded     how many produce the SAME answer once the evaluator's own
                 extractor has run. This is the one that moves a table cell,
                 and it is the number to judge the change by.
    """
    global _VERIFIED
    _VERIFIED = True

    _check_padding_is_masked(model, tokenizer, prompts, device=device,
                             max_input_tokens=max_input_tokens)

    n = min(len(prompts), 8)
    same_raw = 0
    same_key = 0
    n_key = 0
    diverge_at = []
    for j in range(n):
        one = _generate_once(
            model, tokenizer, [prompts[j]], device=device,
            max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens,
            gen_kwargs=gen_kwargs,
        )[0]
        a, b = batched[j], one
        d = _first_divergence(a, b)
        if d < 0:
            same_raw += 1
        else:
            diverge_at.append((d, max(len(a), len(b))))
        if score_key is not None:
            n_key += 1
            try:
                ka, kb = score_key(a), score_key(b)
            except Exception:  # an extractor that throws is not our business
                n_key -= 1
            else:
                if ka == kb:
                    same_key += 1
                else:
                    logger.warning(
                        "  graded answer DIFFERS on example %d: batched=%r "
                        "single=%r  (%s)", j, ka, kb,
                        "raw text is identical" if d < 0
                        else f"texts first differ at char {d}",
                    )

    if same_raw == n:
        logger.info("  batched generation verified: %d/%d continuations "
                    "character-identical to one-at-a-time decoding", n, n)
    else:
        where = ", ".join(f"{d}/{t}" for d, t in diverge_at[:5])
        logger.info(
            "  batched vs one-at-a-time: %d/%d continuations identical; the "
            "other %d fork at char %s (position/length). Left padding is "
            "exact in real arithmetic and the padding check above passed, so "
            "this is float reassociation across batch shapes — a long greedy "
            "chain that passes near a tie takes the other branch.",
            same_raw, n, n - same_raw, where,
        )

    # The graded verdict is reported whenever anything at all disagreed. It
    # is the only one that can move a table cell, so it must not be skipped
    # just because the raw texts happened to match — an extractor is a pure
    # function of the text, and if it disagrees on identical text, the
    # extractor itself is nondeterministic and that is worse.
    if n_key and (same_key < n_key or same_raw < n):
        if same_key == n_key:
            logger.info(
                "  and it does not reach the score: the extracted answer is "
                "identical on %d/%d — those forks are downstream of the "
                "answer, in the trailing reasoning.", same_key, n_key,
            )
        else:
            logger.warning(
                "  AND IT REACHES THE SCORE: the extracted answer differs on "
                "%d/%d examples. Do not report this run. Re-run with "
                "TAG_EVAL_GEN_BS=1, or lower it until the graded answers "
                "agree.", n_key - same_key, n_key,
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
    score_key: Optional[Callable[[str], Any]] = None,
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
                score_key=score_key,
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
