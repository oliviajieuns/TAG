"""SelectIT scoring — token-level + sentence-level uncertainty.

Faithful port of the official SelectIT implementation in
https://github.com/Blue-Raincoat/SelectIT/blob/main/self_reflection/{token_level,sentence_level}.py

Algorithm sketch (paper Eq.2 / Eq.4):
    1. Build a rating prompt:
           <rating_template>\\nInstruction:<ins>\\nResponse:<res>\\nThe answer is: \\n
    2. Forward the model once; take the last-position logits and softmax
       over the full vocab.
    3. Extract probabilities at the 5 token ids that encode the digits
       "1".."5". Renormalise with `exp(p / sum(p))` then divide by the sum
       again — this "double softmax" sharpens the 5-way distribution. (This
       quirk is in the official repo; preserved for paper-faithful scoring.)
    4. token_level_score = (argmax_rating + 1) * mean(p[argmax] - p[j])
       i.e. the predicted rating (1..5) weighted by how confident the model
       is in that rating.
    5. sentence_level_score = avg(token_scores_over_k_templates) /
                              (1 + alpha * std(...))
       Uses k different rating templates per sample for a robustness signal.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


def resolve_rating_token_ids(tokenizer) -> List[int]:
    """Return the token ids for the bare digits "1".."5".

    On LLaMA-2 SentencePiece, `tokenizer.encode("1", add_special_tokens=False)`
    actually returns ``[29871, 29896]`` — the SP model prefixes a leading
    whitespace marker (▁, id 29871) before the digit. The official SelectIT
    code sidesteps this by hard-coding ``[29896, 29906, 29941, 29946, 29945]``.

    Strategy here:
        1. Try `convert_tokens_to_ids(digit)`. SP-based tokenizers map the bare
           digit token "1" directly to its single vocab id (29896 for LLaMA-2)
           without any whitespace prefix — this matches the official repo's
           hard-coded constant exactly.
        2. Fallback: `encode(digit, add_special_tokens=False)` and take the
           LAST token — that strips a leading ▁ marker if present, leaving the
           digit's vocab id.
        3. Raise only if both yield no usable id.
    """
    ids: List[int] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for digit in ("1", "2", "3", "4", "5"):
        # (1) direct vocab lookup
        candidate = tokenizer.convert_tokens_to_ids(digit)
        if candidate is not None and candidate != unk_id and isinstance(candidate, int):
            ids.append(int(candidate))
            continue
        # (2) encode-and-take-last
        toks = tokenizer.encode(digit, add_special_tokens=False)
        if not toks:
            raise ValueError(
                f"Tokenizer produced empty encoding for rating digit {digit!r}."
            )
        ids.append(int(toks[-1]))
    if len(set(ids)) != 5:
        raise ValueError(
            f"Resolved rating token ids are not unique: {ids}. This tokenizer "
            f"collapses digit tokens — SelectIT scoring won't work unmodified."
        )
    # Audit-2 guard: confirm each resolved id actually decodes to the bare
    # digit. On exotic tokenizers `convert_tokens_to_ids("1")` or the
    # encode-last-token fallback can quietly return a multi-digit token
    # like "10" — making `pro_softmax` aggregate the wrong vocabulary
    # entries and producing a meaningless score.
    for digit, tid in zip(("1", "2", "3", "4", "5"), ids):
        decoded = tokenizer.decode([tid]).strip()
        if decoded != digit:
            raise ValueError(
                f"Rating token id {tid} decodes to {decoded!r}, not {digit!r}. "
                f"Tokenizer doesn't have a single-digit token for {digit} — "
                f"SelectIT scoring needs a custom resolver for this model."
            )
    return ids


def build_rating_prompt(rating_template: str, instruction: str, response: str) -> str:
    """Assemble the rating prompt exactly as in the official repo."""
    return (
        f"{rating_template.rstrip()}\n"
        f"Instruction:{instruction}\n"
        f"Response:{response}\n"
        f"The answer is: \n"
    )


def _double_softmax(probs5: np.ndarray) -> np.ndarray:
    """The paper's extra normalisation pass: `exp(p/sum(p))` then renormalise.

    NaN guard (audit-2): if the model puts ~all probability mass on
    non-rating tokens (newline, "The", etc.), `probs5.sum()` underflows to
    0 and `np.exp(0/0)` propagates NaN through the score. Argsort over
    NaN is implementation-defined → returns a uniform distribution so the
    sample gets a deterministic (low) score instead of a random rank.
    """
    s = probs5.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(probs5, 1.0 / len(probs5))
    e = np.exp(probs5 / s)
    return e / e.sum()


def _token_score_from_probs(p5: np.ndarray, k: int = 5) -> float:
    """Eq.2 — predicted rating × mean-confidence gap.

    `(score_base + 1) * sum(p[score_base] - p[j]) / (k - 1)`
    The j == score_base term contributes 0; sum over all 5 then divide by
    (k-1) gives the mean gap against the other 4 ratings.
    """
    base = int(np.argmax(p5))
    diff_sum = float(np.sum(p5[base] - p5))
    confidence = diff_sum / (k - 1)
    return float((base + 1) * confidence)


@torch.no_grad()
def selectit_scores(
    model,
    tokenizer,
    device,
    *,
    rating_templates: Sequence[str],
    instructions: Sequence[str],
    responses: Sequence[str],
    level: str = "token",
    alpha: float = 0.2,
    max_length: int = 2048,
    batch_size: int = 8,
    log_every: int = 200,
) -> List[float]:
    """Compute per-sample SelectIT scores (batched, paper-faithful).

    Args:
        level: "token" — one rating template per sample (cycled through
            ``rating_templates``). "sentence" — k templates per sample,
            then averaged with std penalty per paper Eq.4.
        alpha: sentence-level std-penalty weight; ignored for token-level.
        max_length: tokenizer truncation cap.
        batch_size: number of prompts forwarded per model call. The original
            implementation forwarded ONE prompt at a time → 50K samples = 50K
            sequential forwards (~3 h on 7B). With batch_size=8 the forward
            count drops ~8× and reaches the same numerics modulo padding.
            Padding fills with EOS via attention_mask; logits at the last
            non-pad position are paper-equivalent to the un-batched call.

    Numerics:
        Identical to the un-batched original — same full-vocab softmax,
        same ``_double_softmax`` 5-way renorm, same ``_token_score_from_probs``
        per-template aggregation. Only the loop structure changed.
    """
    if len(instructions) != len(responses):
        raise ValueError(
            f"instructions/responses length mismatch: {len(instructions)} vs {len(responses)}"
        )
    n_samples = len(instructions)
    rating_ids = resolve_rating_token_ids(tokenizer)
    k = len(rating_ids)  # = 5
    rating_ids_t = torch.tensor(rating_ids, device=device, dtype=torch.long)

    if level == "token":
        n_per_sample = 1
        templates_per_sample = [
            (rating_templates[i % len(rating_templates)],) for i in range(n_samples)
        ]
    elif level == "sentence":
        if len(rating_templates) < k:
            raise ValueError(
                f"sentence-level needs at least {k} rating prompts; got {len(rating_templates)}."
            )
        n_per_sample = k
        first_k = tuple(rating_templates[:k])
        templates_per_sample = [first_k for _ in range(n_samples)]
    else:
        raise ValueError(f"level must be 'token' or 'sentence', got {level!r}")

    # Flatten all (sample × template) prompts so we can batch-forward them
    # contiguously. We re-aggregate per sample at the end.
    all_prompts: List[str] = []
    for i in range(n_samples):
        for tpl in templates_per_sample[i]:
            all_prompts.append(build_rating_prompt(tpl, instructions[i], responses[i]))
    total_prompts = len(all_prompts)

    # Tokenizer state — restore on exit so we don't surprise callers that
    # share this tokenizer with other code paths.
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    orig_pad_token = tokenizer.pad_token
    tokenizer.padding_side = "right"   # need attention_mask.sum(-1)-1 = last
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_was_training = model.training
    model.eval()
    try:
        per_prompt_scores: List[float] = []
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            batch_prompts = all_prompts[batch_start:batch_end]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**enc)
            # Last non-pad position per row (right padding → sum-1).
            last_idx = enc["attention_mask"].sum(dim=1).clamp_min(1) - 1
            B = out.logits.size(0)
            bidx = torch.arange(B, device=device)
            last_logits = out.logits[bidx, last_idx, :].float()  # (B, V)
            sm = torch.softmax(last_logits, dim=-1)              # (B, V)
            probs5_batch = sm[:, rating_ids_t].cpu().numpy()     # (B, 5)
            for j in range(B):
                p5 = _double_softmax(probs5_batch[j])
                per_prompt_scores.append(_token_score_from_probs(p5, k=k))

            # Log roughly every `log_every` prompts (rounded to batch boundary).
            if (batch_end // log_every) > (batch_start // log_every):
                logger.info(
                    "SelectIT batched %d / %d (last=%.4f)",
                    batch_end, total_prompts, per_prompt_scores[-1],
                )

        # Aggregate per sample. Token-level: 1 template/sample → first score.
        # Sentence-level: k templates → avg / (1 + alpha · std) (paper Eq.4).
        scores: List[float] = []
        for i in range(n_samples):
            s = per_prompt_scores[i * n_per_sample : (i + 1) * n_per_sample]
            if level == "token":
                scores.append(s[0])
            else:
                avg = float(np.mean(s))
                std = float(np.std(s))
                scores.append(avg / (1.0 + alpha * std))
        return scores
    finally:
        if model_was_training:
            model.train()
        tokenizer.padding_side = orig_padding_side
        tokenizer.pad_token = orig_pad_token


def select_top_proportion(scores: Sequence[float], proportion: float) -> List[int]:
    """Return indices of the top-`proportion` items by score (descending)."""
    n = len(scores)
    k = max(1, int(n * proportion))
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return order[:k].tolist()
