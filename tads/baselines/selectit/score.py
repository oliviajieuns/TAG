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
    """Return the token ids for the bare digits "1".."5" using the model's
    own tokenizer. On LLaMA-2 SentencePiece this yields [29896, 29906, 29941,
    29946, 29945] — matching the hard-coded ids in the official repo.

    Raises if any digit is not a single token (some BPE tokenizers split it
    differently — those models need a per-tokenizer scoring strategy).
    """
    ids: List[int] = []
    for digit in ("1", "2", "3", "4", "5"):
        toks = tokenizer.encode(digit, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                f"Tokenizer split rating digit {digit!r} into {toks} — "
                f"SelectIT scoring expects single-token rating digits. "
                f"This tokenizer may need a custom rating-id resolver."
            )
        ids.append(int(toks[0]))
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
    """The paper's extra normalisation pass: `exp(p/sum(p))` then renormalise."""
    e = np.exp(probs5 / probs5.sum())
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
    log_every: int = 200,
) -> List[float]:
    """Compute per-sample SelectIT scores.

    Args:
        level: "token" — one rating template per sample (cycled through
            ``rating_templates``). "sentence" — k templates per sample,
            then averaged with std penalty per paper Eq.4.
        alpha: sentence-level std-penalty weight; ignored for token-level.
        max_length: tokenizer truncation cap.
    """
    if len(instructions) != len(responses):
        raise ValueError(
            f"instructions/responses length mismatch: {len(instructions)} vs {len(responses)}"
        )
    n_samples = len(instructions)
    rating_ids = resolve_rating_token_ids(tokenizer)
    k = len(rating_ids)  # = 5

    if level == "token":
        templates_per_sample = [
            (rating_templates[i % len(rating_templates)],) for i in range(n_samples)
        ]
    elif level == "sentence":
        if len(rating_templates) < k:
            raise ValueError(
                f"sentence-level needs at least {k} rating prompts; got {len(rating_templates)}."
            )
        first_k = tuple(rating_templates[:k])
        templates_per_sample = [first_k for _ in range(n_samples)]
    else:
        raise ValueError(f"level must be 'token' or 'sentence', got {level!r}")

    model_was_training = model.training
    model.eval()
    try:
        scores: List[float] = []
        for i in range(n_samples):
            per_template = []
            for tpl in templates_per_sample[i]:
                prompt = build_rating_prompt(tpl, instructions[i], responses[i])
                inp = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                out = model(**inp)
                # out.logits: (1, T, V)
                logits = out.logits[:, -1, :]
                sm = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                probs5 = np.asarray([sm[t] for t in rating_ids], dtype=np.float64)
                probs5 = _double_softmax(probs5)
                per_template.append(_token_score_from_probs(probs5, k=k))

            if level == "token":
                scores.append(per_template[0])
            else:
                avg = float(np.mean(per_template))
                std = float(np.std(per_template))
                scores.append(avg / (1.0 + alpha * std))

            if (i + 1) % log_every == 0:
                logger.info(
                    "SelectIT scoring %d / %d (last=%.4f)",
                    i + 1, n_samples, scores[-1],
                )
        return scores
    finally:
        if model_was_training:
            model.train()


def select_top_proportion(scores: Sequence[float], proportion: float) -> List[int]:
    """Return indices of the top-`proportion` items by score (descending)."""
    n = len(scores)
    k = max(1, int(n * proportion))
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return order[:k].tolist()
