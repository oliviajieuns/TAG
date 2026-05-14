"""Shared supervised fine-tuning prompts and assistant-only label masking.

Prompt styles are **per model family** (not one string for all non-Qwen models):

- ``llama_user_assistant`` / ``deepseek_user_assistant``: ``<|user|>`` … ``<|assistant|>``
- ``mistral_instruct``: ``<s>[INST] … [/INST] …</s>``
- ``qwen_chatml``: Qwen ChatML (``im_start`` / ``im_end``)
- ``alpaca_default``: Stanford Alpaca ``### Instruction:`` / ``### Response:`` template

The first four styles are model-family specific. ``alpaca_default`` is the
canonical Alpaca template; used when ``prompt_style`` is not set explicitly.
TyDiQA / HumanEval / GSM8K all support the four model-family styles.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IM_START = "<|im_" + "start|>"
IM_END = "<|im_" + "end|>"
ASSISTANT_TAG = "<|assistant|>"

# Recognised model_id → prompt_style mapping (extend as needed).
PROMPT_STYLE_BY_MODEL_ID: Dict[str, str] = {
    "llama2-7b":   "llama_user_assistant",
    "mistral-7b":  "mistral_instruct",
    "qwen2.5-7b":  "qwen_chatml",
    "qwen2.5-14b": "qwen_chatml",
    "qwen2.5-0.5b": "qwen_chatml",
    "deepseek-7b": "deepseek_user_assistant",
}


def prompt_style_for_model_id(model_id: str) -> str:
    if model_id not in PROMPT_STYLE_BY_MODEL_ID:
        raise KeyError(
            f"Unknown model_id={model_id!r}; add it to PROMPT_STYLE_BY_MODEL_ID "
            f"in tads/data/sft_prompts.py",
        )
    return PROMPT_STYLE_BY_MODEL_ID[model_id]


# =============================================================================
# Alpaca (training data)
# =============================================================================
ALPACA_PROMPT_PREFIX = (
    "Below is an instruction that describes a task"
    "{input_part}. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def alpaca_input_part(inp: str) -> str:
    """Return the optional ``, using the input below as context\\n### Input:\\n…``
    suffix for the Alpaca prompt prefix. Empty string if ``inp`` is blank."""
    inp = (inp or "").strip()
    if not inp:
        return ""
    return f", using the input below as context\n\n### Input:\n{inp}"


# Backwards-compatible alias (older code may import the underscored name).
_alpaca_input_part = alpaca_input_part


def alpaca_prompt_text(example: Dict[str, Any]) -> str:
    """Return only the *prompt* portion (everything up to and including
    ``### Response:\\n`` for ``alpaca_default``)."""
    return ALPACA_PROMPT_PREFIX.format(
        input_part=alpaca_input_part(example.get("input", "")),
        instruction=example["instruction"],
    )


# Prompt styles whose template *already* begins with the model's BOS token
# (so we must NOT let the tokenizer prepend another one).
_TEMPLATE_HAS_BOS = {"mistral_instruct"}


def tokenize_alpaca(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    *,
    prompt_style: str = "alpaca_default",
) -> Dict[str, Any]:
    """Tokenise an Alpaca example with separate prompt / response encoding.

    Returns dict with ``input_ids``, ``attention_mask``, ``labels``.

    Why separate encoding? Llama-2's SentencePiece is context-dependent —
    tokenising the full string and searching for a marker fails because the
    marker's token sequence isn't reproduced verbatim inside the longer
    string. Encoding prompt and response separately and concatenating is
    the Stanford Alpaca canonical approach and works for every tokenizer.
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )
    eos_id = tokenizer.eos_token_id

    instruction = example["instruction"]
    inp = example.get("input", "") or ""
    output = example["output"]

    if prompt_style == "alpaca_default":
        prompt_text = ALPACA_PROMPT_PREFIX.format(
            input_part=alpaca_input_part(inp),
            instruction=instruction,
        )
        response_text = output

    elif prompt_style == "qwen_chatml":
        user = instruction + (f"\n\n{inp}" if inp.strip() else "")
        prompt_text = f"{IM_START}user\n{user}\n{IM_END}\n{IM_START}assistant\n"
        response_text = output

    elif prompt_style == "mistral_instruct":
        user = instruction + (f"\n\n{inp}" if inp.strip() else "")
        prompt_text = f"<s>[INST] {user} [/INST] "
        response_text = output

    elif prompt_style in ("llama_user_assistant", "deepseek_user_assistant"):
        user = instruction + (f"\n\n{inp}" if inp.strip() else "")
        prompt_text = f"<|user|>\n{user}\n\n<|assistant|>\n"
        response_text = output

    else:
        raise ValueError(
            f"Unknown prompt_style={prompt_style!r}; expected one of "
            f"alpaca_default, qwen_chatml, mistral_instruct, "
            f"llama_user_assistant, deepseek_user_assistant",
        )

    # If the template already contains the model's BOS (e.g. Mistral's "<s>"),
    # disable add_special_tokens so we don't get a doubled BOS token.
    prompt_add_special = prompt_style not in _TEMPLATE_HAS_BOS
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=prompt_add_special, return_tensors=None,
    )["input_ids"]
    response_ids = tokenizer(
        response_text, add_special_tokens=False, return_tensors=None,
    )["input_ids"]
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]

    pad_len = max_seq_len - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids = input_ids + [pad_id] * pad_len
    labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =============================================================================
# GSM8K (evaluation prompts, optional training prompts)
# =============================================================================
GSM8K_COT_8SHOT: List[tuple] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]


def build_cot_prompt_prefix(question: str) -> str:
    """8-shot CoT prefix terminating with ``Q: <question>\\n\\n A: ``."""
    parts: List[str] = []
    for q, a in GSM8K_COT_8SHOT:
        parts.append(f"Q: {q}\n\n A: {a}\n\n")
    parts.append(f"Q: {question}\n\n A: ")
    return "".join(parts)


# =============================================================================
# TyDiQA (evaluation prompts)
# =============================================================================
def tydiqa_user_block(
    context: str,
    question: str,
    *,
    demos: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    """Build the user-side text for a TyDiQA query.

    When ``demos`` is provided, the prefix becomes a flat 5-shot prompt:
        Answer ...
        Context: <c1>  Question: <q1>  Answer: <a1>
        Context: <c2>  Question: <q2>  Answer: <a2>
        ...
        Context: <c_test>  Question: <q_test>  Answer:
    Matches the standard lm-eval-harness layout for SQuAD-style extractive QA.
    """
    ctx = (context or "").strip()
    head = "Answer using the passage when possible. Reply with a short extractive span."
    if not demos:
        return f"{head}\n\nContext:\n{ctx}\n\nQuestion:\n{question}"
    chunks = [head + "\n\n"]
    for d_ctx, d_q, d_a in demos:
        chunks.append(
            f"Context:\n{(d_ctx or '').strip()}\n\n"
            f"Question:\n{d_q}\n\n"
            f"Answer: {d_a}\n\n"
        )
    chunks.append(
        f"Context:\n{ctx}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    return "".join(chunks)


def tydiqa_generation_prefix(
    context: str,
    question: str,
    *,
    prompt_style: str,
    demos: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    """TyDiQA generation prefix with optional 5-shot demonstrations.

    ``demos`` is an iterable of ``(context, question, answer)`` triples — when
    provided the evaluator becomes paper-faithful 5-shot (NAIT Appendix D).
    """
    user = tydiqa_user_block(context, question, demos=demos)
    if prompt_style == "qwen_chatml":
        return f"{IM_START}user\n{user}\n{IM_END}\n{IM_START}assistant\n"
    if prompt_style == "mistral_instruct":
        return f"<s>[INST] {user} [/INST] "
    if prompt_style in ("llama_user_assistant", "deepseek_user_assistant", "alpaca_default"):
        return f"<|user|>\n{user}\n<|assistant|>\n"
    raise ValueError(f"Unknown prompt_style={prompt_style!r} for TyDiQA eval prefix")


# =============================================================================
# HumanEval (evaluation prompts)
# =============================================================================
def humaneval_generation_prefix(
    code_prompt: str, *, prompt_style: str = "alpaca_default",
) -> str:
    """Build the generation prefix for HumanEval — raw code prompt for the
    code-completion task. For chat-style prompts we wrap minimally."""
    if prompt_style == "alpaca_default":
        # Most code-completion benches expect the raw function signature
        # as the prefix, so return it as-is.
        return code_prompt
    user = f"Complete the following Python function:\n\n{code_prompt}"
    if prompt_style == "qwen_chatml":
        return f"{IM_START}user\n{user}\n{IM_END}\n{IM_START}assistant\n"
    if prompt_style == "mistral_instruct":
        return f"<s>[INST] {user} [/INST] "
    if prompt_style in ("llama_user_assistant", "deepseek_user_assistant"):
        return f"<|user|>\n{user}\n<|assistant|>\n"
    raise ValueError(f"Unknown prompt_style={prompt_style!r} for HumanEval")
