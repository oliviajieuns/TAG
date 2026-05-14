"""Tokenize Alpaca examples; verify response masking for each prompt_style.

These tests don't load a real tokenizer; they use a tiny stub that exposes
the minimal interface tokenize_alpaca needs. The goal is to verify the
prompt/response split and label masking behaviour, not byte-level tokens.
"""
from __future__ import annotations

import pytest

from tads.data.sft_prompts import tokenize_alpaca


class _StubTokenizer:
    """Character-level tokenizer for unit tests. Adds <s> if add_special_tokens."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self._encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return self._encode(text, add_special_tokens=add_special_tokens)

    def _encode(self, text, *, add_special_tokens):
        # Map char → its ASCII ord (offset to avoid 0/1 collisions with pad/eos).
        ids = [(ord(c) + 8) for c in text]
        if add_special_tokens:
            ids = [2] + ids  # fake BOS
        return ids


EXAMPLE = {
    "instruction": "Sum two numbers.",
    "input": "1 and 2",
    "output": "3",
}


@pytest.mark.parametrize(
    "style",
    [
        "alpaca_default",
        "qwen_chatml",
        "mistral_instruct",
        "llama_user_assistant",
        "deepseek_user_assistant",
    ],
)
def test_tokenize_alpaca_masking(style):
    tok = _StubTokenizer()
    out = tokenize_alpaca(EXAMPLE, tok, max_seq_len=256, prompt_style=style)
    assert set(out) == {"input_ids", "attention_mask", "labels"}
    assert len(out["input_ids"]) == 256
    assert len(out["labels"]) == 256

    labels = out["labels"]
    # At least one response token survives masking.
    assert any(l != -100 for l in labels), f"{style}: no unmasked tokens"
    # The first non-pad token of the prompt must be masked.
    am = out["attention_mask"]
    first_real = am.index(1) if 1 in am else 0
    assert labels[first_real] == -100, f"{style}: first real token not masked"


def test_alpaca_default_prompt_response_split():
    tok = _StubTokenizer()
    out = tokenize_alpaca(EXAMPLE, tok, max_seq_len=512, prompt_style="alpaca_default")
    n_real = sum(out["attention_mask"])
    n_unmasked = sum(1 for l in out["labels"] if l != -100)
    # Response "3" + EOS = at least 2 real label tokens, fewer than the whole sequence.
    assert 1 <= n_unmasked < n_real
