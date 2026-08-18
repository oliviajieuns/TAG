"""Batched greedy decoding must equal one-at-a-time decoding.

The evaluators went from one ``model.generate`` per example to batches of
16 (tag/evals/_gen.py). That is a >5x change in eval wall-clock and a 0x
change in what is computed — provided the padding is on the LEFT and is
masked. If it is not, every benchmark number silently shifts and the shift
is invisible in the summary JSON.

These tests pin the two properties a table row depends on:

  * the batched continuations are token-identical to the unbatched ones,
  * and they come back in the caller's order, not the internal
    longest-first order the batcher uses.

A character-level stub tokenizer keeps this offline and fast; the model is
a real (tiny, random) GPT-2, so the padding/attention-mask path under test
is the production one.
"""
from __future__ import annotations

import torch
import pytest
import transformers

from tag.evals._gen import generate_texts


_CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?\n")
_PAD, _EOS = len(_CHARS), len(_CHARS) + 1
_VOCAB = _EOS + 1


class _CharTokenizer:
    """The slice of the tokenizer API ``generate_texts`` actually uses."""

    padding_side = "right"
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = _PAD
    eos_token_id = _EOS

    def __call__(self, text, return_tensors=None, padding=False,
                 truncation=False, max_length=None, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [[_CHARS.index(c) for c in t if c in _CHARS] for t in texts]
        if truncation and max_length:
            # Match HF: keep the tail-most `max_length` tokens only when the
            # tokenizer says so; default HF drops from the right.
            ids = [x[:max_length] for x in ids]
        if return_tensors is None:
            return {"input_ids": ids[0] if isinstance(text, str) else ids}
        width = max(len(x) for x in ids) if padding else len(ids[0])
        out, mask = [], []
        for x in ids:
            pad = width - len(x)
            if self.padding_side == "left":
                out.append([_PAD] * pad + x)
                mask.append([0] * pad + [1] * len(x))
            else:
                out.append(x + [_PAD] * pad)
                mask.append([1] * len(x) + [0] * pad)
        return {
            "input_ids": torch.tensor(out),
            "attention_mask": torch.tensor(mask),
        }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            _CHARS[int(i)] for i in ids
            if not (skip_special_tokens and int(i) in (_PAD, _EOS))
        )


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(7)
    cfg = transformers.GPT2Config(
        vocab_size=_VOCAB, n_positions=128, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=_EOS, eos_token_id=_EOS,
    )
    model = transformers.GPT2LMHeadModel(cfg).eval()
    return model, _CharTokenizer()


_PROMPTS = [
    "the quick brown fox",
    "a",
    "hello there, how are you doing today?",
    "some medium length prompt here",
    "why",
    "the rain in spain falls mainly on the plain, they say",
    "no",
    "counting one two three four five six seven",
    "batching must not change the answer",
]


def _single(model, tok, prompts, n_new):
    """One prompt per generate() — the path the evaluators used to take."""
    outs = []
    for p in prompts:
        enc = tok([p], return_tensors="pt")
        out = model.generate(
            **enc, max_new_tokens=n_new, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        outs.append(tok.decode(out[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    return outs


def test_batched_matches_unbatched(tiny, monkeypatch):
    model, tok = tiny
    monkeypatch.setenv("TAG_EVAL_GEN_VERIFY", "0")
    ref = _single(model, tok, _PROMPTS, 12)
    got = generate_texts(
        model, tok, _PROMPTS, device="cpu",
        max_new_tokens=12, max_input_tokens=64, batch_size=4,
    )
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        assert a == b, f"prompt {i}: batched {a!r} != unbatched {b!r}"


def test_results_come_back_in_caller_order(tiny, monkeypatch):
    """The batcher sorts longest-first internally; callers zip the result
    against their own example list, so the mapping must be restored."""
    model, tok = tiny
    monkeypatch.setenv("TAG_EVAL_GEN_VERIFY", "0")
    per_prompt = {
        p: generate_texts(model, tok, [p], device="cpu", max_new_tokens=8,
                          max_input_tokens=64, batch_size=1)[0]
        for p in _PROMPTS
    }
    got = generate_texts(
        model, tok, _PROMPTS, device="cpu", max_new_tokens=8,
        max_input_tokens=64, batch_size=5,
    )
    for p, g in zip(_PROMPTS, got):
        assert g == per_prompt[p]


def test_batch_size_one_is_the_old_path(tiny, monkeypatch):
    model, tok = tiny
    monkeypatch.setenv("TAG_EVAL_GEN_VERIFY", "0")
    ref = _single(model, tok, _PROMPTS[:4], 10)
    got = generate_texts(model, tok, _PROMPTS[:4], device="cpu",
                         max_new_tokens=10, max_input_tokens=64, batch_size=1)
    assert got == ref


def test_env_override(monkeypatch):
    from tag.evals import _gen
    monkeypatch.setenv("TAG_EVAL_GEN_BS", "7")
    assert _gen.gen_batch_size() == 7
    monkeypatch.setenv("TAG_EVAL_GEN_BS", "nonsense")
    assert _gen.gen_batch_size(16) == 16
    monkeypatch.delenv("TAG_EVAL_GEN_BS")
    assert _gen.gen_batch_size(16) == 16


def test_empty_prompts_is_empty(tiny):
    model, tok = tiny
    assert generate_texts(model, tok, [], device="cpu", max_new_tokens=4,
                          max_input_tokens=64) == []
