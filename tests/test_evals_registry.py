"""Evaluator registry tests."""
from __future__ import annotations

import pytest

from tag.evals import get_evaluator, list_evaluators
from tag.evals.mmlu import MMLUEvaluator


def test_registry_lists_known_benchmarks():
    names = list_evaluators()
    for expected in ("mmlu", "gsm8k", "humaneval", "tydiqa", "bbh", "lm_harness"):
        assert expected in names


def test_registry_returns_correct_class():
    ev = get_evaluator("mmlu")
    assert isinstance(ev, MMLUEvaluator)
    assert ev.name == "mmlu"


def test_registry_raises_on_unknown():
    with pytest.raises(KeyError):
        get_evaluator("nope")
