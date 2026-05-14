"""Benchmark evaluators (registry-based dispatch).

Importing this package eagerly imports each benchmark module so that
``@register(name)`` decorators populate the registry. New benchmarks
need only add a file under ``tads/evals/`` and decorate their class.
"""
from .base import BenchmarkEvaluator, register, get_evaluator, list_evaluators

# Eager imports populate the registry.
from . import mmlu, gsm8k, humaneval, tydiqa, bbh, lm_harness  # noqa: F401

__all__ = [
    "BenchmarkEvaluator",
    "register",
    "get_evaluator",
    "list_evaluators",
]
