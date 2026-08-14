"""Benchmark evaluators (registry-based dispatch).

Importing this package eagerly imports each benchmark module so that
``@register(name)`` decorators populate the registry. New benchmarks
need only add a file under ``tag/evals/`` and decorate their class.
"""
from .base import BenchmarkEvaluator, register, get_evaluator, list_evaluators

# Eager imports populate the registry.
from . import (  # noqa: F401
    mmlu,
    mmlu_pro,
    gsm8k,
    svamp,
    humaneval,
    mbpp,
    tydiqa,
    xquad,
    bbh,
    lm_harness,
)

__all__ = [
    "BenchmarkEvaluator",
    "register",
    "get_evaluator",
    "list_evaluators",
]
