"""Benchmark evaluator base class + registry.

Add a new benchmark in three steps:
    1. Create ``tads/evals/<name>.py`` with a class extending
       :class:`BenchmarkEvaluator`.
    2. Decorate the class with ``@register("<name>")``.
    3. Add the new module name to the eager-import list in
       ``tads/evals/__init__.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


_REGISTRY: Dict[str, Type["BenchmarkEvaluator"]] = {}


def register(name: str):
    """Class decorator: register an evaluator under ``name``."""
    def deco(cls: Type["BenchmarkEvaluator"]):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_evaluator(name: str) -> "BenchmarkEvaluator":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark: {name!r}. Registered: {sorted(_REGISTRY)}",
        )
    return _REGISTRY[name]()


def list_evaluators() -> list:
    return sorted(_REGISTRY)


class BenchmarkEvaluator(ABC):
    """Common interface for benchmark evaluators."""

    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",
        data_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run evaluation; return a summary dict with at least a headline metric."""
