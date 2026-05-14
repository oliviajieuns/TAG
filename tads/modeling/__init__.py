"""Tokenizer / model loading with full-FT or LoRA, single-GPU or DDP.

``build_lora_config`` is intentionally re-exported via lazy attribute access
so that importing this package does NOT pull in the ``peft`` library. Full-FT
runs therefore tolerate PEFT version mismatches.
"""
from .loader import (
    load_tokenizer,
    load_model,
    load_for_eval,
    get_hidden_size,
)

__all__ = [
    "load_tokenizer",
    "load_model",
    "load_for_eval",
    "get_hidden_size",
    "build_lora_config",
]


def __getattr__(name):
    if name == "build_lora_config":
        from .lora import build_lora_config
        return build_lora_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
