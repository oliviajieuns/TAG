"""LoRA configuration helpers.

YAML ``lora:`` section → ``peft.LoraConfig``. Accepts both ``alpha`` and
``lora_alpha`` keys; both ``dropout`` and ``lora_dropout``; ``r`` is required.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from peft import LoraConfig, TaskType


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_lora_config(
    lora_cfg: Optional[Dict[str, Any]],
    *,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Build a ``LoraConfig`` from a YAML-shaped dict."""
    cfg = dict(lora_cfg or {})
    r = int(cfg.get("r", 16))
    alpha = int(cfg.get("alpha", cfg.get("lora_alpha", 32)))
    dropout = float(cfg.get("dropout", cfg.get("lora_dropout", 0.05)))
    target_modules = cfg.get("target_modules", DEFAULT_TARGET_MODULES)
    bias = cfg.get("bias", "none")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias=bias,
        task_type=task_type,
    )
