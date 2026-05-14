"""Utility functions: seeding, logging, CUDA memory, DDP helpers, config loading.

`load_config` supports multi-level inheritance via a list-form `defaults:` key
and `${oc.env:VAR,default}` environment variable interpolation in string values.
"""
from __future__ import annotations

import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------------- seed
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (CUDA) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- CUDA
def cuda_mem_str() -> str:
    """Return human-readable CUDA memory string."""
    if not torch.cuda.is_available():
        return "mem=N/A"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"mem={allocated:.2f}/{reserved:.2f}GB"


# ----------------------------------------------------------------------- DDP
def is_main_process() -> bool:
    """True if not running under torch.distributed, or rank 0 if we are."""
    return not dist.is_initialized() or dist.get_rank() == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# --------------------------------------------------------------------- logger
def setup_logger(
    log_dir: str,
    name: str = "tads",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a file + stderr logger. Safe under DDP (rank-0 file write)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_r{rank()}" if dist.is_initialized() else ""
    log_path = Path(log_dir) / f"{name}_{ts}{suffix}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(name)


# --------------------------------------------------------------------- backup
def backup_latest_if_exists(output_dir: str) -> None:
    """Copy latest.pt / agent.pt / metrics.json into backup_<ts>/ if present."""
    latest = Path(output_dir) / "latest.pt"
    if not latest.exists():
        return
    import shutil
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(output_dir) / f"backup_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("latest.pt", "agent.pt", "metrics.json"):
        src = Path(output_dir) / name
        if src.exists():
            shutil.copy(src, backup_dir / name)
    logging.getLogger(__name__).warning(
        "Previous checkpoint backed up to %s", backup_dir,
    )


# --------------------------------------------------------------------- config
_ENV_PAT = re.compile(r"\$\{oc\.env:([A-Z_][A-Z0-9_]*)(?:,([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    """Replace ${oc.env:VAR,default} placeholders inside strings, recursively."""
    if isinstance(value, str):
        return _ENV_PAT.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge: override wins, nested dicts merged in place."""
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(ref: str, anchor: Path) -> Path:
    """Find an existing path for ``ref`` relative to a few candidate roots."""
    p = Path(ref)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([
        Path.cwd() / ref,
        anchor.parent / ref,
        anchor.parent.parent / ref,
        anchor.parent.parent.parent / ref,
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Config defaults reference {ref!r} could not be resolved. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML with support for ``defaults:`` (str or list) and env interp.

    A ``defaults:`` value can be:
        - a single string (path to one parent YAML), or
        - a list of strings (multiple parents, merged in order; later wins).
    Local keys (alongside ``defaults:``) override the merged parents.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    with open(cfg_path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return _resolve_env(cfg)

    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise TypeError(
            f"`defaults` must be a string or list of strings; got {type(defaults).__name__}",
        )

    merged: Dict[str, Any] = {}
    for ref in defaults:
        parent_path = _resolve_path(ref, cfg_path)
        parent_cfg = load_config(str(parent_path))
        merged = _deep_merge(merged, parent_cfg)

    merged = _deep_merge(merged, cfg)
    return _resolve_env(merged)
