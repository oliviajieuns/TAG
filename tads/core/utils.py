"""Utility functions: seeding, logging, CUDA memory, DDP helpers, config loading.

`load_config` supports multi-level inheritance via a list-form `defaults:` key
and `${oc.env:VAR,default}` environment variable interpolation in string values.
"""
from __future__ import annotations

import logging
import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------- warning dedup
class _DedupLogFilter(logging.Filter):
    """logging.Filter that lets each unique (logger, level, message) through once.

    Some libraries (HF transformers, datasets, peft) emit the same warning
    once per call inside the data-loader inner loop. This filter collapses
    those to a single line per unique message per process.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[Tuple[str, int, str]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        # Only deduplicate warnings/info — never suppress errors.
        if record.levelno >= logging.ERROR:
            return True
        key = (record.name, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def quiet_repeated_warnings(
    logger_names: Tuple[str, ...] = (
        "transformers",
        "transformers.tokenization_utils_base",
        "datasets",
        "peft",
        "accelerate",
        "py.warnings",
    ),
) -> None:
    """Show each unique warning at most once per process.

    - Attaches a dedup filter to common noisy loggers.
    - Routes `warnings.warn(...)` through the `py.warnings` logger so the
      same filter applies, and sets the stdlib `warnings` filter to
      "default" (one entry per (category, module, lineno)).
    """
    f = _DedupLogFilter()
    for name in logger_names:
        logging.getLogger(name).addFilter(f)

    # Bridge stdlib warnings → logging, so the dedup filter catches them.
    logging.captureWarnings(True)
    warnings.simplefilter("default")


# ----------------------------------------------------------------- coredumps
def disable_coredumps() -> None:
    """Set RLIMIT_CORE to (0, 0) on the current process and its children.

    Why this lives in Python (not just scripts/setup_env.sh's ``ulimit -c 0``):
    a single 7B-DDP rank that segfaults dumps its full virtual address space
    — model weights + bnb 8-bit optimiser state + grad buffers + CUDA-mapped
    VRAM — and ends up around 240 GB per rank. With 4 ranks that's ~1 TB
    landing on the 50 GB user-volume, which then ENOSPC's everything else.

    The shell-level ``ulimit -c 0`` works ONLY if the user actually sourced
    setup_env.sh in the launching shell. Real cluster launches (cron jobs,
    tmux sessions reopened later, `python -m tads.train` started from a
    fresh login) frequently skip that. Enforce the limit from inside the
    Python entry point so it can't be bypassed.

    Opt out (debugging) by exporting ``TADS_ENABLE_COREDUMPS=1`` before
    launching — matches the setup_env.sh contract.

    Cross-platform: ``resource`` is POSIX-only. No-op on Windows.
    """
    if os.environ.get("TADS_ENABLE_COREDUMPS", "0") == "1":
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        # ImportError: Windows. ValueError/OSError: hardened systems where
        # the soft limit can't be lowered (rare). Either way the worst case
        # is the legacy shell-level ulimit takes over.
        pass


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
    # `force=True` swaps the root logger's handler list but does NOT call
    # .close() on the old ones — repeat calls (back-to-back eval invocations
    # in the same process, pytest sessions) leak FileHandler fds and keep
    # writing to stale log files. Close + clear first so each setup_logger
    # call is a clean re-config.
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
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


# Project root = parent of the `tads/` package directory.
# utils.py lives at <root>/tads/core/utils.py, so go up three levels.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _collect_path_candidates(ref: str, anchor: Optional[Path] = None) -> List[Path]:
    """Build the ordered list of locations to try for a relative path `ref`.

    Order:
      1. Absolute (if `ref` is already absolute).
      2. Current working directory.
      3. The tads/ package root (so users can `python -m tads.train` from
         ANY directory — including their home — without ``cd``-ing first).
      4. Walking up from the anchor file's directory toward `/` (so a
         ``defaults: configs/base.yaml`` inside a deeply nested experiment
         YAML can still locate the top-level configs/ folder).
    """
    p = Path(ref)
    out: List[Path] = []
    if p.is_absolute():
        out.append(p)
        return out

    out.append(Path.cwd() / ref)
    out.append(_PROJECT_ROOT / ref)
    if anchor is not None:
        cur = anchor.parent if anchor.is_file() else anchor
        # Cap the walk-up depth so we never traverse the entire filesystem.
        for _ in range(10):
            out.append(cur / ref)
            if cur.parent == cur:
                break
            cur = cur.parent
    return out


def _find_existing(ref: str, anchor: Optional[Path] = None) -> Optional[Path]:
    for c in _collect_path_candidates(ref, anchor):
        if c.exists():
            return c
    return None


def _resolve_path(ref: str, anchor: Path) -> Path:
    """Find an existing path for ``ref`` (raises FileNotFoundError otherwise)."""
    found = _find_existing(ref, anchor=anchor)
    if found is not None:
        return found
    tried = _collect_path_candidates(ref, anchor=anchor)
    raise FileNotFoundError(
        f"Config defaults reference {ref!r} could not be resolved.\n"
        f"  cwd          = {Path.cwd()}\n"
        f"  project_root = {_PROJECT_ROOT}\n"
        f"  tried        = {[str(c) for c in tried]}"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML with support for ``defaults:`` (str or list) and env interp.

    A ``defaults:`` value can be:
        - a single string (path to one parent YAML), or
        - a list of strings (multiple parents, merged in order; later wins).
    Local keys (alongside ``defaults:``) override the merged parents.
    """
    # Resolve the entry-point YAML by trying cwd, project root, and (if it
    # looks like it might be relative to another loaded yaml) walking up.
    resolved = _find_existing(config_path)
    if resolved is None:
        tried = _collect_path_candidates(config_path)
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"  cwd          = {Path.cwd()}\n"
            f"  project_root = {_PROJECT_ROOT}\n"
            f"  tried        = {[str(c) for c in tried]}"
        )
    cfg_path = resolved
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
