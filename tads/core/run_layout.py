"""Run-directory layout for history-preserving training.

Each invocation of ``tads.train`` writes its checkpoints under

    <output_dir>/runs/<run_tag>/

where ``output_dir`` is the per-experiment dir (``OUTPUT_ROOT/output_subdir``)
and ``run_tag`` is either user-provided (``--run_tag``) or an auto timestamp
``YYYYMMDD_HHMMSS``. This keeps every prior training intact while making the
most recent one easy to find via the ``_latest`` symlink:

    <output_dir>/_latest -> runs/<run_tag>

Each run dir also contains a ``cfg.yaml`` snapshot of the resolved
hyperparameters that produced its checkpoints, so a later eval / audit knows
exactly which knobs were used. The same ``_complete`` sentinel mechanic from
the previous flat layout is kept, scoped to the run dir, so resume picks the
last successfully-saved epoch *within the current run* (not across runs —
crossing runs would mix hyperparameter regimes silently).

The ``_latest`` pointer is a symlink when the filesystem accepts one, and
falls back to a ``_latest.txt`` file containing the run_tag string for SMB /
some Windows mounts. Either form is read transparently by ``resolve_latest``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# A run_tag becomes part of a directory name and (possibly) a symlink target,
# so we restrict it to a portable subset.
_RUN_TAG_PAT = re.compile(r"^[A-Za-z0-9._-]+$")


def make_run_tag(suffix: str = "") -> str:
    """Return ``YYYYMMDD_HHMMSS`` (optionally suffixed with ``_<suffix>``)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not suffix:
        return ts
    if not _RUN_TAG_PAT.match(suffix):
        raise ValueError(
            f"run_tag suffix must match {_RUN_TAG_PAT.pattern}; got {suffix!r}",
        )
    return f"{ts}_{suffix}"


def _runs_root(output_dir: Path) -> Path:
    return output_dir / "runs"


def run_dir_for(output_dir: Path, run_tag: str) -> Path:
    """``<output_dir>/runs/<run_tag>``."""
    if not _RUN_TAG_PAT.match(run_tag):
        raise ValueError(
            f"run_tag must match {_RUN_TAG_PAT.pattern}; got {run_tag!r}",
        )
    return _runs_root(output_dir) / run_tag


def latest_pointer(output_dir: Path) -> Path:
    """Path of the symlink (``_latest``) regardless of whether it currently exists."""
    return output_dir / "_latest"


def list_runs(output_dir: Path) -> List[Tuple[str, Path]]:
    """Return ``[(run_tag, path)]`` sorted ascending (oldest first)."""
    root = _runs_root(output_dir)
    if not root.is_dir():
        return []
    return sorted(
        ((p.name, p) for p in root.iterdir() if p.is_dir()),
        key=lambda x: x[0],
    )


def resolve_latest(output_dir: Path) -> Optional[Path]:
    """Return the ``run_dir`` the ``_latest`` pointer resolves to, or None.

    Tries the symlink first, then the ``_latest.txt`` fallback. Returns the
    *resolved* path, not the symlink itself.
    """
    link = latest_pointer(output_dir)
    if link.is_symlink():
        target = link.resolve()
        return target if target.exists() else None
    if link.is_dir():
        # Some filesystems may have materialised it as a real directory; treat
        # that as "this IS the latest run" only if it contains epoch_*/ .
        if any(link.glob("epoch_*")):
            return link
    txt = output_dir / "_latest.txt"
    if txt.is_file():
        try:
            tag = txt.read_text().strip()
        except OSError:
            return None
        if tag and _RUN_TAG_PAT.match(tag):
            rd = run_dir_for(output_dir, tag)
            return rd if rd.exists() else None
    return None


def update_latest(output_dir: Path, run_tag: str) -> str:
    """Atomically point ``_latest`` at ``runs/<run_tag>``.

    Returns ``"symlink"`` or ``"textfile"`` depending on which mechanism
    succeeded — useful for log messages.
    """
    target_rel = Path("runs") / run_tag
    link = latest_pointer(output_dir)
    tmp_link = output_dir / "_latest.tmp"

    # Clean any leftover tmp from a prior crashed attempt.
    if tmp_link.is_symlink() or tmp_link.exists():
        try:
            tmp_link.unlink()
        except OSError:
            pass

    try:
        os.symlink(target_rel, tmp_link)
        os.replace(tmp_link, link)  # atomic POSIX rename
        return "symlink"
    except (OSError, NotImplementedError):
        # Fall back to a plain text file — works on FSes that disallow symlinks.
        if tmp_link.is_symlink() or tmp_link.exists():
            try:
                tmp_link.unlink()
            except OSError:
                pass

    txt = output_dir / "_latest.txt"
    txt_tmp = output_dir / "_latest.txt.tmp"
    with open(txt_tmp, "w") as f:
        f.write(run_tag)
        f.flush()
        os.fsync(f.fileno())
    os.replace(txt_tmp, txt)
    return "textfile"


def save_cfg_snapshot(run_dir_path: Path, cfg: Dict[str, Any]) -> Path:
    """Persist the resolved cfg dict that produced this run's checkpoints.

    Written atomically (.tmp + rename + fsync). Re-running save with the same
    cfg is a no-op-ish overwrite. Saved as YAML for human readability and as
    JSON for tooling that doesn't want to depend on PyYAML.
    """
    run_dir_path.mkdir(parents=True, exist_ok=True)

    yaml_tmp = run_dir_path / "cfg.yaml.tmp"
    yaml_path = run_dir_path / "cfg.yaml"
    with open(yaml_tmp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(yaml_tmp, yaml_path)

    json_tmp = run_dir_path / "cfg.json.tmp"
    json_path = run_dir_path / "cfg.json"
    with open(json_tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(json_tmp, json_path)

    return yaml_path


def load_cfg_snapshot(run_dir_path: Path) -> Optional[Dict[str, Any]]:
    """Read back the snapshot saved by :func:`save_cfg_snapshot`, or None."""
    yaml_path = run_dir_path / "cfg.yaml"
    if yaml_path.is_file():
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    json_path = run_dir_path / "cfg.json"
    if json_path.is_file():
        with open(json_path) as f:
            return json.load(f)
    return None


def find_latest_complete_epoch(run_dir_path: Path) -> Tuple[int, Optional[Path]]:
    """Largest ``epoch_N`` under ``run_dir_path`` whose ``_complete`` sentinel
    is present. Returns ``(0, None)`` if nothing usable.

    A checkpoint is considered "complete" only when both:
      * ``_complete`` sentinel file exists (atomic-rename'd at the end of
        the save sequence)
      * ``config.json`` (full FT) or ``adapter_config.json`` (LoRA) exists

    so partial / mid-save dirs are skipped and the next run picks up
    cleanly from the last good epoch.
    """
    if not run_dir_path.exists():
        return 0, None
    epochs: List[Tuple[int, Path]] = []
    for p in run_dir_path.glob("epoch_*"):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.replace("epoch_", ""))
        except ValueError:
            continue
        if not (p / "_complete").exists():
            continue
        if (p / "config.json").exists() or (p / "adapter_config.json").exists():
            epochs.append((n, p))
    if not epochs:
        return 0, None
    epochs.sort()
    return epochs[-1]


def resolve_eval_ckpt(
    output_dir: Path,
    explicit_ckpt: Optional[str] = None,
    epoch: Optional[int] = None,
) -> Path:
    """Resolve the checkpoint dir an evaluator should load.

    Priority:
      1. ``explicit_ckpt`` if given AND it points at an existing dir (used
         as-is — supports legacy flat layouts and ad-hoc paths).
      2. If ``explicit_ckpt`` is the experiment ``output_dir`` itself, resolve
         to ``_latest/epoch_<epoch or max>``.
      3. Otherwise resolve to ``output_dir/_latest/epoch_<epoch or max>``.

    Raises ``FileNotFoundError`` with a precise message listing what was
    tried, so a missing ``_latest`` pointer doesn't surface as an opaque
    HF "config.json not found" deeper in the loader.
    """
    if explicit_ckpt:
        p = Path(explicit_ckpt)
        if p.is_dir() and (p / "config.json").exists() or (p / "adapter_config.json").exists():
            return p
        # Fall through if explicit_ckpt is actually the experiment dir.
        if p.is_dir() and (p / "_latest").exists() or (p / "_latest.txt").exists():
            output_dir = p
            explicit_ckpt = None
        elif not p.exists():
            raise FileNotFoundError(
                f"--ckpt {explicit_ckpt!r} does not exist. "
                f"Either pass an existing epoch dir, or pass the experiment "
                f"output dir and let it resolve via _latest."
            )

    latest_run = resolve_latest(output_dir)
    if latest_run is None:
        raise FileNotFoundError(
            f"No _latest pointer under {output_dir}. "
            f"Looked for {output_dir / '_latest'} (symlink/dir) and "
            f"{output_dir / '_latest.txt'}. Train first, or pass --ckpt explicitly."
        )

    if epoch is not None:
        target = latest_run / f"epoch_{epoch}"
        if not target.exists():
            raise FileNotFoundError(
                f"epoch_{epoch} not found in {latest_run}. "
                f"Available: {sorted(p.name for p in latest_run.glob('epoch_*'))}"
            )
        return target

    n, p = find_latest_complete_epoch(latest_run)
    if p is None:
        raise FileNotFoundError(
            f"No complete epoch_N/_complete checkpoint inside {latest_run}.",
        )
    return p
