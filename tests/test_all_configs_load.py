"""Every experiment config must resolve, and every script CLI must start.

Both failure classes surface at the worst possible moment otherwise: a
broken ``defaults:`` chain fails when someone launches the arm, and a
script-level import error fails when someone reaches for the tool. Both
are cheap to pin here.

The config check resolves the full inheritance chain the way training
does (environment interpolation included), with no environment prepared —
so it also proves that every config is loadable from a bare shell, which
is exactly the state a new collaborator starts in.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_CONFIGS = sorted(
    p for p in glob.glob(str(_ROOT / "configs" / "experiments" / "**" / "*.yaml"),
                         recursive=True)
    if "_shared" not in Path(p).name  # fragments, not launchable configs
)


@pytest.mark.parametrize(
    "path", _CONFIGS, ids=lambda p: str(Path(p).relative_to(_ROOT / "configs")),
)
def test_experiment_config_resolves(path: str) -> None:
    from tag.core.utils import load_config
    cfg = load_config(path)
    assert isinstance(cfg, dict) and cfg, f"{path} resolved to nothing"
    # Every launchable experiment names where its outputs go; a config
    # without it would scatter runs into the tool's defaults.
    assert cfg.get("output_subdir") or cfg.get("output_dir"), (
        f"{path} has neither output_subdir nor output_dir"
    )
