"""Config loader: defaults chain, deep-merge, env interpolation."""
from __future__ import annotations

import os
from pathlib import Path

from tag.core.utils import load_config


def _write(p: Path, content: str) -> Path:
    p.write_text(content)
    return p


def test_load_config_defaults_chain(tmp_path: Path):
    a = _write(tmp_path / "a.yaml", "x: 1\nseed: 42\nlora:\n  r: 8\n")
    b = _write(
        tmp_path / "b.yaml",
        f"defaults: a.yaml\nseed: 100\nlora:\n  alpha: 16\n",
    )
    cfg = load_config(str(b))
    assert cfg["x"] == 1            # inherited
    assert cfg["seed"] == 100       # overridden
    assert cfg["lora"]["r"] == 8    # inherited (deep merge)
    assert cfg["lora"]["alpha"] == 16


def test_load_config_list_defaults(tmp_path: Path):
    a = _write(tmp_path / "a.yaml", "common: 1\nshared: A\n")
    b = _write(tmp_path / "b.yaml", "shared: B\n")
    c = _write(
        tmp_path / "c.yaml",
        "defaults:\n  - a.yaml\n  - b.yaml\nlocal: y\n",
    )
    cfg = load_config(str(c))
    assert cfg["common"] == 1
    assert cfg["shared"] == "B"   # b.yaml wins over a.yaml
    assert cfg["local"] == "y"


def test_env_interpolation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MY_VAR", "/from/env")
    f = _write(
        tmp_path / "cfg.yaml",
        "model_path: ${oc.env:MY_VAR,/default}\n"
        "fallback_path: ${oc.env:NOPE,/fallback}\n",
    )
    cfg = load_config(str(f))
    assert cfg["model_path"] == "/from/env"
    assert cfg["fallback_path"] == "/fallback"
