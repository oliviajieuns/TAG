"""Config-resolution regression tests for the lowq 0.5B experiment arms.

Plan §5.2 (CIKM lesson §2.3): every lowq arm must inherit its training
hyperparameters from configs/experiments/lowq/_shared_light_05b.yaml and
differ ONLY in the selection method and its score parameters — the previous
paper died partly because one arm silently ran with its own optimizer
settings. These tests pin the RESOLVED configs (through
tads.core.utils.load_config, the exact loader tads.train uses), so a
defaults-order mistake — e.g. listing the shared fragment before the method
fragment, which silently resolved episode_batch_size to the 7B-scale 16
(~8x activation memory on the light GPU class, adversarial review 2026-08)
— fails here instead of on the GPU.

Also guards the ${oc.env:TADS_RELIABILITY_SCALE,} plumbing: with the env
var unset the placeholder resolves to the EMPTY STRING, and
tads.pipelines.selection._resolve_reliability_scale must treat that as
"unset" instead of calling float('') — which crashed every MVF run at
epoch-1 selection before the guard existed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tads.core.utils import load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOWQ_DIR = _REPO_ROOT / "configs" / "experiments" / "lowq"

_ARMS = [
    "light_random_05b",
    "light_full_polluted_05b",
    "light_oracle_clean_05b",
    "light_tads_legacy_05b",
    "light_tads_mvf_05b",
    "light_tads_mvf_static_05b",
]


def _load(name: str) -> dict:
    return load_config(str(_LOWQ_DIR / f"{name}.yaml"))


# --------------------------------------------------------------------------
# Shared training pins: identical across ALL arms, whatever the method
# fragment says (method FIRST, shared fragment LAST in every arm's defaults)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ARMS)
def test_shared_training_pins_win_over_method_fragments(name):
    cfg = _load(name)
    # The 7B method fragments carry episode_batch_size: 16 — the shared
    # light fragment's 2 must win in every arm.
    assert cfg["episode_batch_size"] == 2
    assert cfg["train_epochs"] == 5
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 2


# --------------------------------------------------------------------------
# Per-arm identity: the ONLY things allowed to differ between arms
# --------------------------------------------------------------------------

def test_random_arm_resolves_method_and_ratio():
    cfg = _load("light_random_05b")
    assert cfg["method"] == "random"
    assert cfg["selection_ratio"] == 0.1


def test_full_polluted_arm_trains_on_everything():
    cfg = _load("light_full_polluted_05b")
    assert cfg["method"] == "full"
    assert cfg["selection_ratio"] == 1.0


def test_oracle_arm_rescales_ratio_to_match_budget():
    cfg = _load("light_oracle_clean_05b")
    # K = 0.10 x N_original = 0.125 x N_clean at 20% corruption.
    assert cfg["selection_ratio"] == 0.125


def test_legacy_arm_stays_on_legacy_score():
    cfg = _load("light_tads_legacy_05b")
    assert cfg["method"] == "tads"
    # The legacy arm must never inherit the MVF score: score_mode is either
    # absent (legacy default) or explicitly "tads".
    assert cfg["tads"].get("score_mode", "tads") == "tads"


def test_mvf_arm_resolves_mvf_score_mode():
    cfg = _load("light_tads_mvf_05b")
    assert cfg["method"] == "tads"
    assert cfg["tads"]["score_mode"] == "mvf"
    assert cfg["selection_ratio"] == 0.1


def test_mvf_static_arm_freezes_selection():
    cfg = _load("light_tads_mvf_static_05b")
    assert cfg["tads"]["score_mode"] == "mvf"
    assert cfg["tads"]["mvf"]["static"] is True


# --------------------------------------------------------------------------
# TADS_RELIABILITY_SCALE plumbing: unset env var -> empty string in the
# resolved config -> None (never float('')) in the selection pipeline
# --------------------------------------------------------------------------

def test_reliability_scale_unset_env_resolves_empty_and_means_unset(monkeypatch):
    from tads.pipelines.selection import _resolve_reliability_scale

    monkeypatch.delenv("TADS_RELIABILITY_SCALE", raising=False)
    cfg = _load("light_tads_mvf_05b")
    scale = cfg["tads"]["mvf"]["reliability_scale"]
    # ${oc.env:TADS_RELIABILITY_SCALE,} resolves to the EMPTY STRING when
    # the env var is unset (utils._resolve_env) — NOT to None.
    assert scale == ""
    # ...and the pipeline must treat that as "no explicit scale", not crash
    # on float('') (confirmed critical: this killed every MVF run once).
    assert _resolve_reliability_scale({"reliability_scale": ""}) is None
    assert _resolve_reliability_scale({"reliability_scale": "  "}) is None
    assert _resolve_reliability_scale({"reliability_scale": None}) is None


def test_reliability_scale_set_env_resolves_to_float(monkeypatch):
    from tads.pipelines.selection import _resolve_reliability_scale

    monkeypatch.setenv("TADS_RELIABILITY_SCALE", "0.25")
    cfg = _load("light_tads_mvf_05b")
    scale = cfg["tads"]["mvf"]["reliability_scale"]
    assert scale == "0.25"
    assert _resolve_reliability_scale({"reliability_scale": scale}) == pytest.approx(0.25)
