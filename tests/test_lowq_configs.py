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
    "light_tag_05b",
    "light_tag_static_05b",
]

# The 7B grid has its own shared fragment (full-FT, 3 epochs) — plan §5.3.
_ARMS_7B = [
    "tag_7b",
    "tag_static_7b",
    "tads_legacy_7b",
    "random_7b",
    "full_polluted_7b",
]


def _load(name: str) -> dict:
    return load_config(str(_LOWQ_DIR / f"{name}.yaml"))


# --------------------------------------------------------------------------
# Shared training pins: identical across ALL arms, whatever the method
# fragment says (method FIRST, shared fragment LAST in every arm's defaults)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ARMS)
def test_shared_training_pins_win_over_method_fragments(name, monkeypatch):
    monkeypatch.delenv("TADS_EPISODE_BS", raising=False)
    cfg = _load(name)
    # The 7B method fragments carry episode_batch_size: 16 — the shared
    # light fragment's default must win in every arm.
    #
    # int(): episode_batch_size comes through ${oc.env:TADS_EPISODE_BS,2}, and
    # the env interpolation is a string substitution, so the resolved value is
    # the STRING "2" when the var is unset. Every consumer wraps it in int()
    # (grep episode_batch_size), and this assertion pins that contract — the
    # same class of trap as the TADS_RELIABILITY_SCALE empty string.
    assert int(cfg["episode_batch_size"]) == 2
    assert cfg["train_epochs"] == 5
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 2


@pytest.mark.parametrize("name", _ARMS)
def test_episode_batch_size_env_override_applies_to_every_arm(name, monkeypatch):
    """The H100 speed knob must move ALL arms together — an arm that missed it
    would differ from its siblings in float reduction order, and the diff
    would be unattributable (plan §5.2)."""
    monkeypatch.setenv("TADS_EPISODE_BS", "64")
    cfg = _load(name)
    assert int(cfg["episode_batch_size"]) == 64


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


def test_tag_arm_resolves_tag_score_mode():
    cfg = _load("light_tag_05b")
    assert cfg["method"] == "tads"
    assert cfg["tads"]["score_mode"] == "tag"
    assert cfg["selection_ratio"] == 0.1
    # TAG keeps the legacy dynamic score intact — the anchor must stay on,
    # otherwise the arm silently degrades to s = G · R (paper Eq. 1 with
    # lam = 0) and the trajectory claim goes untested.
    assert cfg["tads"]["use_anchor"] is True
    assert cfg["tads"]["lam"] == 1.0


def test_tag_static_arm_freezes_selection():
    cfg = _load("light_tag_static_05b")
    assert cfg["tads"]["score_mode"] == "tag"
    assert cfg["tads"]["tag"]["static"] is True


def test_tag_arm_carries_span_and_gate_params():
    cfg = _load("light_tag_05b")
    tag = cfg["tads"]["tag"]
    assert tag["span_tokens"] == 16
    assert tag["tau_mode"] == "per_token"
    assert tag["tail_mode"] == "min"
    assert tag["include_eos"] is False


def test_tag_and_mvf_arms_do_not_share_a_score_block():
    """A TAG arm must not inherit MVF's parameters (or vice versa) — the two
    modes are mutually exclusive and their configs are separate subtrees."""
    tag_cfg = _load("light_tag_05b")
    mvf_cfg = _load("light_tads_mvf_05b")
    assert "mvf" not in tag_cfg["tads"]
    assert "tag" not in mvf_cfg["tads"]


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


def test_gate_scale_unset_env_resolves_empty_and_means_unset(monkeypatch):
    """Same empty-string trap as TADS_RELIABILITY_SCALE, new env var."""
    from tads.pipelines.selection import _resolve_gate_scale

    monkeypatch.delenv("TADS_GATE_SCALE", raising=False)
    monkeypatch.delenv("TADS_GATE_REF", raising=False)
    cfg = _load("light_tag_05b")
    assert cfg["tads"]["tag"]["gate_scale"] == ""
    assert cfg["tads"]["tag"]["gate_ref_file"] == ""
    assert _resolve_gate_scale({"gate_scale": ""}) is None
    assert _resolve_gate_scale({"gate_scale": "  "}) is None
    assert _resolve_gate_scale({"gate_scale": None}) is None
    # An empty gate_ref_file must not be treated as a path to torch.load.
    assert _resolve_gate_scale({"gate_scale": "", "gate_ref_file": ""}) is None


def test_gate_scale_set_env_resolves_to_float(monkeypatch):
    from tads.pipelines.selection import _resolve_gate_scale

    monkeypatch.setenv("TADS_GATE_SCALE", "0.15")
    cfg = _load("light_tag_05b")
    assert cfg["tads"]["tag"]["gate_scale"] == "0.15"
    assert _resolve_gate_scale(
        {"gate_scale": cfg["tads"]["tag"]["gate_scale"]}
    ) == pytest.approx(0.15)


# --------------------------------------------------------------------------
# Regression: an UNPINNED gate scale must survive to epoch 2
# --------------------------------------------------------------------------

def test_unpinned_gate_scale_is_adopted_from_the_cache(tmp_path, monkeypatch):
    """With no gate_scale/gate_ref_file (the shipped default) the scale is
    derived in-pool at epoch 1 and stored in the cache. At epoch 2 the
    requested scale is still None, so a naive identity comparison misses the
    cache, tries to recompute, and hits the base-checkpoint hard error —
    killing every default TAG run at epoch 2. The cached scale IS this run's
    base-checkpoint derivation and must be adopted."""
    import torch
    from tads.core.gate import GateConfig, save_gate_cache
    from tads.pipelines.selection import _prepare_tag

    n = 6
    cfg_used = GateConfig(span_tokens=4, scale=0.0123)
    result = {
        "gate": torch.full((n,), 0.5),
        "completeness": torch.ones(n),
        "delta_bar": torch.zeros(n), "delta_min": torch.zeros(n),
        "delta_hat": torch.zeros(n), "n_spans": torch.ones(n, dtype=torch.long),
        "n_common": torch.full((n,), 8, dtype=torch.long),
        "undefined": torch.zeros(n, dtype=torch.bool),
        "empty_c": torch.zeros(n, dtype=torch.bool),
    }
    save_gate_cache(tmp_path, result=result, cfg=cfg_used, epoch=1)

    monkeypatch.delenv("TADS_GATE_SCALE", raising=False)
    monkeypatch.delenv("TADS_GATE_REF", raising=False)
    tag_ctx = {
        "completeness": torch.ones(n),
        "cf_datasets": [],                    # must not be needed: cache hit
        "params": {"span_tokens": 4, "gate_scale": "", "gate_ref_file": ""},
    }
    tag = _prepare_tag(
        tag_ctx, model=None, cfg={"output_dir": str(tmp_path)},
        epoch=2, device="cpu", n_pool=n,
    )
    assert tag["gate"] is not None
    assert tag["gate_config"].scale == pytest.approx(0.0123)


def test_tag_arm_uses_neutral_undefined_policy():
    cfg = _load("light_tag_05b")
    assert cfg["tads"]["tag"]["undefined_policy"] == "neutral"
    assert cfg["tads"]["tag"]["undefined_gate_value"] == 0.6


def test_missing_gate_ref_file_is_an_actionable_error_not_a_silent_fallback(tmp_path):
    """A configured-but-absent gate_ref_file must fail loudly. Falling back to
    in-pool self-calibration would silently produce a reported run whose gate
    means something different from what the config says."""
    from tads.pipelines.selection import _resolve_gate_scale

    missing = tmp_path / "delta_hat_missing.pt"
    with pytest.raises(FileNotFoundError, match="calibrate_reliability.py --mode tag"):
        _resolve_gate_scale({"gate_scale": "", "gate_ref_file": str(missing)})


def test_mvf_reference_is_rejected_by_the_tag_gate(tmp_path):
    """The MVF artifact stores raw Delta_L in nats, the TAG one a ratio.
    Feeding the former to the gate would calibrate s an order of magnitude
    too high and effectively disable it, with no visible symptom."""
    import torch
    from tads.pipelines.selection import _resolve_gate_scale

    ref = tmp_path / "delta_mvf.pt"
    torch.save({"delta": torch.linspace(0.1, 2.0, 200)}, ref)
    with pytest.raises(ValueError, match="delta_hat"):
        _resolve_gate_scale({"gate_scale": "", "gate_ref_file": str(ref)})


def test_gate_ref_with_a_different_span_config_is_rejected(tmp_path):
    """s is a quantile of Delta_hat, whose distribution depends on the span
    partition — calibrating at W=16 and gating at W=32 mis-scales every gate
    value in the run."""
    import torch
    from tads.core.gate import GateConfig
    from tads.pipelines.selection import _resolve_gate_scale

    ref = tmp_path / "delta_hat.pt"
    torch.save(
        {
            "delta_hat": torch.linspace(0.05, 0.95, 200),
            "gate_config": GateConfig(span_tokens=16, scale=1.0).identity(),
        },
        ref,
    )
    # Same partition: accepted.
    assert _resolve_gate_scale(
        {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 16}
    ) > 0
    # Different W: rejected.
    with pytest.raises(ValueError, match="different span configuration"):
        _resolve_gate_scale(
            {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 32}
        )


# --------------------------------------------------------------------------
# 7B grid
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ARMS_7B)
def test_7b_arms_share_training_pins(name, monkeypatch):
    for v in ("TADS_EPISODE_BS_7B", "TADS_GRAD_ACCUM_7B"):
        monkeypatch.delenv(v, raising=False)
    cfg = _load(name)
    assert cfg["train_epochs"] == 3, "plan §5.3 pre-registers 3 epochs at 7B"
    assert cfg["training_mode"] == "full"
    # Without 8-bit AdamW the fp32 optimizer state alone (~56 GB) does not
    # fit beside 7B weights on an 80GB card.
    assert cfg["use_8bit_optimizer"] is True
    assert cfg["model_key"] == "qwen2.5-7b"
    # effective batch = batch_size x grad_accum x world_size, held at 128
    # with the default one-arm-per-GPU layout (world_size 1).
    assert int(cfg["batch_size"]) * int(cfg["grad_accum"]) == 128


@pytest.mark.parametrize("name", _ARMS_7B)
def test_7b_ddp_override_keeps_the_effective_batch(name, monkeypatch):
    """One arm across 4 DDP ranks must reach the same effective batch as one
    arm per GPU, or the two layouts are not comparable."""
    monkeypatch.setenv("TADS_GRAD_ACCUM_7B", "4")
    cfg = _load(name)
    assert int(cfg["batch_size"]) * int(cfg["grad_accum"]) * 4 == 128


def test_7b_tag_arm_resolves_tag_score_mode():
    cfg = _load("tag_7b")
    assert cfg["method"] == "tads"
    assert cfg["tads"]["score_mode"] == "tag"
    assert cfg["tads"]["use_anchor"] is True
    assert cfg["selection_ratio"] == 0.1


def test_7b_tag_arms_accept_a_shared_gate_cache(monkeypatch):
    """G depends only on (pool, base checkpoint, gate config), so every arm
    and seed must be able to point at ONE precomputed cache."""
    monkeypatch.setenv("TADS_GATE_CACHE", "/tmp/shared_gate.pt")
    for name in ("tag_7b", "tag_static_7b"):
        cfg = _load(name)
        assert cfg["tads"]["tag"]["gate_cache_file"] == "/tmp/shared_gate.pt"


def test_7b_legacy_arm_has_no_gate():
    """TAG - legacy is the pair that isolates the gate, so legacy must carry
    no tag block at all."""
    cfg = _load("tads_legacy_7b")
    assert cfg["tads"].get("score_mode", "tads") == "tads"
    assert "tag" not in cfg["tads"]


# --------------------------------------------------------------------------
# A missing model path must say so, not become an HF repo id
# --------------------------------------------------------------------------

def test_missing_local_model_path_is_an_actionable_error():
    """transformers interprets a nonexistent local path as a hub repo id and
    raises HFValidationError('Repo id must be in the form ...'), which names a
    path the user never chose and says nothing about the real cause — an
    unset MODEL_PATH_* leaving the config default in place."""
    from tads.modeling.loader import _resolve_local_path

    with pytest.raises(FileNotFoundError, match="MODEL_PATH_QWEN25_7B"):
        _resolve_local_path("/group-volume/nait-models/qwen2.5-7b")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_local_path("/definitely/not/here")


def test_bare_repo_id_is_not_treated_as_a_missing_path():
    """With local_files_only=True transformers can still resolve 'org/name'
    from the HF cache, so it must pass through untouched."""
    from tads.modeling.loader import _resolve_local_path

    assert _resolve_local_path("Qwen/Qwen2.5-7B") == "Qwen/Qwen2.5-7B"


def test_case_variant_resolution_still_works(tmp_path):
    from tads.modeling.loader import _resolve_local_path

    (tmp_path / "MyModel").mkdir()
    assert _resolve_local_path(str(tmp_path / "mymodel")) == str(tmp_path / "MyModel")
