"""Config-resolution regression tests for the lowq 0.5B experiment arms.

Plan §5.2 (CIKM lesson §2.3): every lowq arm must inherit its training
hyperparameters from configs/experiments/lowq/_shared_light_05b.yaml and
differ ONLY in the selection method and its score parameters — the previous
paper died partly because one arm silently ran with its own optimizer
settings. These tests pin the RESOLVED configs (through
tag.core.utils.load_config, the exact loader tag.train uses), so a
defaults-order mistake — e.g. listing the shared fragment before the method
fragment, which silently resolved episode_batch_size to the 7B-scale 16
(~8x activation memory on the light GPU class, adversarial review 2026-08)
— fails here instead of on the GPU.

Also guards the ${oc.env:TAG_RELIABILITY_SCALE,} plumbing: with the env
var unset the placeholder resolves to the EMPTY STRING, and
tag.pipelines.selection._resolve_reliability_scale must treat that as
"unset" instead of calling float('') — which crashed every MVF run at
epoch-1 selection before the guard existed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tag.core.utils import load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOWQ_DIR = _REPO_ROOT / "configs" / "experiments" / "lowq"

_ARMS = [
    "light_random_05b",
    "light_full_polluted_05b",
    "light_oracle_clean_05b",
    "light_legacy_05b",
    "light_mvf_05b",
    "light_mvf_static_05b",
    "light_tag_05b",
    "light_tag_static_05b",
]

# The 7B grid has its own shared fragment (full-FT, 3 epochs) — plan §5.3.
_ARMS_7B = [
    "tag_7b",
    "tag_static_7b",
    "legacy_7b",
    "random_7b",
    "full_polluted_7b",
    "tag_nonull_7b",
    "tag_bar_7b",
    "tag_prefix_7b",
]


def _load(name: str) -> dict:
    return load_config(str(_LOWQ_DIR / f"{name}.yaml"))


# --------------------------------------------------------------------------
# Shared training pins: identical across ALL arms, whatever the method
# fragment says (method FIRST, shared fragment LAST in every arm's defaults)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ARMS)
def test_shared_training_pins_win_over_method_fragments(name, monkeypatch):
    monkeypatch.delenv("TAG_EPISODE_BS", raising=False)
    cfg = _load(name)
    # The 7B method fragments carry episode_batch_size: 16 — the shared
    # light fragment's default must win in every arm.
    #
    # int(): episode_batch_size comes through ${oc.env:TAG_EPISODE_BS,2}, and
    # the env interpolation is a string substitution, so the resolved value is
    # the STRING "2" when the var is unset. Every consumer wraps it in int()
    # (grep episode_batch_size), and this assertion pins that contract — the
    # same class of trap as the TAG_RELIABILITY_SCALE empty string.
    assert int(cfg["episode_batch_size"]) == 2
    assert cfg["train_epochs"] == 5
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 2


@pytest.mark.parametrize("name", _ARMS)
def test_episode_batch_size_env_override_applies_to_every_arm(name, monkeypatch):
    """The H100 speed knob must move ALL arms together — an arm that missed it
    would differ from its siblings in float reduction order, and the diff
    would be unattributable (plan §5.2)."""
    monkeypatch.setenv("TAG_EPISODE_BS", "64")
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
    cfg = _load("light_legacy_05b")
    assert cfg["method"] == "selection"
    # The legacy arm must never inherit the MVF score: score_mode is either
    # absent (legacy default) or explicitly "legacy".
    assert cfg["selection"].get("score_mode", "legacy") == "legacy"


def test_mvf_arm_resolves_mvf_score_mode():
    cfg = _load("light_mvf_05b")
    assert cfg["method"] == "selection"
    assert cfg["selection"]["score_mode"] == "mvf"
    assert cfg["selection_ratio"] == 0.1


def test_mvf_static_arm_freezes_selection():
    cfg = _load("light_mvf_static_05b")
    assert cfg["selection"]["score_mode"] == "mvf"
    assert cfg["selection"]["mvf"]["static"] is True


def test_tag_arm_resolves_tag_score_mode():
    cfg = _load("light_tag_05b")
    assert cfg["method"] == "selection"
    assert cfg["selection"]["score_mode"] == "tag"
    assert cfg["selection_ratio"] == 0.1
    # TAG keeps the legacy dynamic score intact — the anchor must stay on,
    # otherwise the arm silently degrades to s = G · R (paper Eq. 1 with
    # lam = 0) and the trajectory claim goes untested.
    assert cfg["selection"]["use_anchor"] is True
    assert cfg["selection"]["lam"] == 1.0


def test_tag_static_arm_freezes_selection():
    cfg = _load("light_tag_static_05b")
    assert cfg["selection"]["score_mode"] == "tag"
    assert cfg["selection"]["tag"]["static"] is True


def test_tag_arm_carries_span_and_gate_params():
    cfg = _load("light_tag_05b")
    tag = cfg["selection"]["tag"]
    assert tag["span_tokens"] == 16
    assert tag["tau_mode"] == "per_token"
    assert tag["tail_mode"] == "min"
    assert tag["include_eos"] is False


def test_tag_and_mvf_arms_do_not_share_a_score_block():
    """A TAG arm must not inherit MVF's parameters (or vice versa) — the two
    modes are mutually exclusive and their configs are separate subtrees."""
    tag_cfg = _load("light_tag_05b")
    mvf_cfg = _load("light_mvf_05b")
    assert "mvf" not in tag_cfg["selection"]
    assert "tag" not in mvf_cfg["selection"]


# --------------------------------------------------------------------------
# TAG_RELIABILITY_SCALE plumbing: unset env var -> empty string in the
# resolved config -> None (never float('')) in the selection pipeline
# --------------------------------------------------------------------------

def test_reliability_scale_unset_env_resolves_empty_and_means_unset(monkeypatch):
    from tag.pipelines.selection import _resolve_reliability_scale

    monkeypatch.delenv("TAG_RELIABILITY_SCALE", raising=False)
    cfg = _load("light_mvf_05b")
    scale = cfg["selection"]["mvf"]["reliability_scale"]
    # ${oc.env:TAG_RELIABILITY_SCALE,} resolves to the EMPTY STRING when
    # the env var is unset (utils._resolve_env) — NOT to None.
    assert scale == ""
    # ...and the pipeline must treat that as "no explicit scale", not crash
    # on float('') (confirmed critical: this killed every MVF run once).
    assert _resolve_reliability_scale({"reliability_scale": ""}) is None
    assert _resolve_reliability_scale({"reliability_scale": "  "}) is None
    assert _resolve_reliability_scale({"reliability_scale": None}) is None


def test_reliability_scale_set_env_resolves_to_float(monkeypatch):
    from tag.pipelines.selection import _resolve_reliability_scale

    monkeypatch.setenv("TAG_RELIABILITY_SCALE", "0.25")
    cfg = _load("light_mvf_05b")
    scale = cfg["selection"]["mvf"]["reliability_scale"]
    assert scale == "0.25"
    assert _resolve_reliability_scale({"reliability_scale": scale}) == pytest.approx(0.25)


def test_gate_scale_unset_env_resolves_empty_and_means_unset(monkeypatch):
    """Same empty-string trap as TAG_RELIABILITY_SCALE, new env var."""
    from tag.pipelines.selection import _resolve_gate_scale

    monkeypatch.delenv("TAG_GATE_SCALE", raising=False)
    monkeypatch.delenv("TAG_GATE_REF", raising=False)
    cfg = _load("light_tag_05b")
    assert cfg["selection"]["tag"]["gate_scale"] == ""
    assert cfg["selection"]["tag"]["gate_ref_file"] == ""
    # null_correction off: this test is about the empty-string trap in the
    # SCALE, and with the correction on an absent reference is a hard error
    # by design (pinned separately below).
    off = {"null_correction": False}
    assert _resolve_gate_scale({"gate_scale": "", **off}) is None
    assert _resolve_gate_scale({"gate_scale": "  ", **off}) is None
    assert _resolve_gate_scale({"gate_scale": None, **off}) is None
    # An empty gate_ref_file must not be treated as a path to torch.load.
    assert _resolve_gate_scale({"gate_scale": "", "gate_ref_file": "", **off}) is None


def test_null_correction_without_a_reference_is_a_hard_error():
    """mu(M) is measured on a CLEAN pool; there is no in-pool fallback for it.

    Self-calibrating the null on the candidate pool would fit the curve to
    data that is 30% corrupted, so the correction would absorb exactly the
    signal the gate exists to find. Silently skipping it instead would give
    back the 60%-of-clean-at-zero behaviour under a config that claims otherwise.
    Neither is acceptable, so this fails loudly with the command that fixes it.
    """
    from tag.pipelines.selection import _resolve_gate_calibration

    with pytest.raises(FileNotFoundError, match="null_correction"):
        _resolve_gate_calibration({"gate_scale": "", "gate_ref_file": ""})
    # Even a PINNED scale does not stand in for the curve — they are separate
    # halves of the calibration.
    with pytest.raises(FileNotFoundError, match="null_correction"):
        _resolve_gate_calibration({"gate_scale": "0.2"})


def test_gate_scale_set_env_resolves_to_float(monkeypatch):
    from tag.pipelines.selection import _resolve_gate_scale

    monkeypatch.setenv("TAG_GATE_SCALE", "0.15")
    cfg = _load("light_tag_05b")
    assert cfg["selection"]["tag"]["gate_scale"] == "0.15"
    assert _resolve_gate_scale(
        {"gate_scale": cfg["selection"]["tag"]["gate_scale"], "null_correction": False}
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
    from tag.core.gate import GateConfig, save_gate_cache
    from tag.pipelines.selection import _prepare_tag

    n = 6
    cfg_used = GateConfig(span_tokens=4, scale=0.0123, null_correction=False)
    result = {
        "gate": torch.full((n,), 0.5),
        "completeness": torch.ones(n),
        "delta_bar": torch.zeros(n), "delta_min": torch.zeros(n),
        "delta_hat": torch.zeros(n), "delta_hat_raw": torch.zeros(n),
        "n_spans": torch.ones(n, dtype=torch.long),
        "n_common": torch.full((n,), 8, dtype=torch.long),
        "undefined": torch.zeros(n, dtype=torch.bool),
        "empty_c": torch.zeros(n, dtype=torch.bool),
    }
    save_gate_cache(tmp_path, result=result, cfg=cfg_used, epoch=1)

    monkeypatch.delenv("TAG_GATE_SCALE", raising=False)
    monkeypatch.delenv("TAG_GATE_REF", raising=False)
    tag_ctx = {
        "completeness": torch.ones(n),
        "cf_datasets": [],                    # must not be needed: cache hit
        "params": {"span_tokens": 4, "gate_scale": "", "gate_ref_file": "",
                   "null_correction": False},
    }
    tag = _prepare_tag(
        tag_ctx, model=None, cfg={"output_dir": str(tmp_path)},
        epoch=2, device="cpu", n_pool=n,
    )
    assert tag["gate"] is not None
    assert tag["gate_config"].scale == pytest.approx(0.0123)


def test_tag_arm_uses_neutral_undefined_policy():
    cfg = _load("light_tag_05b")
    assert cfg["selection"]["tag"]["undefined_policy"] == "neutral"
    assert cfg["selection"]["tag"]["undefined_gate_value"] == 0.6


def test_missing_gate_ref_file_is_an_actionable_error_not_a_silent_fallback(tmp_path):
    """A configured-but-absent gate_ref_file must fail loudly. Falling back to
    in-pool self-calibration would silently produce a reported run whose gate
    means something different from what the config says."""
    from tag.pipelines.selection import _resolve_gate_scale

    missing = tmp_path / "delta_hat_missing.pt"
    with pytest.raises(FileNotFoundError, match="calibrate_reliability.py --mode tag"):
        _resolve_gate_scale({"gate_scale": "", "gate_ref_file": str(missing)})


def test_mvf_reference_is_rejected_by_the_tag_gate(tmp_path):
    """The MVF artifact stores raw Delta_L in nats, the TAG one a ratio.
    Feeding the former to the gate would calibrate s an order of magnitude
    too high and effectively disable it, with no visible symptom."""
    import torch
    from tag.pipelines.selection import _resolve_gate_scale

    ref = tmp_path / "delta_mvf.pt"
    torch.save({"delta": torch.linspace(0.1, 2.0, 200)}, ref)
    with pytest.raises(ValueError, match="delta_hat"):
        _resolve_gate_scale({"gate_scale": "", "gate_ref_file": str(ref)})


def test_gate_ref_with_a_different_span_config_is_rejected(tmp_path):
    """s is a quantile of Delta_hat, whose distribution depends on the span
    partition — calibrating at W=16 and gating at W=32 mis-scales every gate
    value in the run."""
    import torch
    from tag.core.gate import GateConfig
    from tag.pipelines.selection import _resolve_gate_scale

    ref = tmp_path / "delta_hat.pt"
    torch.save(
        {
            "delta_hat": torch.linspace(0.05, 0.95, 200),
            "gate_config": GateConfig(
                span_tokens=16, scale=1.0, null_correction=False,
            ).identity(),
        },
        ref,
    )
    off = {"null_correction": False}
    # Same partition: accepted.
    assert _resolve_gate_scale(
        {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 16, **off}
    ) > 0
    # Different W: rejected.
    with pytest.raises(ValueError, match="different span configuration"):
        _resolve_gate_scale(
            {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 32, **off}
        )


# --------------------------------------------------------------------------
# 7B grid
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ARMS_7B)
def test_7b_arms_share_training_pins(name, monkeypatch):
    for v in ("TAG_EPISODE_BS_7B", "TAG_GRAD_ACCUM_7B", "TAG_BATCH_7B"):
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
    # Qwen2.5's 151643-token vocab makes the fp32 logits tensor ~4.7x a
    # Llama-2 one, and full_ft.yaml's 8 OOMs on an 80GB H100 mid-epoch.
    assert int(cfg["batch_size"]) <= 4


@pytest.mark.parametrize("name", _ARMS_7B)
def test_7b_ddp_override_keeps_the_effective_batch(name, monkeypatch):
    """One arm across 4 DDP ranks must reach the same effective batch as one
    arm per GPU, or the two layouts are not comparable."""
    monkeypatch.delenv("TAG_BATCH_7B", raising=False)
    monkeypatch.setenv("TAG_GRAD_ACCUM_7B", "8")
    cfg = _load(name)
    assert int(cfg["batch_size"]) * int(cfg["grad_accum"]) * 4 == 128


def test_7b_tag_arm_resolves_tag_score_mode():
    cfg = _load("tag_7b")
    assert cfg["method"] == "selection"
    assert cfg["selection"]["score_mode"] == "tag"
    assert cfg["selection"]["use_anchor"] is True
    assert cfg["selection_ratio"] == 0.1


def test_7b_tag_arms_accept_a_shared_gate_cache(monkeypatch):
    """G depends only on (pool, base checkpoint, gate config), so every arm
    and seed must be able to point at ONE precomputed cache."""
    monkeypatch.setenv("TAG_GATE_CACHE", "/tmp/shared_gate.pt")
    for name in ("tag_7b", "tag_static_7b"):
        cfg = _load(name)
        assert cfg["selection"]["tag"]["gate_cache_file"] == "/tmp/shared_gate.pt"


def test_7b_legacy_arm_has_no_gate():
    """TAG - legacy is the pair that isolates the gate, so legacy must carry
    no tag block at all."""
    cfg = _load("legacy_7b")
    assert cfg["selection"].get("score_mode", "legacy") == "legacy"
    assert "tag" not in cfg["selection"]


# --------------------------------------------------------------------------
# A missing model path must say so, not become an HF repo id
# --------------------------------------------------------------------------

def test_missing_local_model_path_is_an_actionable_error():
    """transformers interprets a nonexistent local path as a hub repo id and
    raises HFValidationError('Repo id must be in the form ...'), which names a
    path the user never chose and says nothing about the real cause — an
    unset MODEL_PATH_* leaving the config default in place."""
    from tag.modeling.loader import _resolve_local_path

    with pytest.raises(FileNotFoundError, match="MODEL_PATH_QWEN25_7B"):
        _resolve_local_path("/group-volume/nait-models/qwen2.5-7b")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_local_path("/definitely/not/here")


def test_bare_repo_id_is_not_treated_as_a_missing_path():
    """With local_files_only=True transformers can still resolve 'org/name'
    from the HF cache, so it must pass through untouched."""
    from tag.modeling.loader import _resolve_local_path

    assert _resolve_local_path("Qwen/Qwen2.5-7B") == "Qwen/Qwen2.5-7B"


def test_case_variant_resolution_still_works(tmp_path):
    from tag.modeling.loader import _resolve_local_path

    (tmp_path / "MyModel").mkdir()
    assert _resolve_local_path(str(tmp_path / "mymodel")) == str(tmp_path / "MyModel")


def test_7b_gate_ref_is_backbone_specific(monkeypatch):
    """One shared TAG_GATE_REF for a per-backbone artifact meant a 7B run
    silently picked up the 0.5B reference — Delta_hat is a property of a
    particular model's likelihoods, so that mis-scales every gate value."""
    monkeypatch.setenv("TAG_GATE_REF", "/ref/delta_hat_05b.pt")
    monkeypatch.setenv("TAG_GATE_REF_7B", "/ref/delta_hat_7b.pt")
    for name in ("tag_7b", "tag_static_7b"):
        assert _load(name)["selection"]["tag"]["gate_ref_file"] == "/ref/delta_hat_7b.pt"
    assert _load("light_tag_05b")["selection"]["tag"]["gate_ref_file"] == "/ref/delta_hat_05b.pt"


# --------------------------------------------------------------------------
# Eq. 5' null correction: config wiring
# --------------------------------------------------------------------------

def test_tag_arms_default_to_the_null_correction():
    for name in ("tag_7b", "tag_static_7b", "light_tag_05b"):
        cfg = _load(name)
        tag = cfg["selection"]["tag"]
        assert tag["null_correction"] is True, name
        assert tag["target_zero_rate"] == 0.05, name
        # Centring puts the target_zero_rate quantile at exactly 0, so s must be
        # derived from a strictly larger one.
        assert tag["target_zero_rate"] < tag["calibration_target_pct"], name


def test_nonull_ablation_arm_differs_in_exactly_one_bit(monkeypatch):
    """The Eq. 5 ablation must be comparable to tag_7b: same optimizer, same
    schedule, same span config — only the correction flipped. It also needs
    its OWN reference and cache, since both s and G differ."""
    for v in ("TAG_GATE_REF_7B", "TAG_GATE_REF_7B_NONULL",
              "TAG_GATE_CACHE", "TAG_GATE_CACHE_NONULL"):
        monkeypatch.delenv(v, raising=False)
    on = _load("tag_7b")
    off = _load("tag_nonull_7b")
    assert on["selection"]["tag"]["null_correction"] is True
    assert off["selection"]["tag"]["null_correction"] is False
    for k in ("batch_size", "grad_accum", "train_epochs", "selection_ratio",
              "training_mode", "model_key", "episode_batch_size"):
        assert on[k] == off[k], k
    for k in ("span_tokens", "tau", "tau_mode", "min_span_tokens",
              "tail_mode", "include_eos", "c_trunc", "undefined_policy"):
        assert on["selection"]["tag"][k] == off["selection"]["tag"][k], k
    assert on["output_subdir"] != off["output_subdir"]
    # Distinct artifacts, so the two arms cannot silently share a gate.
    monkeypatch.setenv("TAG_GATE_REF_7B", "/tmp/on_ref.pt")
    monkeypatch.setenv("TAG_GATE_REF_7B_NONULL", "/tmp/off_ref.pt")
    monkeypatch.setenv("TAG_GATE_CACHE", "/tmp/on_gate.pt")
    monkeypatch.setenv("TAG_GATE_CACHE_NONULL", "/tmp/off_gate.pt")
    assert _load("tag_7b")["selection"]["tag"]["gate_ref_file"] == "/tmp/on_ref.pt"
    assert _load("tag_nonull_7b")["selection"]["tag"]["gate_ref_file"] == "/tmp/off_ref.pt"
    assert _load("tag_7b")["selection"]["tag"]["gate_cache_file"] == "/tmp/on_gate.pt"
    assert _load("tag_nonull_7b")["selection"]["tag"]["gate_cache_file"] == "/tmp/off_gate.pt"


def test_gate_ref_carrying_a_null_curve_calibrates_s_on_the_centred_statistic(tmp_path):
    """s is a quantile of what Eq. 6 SEES. Deriving it from the uncentred
    reference and then centring at gate time would shift every value by mu
    while leaving the scale that interprets them put."""
    import torch
    from tag.core.gate import GateConfig, fit_calibration
    from tag.pipelines.selection import _resolve_gate_calibration

    g = torch.Generator().manual_seed(0)
    n_spans = torch.randint(1, 30, (4000,), generator=g)
    # A raw statistic that drifts down with span count, like the real one.
    raw = 0.3 - 0.02 * n_spans.float() + 0.15 * torch.randn(4000, generator=g)
    fit = fit_calibration(raw, n_spans, span_tokens=16, target_zero_rate=0.05)

    ref = tmp_path / "delta_hat.pt"
    torch.save(
        {
            "delta_hat": raw,
            "n_spans": n_spans,
            "null": fit["null"].to_dict(),
            "gate_config": GateConfig(
                span_tokens=16, scale=1.0, null=fit["null"], target_zero_rate=0.05,
            ).identity(),
        },
        ref,
    )
    params = {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 16,
              "target_zero_rate": 0.05}
    scale, null = _resolve_gate_calibration(params)
    assert null == fit["null"]
    assert scale == pytest.approx(fit["scale"], rel=1e-6)

    # A reference with no curve is refused rather than silently uncorrected.
    bare = tmp_path / "bare.pt"
    torch.save({"delta_hat": raw}, bare)
    with pytest.raises(ValueError, match="no null curve"):
        _resolve_gate_calibration({**params, "gate_ref_file": str(bare)})
    # ...and one fit at another target_zero_rate is refused too.
    with pytest.raises(ValueError, match="target_zero_rate"):
        _resolve_gate_calibration({**params, "target_zero_rate": 0.10})


def test_calibration_check_ignores_fields_the_config_never_reads(tmp_path):
    """A refit artifact recorded the sweep script's default tail_quantile=0.25
    while the arm inherited 0.0 — under tail_mode: none, where NEITHER value
    is ever read — and the run died with 'calibrated under a different span
    configuration'. Comparing a field the code path provably ignores turns a
    difference that changes nothing into a hard error."""
    import torch
    from tag.core.gate import GateConfig
    from tag.pipelines.selection import (
        _effective_calibration_fields, _resolve_gate_scale,
    )

    # tail_quantile is read only by tail_gain(mode="quantile").
    assert "tail_quantile" not in _effective_calibration_fields({"tail_mode": "none"})
    assert "tail_quantile" not in _effective_calibration_fields({"tail_mode": "min"})
    assert "tail_quantile" in _effective_calibration_fields({"tail_mode": "quantile"})
    # C_i exists only to mask the spans the tail test reads.
    for f in ("tau", "tau_mode", "min_span_tokens"):
        assert f not in _effective_calibration_fields({"tail_mode": "none"})
        assert f in _effective_calibration_fields({"tail_mode": "min"})
    # span_tokens always matters: it sets M, and mu(M) bins on it.
    for tm in ("none", "min", "quantile"):
        eff = _effective_calibration_fields({"tail_mode": tm})
        assert "span_tokens" in eff and "prefix_tokens" in eff and "tail_mode" in eff

    ref = tmp_path / "delta_hat.pt"
    torch.save(
        {
            "delta_hat": torch.linspace(0.05, 0.95, 200),
            "gate_config": GateConfig(
                span_tokens=16, tail_mode="none", tail_quantile=0.25,
                scale=1.0, null_correction=False,
            ).identity(),
        },
        ref,
    )
    base = {"gate_scale": "", "gate_ref_file": str(ref), "span_tokens": 16,
            "tail_mode": "none", "null_correction": False}
    # The exact failure: differs only in a field tail_mode: none never reads.
    assert _resolve_gate_scale({**base, "tail_quantile": 0.0}) > 0
    # A field it DOES read still fails loudly.
    with pytest.raises(ValueError, match="different span configuration"):
        _resolve_gate_scale({**base, "span_tokens": 32})
    with pytest.raises(ValueError, match="different span configuration"):
        _resolve_gate_scale({**base, "tail_mode": "min"})
