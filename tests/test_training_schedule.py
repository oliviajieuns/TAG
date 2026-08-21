from __future__ import annotations

import pytest
import torch

from tag.core.schedulers import (
    get_cosine_schedule_with_warmup,
    optimizer_steps_per_epoch,
)
from tag.pipelines.sft import _accumulation_window_size


def test_table2_step_planning_uses_every_partial_group():
    # Both allocated layouts preserve effective batch 64 and therefore must
    # produce the same number of optimiser updates.
    assert optimizer_steps_per_epoch(5200, batch_size=4, grad_accum=4, world_size=4) == 82
    assert optimizer_steps_per_epoch(5200, batch_size=8, grad_accum=4, world_size=2) == 82

    # Historical effective-batch-128 layout executes 41, not floor(40.625).
    assert optimizer_steps_per_epoch(5200, batch_size=4, grad_accum=8, world_size=4) == 41
    assert optimizer_steps_per_epoch(5200, batch_size=8, grad_accum=8, world_size=2) == 41


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0, "batch_size": 4, "grad_accum": 4, "world_size": 4},
        {"n_samples": 10, "batch_size": 0, "grad_accum": 4, "world_size": 4},
        {"n_samples": 10, "batch_size": 4, "grad_accum": 0, "world_size": 4},
        {"n_samples": 10, "batch_size": 4, "grad_accum": 4, "world_size": 0},
    ],
)
def test_step_planning_rejects_nonpositive_inputs(kwargs):
    with pytest.raises(ValueError):
        optimizer_steps_per_epoch(**kwargs)


def test_partial_accumulation_window_uses_actual_denominator():
    assert [_accumulation_window_size(i, 10, 4) for i in range(10)] == [
        4, 4, 4, 4,
        4, 4, 4, 4,
        2, 2,
    ]
    # Proposed Table-2 loader: 325 batches, ga=4, final window has one batch.
    assert _accumulation_window_size(324, 325, 4) == 1
    # Historical ga=8 layout: the final update contains five batches.
    assert _accumulation_window_size(324, 325, 8) == 5


def test_cosine_floor_reaches_ten_percent_and_does_not_rebound():
    p = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([p], lr=2.0e-5)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=15,
        num_training_steps=246,
        min_lr_ratio=0.10,
    )
    for _ in range(246):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(2.0e-6)

    # A caller accidentally stepping beyond the plan remains at the floor;
    # it must not enter a new cosine lobe and raise the LR again.
    for _ in range(100):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(2.0e-6)


def test_zero_floor_preserves_historical_endpoint():
    p = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([p], lr=2.0e-5)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=7,
        num_training_steps=123,
    )
    for _ in range(123):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-15)


def test_cosine_floor_rejects_invalid_ratio():
    p = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([p], lr=1.0)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=1,
            num_training_steps=2,
            min_lr_ratio=1.01,
        )


def test_tuning_config_resolves_to_isolated_bs64_arm():
    from tag.core.utils import load_config

    cfg = load_config(
        "configs/experiments/main_7b/llama2/tag_10_schedfloor_bs64.yaml",
    )
    assert cfg["output_subdir"] == "main_7b/llama2/tag_10_schedfloor_bs64"
    assert (cfg["batch_size"], cfg["grad_accum"]) == (4, 4)
    assert cfg["min_lr_ratio"] == pytest.approx(0.10)
    assert cfg["adamw_foreach"] is False
    assert cfg["selection_ratio"] == pytest.approx(0.10)
    assert cfg["selection"]["score_mode"] == "tag"
    assert cfg["selection"]["tag"]["prefix_tokens"] == 32
    assert cfg["selection"]["tag"]["gate_power"] == pytest.approx(1.0)
    assert cfg["selection"]["tag"]["gate_strength"] == pytest.approx(1.0)


def test_gate_weakening_arms_and_matched_control_share_training_recipe():
    from tag.core.utils import load_config

    root = "configs/experiments/main_7b/llama2"
    strong = load_config(f"{root}/tag_10_schedfloor_bs64.yaml")
    weak = load_config(f"{root}/tag_10_weakpower50_bs64.yaml")
    soft = load_config(f"{root}/tag_10_softmix50_bs64.yaml")
    control = load_config(f"{root}/legacy_10_schedfloor_bs64.yaml")

    for cfg in (strong, weak, soft, control):
        assert (cfg["batch_size"], cfg["grad_accum"]) == (4, 4)
        assert cfg["min_lr_ratio"] == pytest.approx(0.10)
        assert cfg["adamw_foreach"] is False
        assert cfg["selection_ratio"] == pytest.approx(0.10)
        assert cfg["train_epochs"] == 3

    assert weak["selection"]["tag"]["gate_power"] == pytest.approx(0.5)
    assert weak["selection"]["tag"]["gate_strength"] == pytest.approx(1.0)
    assert soft["selection"]["tag"]["gate_power"] == pytest.approx(1.0)
    assert soft["selection"]["tag"]["gate_strength"] == pytest.approx(0.5)
    assert control["selection"].get("score_mode", "legacy") == "legacy"
    assert control["selection"]["lam"] == pytest.approx(1.0)
    assert control["selection"]["use_anchor"] is True
