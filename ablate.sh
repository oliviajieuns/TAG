#!/usr/bin/env bash

# Shared launcher for the matched Table-2 clean-pool gate sweep.
# Use the short wrappers instead of calling this file directly:
#
#   S=42 bash weak.sh          # sqrt(G), exact zero preserved
#   S=42 bash soft.sh          # 0.5 + 0.5G, diagnostic soft gate
#   S=42 bash ctl.sh           # matched R x A control
#
# Each training run automatically starts its independent full 8-task eval.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run with bash, do not source this file" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
FRESH=${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
NCCL_FIX=/group-volume/jieuns.shin/Unlearning/experiment/benchmark/openunlearning/third_party/open-unlearning/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
HIST_GATE="$OLD/pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt"
CONTROL_CFG="$REPO/configs/experiments/main_7b/llama2/legacy_10_schedfloor_bs64.yaml"
STRONG_CFG="$REPO/configs/experiments/main_7b/llama2/tag_10_schedfloor_bs64.yaml"
GRAD_ACCUM=4
MIN_LR_RATIO=0.10

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve() {
  HOST=$(hostname -s)
  SEED=${S:-42}
  ARM=${B:-weak}
  case "$SEED" in
    1|7|42) ;;
    *) die "S must be 1, 7, or 42" ;;
  esac

  case "$ARM" in
    weak)
      CFG="$REPO/configs/experiments/main_7b/llama2/tag_10_weakpower50_bs64.yaml"
      CELL="main_7b/llama2/tag_10_weakpower50_bs64_seed${SEED}"
      RUN_PREFIX=tag10weak
      LOG_GROUP=table2_tag_weakpower50
      GATE_POWER=0.5
      GATE_STRENGTH=1.0
      ;;
    soft)
      CFG="$REPO/configs/experiments/main_7b/llama2/tag_10_softmix50_bs64.yaml"
      CELL="main_7b/llama2/tag_10_softmix50_bs64_seed${SEED}"
      RUN_PREFIX=tag10soft
      LOG_GROUP=table2_tag_softmix50
      GATE_POWER=1.0
      GATE_STRENGTH=0.5
      ;;
    ctl)
      CFG="$CONTROL_CFG"
      CELL="main_7b/llama2/legacy_10_schedfloor_bs64_seed${SEED}"
      RUN_PREFIX=ra64
      LOG_GROUP=table2_ra_matched64
      GATE_POWER=none
      GATE_STRENGTH=none
      ;;
    *) die "B must be weak, soft, or ctl" ;;
  esac
  # Private per arm+seed: _prepare_tag may legitimately reserialise a cache
  # after a config-identity check.  Sharing one writable copy across the
  # 2/4/4-GPU nodes would race on its fixed .pt.tmp path and could let one
  # ablation alter another arm's input.
  RAW_GATE="$FRESH/cache/table2_gate_sweep/$ARM/seed${SEED}/tag_gate_llama2-7b_prefix32.pt"

  mapfile -t VISIBLE_GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '
  )
  NPROC=${#VISIBLE_GPUS[@]}
  case "$NPROC" in
    2) BATCH=8 ;;
    4) BATCH=4 ;;
    *) die "$HOST sees $NPROC GPU(s); this sweep accepts exactly 2 or 4" ;;
  esac
  GPUS=$(IFS=,; echo "${VISIBLE_GPUS[*]}")

  cd "$REPO"
  PIN=$(git rev-parse HEAD)
  SHORT_PIN=${PIN:0:7}
  case "$ARM" in
    weak) KNOB=gpow050 ;;
    soft) KNOB=gsoft050 ;;
    ctl) KNOB=nogate ;;
  esac
  RUN_TAG="${RUN_PREFIX}_${SHORT_PIN}_seed${SEED}_${NPROC}g_b${BATCH}_ga4_floor010_bs64_${KNOB}"
  RUN_DIR="$FRESH/runs/$CELL/runs/$RUN_TAG"
  LOG_DIR="$FRESH/logs/$LOG_GROUP"
  LOG="$LOG_DIR/$RUN_TAG.log"
  STATE_DIR="$FRESH/status/$LOG_GROUP"
}

install_gate_copy() {
  [[ "$ARM" == ctl ]] && return 0
  mkdir -p "$(dirname "$RAW_GATE")"
  if [[ ! -f "$RAW_GATE" ]]; then
    local tmp="${RAW_GATE}.copy.$(hostname -s).$$"
    cp --reflink=auto "$HIST_GATE" "$tmp"
    cmp -s "$tmp" "$HIST_GATE" || die "raw gate copy verification failed"
    mv "$tmp" "$RAW_GATE"
    echo "RAW_GATE_INSTALLED=$RAW_GATE"
  fi
  # A harmless cache re-save may normalise metadata ordering/keys and change
  # file bytes.  Validate the scientific payload rather than requiring a
  # byte-identical serialization; this also permits safe reuse after such a
  # re-save while still rejecting a transformed or wrong-pool gate.
  "$PY" - "$RAW_GATE" "$HIST_GATE" <<'PY'
import sys
import torch

actual = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
historical = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert torch.equal(actual["gate"], historical["gate"])
assert torch.equal(actual["delta_hat"], historical["delta_hat"])
for key in ("model_path", "pool_files", "n_pool"):
    assert (actual.get("identity") or {}).get(key) == (historical.get("identity") or {}).get(key)
print("RAW_GATE_PAYLOAD_OK")
PY
}

check_pair() {
  local candidate="$CFG"
  [[ "$ARM" != ctl ]] || candidate="$STRONG_CFG"
  TAG_MAIN_POOL="$OLD/pools/alpaca_gpt4/pool.json" \
  TAG_MAIN_CF="$OLD/pools/alpaca_gpt4/counterfactual.json" \
  TAG_GATE_REF_LLAMA2="$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt" \
  TAG_GATE_CACHE_LLAMA2="$RAW_GATE" \
  TAG_GATE_SCALE= \
  PYTHONPATH="$REPO" "$PY" \
    "$REPO/scripts/check_row_pair.py" "$CONTROL_CFG" "$candidate"
}

preflight() {
  resolve
  cd "$REPO"
  git diff --quiet || die "tracked code changes present"
  git diff --cached --quiet || die "staged code changes present"
  [[ -x "$PY" ]] || die "python missing"
  [[ -f "$CFG" && -f "$CONTROL_CFG" && -f "$STRONG_CFG" ]] || \
    die "sweep config missing"
  [[ -f "$NCCL_FIX" ]] || die "NCCL compatibility library missing"
  [[ -f "$OLD/pools/alpaca_gpt4/pool.json" ]] || die "main pool missing"
  [[ -f "$OLD/pools/alpaca_gpt4/counterfactual.json" ]] || \
    die "counterfactual pool missing"
  [[ -f "$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt" ]] || \
    die "gate reference missing"
  [[ -f "$HIST_GATE" ]] || die "historical gate missing"
  install_gate_copy
  check_pair
  [[ ! -e "$RUN_DIR" ]] || \
    die "$ARM run already exists: $RUN_DIR (use status)"

  local free
  while read -r free; do
    [[ "$free" -ge 78000 ]] || die "every visible GPU must be idle (${free} MiB free)"
  done < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  mkdir -p "$LOG_DIR" "$STATE_DIR" "$FRESH/locks"
}

validate_training() {
  [[ "$(<"$RUN_DIR/epoch_last/_complete")" == 3 ]] || \
    die "epoch_last is not epoch 3"
  [[ ! -e "$RUN_DIR/epoch_last/_save_errors.json" ]] || \
    die "checkpoint save errors present"

  PYTHONPATH="$REPO" "$PY" - "$RUN_DIR" "$ARM" "$SEED" "$NPROC" \
    "$BATCH" "$PIN" "$OLD" "$RAW_GATE" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch
from tag.core.schedulers import optimizer_steps_per_epoch

run, arm = Path(sys.argv[1]), sys.argv[2]
seed, world, batch = map(int, sys.argv[3:6])
pin, old, raw_gate = sys.argv[6], Path(sys.argv[7]), Path(sys.argv[8])
cfg = json.loads((run / "cfg.json").read_text())

assert cfg["seed"] == seed and cfg["git_sha"] == pin
assert (cfg["launch_world_size"], cfg["batch_size"], cfg["grad_accum"]) == (world, batch, 4)
assert world * batch * 4 == 64
assert math.isclose(float(cfg["min_lr_ratio"]), 0.10)
assert cfg.get("adamw_foreach") is False and cfg["use_8bit_optimizer"] is False
assert cfg["method"] == "selection" and cfg["selection_ratio"] == 0.1
assert cfg["train_epochs"] == 3 and cfg["training_mode"] == "full"
assert cfg["model_path"] == "/group-volume/models/Llama-2-7b-hf"
assert cfg["data_files"] == str(old / "pools/alpaca_gpt4/pool.json")

selection = cfg["selection"]
if arm in {"weak", "soft"}:
    assert selection["score_mode"] == "tag"
    tag = selection["tag"]
    expected = (0.5, 1.0) if arm == "weak" else (1.0, 0.5)
    assert (float(tag["gate_power"]), float(tag["gate_strength"])) == expected
    assert tag["counterfactual_data_files"] == str(old / "pools/alpaca_gpt4/counterfactual.json")
    assert tag["gate_ref_file"] == str(old / "pools/clean_ref/delta_hat_llama2_prefix.pt")
    assert Path(tag["gate_cache_file"]) == raw_gate
    assert str(tag.get("gate_scale") or "").strip() == ""
else:
    assert selection.get("score_mode", "legacy") == "legacy"
    assert math.isclose(float(selection["lam"]), 1.0) and selection["use_anchor"] is True

rows = json.loads((run / "metrics.json").read_text())
assert [row["epoch"] for row in rows] == [1, 2, 3]
for row in rows:
    assert row["n_total"] == 52002 and row["selected_n"] == 5200
    assert math.isfinite(float(row["train_loss"]))
    if arm == "ctl":
        assert row.get("score_mode", "legacy") == "legacy"
        assert "gate_raw_zero_frac" not in row and "gate_zero_frac" not in row
        continue
    assert row["score_mode"] == "tag" and row["n_zero_weight_selected"] == 0
    assert 0.045 <= row["gate_raw_zero_frac"] <= 0.055
    assert row["n_raw_admissible"] + round(52002 * row["gate_raw_zero_frac"]) == 52002
    assert row["gate_mean"] >= row["gate_raw_mean"]
    if arm == "weak":
        assert row["gate_power"] == 0.5 and row["gate_strength"] == 1.0
        assert math.isclose(row["gate_zero_frac"], row["gate_raw_zero_frac"], abs_tol=1e-9)
        assert row["n_admissible"] == row["n_raw_admissible"]
    else:
        assert row["gate_power"] == 1.0 and row["gate_strength"] == 0.5
        assert row["gate_zero_frac"] == 0.0 and row["n_admissible"] == 52002

for epoch in (1, 2, 3):
    ids = json.loads((run / f"selected_indices_epoch{epoch}.json").read_text())
    assert len(ids) == len(set(ids)) == 5200
    assert all(type(x) is int and 0 <= x < 52002 for x in ids)

steps = optimizer_steps_per_epoch(5200, batch, 4, world)
assert steps == 82
scheduler = torch.load(run / "epoch_last/scheduler.pt", map_location="cpu", weights_only=False)
assert scheduler["last_epoch"] == 246
assert math.isclose(float(scheduler["_last_lr"][0]), float(cfg["learning_rate"]) * 0.10, rel_tol=1e-5)
weights = list((run / "epoch_last").glob("*.safetensors"))
assert weights and sum(p.stat().st_size for p in weights) > 10_000_000_000

if arm != "ctl":
    # Serialization bytes may change if the loader harmlessly re-saves a
    # cache, so compare the scientific payload.  In particular, this catches
    # an accidental write of sqrt(G) / soft(G) into the raw cache.
    actual = torch.load(raw_gate, map_location="cpu", weights_only=False)
    historical = torch.load(
        old / "pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt",
        map_location="cpu", weights_only=False,
    )
    assert torch.equal(actual["gate"], historical["gate"])
    assert torch.equal(actual["delta_hat"], historical["delta_hat"])
    for key in ("model_path", "pool_files", "n_pool"):
        assert (actual.get("identity") or {}).get(key) == (historical.get("identity") or {}).get(key)
    assert "gate_power" not in actual["config"] and "gate_strength" not in actual["config"]

print("ABLATION_TRAIN_VALIDATED", arm, run, "steps=246 effective_batch=64")
PY
  touch "$RUN_DIR/TRAIN_VALIDATED"
}

resolve_status_run() {
  # Status is often checked from a different allocated node, or after the
  # checkout advanced by one commit.  Launch identity remains exact, but a
  # status query must not reconstruct a fake 2-GPU/current-HEAD run tag and
  # report NOT_STARTED for the real 4-GPU/older-HEAD run.
  [[ -e "$RUN_DIR" ]] && return 0
  local root="$FRESH/runs/$CELL/runs" candidate newest=""
  shopt -s nullglob
  for candidate in "$root"/${RUN_PREFIX}_*; do
    [[ -d "$candidate" ]] || continue
    if [[ -z "$newest" || "$candidate" -nt "$newest" ]]; then
      newest=$candidate
    fi
  done
  [[ -n "$newest" ]] || return 0
  RUN_DIR=$newest
  RUN_TAG=$(basename "$RUN_DIR")
  LOG="$LOG_DIR/$RUN_TAG.log"
}

worker() {
  preflight
  exec 9>"$FRESH/locks/$RUN_TAG.lock"
  flock -n 9 || die "worker already active"
  printf 'host=%s\npid=%s\nstarted_utc=%s\n' \
    "$HOST" "$$" "$(date -u +%FT%TZ)" >"$STATE_DIR/$RUN_TAG.running"

  on_exit() {
    local rc=$?
    rm -f "$STATE_DIR/$RUN_TAG.running"
    printf 'host=%s\nrc=%s\nfinished_utc=%s\n' \
      "$HOST" "$rc" "$(date -u +%FT%TZ)" >"$STATE_DIR/$RUN_TAG.exit"
    echo "JOB_EXIT_CODE=$rc"
  }
  trap on_exit EXIT

  export TAG_WORKSPACE="$FRESH"
  export HF_HOME="$FRESH/hf_home"
  export TAG_ENV_RESET=1
  source "$REPO/scripts/gpu_cloud/env.sh"
  unset TAG_ENV_RESET
  export TAG_MAIN_POOL="$OLD/pools/alpaca_gpt4/pool.json"
  export TAG_MAIN_CF="$OLD/pools/alpaca_gpt4/counterfactual.json"
  export TAG_GATE_REF_LLAMA2="$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt"
  export TAG_GATE_CACHE_LLAMA2="$RAW_GATE"
  unset TAG_GATE_SCALE
  export PYTHONHASHSEED=0
  export TAG_ENABLE_NO_SYNC=0
  export TAG_NCCL_REINIT=0

  echo "ABLATION_TRAIN_START arm=$ARM host=$HOST seed=$SEED nproc=$NPROC batch=$BATCH grad_accum=4 effective_batch=64"
  echo "GATE_POWER=$GATE_POWER GATE_STRENGTH=$GATE_STRENGTH"
  echo "RUN_DIR=$RUN_DIR"

  cd "$REPO"
  LD_PRELOAD="$NCCL_FIX${LD_PRELOAD:+:$LD_PRELOAD}" \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  "$PY" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    -m tag.train \
    --config "$CFG" \
    --run_tag "$RUN_TAG" \
    --override \
      seed="$SEED" \
      batch_size="$BATCH" \
      grad_accum="$GRAD_ACCUM" \
      min_lr_ratio="$MIN_LR_RATIO" \
      adamw_foreach=false \
      output_subdir="$CELL" \
      git_sha="$PIN" \
      launch_world_size="$NPROC"

  validate_training
  printf 'validated_utc=%s\nrun_dir=%s\n' \
    "$(date -u +%FT%TZ)" "$RUN_DIR" >"$STATE_DIR/$RUN_TAG.validated"
  echo "ABLATION_TRAINING_COMPLETE arm=$ARM seed=$SEED run_dir=$RUN_DIR"
  flock -u 9
  exec 9>&-

  if [[ "${ABLATION_AUTO_EVAL:-1}" == 1 ]]; then
    echo "AUTO_EVAL_START arm=$ARM seed=$SEED"
    # CUDA contexts can remain visible for a few seconds after torchrun exits.
    # Give the driver up to one minute to release at least one lane before the
    # eval queue performs its strict 78-GB-idle check.
    local attempt idle_count=0
    for attempt in {1..12}; do
      idle_count=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '$1 >= 78000 {n++} END {print n+0}')
      [[ "$idle_count" -gt 0 ]] && break
      sleep 5
    done
    if ! A="$ARM" S="$SEED" bash "$REPO/all.sh"; then
      echo "AUTO_EVAL_NOT_STARTED: S=$SEED bash ${ARM}.sh eval"
    fi
  fi
}

status() {
  resolve
  resolve_status_run
  echo "host=$HOST arm=$ARM seed=$SEED effective_batch=64"
  echo "run_dir=$RUN_DIR"
  pgrep -af "tag.train.*$RUN_TAG|torch.distributed.run.*$RUN_TAG" || true
  if [[ -f "$RUN_DIR/TRAIN_VALIDATED" ]]; then
    echo "STATUS=TRAIN_VALIDATED"
    A="$ARM" S="$SEED" bash "$REPO/all.sh" status || true
  elif [[ -f "$STATE_DIR/$RUN_TAG.running" ]]; then
    echo "STATUS=RUNNING"
    cat "$STATE_DIR/$RUN_TAG.running"
  elif [[ -f "$STATE_DIR/$RUN_TAG.exit" ]]; then
    echo "STATUS=EXITED"
    cat "$STATE_DIR/$RUN_TAG.exit"
  else
    echo "STATUS=NOT_STARTED"
  fi
  [[ ! -f "$LOG" ]] || tail -n 60 "$LOG"
}

launch() {
  preflight
  nohup bash "$0" --worker >"$LOG" 2>&1 </dev/null &
  local pid=$!
  sleep 5
  if kill -0 "$pid" 2>/dev/null; then
    echo "STARTED arm=$ARM seed=$SEED host=$HOST pid=$pid"
    echo "STATUS: S=$SEED bash ${ARM}.sh status"
  else
    echo "START_FAILED"
    tail -n 100 "$LOG" || true
    exit 1
  fi
}

case "${1:-}" in
  --worker) worker ;;
  status) status ;;
  eval)
    resolve
    A="$ARM" S="$SEED" exec bash "$REPO/all.sh"
    ;;
  test)
    cd "$REPO"
    exec "$PY" -m pytest -q \
      tests/test_selector_tag.py \
      tests/test_training_schedule.py \
      tests/test_gate_params_forwarded.py
    ;;
  "") launch ;;
  *) die "usage: S={1|7|42} bash ${B:-weak}.sh [status|eval|test]" ;;
esac
