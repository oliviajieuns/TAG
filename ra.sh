#!/usr/bin/env bash

# Repeat the original Table-2 R x A (legacy/TADS) arm, then automatically
# launch its full eight-benchmark evaluation.
#
#   S=42 bash ra.sh
#   S=42 bash ra.sh status

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run with: bash ra.sh" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
FRESH=${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
CFG="$REPO/configs/experiments/main_7b/llama2/legacy_10.yaml"
TAG_CFG="$REPO/configs/experiments/main_7b/llama2/tag_10.yaml"
NCCL_FIX=/group-volume/jieuns.shin/Unlearning/experiment/benchmark/openunlearning/third_party/open-unlearning/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve() {
  HOST=$(hostname -s)
  SEED=${S:-42}
  case "$SEED" in
    1|7|42) ;;
    *) die "S must be 1, 7, or 42" ;;
  esac

  mapfile -t VISIBLE_GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '
  )
  NPROC=${#VISIBLE_GPUS[@]}
  case "$NPROC" in
    2) BATCH=8 ;;
    4) BATCH=4 ;;
    *) die "$HOST sees $NPROC GPU(s); R x A repeat accepts exactly 2 or 4" ;;
  esac
  GPUS=$(IFS=,; echo "${VISIBLE_GPUS[*]}")

  cd "$REPO"
  PIN=$(git rev-parse HEAD)
  SHORT_PIN=${PIN:0:7}
  CELL="main_7b/llama2/legacy_10_repeat_seed${SEED}"
  RUN_TAG="ra_${SHORT_PIN}_seed${SEED}_${NPROC}g_b${BATCH}_ga8_bs128"
  RUN_DIR="$FRESH/runs/$CELL/runs/$RUN_TAG"
  LOG_DIR="$FRESH/logs/table2_ra_repeat"
  LOG="$LOG_DIR/$RUN_TAG.log"
  STATE_DIR="$FRESH/status/table2_ra_repeat"
}

preflight() {
  resolve
  cd "$REPO"
  git diff --quiet || die "tracked code changes present"
  git diff --cached --quiet || die "staged code changes present"
  [[ -x "$PY" ]] || die "python missing"
  [[ -f "$CFG" && -f "$TAG_CFG" ]] || die "Table-2 config missing"
  [[ -f "$NCCL_FIX" ]] || die "NCCL compatibility library missing"
  [[ -f "$OLD/pools/alpaca_gpt4/pool.json" ]] || die "Alpaca-GPT4 pool missing"
  [[ ! -e "$RUN_DIR" ]] || die "R x A repeat already exists; use status"

  local free
  while read -r free; do
    [[ "$free" -ge 78000 ]] || die "every visible GPU must be idle (${free} MiB free)"
  done < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )

  TAG_MAIN_POOL="$OLD/pools/alpaca_gpt4/pool.json" \
  TAG_MAIN_CF="$OLD/pools/alpaca_gpt4/counterfactual.json" \
  TAG_GATE_REF_LLAMA2="$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt" \
  TAG_GATE_CACHE_LLAMA2="$OLD/pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt" \
  PYTHONPATH="$REPO" "$PY" scripts/check_row_pair.py "$CFG" "$TAG_CFG"

  mkdir -p "$LOG_DIR" "$STATE_DIR" "$FRESH/locks"
}

validate_training() {
  [[ "$(<"$RUN_DIR/epoch_last/_complete")" == 3 ]] || die "epoch_last is not epoch 3"
  [[ ! -e "$RUN_DIR/epoch_last/_save_errors.json" ]] || die "checkpoint save errors present"

  PYTHONPATH="$REPO" "$PY" - "$RUN_DIR" "$SEED" "$NPROC" "$BATCH" \
    "$PIN" "$OLD" <<'PY'
import json, math, sys
from pathlib import Path

import torch
from tag.core.schedulers import optimizer_steps_per_epoch

run = Path(sys.argv[1])
seed, world, batch = map(int, sys.argv[2:5])
pin, old = sys.argv[5], Path(sys.argv[6])
cfg = json.loads((run / "cfg.json").read_text())

assert cfg["seed"] == seed and cfg["git_sha"] == pin
assert (cfg["launch_world_size"], cfg["batch_size"], cfg["grad_accum"]) == (world, batch, 8)
assert world * batch * 8 == 128
assert cfg["method"] == "selection" and cfg["selection_ratio"] == 0.1
assert cfg["train_epochs"] == 3 and cfg["training_mode"] == "full"
assert cfg["model_path"] == "/group-volume/models/Llama-2-7b-hf"
assert cfg["data_files"] == str(old / "pools/alpaca_gpt4/pool.json")
assert cfg["use_8bit_optimizer"] is False and cfg.get("adamw_foreach") is False
assert math.isclose(float(cfg["warmup_ratio"]), 0.06)
assert math.isclose(float(cfg["gradient_clip"]), 0.5)
assert math.isclose(float(cfg.get("min_lr_ratio", 0.0)), 0.0)
selection = cfg["selection"]
assert selection.get("score_mode", "legacy") == "legacy"
assert math.isclose(float(selection["lam"]), 1.0) and selection["use_anchor"] is True

rows = json.loads((run / "metrics.json").read_text())
assert [row["epoch"] for row in rows] == [1, 2, 3]
for row in rows:
    assert row["n_total"] == 52002 and row["selected_n"] == 5200
    assert math.isfinite(float(row["train_loss"]))
    assert "gate_zero_frac" not in row and "n_zero_weight_selected" not in row

for epoch in (1, 2, 3):
    ids = json.loads((run / f"selected_indices_epoch{epoch}.json").read_text())
    assert len(ids) == len(set(ids)) == 5200
    assert all(type(x) is int and 0 <= x < 52002 for x in ids)

steps_per_epoch = optimizer_steps_per_epoch(5200, batch, 8, world)
assert steps_per_epoch == 41
scheduler = torch.load(
    run / "epoch_last/scheduler.pt", map_location="cpu", weights_only=False,
)
assert scheduler["last_epoch"] == 123
assert math.isclose(float(scheduler["_last_lr"][0]), 0.0, abs_tol=1e-12)

weights = list((run / "epoch_last").glob("*.safetensors"))
assert weights and sum(path.stat().st_size for path in weights) > 10_000_000_000
print("RA_TRAIN_VALIDATED", run, "steps_per_epoch=41 total_steps=123 effective_batch=128")
PY
  touch "$RUN_DIR/TRAIN_VALIDATED"
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
  export PYTHONHASHSEED=0
  export TAG_ENABLE_NO_SYNC=0
  export TAG_NCCL_REINIT=0

  echo "RA_TRAIN_START host=$HOST seed=$SEED nproc=$NPROC batch=$BATCH grad_accum=8 effective_batch=128"
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
      grad_accum=8 \
      adamw_foreach=false \
      output_subdir="$CELL" \
      git_sha="$PIN" \
      launch_world_size="$NPROC"

  validate_training
  printf 'validated_utc=%s\nrun_dir=%s\n' \
    "$(date -u +%FT%TZ)" "$RUN_DIR" >"$STATE_DIR/$RUN_TAG.validated"
  echo "RA_TRAINING_COMPLETE seed=$SEED run_dir=$RUN_DIR"
  flock -u 9
  exec 9>&-

  if [[ "${RA_AUTO_EVAL:-1}" == 1 ]]; then
    echo "RA_AUTO_EVAL_START"
    S="$SEED" bash "$REPO/raeval.sh" || \
      echo "RA_AUTO_EVAL_NOT_STARTED: run S=$SEED bash raeval.sh manually"
  fi
}

status() {
  resolve
  echo "host=$HOST seed=$SEED effective_batch=128"
  echo "run_dir=$RUN_DIR"
  pgrep -af "tag.train.*$RUN_TAG|torch.distributed.run.*$RUN_TAG" || true
  if [[ -f "$RUN_DIR/TRAIN_VALIDATED" ]]; then
    echo "STATUS=RA_TRAIN_VALIDATED"
    S="$SEED" bash "$REPO/raeval.sh" status || true
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
    echo "STARTED R×A seed=$SEED host=$HOST pid=$pid"
    echo "STATUS: S=$SEED bash ra.sh status"
  else
    echo "START_FAILED"
    tail -n 100 "$LOG" || true
    exit 1
  fi
}

case "${1:-}" in
  --worker) worker ;;
  status) status ;;
  "") launch ;;
  *) die "usage: S={1|7|42} bash ra.sh [status]" ;;
esac
