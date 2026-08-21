#!/usr/bin/env bash

# Quick, paper-comparable pilot evaluation. One model load, fixed order:
# MBPP -> GSM8K -> MMLU.
#
#   S=42 bash eval.sh
#   S=42 bash eval.sh status
#
# G selects the physical GPU (default 0).

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run with: bash eval.sh" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
FRESH=${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
CFG="$REPO/configs/experiments/main_7b/llama2/tag_10_schedfloor_bs64.yaml"

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve() {
  SEED=${S:-42}
  GPU=${G:-0}
  case "$SEED" in
    1|7|42) ;;
    *) die "S must be 1, 7, or 42" ;;
  esac
  [[ "$GPU" =~ ^[0-9]+$ ]] || die "G must be a GPU index"

  CELL="$FRESH/runs/main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}/runs"
  RUN=""
  local candidate
  shopt -s nullglob
  for candidate in "$CELL"/tag10tune_*; do
    [[ -f "$candidate/TRAIN_VALIDATED" ]] || continue
    if [[ -z "$RUN" || "$candidate/TRAIN_VALIDATED" -nt "$RUN/TRAIN_VALIDATED" ]]; then
      RUN=$candidate
    fi
  done
  [[ -n "$RUN" ]] || die "no validated tuning run for seed $SEED"

  CKPT="$RUN/epoch_last"
  RUN_NAME=$(basename "$RUN")
  EVAL_TAG="quick3_${RUN_NAME}"
  OUT="$FRESH/eval-results/main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}_quick3"
  EVAL_DIR="$OUT/runs/$EVAL_TAG"
  LOG_DIR="$FRESH/logs/table2_tag_tune_eval"
  LOG="$LOG_DIR/seed${SEED}_${RUN_NAME}_quick3.log"
  STATE="$FRESH/status/table2_tag_tune_eval/seed${SEED}_${RUN_NAME}_quick3"
}

check_training() {
  [[ "$(<"$CKPT/_complete")" == 3 ]] || die "training checkpoint is not epoch 3"
  [[ ! -e "$CKPT/_save_errors.json" ]] || die "training checkpoint has save errors"
  "$PY" - "$RUN" "$SEED" <<'PY'
import json, math, sys
from pathlib import Path

run, seed = Path(sys.argv[1]), int(sys.argv[2])
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed
assert cfg["training_mode"] == "full"
assert cfg["selection_ratio"] == 0.1
assert cfg["grad_accum"] == 4
assert cfg["batch_size"] * cfg["launch_world_size"] * cfg["grad_accum"] == 64
assert math.isclose(float(cfg["min_lr_ratio"]), 0.10)
assert cfg.get("adamw_foreach") is False
print("TRAIN_INPUT_OK", run / "epoch_last")
PY
}

preflight() {
  resolve
  [[ -x "$PY" ]] || die "python missing"
  [[ -f "$CFG" ]] || die "config missing"
  [[ -d "$OLD/eval-data/mmlu" ]] || die "MMLU data missing"
  [[ -d /group-volume/datasets/gsm8k/datasets/openai/gsm8k ]] || die "GSM8K data missing"
  [[ -d /group-volume/datasets/mbpp ]] || die "MBPP data missing"
  [[ ! -e "$EVAL_DIR" ]] || die "evaluation already exists; use status"
  cd "$REPO"
  git diff --quiet || die "tracked code changes present"
  git diff --cached --quiet || die "staged code changes present"
  nvidia-smi -i "$GPU" >/dev/null || die "GPU $GPU not visible"
  local free
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [[ "$free" -ge 78000 ]] || die "GPU $GPU is busy (${free} MiB free)"
  check_training
  mkdir -p "$LOG_DIR" "$(dirname "$STATE")" "$FRESH/locks"
}

validate_eval() {
  "$PY" - "$EVAL_DIR" "$EVAL_TAG" "$CKPT" "$SEED" <<'PY'
import json, math, sys
from pathlib import Path

run, tag, ckpt, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
assert (run / "_complete").read_text().strip() == tag
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed and cfg["ckpt"] == ckpt
summaries = list(run.glob("*-eval_summary.json"))
assert len(summaries) == 1, summaries
payload = json.loads(summaries[0].read_text())
assert payload["limit"] is None and payload["failures"] == []
assert [x["benchmark"] for x in payload["summaries"]] == ["mbpp", "gsm8k", "mmlu"]

expected = {"mbpp": 257, "gsm8k": 1319, "mmlu": 14042}
scores = {}
for bench, n in expected.items():
    paths = list(run.glob(f"*-{bench}.json"))
    assert len(paths) == 1, (bench, paths)
    result = json.loads(paths[0].read_text())
    score = 100.0 * float(result["accuracy"])
    assert math.isfinite(score) and 0.0 <= score <= 100.0
    if bench == "mbpp":
        assert result["total_questions"] == n and result["generation_batch_size"] == 16
    elif bench == "gsm8k":
        assert result["total"] == n and result["generation_batch_size"] == 16
    else:
        assert result["total_questions"] == n and result["num_subjects"] == 57
    scores[bench] = score

text = " ".join(f"{b.upper()}={scores[b]:.2f}" for b in ("mbpp", "gsm8k", "mmlu"))
(run / "SCORES.txt").write_text(text + "\n")
(run / "EVAL_VALIDATED").write_text("ok\n")
print("EVAL_VALIDATED", text)
PY
}

worker() {
  preflight
  exec 9>"$FRESH/locks/seed${SEED}_${RUN_NAME}_quick3.lock"
  flock -n 9 || die "evaluation worker already running"
  printf 'host=%s\npid=%s\nstarted_utc=%s\n' \
    "$(hostname -s)" "$$" "$(date -u +%FT%TZ)" >"$STATE.running"

  on_exit() {
    local rc=$?
    rm -f "$STATE.running"
    printf 'host=%s\nrc=%s\nfinished_utc=%s\n' \
      "$(hostname -s)" "$rc" "$(date -u +%FT%TZ)" >"$STATE.exit"
    echo "JOB_EXIT_CODE=$rc"
  }
  trap on_exit EXIT

  export TAG_WORKSPACE="$FRESH"
  export HF_HOME="$FRESH/hf_home"
  export TAG_ENV_RESET=1
  source "$REPO/scripts/gpu_cloud/env.sh"
  unset TAG_ENV_RESET

  echo "EVAL_START seed=$SEED gpu=$GPU order=mbpp,gsm8k,mmlu"
  echo "CKPT=$CKPT"
  echo "EVAL_DIR=$EVAL_DIR"

  cd "$REPO"
  env -u LD_PRELOAD -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$REPO" \
    HF_HOME="$FRESH/hf_home" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    TAG_EVAL_GEN_BS=16 \
    TAG_GSM8K_USE_SFT_WRAP=0 \
    TAG_MBPP_USE_SFT_WRAP=0 \
    "$PY" -m tag.eval \
      --config "$CFG" \
      --ckpt "$CKPT" \
      --benchmarks mbpp,gsm8k,mmlu \
      --out_dir "$OUT" \
      --eval_tag "$EVAL_TAG" \
      --training_mode full \
      --cuda_device 0 \
      --mbpp_data_dir /group-volume/datasets/mbpp \
      --gsm8k_data_dir /group-volume/datasets/gsm8k/datasets/openai/gsm8k \
      --mmlu_data_dir "$OLD/eval-data/mmlu"

  validate_eval
  echo "EVALUATION_COMPLETE seed=$SEED"
}

status() {
  resolve
  echo "seed=$SEED gpu=$GPU"
  echo "checkpoint=$CKPT"
  echo "eval_dir=$EVAL_DIR"
  if [[ -f "$EVAL_DIR/EVAL_VALIDATED" ]]; then
    echo "STATUS=EVAL_VALIDATED"
    cat "$EVAL_DIR/SCORES.txt"
  elif [[ -f "$STATE.running" ]]; then
    echo "STATUS=RUNNING"
    cat "$STATE.running"
  elif [[ -f "$STATE.exit" ]]; then
    echo "STATUS=EXITED"
    cat "$STATE.exit"
  elif [[ -e "$EVAL_DIR" ]]; then
    echo "STATUS=INCOMPLETE"
  else
    echo "STATUS=NOT_STARTED"
  fi
  if [[ -f "$LOG" ]]; then
    echo "===== LOG TAIL ====="
    tail -n 40 "$LOG"
  fi
}

launch() {
  preflight
  nohup bash "$0" --worker >"$LOG" 2>&1 </dev/null &
  local pid=$!
  sleep 4
  if kill -0 "$pid" 2>/dev/null; then
    echo "STARTED seed=$SEED pid=$pid order=MBPP,GSM8K,MMLU"
    echo "STATUS: S=$SEED bash eval.sh status"
  else
    echo "START_FAILED"
    tail -n 80 "$LOG" || true
    exit 1
  fi
}

case "${1:-}" in
  --worker) worker ;;
  status) status ;;
  "") launch ;;
  *) die "usage: S={1|7|42} bash eval.sh [status]" ;;
esac
