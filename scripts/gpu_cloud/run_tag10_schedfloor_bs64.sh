#!/usr/bin/env bash

# Launch the isolated TAG tuning pilot on any idle 2- or 4-H100 node.
#
#   2 GPUs: batch 8 x grad_accum 4 x world 2 = effective batch 64
#   4 GPUs: batch 4 x grad_accum 4 x world 4 = effective batch 64
#
# The code checkout may be separate from the active /tag2 checkout.  Outputs
# and immutable data assets stay in the shared /tag2/workspace by default.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run this file with: bash ${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
FRESH=${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
CFG=configs/experiments/main_7b/llama2/tag_10_schedfloor_bs64.yaml
NCCL_FIX=/group-volume/jieuns.shin/Unlearning/experiment/benchmark/openunlearning/third_party/open-unlearning/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
GRAD_ACCUM=4
MIN_LR_RATIO=0.10

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve_run() {
  HOST=$(hostname -s)
  SEED=${TAG10_TUNE_SEED:-42}
  case "$SEED" in
    1|7|42) ;;
    *) die "TAG10_TUNE_SEED must be 1, 7, or 42; got $SEED" ;;
  esac

  mapfile -t VISIBLE_GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '
  )
  NPROC=${#VISIBLE_GPUS[@]}
  case "$NPROC" in
    2) BATCH=8 ;;
    4) BATCH=4 ;;
    *) die "$HOST sees $NPROC GPU(s); this pilot accepts exactly 2 or 4" ;;
  esac
  GPUS=$(IFS=,; echo "${VISIBLE_GPUS[*]}")

  cd "$REPO"
  PIN=$(git rev-parse HEAD)
  SHORT_PIN=${PIN:0:7}
  CELL="main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}"
  RUN_TAG="tag10tune_${SHORT_PIN}_seed${SEED}_${NPROC}g_b${BATCH}_ga4_floor010_bs64"
  RUN_DIR="$FRESH/runs/$CELL/runs/$RUN_TAG"
  LOG_DIR="$FRESH/logs/table2_tag_tune"
  LOG="$LOG_DIR/$RUN_TAG.log"
  STATE_DIR="$FRESH/status/table2_tag_tune"
  GATE="$FRESH/cache/table2_tag_tune/gates/seed${SEED}/tag_gate_llama2-7b_prefix32.pt"
  HIST_GATE="$OLD/pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt"
}

install_gate_copy() {
  mkdir -p "$(dirname "$GATE")"
  if [[ ! -f "$GATE" ]]; then
    local tmp="${GATE}.copy.$(hostname -s).$$"
    cp --reflink=auto "$HIST_GATE" "$tmp"
    cmp -s "$tmp" "$HIST_GATE" || die "gate copy verification failed: $tmp"
    mv "$tmp" "$GATE"
    echo "GATE_COPY_INSTALLED=$GATE"
  fi
  cmp -s "$GATE" "$HIST_GATE" || \
    die "tuning gate differs from validated historical gate: $GATE"
}

preflight() {
  resolve_run
  cd "$REPO"
  git diff --quiet || die "tracked worktree changes present under $REPO"
  git diff --cached --quiet || die "staged changes present under $REPO"
  [[ -x "$PY" ]] || die "python missing: $PY"
  [[ -f "$CFG" ]] || die "config missing: $CFG"
  [[ -f "$NCCL_FIX" ]] || die "NCCL fix missing: $NCCL_FIX"
  [[ -f "$OLD/pools/alpaca_gpt4/pool.json" ]] || die "main pool missing"
  [[ -f "$OLD/pools/alpaca_gpt4/counterfactual.json" ]] || die "counterfactual missing"
  [[ -f "$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt" ]] || die "gate reference missing"
  [[ -f "$HIST_GATE" ]] || die "historical gate missing"
  install_gate_copy

  mapfile -t free_mib < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  local mib
  for mib in "${free_mib[@]}"; do
    [[ "$mib" -ge 78000 ]] || \
      die "a GPU has only ${mib} MiB free; every visible GPU must be idle"
  done
  [[ ! -e "$RUN_DIR" ]] || \
    die "pilot run already exists: $RUN_DIR (use --status; do not overwrite)"
  mkdir -p "$LOG_DIR" "$STATE_DIR" "$FRESH/locks"
}

validate_training() {
  [[ "$(<"$RUN_DIR/epoch_last/_complete")" == 3 ]] || \
    die "epoch_last is not sealed at epoch 3"
  [[ ! -e "$RUN_DIR/epoch_last/_save_errors.json" ]] || \
    die "checkpoint save errors present"

  "$PY" - "$RUN_DIR" "$SEED" "$NPROC" "$BATCH" "$PIN" "$OLD" "$GATE" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from tag.core.schedulers import optimizer_steps_per_epoch

run = Path(sys.argv[1])
seed, world, batch = map(int, sys.argv[2:5])
pin, old, gate = sys.argv[5], Path(sys.argv[6]), Path(sys.argv[7])
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

tag = cfg["selection"]["tag"]
assert cfg["selection"]["score_mode"] == "tag"
assert tag["counterfactual_data_files"] == str(old / "pools/alpaca_gpt4/counterfactual.json")
assert tag["gate_ref_file"] == str(old / "pools/clean_ref/delta_hat_llama2_prefix.pt")
assert Path(tag["gate_cache_file"]) == gate
assert (tag["prefix_tokens"], tag["span_tokens"]) == (32, 16)
assert (tag["tau"], tag["tau_mode"], tag["tail_mode"]) == (0.5, "per_token", "none")
assert tag["null_correction"] is True and tag["target_zero_rate"] == 0.05

rows = json.loads((run / "metrics.json").read_text())
assert [row["epoch"] for row in rows] == [1, 2, 3]
for row in rows:
    assert row["n_total"] == 52002 and row["selected_n"] == 5200
    assert row["score_mode"] == "tag" and row["n_zero_weight_selected"] == 0
    assert 0.045 <= row["gate_zero_frac"] <= 0.055
    assert math.isfinite(row["train_loss"])

for epoch in (1, 2, 3):
    ids = json.loads((run / f"selected_indices_epoch{epoch}.json").read_text())
    assert len(ids) == len(set(ids)) == 5200
    assert all(type(x) is int and 0 <= x < 52002 for x in ids)

steps_per_epoch = optimizer_steps_per_epoch(5200, batch, 4, world)
assert steps_per_epoch == 82
expected_total = steps_per_epoch * 3
scheduler = torch.load(
    run / "epoch_last/scheduler.pt", map_location="cpu", weights_only=False,
)
assert scheduler["last_epoch"] == expected_total, scheduler["last_epoch"]
last_lr = float(scheduler["_last_lr"][0])
base_lr = float(cfg["learning_rate"])
assert math.isclose(last_lr, base_lr * 0.10, rel_tol=1e-5, abs_tol=1e-12)

weights = list((run / "epoch_last").glob("*.safetensors"))
assert weights and sum(p.stat().st_size for p in weights) > 10_000_000_000

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

assert sha(gate) == sha(old / "pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt")
print(
    "TRAIN_VALIDATED",
    run,
    f"steps_per_epoch={steps_per_epoch}",
    f"total_steps={expected_total}",
    f"last_lr={last_lr:.3e}",
)
PY
  touch "$RUN_DIR/TRAIN_VALIDATED"
}

worker() {
  preflight
  exec 9>"$FRESH/locks/$RUN_TAG.lock"
  flock -n 9 || die "worker already active for $RUN_TAG"
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
  export TAG_GATE_CACHE_LLAMA2="$GATE"
  export PYTHONHASHSEED=0
  export TAG_ENABLE_NO_SYNC=0
  export TAG_NCCL_REINIT=0

  echo "TRAIN_START host=$HOST seed=$SEED nproc=$NPROC batch=$BATCH grad_accum=$GRAD_ACCUM effective_batch=64 min_lr_ratio=$MIN_LR_RATIO"
  echo "CODE_HEAD=$PIN"
  echo "RUN_DIR=$RUN_DIR"
  echo "GATE=$GATE"

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
  echo "TRAINING_COMPLETE seed=$SEED run_dir=$RUN_DIR"
}

status() {
  resolve_run
  echo "host=$HOST seed=$SEED effective_batch=64"
  echo "code_head=$PIN"
  echo "run_dir=$RUN_DIR"
  echo "log=$LOG"
  pgrep -af "tag.train.*$RUN_TAG|torch.distributed.run.*$RUN_TAG" || true
  if [[ -f "$RUN_DIR/TRAIN_VALIDATED" ]]; then
    echo "STATUS=TRAIN_VALIDATED"
  elif [[ -f "$STATE_DIR/$RUN_TAG.running" ]]; then
    echo "STATUS=RUNNING"
    cat "$STATE_DIR/$RUN_TAG.running"
  elif [[ -f "$STATE_DIR/$RUN_TAG.exit" ]]; then
    echo "STATUS=EXITED"
    cat "$STATE_DIR/$RUN_TAG.exit"
  else
    echo "STATUS=NOT_STARTED"
  fi
  if [[ -f "$LOG" ]]; then
    echo "===== LOG TAIL ====="
    tail -n 60 "$LOG"
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
}

launch() {
  preflight
  nohup bash "$0" --worker >"$LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$LOG.pid"
  sleep 5
  if kill -0 "$pid" 2>/dev/null; then
    echo "STARTED host=$HOST seed=$SEED pid=$pid"
    echo "LOG=$LOG"
    echo "STATUS: TAG10_TUNE_SEED=$SEED bash $0 --status"
  else
    echo "START_FAILED host=$HOST seed=$SEED"
    tail -n 100 "$LOG" || true
    exit 1
  fi
}

case "${1:-}" in
  --worker) worker ;;
  --status) status ;;
  "") launch ;;
  *) die "usage: TAG10_TUNE_SEED={1|7|42} bash $0 [--status]" ;;
esac
