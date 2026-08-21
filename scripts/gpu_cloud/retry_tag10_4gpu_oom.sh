#!/usr/bin/env bash

# Retry the failed Table-2 TAG seed on each 4-GPU node with the same
# effective global batch (4 x 2 x 16 = 128) and a lower activation peak.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run this file with: bash ${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=/group-volume/jieuns.shin/tag2
FRESH="$REPO/workspace"
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
PIN=ecde57be15c977be640f4e1ed9857a60a9860c12
CFG=configs/experiments/main_7b/llama2/tag_10.yaml
NCCL_FIX=/group-volume/jieuns.shin/Unlearning/experiment/benchmark/openunlearning/third_party/open-unlearning/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
NPROC=4
BATCH=2
GRAD_ACCUM=16
GPUS=0,1,2,3

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve_node() {
  HOST=$(hostname -s)
  case "$HOST" in
    run270622*) SEED=7 ;;
    run270630*) SEED=42 ;;
    *) die "this retry is only for run270622 (seed 7) or run270630 (seed 42); host=$HOST" ;;
  esac
  CELL="main_7b/llama2/tag_10_seed${SEED}"
  RUN_TAG="tag10_ecde57b_seed${SEED}_4g_b2_ga16_ncclfix"
  RUN_DIR="$FRESH/runs/$CELL/runs/$RUN_TAG"
  LOG_DIR="$FRESH/logs/table2_tag"
  LOG="$LOG_DIR/$RUN_TAG.log"
  GATE="$FRESH/cache/table2_tag/gates/seed${SEED}/tag_gate_llama2-7b_prefix32.pt"
  STATE_DIR="$FRESH/status/table2_tag"
}

preflight() {
  resolve_node
  cd "$REPO"
  [[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "repo HEAD is not $PIN"
  git diff --quiet || die "tracked worktree changes present"
  git diff --cached --quiet || die "staged changes present"
  [[ -x "$PY" ]] || die "python missing: $PY"
  [[ -f "$CFG" ]] || die "config missing: $CFG"
  [[ -f "$NCCL_FIX" ]] || die "NCCL fix missing: $NCCL_FIX"
  [[ -f "$OLD/pools/alpaca_gpt4/pool.json" ]] || die "main pool missing"
  [[ -f "$OLD/pools/alpaca_gpt4/counterfactual.json" ]] || die "counterfactual missing"
  [[ -f "$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt" ]] || die "gate reference missing"
  [[ -f "$OLD/pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt" ]] || die "historical gate missing"
  [[ -f "$GATE" ]] || die "seed gate missing: $GATE"
  cmp -s "$GATE" "$OLD/pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt" || \
    die "seed gate differs from validated historical gate: $GATE"
  mapfile -t visible_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  [[ "${#visible_gpus[@]}" -eq 4 ]] || die "$HOST sees ${#visible_gpus[@]} GPU(s), expected 4"
  mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  for mib in "${free_mib[@]}"; do
    [[ "$mib" -ge 78000 ]] || die "a GPU has only ${mib} MiB free; all four must be idle before retry"
  done
  [[ ! -e "$RUN_DIR" ]] || die "new retry run already exists: $RUN_DIR (use --status; do not overwrite)"
  mkdir -p "$LOG_DIR" "$STATE_DIR" "$FRESH/locks"
}

validate_training() {
  [[ "$(<"$RUN_DIR/epoch_last/_complete")" == 3 ]] || die "epoch_last is not sealed at epoch 3"
  [[ ! -e "$RUN_DIR/epoch_last/_save_errors.json" ]] || die "checkpoint save errors present"
  "$PY" - "$RUN_DIR" "$SEED" "$PIN" "$OLD" "$GATE" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

run, seed, pin, old, gate = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed and cfg["git_sha"] == pin
assert (cfg["launch_world_size"], cfg["batch_size"], cfg["grad_accum"]) == (4, 2, 16)
assert cfg["launch_world_size"] * cfg["batch_size"] * cfg["grad_accum"] == 128
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
    p = run / f"selected_indices_epoch{epoch}.json"
    ids = json.loads(p.read_text())
    assert len(ids) == len(set(ids)) == 5200
    assert all(type(x) is int and 0 <= x < 52002 for x in ids)
weights = list((run / "epoch_last").glob("*.safetensors"))
assert weights and sum(p.stat().st_size for p in weights) > 10_000_000_000
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
assert sha(gate) == sha(old / "pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt")
print("TRAIN_VALIDATED", run)
PY
  touch "$RUN_DIR/TRAIN_VALIDATED"
}

worker() {
  preflight
  exec 9>"$FRESH/locks/$RUN_TAG.lock"
  flock -n 9 || die "retry worker already active for $RUN_TAG"
  printf 'host=%s\npid=%s\nstarted_utc=%s\n' "$HOST" "$$" "$(date -u +%FT%TZ)" >"$STATE_DIR/$RUN_TAG.running"
  on_exit() {
    local rc=$?
    rm -f "$STATE_DIR/$RUN_TAG.running"
    printf 'host=%s\nrc=%s\nfinished_utc=%s\n' "$HOST" "$rc" "$(date -u +%FT%TZ)" >"$STATE_DIR/$RUN_TAG.exit"
    echo "JOB_EXIT_CODE=$rc"
  }
  trap on_exit EXIT

  export TAG_WORKSPACE="$FRESH"
  export HF_HOME="$FRESH/hf_home"
  export TAG_ENV_RESET=1
  source scripts/gpu_cloud/env.sh
  unset TAG_ENV_RESET

  export TAG_MAIN_POOL="$OLD/pools/alpaca_gpt4/pool.json"
  export TAG_MAIN_CF="$OLD/pools/alpaca_gpt4/counterfactual.json"
  export TAG_GATE_REF_LLAMA2="$OLD/pools/clean_ref/delta_hat_llama2_prefix.pt"
  export TAG_GATE_CACHE_LLAMA2="$GATE"
  export PYTHONHASHSEED=0
  export TAG_ENABLE_NO_SYNC=0
  export TAG_NCCL_REINIT=0
  export TAG_DL_NUM_WORKERS=0

  echo "TRAIN_START host=$HOST seed=$SEED nproc=$NPROC batch=$BATCH grad_accum=$GRAD_ACCUM effective_batch=$((NPROC * BATCH * GRAD_ACCUM))"
  echo "RUN_DIR=$RUN_DIR"
  echo "GATE=$GATE"
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
      output_subdir="$CELL" \
      git_sha="$PIN" \
      launch_world_size="$NPROC"

  validate_training
  printf 'validated_utc=%s\nrun_dir=%s\n' "$(date -u +%FT%TZ)" "$RUN_DIR" >"$STATE_DIR/$RUN_TAG.validated"
  echo "TRAINING_COMPLETE seed=$SEED run_dir=$RUN_DIR"
}

status() {
  resolve_node
  echo "host=$HOST seed=$SEED"
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
    echo "STATUS: bash $0 --status"
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
  *) die "usage: bash $0 [--status]" ;;
esac
