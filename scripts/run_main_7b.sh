#!/usr/bin/env bash
# Run the main 7B experiment matrix on 4 × A100 80GB.
#
# Default mode is sequential DDP — each (model, method) experiment uses all
# 4 GPUs together via `torchrun --nproc_per_node=4`. This minimises wall
# time per experiment without changing the paper-faithful effective batch.
#
# Filter via env vars:
#   MODELS="llama2 qwen25" METHODS="tads_50 random_50" \
#       bash scripts/run_main_7b.sh
#
# Legacy parallel mode (1 GPU per concurrent experiment, e.g. for the
# 0.5B / LoRA experiments) is still available via `--parallel`.
set -euo pipefail

cd "$(dirname "$0")/.."

# Load paths if not already exported.
if [ -z "${OUTPUT_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS:-"llama2 qwen25 mistral deepseek"}
# Default methods = the 4-experiment main matrix:
#   full_100        — full data (selection_ratio 1.0)
#   random_10       — random 10% baseline
#   data_agent_10   — ours, λ=0, 10%
#   tads_10         — ours, λ=1, 10%
# The legacy 50% selection variants (tads_50/data_agent_50/random_50) are
# still available; set METHODS=... to include them when needed.
METHODS=${METHODS:-"full_100 random_10 data_agent_10 tads_10"}
NPROC=${NPROC:-4}                # GPUs per DDP job
MASTER_PORT=${MASTER_PORT:-29500}
PARALLEL=0
for arg in "$@"; do
  case "$arg" in
    --parallel) PARALLEL=1 ;;
    --nproc=*) NPROC="${arg#*=}" ;;
    -h|--help)
      sed -n '2,22p' "$0"; exit 0 ;;
  esac
done

mkdir -p logs

run_ddp() {
  local model=$1 method=$2
  local cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  local log="logs/main_7b_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi
  echo "=== ${model} / ${method} (torchrun --nproc_per_node=${NPROC}) ==="
  torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m tads.train --config "$cfg" 2>&1 | tee -a "$log"
}

run_parallel_single_gpu() {
  local model=$1 method=$2 gpu=$3
  local cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  local log="logs/main_7b_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi
  echo "=== ${model} / ${method} (GPU ${gpu}, background) ==="
  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup python -m tads.train --config "$cfg" \
    >> "$log" 2>&1 &
}

if [ "$PARALLEL" = "1" ]; then
  gpu=0
  pids=()
  for model in $MODELS; do
    for method in $METHODS; do
      run_parallel_single_gpu "$model" "$method" "$gpu"
      pids+=($!)
      gpu=$((gpu + 1))
    done
  done
  echo "Launched ${#pids[@]} jobs in background. PIDs: ${pids[*]}"
  echo "Tail logs/main_7b_*.log for progress."
  wait "${pids[@]}"
else
  for model in $MODELS; do
    for method in $METHODS; do
      run_ddp "$model" "$method"
    done
  done
fi
