#!/usr/bin/env bash
# Run the main 7B experiment matrix: 4 models × 4 methods (= 16 runs).
#
# Default: sequential on a single GPU (CUDA_VISIBLE_DEVICES respected if set).
# Parallel mode: pass `--parallel` to launch each model on a separate GPU
#                (assumes >= 4 GPUs visible; 1 model per GPU).
#
# Filter by model or method via env vars MODELS / METHODS, e.g.:
#   MODELS="llama2 qwen25" METHODS="tads data_agent" bash scripts/run_main_7b.sh
#
# Override which experiment names to use via METHODS (any of:
#   tads_50, data_agent_50, random_50, full_100).
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=${MODELS:-"llama2 qwen25 mistral deepseek"}
METHODS=${METHODS:-"tads_50 data_agent_50 random_50 full_100"}
PARALLEL=0
for arg in "$@"; do
  case "$arg" in
    --parallel) PARALLEL=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
  esac
done

run_one() {
  local model=$1
  local method=$2
  local gpu=$3
  local cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  local log="logs/main_7b_${model}_${method}.log"
  mkdir -p logs

  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi

  echo "=== ${model} / ${method} (GPU ${gpu}) ===" | tee -a "$log"
  if [ "$PARALLEL" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" \
      nohup python -m tads.train --config "$cfg" \
      >> "$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" \
      python -m tads.train --config "$cfg" \
      2>&1 | tee -a "$log"
  fi
}

gpu=0
pids=()
for model in $MODELS; do
  for method in $METHODS; do
    run_one "$model" "$method" "$gpu"
    if [ "$PARALLEL" = "1" ]; then
      pids+=($!)
      gpu=$((gpu + 1))
    fi
  done
done

if [ "$PARALLEL" = "1" ]; then
  echo "Launched ${#pids[@]} jobs in background. PIDs: ${pids[*]}"
  echo "Tail logs/main_7b_*.log for progress."
  wait "${pids[@]}"
fi
