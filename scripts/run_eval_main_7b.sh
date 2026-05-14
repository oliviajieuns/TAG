#!/usr/bin/env bash
# Evaluate every checkpoint produced by run_main_7b.sh.
#
# For each (model, method) in main_7b/, locate the most recent saved
# epoch under ${OUTPUT_ROOT}/main_7b/<model>/<method>/epoch_* and run
# `python -m tads.eval` on it. Results land under
# ${EVAL_RESULTS_ROOT}/<model>/<method>/.
#
# Filters:
#   MODELS="llama2 qwen25" METHODS="tads_50" \
#       BENCHMARKS="mmlu,gsm8k" bash scripts/run_eval_main_7b.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS:-"llama2 qwen25 mistral deepseek"}
METHODS=${METHODS:-"tads_50 data_agent_50 random_50 full_100"}
BENCHMARKS=${BENCHMARKS:-"mmlu,gsm8k,humaneval,tydiqa"}
LIMIT=${LIMIT:-}          # set e.g. LIMIT=200 for smoke runs
CUDA=${CUDA_VISIBLE_DEVICES:-0}
PARALLEL=0
for arg in "$@"; do
  case "$arg" in
    --parallel) PARALLEL=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
  esac
done

mkdir -p logs

latest_epoch() {
  local dir=$1
  ls -1d "$dir"/epoch_* 2>/dev/null | sort -V | tail -n 1
}

eval_one() {
  local model=$1 method=$2 gpu=$3
  local cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  local ckpt_root="${OUTPUT_ROOT}/main_7b/${model}/${method}"
  local out_dir="${EVAL_RESULTS_ROOT}/${model}/${method}"
  local log="logs/eval_main_7b_${model}_${method}.log"

  local ckpt
  ckpt=$(latest_epoch "$ckpt_root")
  if [ -z "$ckpt" ]; then
    echo "[skip] no checkpoint under $ckpt_root" >&2
    return 0
  fi
  mkdir -p "$out_dir"

  echo "=== eval ${model}/${method}  ckpt=${ckpt} -> ${out_dir} ==="
  local extra_args=()
  if [ -n "$LIMIT" ]; then extra_args+=(--limit "$LIMIT"); fi

  if [ "$PARALLEL" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tads.eval \
      --config "$cfg" --ckpt "$ckpt" --benchmarks "$BENCHMARKS" \
      --out_dir "$out_dir" --cuda_device 0 \
      "${extra_args[@]}" >> "$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" python -m tads.eval \
      --config "$cfg" --ckpt "$ckpt" --benchmarks "$BENCHMARKS" \
      --out_dir "$out_dir" --cuda_device 0 \
      "${extra_args[@]}" 2>&1 | tee -a "$log"
  fi
}

if [ "$PARALLEL" = "1" ]; then
  gpu=0
  pids=()
  for model in $MODELS; do
    for method in $METHODS; do
      eval_one "$model" "$method" "$gpu"
      pids+=($!)
      gpu=$((gpu + 1))
    done
  done
  echo "Launched ${#pids[@]} eval jobs. Tail logs/eval_main_7b_*.log for progress."
  wait "${pids[@]}"
else
  for model in $MODELS; do
    for method in $METHODS; do
      eval_one "$model" "$method" "$CUDA"
    done
  done
fi
