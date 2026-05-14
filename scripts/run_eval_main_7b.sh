#!/usr/bin/env bash
# Evaluate every checkpoint produced by run_main_7b.sh.
#
# For each (model, method) in main_7b/, locate the most recent saved
# epoch under ${OUTPUT_ROOT}/main_7b/<model>/<method>/epoch_* and run
# `python -m tads.eval` on it. Results land under
# ${EVAL_RESULTS_ROOT}/<model>/<method>/.
#
# GPU selection
# -------------
#   bash scripts/run_eval_main_7b.sh --gpus 0
#   bash scripts/run_eval_main_7b.sh --gpus 4,5,6,7 --parallel
#   GPUS="0,1" bash scripts/run_eval_main_7b.sh --parallel
# Sequential mode uses the first GPU in the list. Parallel mode cycles
# concurrent jobs through every GPU in the list (one job per GPU at a time).
#
# Filters
# -------
#   MODELS="llama2 qwen25" METHODS="tads_10" \
#       BENCHMARKS="mmlu,gsm8k" bash scripts/run_eval_main_7b.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS:-"llama2 qwen25 mistral deepseek"}
METHODS=${METHODS:-"full_100 random_10 data_agent_10 tads_10"}
BENCHMARKS=${BENCHMARKS:-"mmlu,gsm8k,humaneval,tydiqa,bbh"}
LIMIT=${LIMIT:-}
GPUS=${GPUS:-"0"}
PARALLEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel)  PARALLEL=1; shift ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --gpus=*)    GPUS="${1#*=}"; shift ;;
    --limit)     LIMIT="$2"; shift 2 ;;
    --limit=*)   LIMIT="${1#*=}"; shift ;;
    -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
    *)           echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done

IFS=',' read -r -a _gpus_arr <<< "$GPUS"

echo "[run_eval] GPUS=$GPUS  PARALLEL=$PARALLEL  BENCHMARKS=$BENCHMARKS"
echo "[run_eval] MODELS=$MODELS"
echo "[run_eval] METHODS=$METHODS"

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

  echo "=== eval ${model}/${method}  ckpt=${ckpt} -> ${out_dir} (GPU ${gpu}) ==="
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
  idx=0
  n_gpus=${#_gpus_arr[@]}
  pids=()
  for model in $MODELS; do
    for method in $METHODS; do
      gpu="${_gpus_arr[$((idx % n_gpus))]}"
      eval_one "$model" "$method" "$gpu"
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  echo "Launched ${#pids[@]} eval jobs (cycled across GPUs ${GPUS}). Tail logs/eval_main_7b_*.log for progress."
  wait "${pids[@]}"
else
  # Sequential: pin every job to the first GPU in the list.
  first_gpu="${_gpus_arr[0]}"
  for model in $MODELS; do
    for method in $METHODS; do
      eval_one "$model" "$method" "$first_gpu"
    done
  done
fi
