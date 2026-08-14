#!/usr/bin/env bash
# Evaluate every checkpoint produced by run_evol_7b.sh.
#
# Mirrors scripts/run_eval_main_7b.sh exactly — same epoch-resolution logic,
# same `python -m tag.eval` invocation — but pulls configs / checkpoints
# from the evol_7b/ tree.
#
# Table 5 reports MMLU / SVAMP / MBPP / TydiQA (the four diagnostic tasks).
# We still evaluate the full 9-benchmark suite so the result file is a
# drop-in replacement for the main_7b pipeline; scripts/make_table.sh
# decides which columns to emit. Override with BENCHMARKS=... to narrow.
#
# GPU selection
# -------------
#   bash scripts/run_eval_evol_7b.sh --gpus 0
#   bash scripts/run_eval_evol_7b.sh --gpus 4,5,6,7 --parallel
#
# Filters
# -------
#   MODELS="llama2" METHODS="legacy_10" \
#       BENCHMARKS="mmlu,svamp,mbpp,tydiqa" bash scripts/run_eval_evol_7b.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS:-"llama2 qwen25"}
METHODS=${METHODS:-"full_100 random_10 data_agent_10 legacy_10"}
# Full 9-benchmark suite (Table 1 / Table 5 share the same eval coverage).
BENCHMARKS=${BENCHMARKS:-"mmlu,mmlu_pro,gsm8k,svamp,humaneval,mbpp,tydiqa,xquad,bbh"}
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

echo "[run_eval_evol] GPUS=$GPUS  PARALLEL=$PARALLEL  BENCHMARKS=$BENCHMARKS"
echo "[run_eval_evol] MODELS=$MODELS"
echo "[run_eval_evol] METHODS=$METHODS"

mkdir -p logs

latest_epoch() {
  # Mirrors scripts/run_eval_main_7b.sh:latest_epoch — kept inline so this
  # script remains self-contained and copy-paste-able onto fresh hosts.
  local ckpt_root=$1
  local run_dir=""

  if [ -L "${ckpt_root}/_latest" ] || [ -d "${ckpt_root}/_latest" ]; then
    run_dir=$(readlink -f "${ckpt_root}/_latest" 2>/dev/null || echo "${ckpt_root}/_latest")
  elif [ -f "${ckpt_root}/_latest.txt" ]; then
    local tag; tag=$(cat "${ckpt_root}/_latest.txt")
    [ -d "${ckpt_root}/runs/${tag}" ] && run_dir="${ckpt_root}/runs/${tag}"
  fi
  if [ -z "$run_dir" ]; then
    run_dir="$ckpt_root"
  fi

  if [ -f "${run_dir}/epoch_last/_complete" ]; then
    echo "${run_dir}/epoch_last"
    return
  fi

  local last=""
  while IFS= read -r p; do
    [ "$(basename "$p")" = "epoch_last" ] && continue
    [ -f "${p}/_complete" ] && last="$p"
  done < <(ls -1d "${run_dir}"/epoch_* 2>/dev/null | sort -V)

  if [ -z "$last" ]; then
    last=$(ls -1d "${run_dir}"/epoch_* 2>/dev/null \
           | grep -v '/epoch_last$' | sort -V | tail -n 1)
  fi
  echo "$last"
}

eval_one() {
  local model=$1 method=$2 gpu=$3
  local cfg="configs/experiments/evol_7b/${model}/${method}.yaml"
  local ckpt_root="${OUTPUT_ROOT}/evol_7b/${model}/${method}"
  local out_dir="${EVAL_RESULTS_ROOT}/evol_7b/${model}/${method}"
  local log="logs/eval_evol_7b_${model}_${method}.log"

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
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tag.eval \
      --config "$cfg" --ckpt "$ckpt" --benchmarks "$BENCHMARKS" \
      --out_dir "$out_dir" --cuda_device 0 \
      "${extra_args[@]}" >> "$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" python -m tag.eval \
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
  echo "Launched ${#pids[@]} eval jobs (cycled across GPUs ${GPUS}). Tail logs/eval_evol_7b_*.log for progress."
  wait "${pids[@]}"
else
  first_gpu="${_gpus_arr[0]}"
  for model in $MODELS; do
    for method in $METHODS; do
      eval_one "$model" "$method" "$first_gpu"
    done
  done
fi
