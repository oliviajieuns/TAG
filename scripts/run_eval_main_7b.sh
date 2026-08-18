#!/usr/bin/env bash
# Evaluate every checkpoint produced by run_main_7b.sh.
#
# For each (model, method) in main_7b/, locate the most recent saved
# epoch under ${OUTPUT_ROOT}/main_7b/<model>/<method>/ — prefers
# epoch_last/ (tag.train layout) and falls back to the largest
# epoch_N/ (comparison-baseline layout) — and run `python -m tag.eval`
# on it. Results land under ${EVAL_RESULTS_ROOT}/<model>/<method>/.
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
#   MODELS="llama2 qwen25" METHODS="legacy_10" \
#       BENCHMARKS="mmlu,gsm8k" bash scripts/run_eval_main_7b.sh
#
# Other experiment sets
# ---------------------
# SET names the directory under configs/experiments/ and under OUTPUT_ROOT.
# Sets without a per-model level (lowq, clean) pass MODELS="":
#   SET=lowq MODELS="" METHODS="tag_prefix_7b legacy_7b" \
#       bash scripts/run_eval_main_7b.sh --gpus 0,1 --parallel
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS-"llama2 qwen25 mistral deepseek"}
# "-" is the no-per-model-level marker; smoke/lowq/clean lay out as
# <set>/<method>. Note ${MODELS-...}, not ${MODELS:-...}: with :- an
# explicit MODELS="" falls back to the default list, which is how a smoke
# run went looking for configs under smoke/llama2/.
[ -z "${MODELS// /}" ] && MODELS="-"
METHODS=${METHODS:-"full_100 random_10 data_agent_10 legacy_10"}
# The paper's Table 2, in its order.
BENCHMARKS=${BENCHMARKS:-"mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad"}
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

# Refuse to start rather than die on the fifth benchmark an hour in. The
# check is on the files each evaluator opens, not on a directory existing.
if ! python scripts/check_eval_data.py --benchmarks "$BENCHMARKS"; then
  exit 2
fi

echo "[run_eval] SET=$SET  GPUS=$GPUS  PARALLEL=$PARALLEL  BENCHMARKS=$BENCHMARKS"
echo "[run_eval] MODELS=$MODELS"
echo "[run_eval] METHODS=$METHODS"

mkdir -p logs

latest_epoch() {
  # Resolve <ckpt_root> -> the sealed ckpt directory.
  # Priority:
  #   (1) new epoch_last/ layout (2026-05-16~):
  #         <ckpt_root>/_latest/epoch_last/_complete exists
  #         → returns <ckpt_root>/_latest/epoch_last
  #   (2) legacy numeric epoch_N/ (largest N with _complete sentinel)
  #   (3) pre-sentinel legacy (largest epoch_N regardless)
  local ckpt_root=$1
  local run_dir=""

  # Resolve _latest pointer (symlink, dir, or _latest.txt fallback).
  if [ -L "${ckpt_root}/_latest" ] || [ -d "${ckpt_root}/_latest" ]; then
    run_dir=$(readlink -f "${ckpt_root}/_latest" 2>/dev/null || echo "${ckpt_root}/_latest")
  elif [ -f "${ckpt_root}/_latest.txt" ]; then
    local tag
    tag=$(cat "${ckpt_root}/_latest.txt")
    [ -d "${ckpt_root}/runs/${tag}" ] && run_dir="${ckpt_root}/runs/${tag}"
  fi
  if [ -z "$run_dir" ]; then
    run_dir="$ckpt_root"
  fi

  # (1) New epoch_last/ layout — wins over any numeric dir.
  if [ -f "${run_dir}/epoch_last/_complete" ]; then
    echo "${run_dir}/epoch_last"
    return
  fi

  # (2) Legacy: largest sealed numeric epoch_N/.
  local last=""
  while IFS= read -r p; do
    [ "$(basename "$p")" = "epoch_last" ] && continue
    [ -f "${p}/_complete" ] && last="$p"
  done < <(ls -1d "${run_dir}"/epoch_* 2>/dev/null | sort -V)

  # (3) Pre-sentinel legacy: largest epoch_N regardless of _complete.
  if [ -z "$last" ]; then
    last=$(ls -1d "${run_dir}"/epoch_* 2>/dev/null \
           | grep -v '/epoch_last$' | sort -V | tail -n 1)
  fi
  echo "$last"
}

eval_one() {
  local model=$1 method=$2 gpu=$3
  local seg=""
  [ "$model" != "-" ] && seg="/${model}"
  local cfg="configs/experiments/${SET}${seg}/${method}.yaml"
  local ckpt_root="${OUTPUT_ROOT}/${SET}${seg}/${method}"
  # <set>/<model>/<method> is the shape make_table_v2 --results-root walks.
  local out_dir="${EVAL_RESULTS_ROOT}/${SET}${seg}/${method}"
  local log="logs/eval_${SET}_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi

  local ckpt
  ckpt=$(latest_epoch "$ckpt_root")
  if [ -z "$ckpt" ]; then
    echo "[skip] no checkpoint under $ckpt_root" >&2
    return 0
  fi
  mkdir -p "$out_dir"

  echo "=== eval ${SET}${seg}/${method}  ckpt=${ckpt} -> ${out_dir} (GPU ${gpu}) ==="
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
      # eval_one returns early when the config or the checkpoint is missing,
      # leaving $! from whatever ran before — or unset, which `set -u` turns
      # into a hard stop that takes the whole run down.
      [ -n "${!:-}" ] && pids+=($!)
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
