#!/usr/bin/env bash
# Run the Evol-Instruct (Table 5) 7B experiment matrix.
#
# Difference vs. scripts/run_main_7b.sh:
#   - Pulls configs from configs/experiments/evol_7b/ (Evol-Instruct dataset).
#   - Launches plain `python -m tag.train` (single-process), NOT torchrun.
#     The DDP path has an unresolved bug; single-process is the supported
#     launcher for tag.train. Data Agent uses its own baselines.data_agent
#     entry point, also single-process.
#
# GPU selection
# -------------
#   bash scripts/run_evol_7b.sh --gpus 0
#   bash scripts/run_evol_7b.sh --gpus 4,5,6,7 --parallel
#   GPUS="0,1" bash scripts/run_evol_7b.sh --parallel
# Sequential mode pins every job to the first GPU in the list. Parallel mode
# cycles concurrent jobs across every GPU (one job per GPU at a time).
#
# Filter examples
# ---------------
#   MODELS="llama2" METHODS="legacy_10 random_10" \
#       bash scripts/run_evol_7b.sh --gpus 0
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

MODELS=${MODELS:-"llama2 qwen25"}
METHODS=${METHODS:-"full_100 random_10 data_agent_10 legacy_10"}
GPUS=${GPUS:-"0"}
PARALLEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel)        PARALLEL=1; shift ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --gpus=*)          GPUS="${1#*=}"; shift ;;
    -h|--help)         sed -n '2,24p' "$0"; exit 0 ;;
    *)                 echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done

IFS=',' read -r -a _gpus_arr <<< "$GPUS"

echo "[run_evol_7b] GPUS=$GPUS  PARALLEL=$PARALLEL"
echo "[run_evol_7b] MODELS=$MODELS"
echo "[run_evol_7b] METHODS=$METHODS"

mkdir -p logs

# Pick the entry-point module by method. data_agent uses the baselines launcher;
# everything else (full / random / legacy) goes through tag.train.
entry_for() {
  case "$1" in
    data_agent*) echo "baselines.data_agent.train" ;;
    *)           echo "tag.train" ;;
  esac
}

extra_args_for() {
  # baselines.data_agent.train requires an explicit --tag (used as the run dir).
  case "$1" in
    data_agent*) echo "--tag DataAgent-PPO" ;;
    *)           echo "" ;;
  esac
}

run_one() {
  local model=$1 method=$2 gpu=$3
  local cfg="configs/experiments/evol_7b/${model}/${method}.yaml"
  local log="logs/evol_7b_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi
  local entry; entry=$(entry_for "$method")
  local extra; extra=$(extra_args_for "$method")

  echo "=== ${model} / ${method} (GPU ${gpu}, $entry) ==="
  if [ "$PARALLEL" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" \
      nohup python -m "$entry" --config "$cfg" $extra \
      >> "$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" \
      python -m "$entry" --config "$cfg" $extra 2>&1 | tee -a "$log"
  fi
}

if [ "$PARALLEL" = "1" ]; then
  idx=0
  n_gpus=${#_gpus_arr[@]}
  pids=()
  for model in $MODELS; do
    for method in $METHODS; do
      gpu="${_gpus_arr[$((idx % n_gpus))]}"
      run_one "$model" "$method" "$gpu"
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  echo "Launched ${#pids[@]} jobs (cycled across GPUs ${GPUS}). PIDs: ${pids[*]}"
  echo "Tail logs/evol_7b_*.log for progress."
  wait "${pids[@]}"
else
  first_gpu="${_gpus_arr[0]}"
  for model in $MODELS; do
    for method in $METHODS; do
      run_one "$model" "$method" "$first_gpu"
    done
  done
fi
