#!/usr/bin/env bash
# Run the main 7B experiment matrix.
#
# Default mode is sequential DDP — each (model, method) experiment uses all
# selected GPUs together via `torchrun`. This minimises wall time per
# experiment without changing the paper-faithful effective batch.
#
# GPU selection
# -------------
#   bash scripts/run_main_7b.sh --gpus 0,1,2,3
#   bash scripts/run_main_7b.sh --gpus 4,5,6,7
#   GPUS="4,5,6,7" bash scripts/run_main_7b.sh
# `nproc_per_node` is derived from the count, so `--gpus 0,1` runs DDP on 2.
# (Override with --nproc=N if you want a different count for some reason.)
#
# Filter examples
# ---------------
#   MODELS="llama2 qwen25" METHODS="legacy_10 random_10" \
#       bash scripts/run_main_7b.sh
#
# SET names the directory under configs/experiments/ and under OUTPUT_ROOT.
# Sets without a per-model level (smoke, lowq, clean) pass MODELS="":
#   SET=smoke MODELS="" METHODS="legacy_q7 tag_q7" \
#       bash scripts/run_main_7b.sh --gpus 0,1,2,3
#
# Legacy parallel mode (1 GPU per concurrent experiment, e.g. for the
# 0.5B / LoRA experiments): pass `--parallel`. Each experiment is then
# bound to a single GPU from the GPUS list, cycled through.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

SET=${SET:-main_7b}
MODELS=${MODELS-"llama2 qwen25 mistral deepseek"}
# "-" is the no-per-model-level marker; smoke/lowq/clean lay out as
# <set>/<method> with no model directory between them.
[ -z "${MODELS// /}" ] && MODELS="-"
# Default methods = 4-experiment main matrix (paper-faithful 10% / full).
METHODS=${METHODS:-"full_100 random_10 data_agent_10 legacy_10"}
GPUS=${GPUS:-"0,1,2,3"}
NPROC=${NPROC:-}                 # if empty, derived from GPUS
# Default master_port: derived from the launching shell's PID so concurrent
# bash invocations (e.g. one tmux pane per model) get distinct ports without
# coordination. Collision probability across simultaneous launches is ~1/1000.
# Override with --master_port=29500 (or env MASTER_PORT=29500) for a fixed
# port — typically only useful for multi-node debugging.
MASTER_PORT=${MASTER_PORT:-$(( 29500 + $$ % 1000 ))}
# Passed straight through to tag.train --override. The one that matters:
# full_ft.yaml sizes batch_size 8 x grad_accum 4 x world_size 4 = 128, so a
# single-GPU run needs grad_accum=16 to keep the same effective batch. Set it
# for EVERY arm of a comparison or the arms differ by more than the method.
#   OVERRIDES="grad_accum=16" SET=smoke MODELS="" ... --parallel
OVERRIDES=${OVERRIDES:-}
PARALLEL=0

# ----- CLI args -----
while [ $# -gt 0 ]; do
  case "$1" in
    --parallel)        PARALLEL=1; shift ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --gpus=*)          GPUS="${1#*=}"; shift ;;
    --nproc)           NPROC="$2"; shift 2 ;;
    --nproc=*)         NPROC="${1#*=}"; shift ;;
    --master_port)     MASTER_PORT="$2"; shift 2 ;;
    --master_port=*)   MASTER_PORT="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,28p' "$0"; exit 0 ;;
    *)                 echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done

# Derive nproc from GPUS unless explicitly set.
if [ -z "$NPROC" ]; then
  NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
fi

# GPUS as an array (e.g. "0,1,2,3" -> (0 1 2 3)) for parallel mode.
IFS=',' read -r -a _gpus_arr <<< "$GPUS"

echo "[run_main_7b] GPUS=$GPUS  NPROC=$NPROC  PARALLEL=$PARALLEL"
echo "[run_main_7b] SET=$SET MODELS=$MODELS OVERRIDES=${OVERRIDES:-<none>}"
echo "[run_main_7b] METHODS=$METHODS"

mkdir -p logs

run_ddp() {
  local model=$1 method=$2
  local seg=""
  [ "$model" != "-" ] && seg="/${model}"
  local cfg="configs/experiments/${SET}${seg}/${method}.yaml"
  local log="logs/${SET}_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi
  echo "=== ${SET}${seg}/${method} (GPUs=${GPUS}, nproc=${NPROC}) ==="
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m tag.train --config "$cfg" \
    ${OVERRIDES:+--override $OVERRIDES} 2>&1 | tee -a "$log"
}

run_parallel_single_gpu() {
  local model=$1 method=$2 gpu=$3
  local seg=""
  [ "$model" != "-" ] && seg="/${model}"
  local cfg="configs/experiments/${SET}${seg}/${method}.yaml"
  local log="logs/${SET}_${model}_${method}.log"
  if [ ! -f "$cfg" ]; then
    echo "[skip] missing config: $cfg" >&2
    return 0
  fi
  echo "=== ${SET}${seg}/${method} (GPU ${gpu}, background) ==="
  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup python -m tag.train --config "$cfg" \
    ${OVERRIDES:+--override $OVERRIDES} \
    >> "$log" 2>&1 &
}

if [ "$PARALLEL" = "1" ]; then
  idx=0
  n_gpus=${#_gpus_arr[@]}
  pids=()
  for model in $MODELS; do
    for method in $METHODS; do
      gpu="${_gpus_arr[$((idx % n_gpus))]}"
      run_parallel_single_gpu "$model" "$method" "$gpu"
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  echo "Launched ${#pids[@]} jobs (cycled across GPUs ${GPUS}). PIDs: ${pids[*]}"
  echo "Tail logs/${SET}_*.log for progress."
  wait "${pids[@]}"
else
  for model in $MODELS; do
    for method in $METHODS; do
      run_ddp "$model" "$method"
    done
  done
fi
