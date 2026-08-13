#!/usr/bin/env bash
# Self-contained environment for running TAG on a generic GPU box
# (Lambda / RunPod / vast.ai / a bare A100 VM).
#
#   source scripts/gpu_cloud/env.sh
#
# Unlike scripts/setup_env.sh — which points at the n9 cluster's
# /group-volume mounts — everything here lives under ONE workspace
# directory, so a fresh machine needs no pre-existing filesystem layout.
#
# Override the workspace before sourcing if you have a big scratch disk:
#   TAG_WORKSPACE=/mnt/data/tag source scripts/gpu_cloud/env.sh

# Resolve the repo root even when sourced from another directory.
_TAG_ENV_SRC="${BASH_SOURCE[0]:-$0}"
export TAG_REPO_ROOT="$(cd "$(dirname "$_TAG_ENV_SRC")/../.." && pwd)"

# Everything the run needs lives here: weights, datasets, pools, outputs.
export TAG_WORKSPACE="${TAG_WORKSPACE:-$TAG_REPO_ROOT/workspace}"

export MODEL_PATH_QWEN25_05B="${MODEL_PATH_QWEN25_05B:-$TAG_WORKSPACE/models/qwen2.5-0.5b}"
export MODEL_PATH_QWEN25_7B="${MODEL_PATH_QWEN25_7B:-$TAG_WORKSPACE/models/qwen2.5-7b}"

export POOLS="${POOLS:-$TAG_WORKSPACE/pools}"
export ALPACA_RAW_JSON="${ALPACA_RAW_JSON:-$TAG_WORKSPACE/datasets/alpaca_gpt4.json}"

# The candidate pool the training run actually selects from. bootstrap.sh
# generates it; until then it does not exist and preflight will say so.
export ALPACA_DATA_FILES="${ALPACA_DATA_FILES:-$POOLS/composite20/pool.json}"
export TADS_CF_FILES="${TADS_CF_FILES:-$POOLS/composite20/counterfactual.json}"
export TADS_DEDUP_FILE="${TADS_DEDUP_FILE:-$POOLS/composite20/dedup_clusters.json}"
export TADS_GATE_REF="${TADS_GATE_REF:-$POOLS/clean_ref/delta_hat_05b.pt}"

# Never let the HF hub be consulted for the TRAINING data — a silent hub
# fallback is how you end up training on a different pool than you think.
# bootstrap.sh flips these off only for the download step.
export ALPACA_DATASET_NAME=""
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$TAG_WORKSPACE/hf_home}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$TAG_WORKSPACE/runs}"
export DATA_CACHE="${DATA_CACHE:-$TAG_WORKSPACE/cache}"

# The PCA in TrajectoryAnchor.update runs torch.linalg.eigh per layer; on
# hosts with many cores each call spawns OMP_NUM_THREADS workers in tight
# succession and can blow past the container's pids limit.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$TAG_WORKSPACE"/{models,datasets,pools,runs,cache,hf_home}

if [ -z "${TAG_ENV_QUIET:-}" ]; then
  echo "[tag-env] workspace : $TAG_WORKSPACE"
  echo "[tag-env] model     : $MODEL_PATH_QWEN25_05B"
  echo "[tag-env] pool      : $ALPACA_DATA_FILES"
  echo "[tag-env] gate ref  : $TADS_GATE_REF"
  echo "[tag-env] outputs   : $OUTPUT_ROOT"
  echo "[tag-env] next      : python scripts/gpu_cloud/preflight.py"
fi
