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
# On a cluster with a per-user group volume, prefer that over the repo dir —
# $HOME is usually small and the repo may sit on an overlay fs.
_tag_pick_workspace() {
  if [ -n "${TAG_WORKSPACE:-}" ]; then echo "$TAG_WORKSPACE"; return; fi
  local u="${USER:-$(id -un 2>/dev/null)}"
  for cand in "/group-volume/$u/tag-workspace" "/mnt/data/tag-workspace"; do
    local parent
    parent="$(dirname "$cand")"
    if [ -d "$parent" ] && [ -w "$parent" ]; then echo "$cand"; return; fi
  done
  echo "$TAG_REPO_ROOT/workspace"
}
# Whether the caller pinned it, BEFORE _tag_pick_workspace fills it in.
if [ -n "${TAG_WORKSPACE:-}" ]; then _TAG_WS_PINNED=1; else _TAG_WS_PINNED=""; fi
export TAG_WORKSPACE="$(_tag_pick_workspace)"

# A workspace sitting beside the repo (../workspace) is a layout the earlier
# n9 bootstrap produced, and _tag_pick_workspace does NOT find it — it would
# silently choose /group-volume/$USER/tag-workspace instead and report every
# pool as missing, inviting a rebuild of assets that already exist. Adopt it
# when it is clearly the one in use, but never over an explicit choice.
if [ -z "$_TAG_WS_PINNED" ] && [ ! -d "$TAG_WORKSPACE/pools" ]; then
  for _cand in "$TAG_REPO_ROOT/../workspace" "$TAG_REPO_ROOT/workspace"; do
    if [ -d "$_cand/pools" ]; then
      TAG_WORKSPACE="$(cd "$_cand" && pwd)"
      export TAG_WORKSPACE
      echo "[tag-env] adopting the existing workspace beside the repo:" >&2
      echo "[tag-env]   $TAG_WORKSPACE" >&2
      echo "[tag-env] (export TAG_WORKSPACE=... before sourcing to override)" >&2
      break
    fi
  done
  unset _cand
fi

# Reuse assets the machine already has rather than re-downloading ~1 GB.
# First existing candidate wins; the workspace path is the download target.
_tag_first_existing() {
  # usage: _tag_first_existing <marker-file-or-empty> <candidate>...
  local marker="$1"; shift
  for c in "$@"; do
    if [ -n "$marker" ]; then
      [ -f "$c/$marker" ] && { echo "$c"; return; }
    else
      [ -f "$c" ] && { echo "$c"; return; }
    fi
  done
  echo ""
}

_TAG_M05B="$(_tag_first_existing config.json \
  "${MODEL_PATH_QWEN25_05B:-/nonexistent}" \
  "$TAG_WORKSPACE/models/qwen2.5-0.5b" \
  /group-volume/nait-models/qwen2.5-0.5b \
  /group-volume/models/Qwen2.5-0.5B \
  /group-volume/models/Qwen2.5-0.5B-Instruct)"
export MODEL_PATH_QWEN25_05B="${_TAG_M05B:-$TAG_WORKSPACE/models/qwen2.5-0.5b}"

_TAG_M7B="$(_tag_first_existing config.json \
  "${MODEL_PATH_QWEN25_7B:-/nonexistent}" \
  "$TAG_WORKSPACE/models/qwen2.5-7b" \
  /group-volume/nait-models/qwen2.5-7b \
  /group-volume/models/Qwen2.5-7B \
  /group-volume/models/Qwen2.5-7B-Instruct)"
export MODEL_PATH_QWEN25_7B="${_TAG_M7B:-$TAG_WORKSPACE/models/qwen2.5-7b}"

export POOLS="${POOLS:-$TAG_WORKSPACE/pools}"

_TAG_RAW="$(_tag_first_existing "" \
  "${ALPACA_RAW_JSON:-/nonexistent}" \
  "$TAG_WORKSPACE/datasets/alpaca_gpt4.json" \
  "/group-volume/${USER:-nobody}/datasets/alpaca_gpt4.json" \
  /group-volume/IT-datasets/alpaca_gpt4/data/train.json)"
export ALPACA_RAW_JSON="${_TAG_RAW:-$TAG_WORKSPACE/datasets/alpaca_gpt4.json}"

# The candidate pool the training run actually selects from. bootstrap.sh
# generates it; until then it does not exist and preflight will say so.
export ALPACA_DATA_FILES="${ALPACA_DATA_FILES:-$POOLS/composite20/pool.json}"
export TADS_CF_FILES="${TADS_CF_FILES:-$POOLS/composite20/counterfactual.json}"
export TADS_DEDUP_FILE="${TADS_DEDUP_FILE:-$POOLS/composite20/dedup_clusters.json}"
# The gate reference is BACKBONE-SPECIFIC: Delta_hat is a property of a
# particular model's likelihoods, so a 0.5B reference mis-scales a 7B gate.
# One shared variable made that mistake silent and easy, so each backbone
# gets its own and the arm configs read the matching one.
export TADS_GATE_REF="${TADS_GATE_REF:-$POOLS/clean_ref/delta_hat_05b.pt}"
export TADS_GATE_REF_7B="${TADS_GATE_REF_7B:-$POOLS/clean_ref/delta_hat_7b.pt}"
# The Eq. 5' ablation arm (tag_nonull_7b) needs a reference fit WITHOUT the
# null correction, and its own gate cache, because G differs.
export TADS_GATE_REF_7B_NONULL="${TADS_GATE_REF_7B_NONULL:-$POOLS/clean_ref/delta_hat_7b_nonull.pt}"
export TADS_GATE_CACHE_NONULL="${TADS_GATE_CACHE_NONULL:-$POOLS/composite20/tag_gate_qwen2.5-7b_nonull.pt}"

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

_tag_mark() { [ -e "$1" ] && echo "found" || echo "MISSING (bootstrap will create)"; }

if [ -z "${TAG_ENV_QUIET:-}" ]; then
  echo "[tag-env] workspace : $TAG_WORKSPACE"
  echo "[tag-env] model 0.5b: $MODEL_PATH_QWEN25_05B  [$(_tag_mark "$MODEL_PATH_QWEN25_05B/config.json")]"
  echo "[tag-env] model 7b  : $MODEL_PATH_QWEN25_7B  [$(_tag_mark "$MODEL_PATH_QWEN25_7B/config.json")]"
  case "$MODEL_PATH_QWEN25_7B" in
    *Instruct*|*instruct*)
      echo "[tag-env] NOTE: the 7B checkpoint is an INSTRUCT model, not a base"
      echo "[tag-env]       model. That is a real deviation from 'SFT from a base"
      echo "[tag-env]       checkpoint' and must be stated in any writeup that"
      echo "[tag-env]       mixes it with base-model numbers."
      ;;
  esac
  echo "[tag-env] raw corpus: $ALPACA_RAW_JSON  [$(_tag_mark "$ALPACA_RAW_JSON")]"
  echo "[tag-env] pool      : $ALPACA_DATA_FILES  [$(_tag_mark "$ALPACA_DATA_FILES")]"
  echo "[tag-env] gate ref 0.5b: $TADS_GATE_REF  [$(_tag_mark "$TADS_GATE_REF")]"
  echo "[tag-env] gate ref 7b  : $TADS_GATE_REF_7B  [$(_tag_mark "$TADS_GATE_REF_7B")]"
  echo "[tag-env] outputs   : $OUTPUT_ROOT"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[tag-env] gpus      : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
  fi
  echo "[tag-env] next      : bash scripts/gpu_cloud/bootstrap.sh && python scripts/gpu_cloud/preflight.py"

  # If the shell's cwd belongs to a DIFFERENT git repo than the one this
  # env.sh came from, every relative command the runbook gives ("git pull",
  # "bash scripts/...") lands somewhere else. That has already happened once:
  # the TAG clone sat one level in, and `git pull origin main` run from its
  # parent updated an unrelated repo while reporting success.
  if command -v git >/dev/null 2>&1; then
    _tag_here="$(git -C . rev-parse --show-toplevel 2>/dev/null || true)"
    _tag_repo="$(git -C "$TAG_REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$_tag_repo" ] && [ "$_tag_here" != "$_tag_repo" ]; then
      echo "" >&2
      echo "[tag-env] WARNING: your shell is NOT inside the TAG checkout." >&2
      echo "[tag-env]   cwd git root : ${_tag_here:-<not a git repo>}" >&2
      echo "[tag-env]   TAG git root : $_tag_repo" >&2
      echo "[tag-env] 'git pull' and 'bash scripts/...' from here will hit the" >&2
      echo "[tag-env] wrong tree. Run:  cd $_tag_repo" >&2
    fi
    unset _tag_here _tag_repo
  fi
fi
