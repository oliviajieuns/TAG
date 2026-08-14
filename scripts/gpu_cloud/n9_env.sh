#!/usr/bin/env bash
# n9-cluster environment for the TAG experiments.
#
#   source /group-volume/jieuns.shin/venvs/exp/bin/activate
#   cd /group-volume/jieuns.shin/tag/tests/tag/TAG
#   source scripts/gpu_cloud/n9_env.sh
#
# This is a thin wrapper over scripts/gpu_cloud/env.sh that pins the paths
# that actually exist on this cluster, instead of letting the generic
# auto-detection guess. Everything the experiment writes stays under
# $TAG_ROOT, so the whole thing is one directory to inspect or delete.
#
# It does NOT activate the venv — do that first, so that whichever python is
# on PATH is the one bootstrap installs into.

_N9_SRC="${BASH_SOURCE[0]:-$0}"
_N9_REPO="$(cd "$(dirname "$_N9_SRC")/../.." && pwd)"

# Root for this experiment: code lives in $TAG_ROOT/TAG, everything the run
# produces lives in $TAG_ROOT/workspace.
export TAG_ROOT="${TAG_ROOT:-/group-volume/jieuns.shin/tag/tests/tag}"
export TAG_WORKSPACE="${TAG_WORKSPACE:-$TAG_ROOT/workspace}"

# Paths found by scripts/gpu_cloud/n9_discover.sh --write. Sourced FIRST so
# the fallbacks below only fill gaps discovery could not.
if [ -f "$TAG_ROOT/discovered_env.sh" ]; then
  # shellcheck disable=SC1090
  . "$TAG_ROOT/discovered_env.sh"
  echo "[n9] using discovered paths from $TAG_ROOT/discovered_env.sh"
else
  echo "[n9] no discovered_env.sh — run: bash scripts/gpu_cloud/n9_discover.sh --write"
fi

# Cluster defaults, used only when discovery did not set them. Only -Instruct
# exists here for 7B; env.sh flags that as a deviation from the paper's
# base-checkpoint setup and preflight repeats it, because it changes what the
# numbers mean.
export MODEL_PATH_QWEN25_7B="${MODEL_PATH_QWEN25_7B:-/group-volume/models/Qwen2.5-7B-Instruct}"
export MODEL_PATH_QWEN25_05B="${MODEL_PATH_QWEN25_05B:-/group-volume/models/Qwen2.5-0.5B-Instruct}"
export ALPACA_RAW_JSON="${ALPACA_RAW_JSON:-/group-volume/datasets/alpaca-cleaned/datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json}"
export HF_HOME="${HF_HOME:-/group-volume/data/hf_home}"

# 4 x H100 80GB: the scoring forwards dominate, so raise the forward-only
# batch well above the small-GPU default. Set here, once, so every arm
# launched from this shell shares it (plan §5.2 comparability).
export TAG_EPISODE_BS="${TAG_EPISODE_BS:-64}"      # 0.5B
export TAG_EPISODE_BS_7B="${TAG_EPISODE_BS_7B:-32}"  # 7B

source "$_N9_REPO/scripts/gpu_cloud/env.sh"

echo "[n9] venv      : ${VIRTUAL_ENV:-NONE — activate it first!}"
echo "[n9] python    : $(command -v python)"
echo "[n9] repo      : $_N9_REPO"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[n9] WARNING: no virtualenv active. Run:"
  echo "[n9]   source /group-volume/jieuns.shin/venvs/exp/bin/activate"
fi
