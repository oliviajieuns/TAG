#!/usr/bin/env bash
# Download AlpaGasus (Chen et al., 2024) pre-filtered selection JSONs.
#
# Usage:
#   bash scripts/download_alpagasus.sh [target_dir]
#
# If target_dir is omitted, falls back to the parent of $ALPAGASUS_FILTERED_FILE
# (so the path setup_env.sh advertises is created without extra args), else
#   /group-volume/IT-datasets/alpagasus
#
# After completion:
#   <target_dir>/
#     chatgpt_9k.json    # primary: ChatGPT-rated ≥4.5 → 9k rows
#     claude_t45.json    # alternative: Claude-rated ≥4.5
#     random_9k.json     # random baseline of matching size
#
# `baselines/alpagasus/train.py` reads whichever filtered JSON the user
# points $ALPAGASUS_FILTERED_FILE (or --filtered_file) at. The default in
# setup_env.sh is chatgpt_9k.json (the paper's primary variant).
#
# Source: github.com/gpt4life/alpagasus  (no auth required, public)
set -euo pipefail

# Resolve target: CLI arg > directory of ALPAGASUS_FILTERED_FILE > default.
if [ $# -ge 1 ] && [ -n "$1" ]; then
  TARGET="$1"
elif [ -n "${ALPAGASUS_FILTERED_FILE:-}" ]; then
  TARGET="$(dirname "$ALPAGASUS_FILTERED_FILE")"
else
  TARGET="/group-volume/IT-datasets/alpagasus"
fi
mkdir -p "$TARGET"

BASE_URL="https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered"

fetch() {
  local name="$1"
  local url="${BASE_URL}/${name}"
  local out="${TARGET}/${name}"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[skip] ${name} already present at ${out}"
    return 0
  fi
  echo "[fetch] ${url} -> ${out}"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

fetch chatgpt_9k.json
fetch claude_t45.json
# The random baseline lives at data/random/random_9k.json, not data/filtered.
RANDOM_URL="https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/random/random_9k.json"
if [ ! -f "${TARGET}/random_9k.json" ] || [ ! -s "${TARGET}/random_9k.json" ]; then
  echo "[fetch] ${RANDOM_URL} -> ${TARGET}/random_9k.json"
  curl -fL --retry 3 --retry-delay 2 -o "${TARGET}/random_9k.json" "$RANDOM_URL"
else
  echo "[skip] random_9k.json already present"
fi

echo ""
echo "AlpaGasus files ready under ${TARGET}:"
ls -lh "${TARGET}"/*.json
echo ""
echo "Next:"
echo "  export ALPAGASUS_FILTERED_FILE=${TARGET}/chatgpt_9k.json"
echo "  python -m baselines.alpagasus.train \\"
echo "      --config configs/experiments/main_7b/llama2/alpagasus.yaml \\"
echo "      --tag AlpaGasus-ChatGPT-9k"
