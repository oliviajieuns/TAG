#!/usr/bin/env bash
# Download Qwen2.5-0.5B (base, NOT -Instruct) into the cluster model path.
#
# Usage:
#   bash scripts/download_qwen25_05b.sh [target_dir]
#
# Default target dir matches the path setup_env.sh advertises:
#     /group-volume/nait-models/qwen2.5-0.5b
# so that MODEL_PATH_QWEN25_05B / configs/models/qwen2.5-0.5b.yaml find the
# weights without further config.
#
# Pre-requisites (1회만):
#   1. `huggingface-cli login` (Qwen2.5-0.5B is public, but a token avoids
#      anonymous rate limits on first download).
#
# Qwen2.5-0.5B is a fully open base model — no gating page to accept.
set -euo pipefail

REPO="Qwen/Qwen2.5-0.5B"

if [ $# -ge 1 ] && [ -n "$1" ]; then
  TARGET="$1"
elif [ -n "${MODEL_PATH_QWEN25_05B:-}" ]; then
  TARGET="$MODEL_PATH_QWEN25_05B"
else
  TARGET="/group-volume/nait-models/qwen2.5-0.5b"
fi
mkdir -p "$(dirname "$TARGET")"

if [ -d "$TARGET" ] && [ -f "$TARGET/config.json" ]; then
  echo "[skip] $TARGET already populated (config.json present)."
  echo "       To re-download: rm -rf $TARGET && rerun this script."
  exit 0
fi

# Temporarily allow network access to HF hub (setup_env.sh pins OFFLINE=1
# project-wide). Only this subprocess is affected.
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "[fetch] $REPO (HF hub) → $TARGET"
python3 - "$TARGET" "$REPO" <<'PY'
import sys
target, repo = sys.argv[1], sys.argv[2]
try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    sys.exit(
        f"[error] `huggingface_hub` package not installed: {e}\n"
        f"        pip install huggingface_hub"
    )
try:
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        local_dir_use_symlinks=False,
        # safetensors weights + tokenizer + config; skip .bin shards if both
        # exist (Qwen2.5-0.5B ships only safetensors anyway).
        allow_patterns=[
            "*.json", "*.txt", "*.safetensors",
            "tokenizer*", "merges.txt", "vocab.json",
            "generation_config.json", "*.model",
        ],
    )
except Exception as e:
    sys.exit(
        f"[error] Failed to fetch {repo} from HF hub: "
        f"{type(e).__name__}: {e}\n"
        f"        If this is an auth/rate-limit issue, run "
        f"`huggingface-cli login` first."
    )
print(f"[done] {repo} → {target}")
PY

echo ""
echo "Qwen2.5-0.5B ready at $TARGET"
echo ""
echo "Next:"
echo "  bash scripts/download_qwen25_05b.sh   # already done"
echo "  source scripts/setup_env.sh"
echo "  CUDA_VISIBLE_DEVICES=0 python -m tads.train \\"
echo "      --config configs/experiments/main_05b/qwen25/tads_10.yaml"
