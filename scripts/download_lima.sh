#!/usr/bin/env bash
# Download LIMA (Zhou et al., 2023) — GAIR/lima, 1030 instruction-response pairs.
#
# Usage:
#   bash scripts/download_lima.sh [target_dir]
#
# If target_dir is omitted, falls back to the parent of $LIMA_DATA_FILES
# (so the path setup_env.sh advertises is created without extra args), else
#   /group-volume/IT-datasets/lima
#
# After completion:
#   <target_dir>/
#     train.jsonl    # exactly the file tads.baselines.lima.data expects
#
# Pre-requisites (1회만):
#   1. `huggingface-cli login`
#      → read token from https://huggingface.co/settings/tokens
#   2. Visit https://huggingface.co/datasets/GAIR/lima
#      → click "Agree and access" (gated dataset)
#
# GAIR/lima is a GATED HF dataset, so a naive `wget` doesn't work — we go
# through the `datasets` Python lib with the user's HF auth token. This
# script temporarily flips HF_*_OFFLINE=0 so the request can actually
# reach the hub even when setup_env.sh defaults to offline.
set -euo pipefail

if [ $# -ge 1 ] && [ -n "$1" ]; then
  TARGET="$1"
elif [ -n "${LIMA_DATA_FILES:-}" ]; then
  TARGET="$(dirname "$LIMA_DATA_FILES")"
else
  TARGET="/group-volume/IT-datasets/lima"
fi
mkdir -p "$TARGET"
OUT="${TARGET}/train.jsonl"

if [ -f "$OUT" ] && [ -s "$OUT" ]; then
  echo "[skip] $OUT already present ($(wc -l < "$OUT") rows)."
  echo ""
  echo "Next: export LIMA_DATA_FILES=$OUT"
  exit 0
fi

# Temporarily allow network access to HF hub (setup_env.sh pins OFFLINE=1
# project-wide). Only this subprocess is affected.
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "[fetch] GAIR/lima (HF hub) → $OUT"
python3 - "$OUT" <<'PY'
import sys
out_path = sys.argv[1]
try:
    from datasets import load_dataset
except ImportError as e:
    sys.exit(
        f"[error] `datasets` package not installed: {e}\n"
        f"        pip install datasets"
    )
try:
    ds = load_dataset("GAIR/lima", split="train")
except Exception as e:
    sys.exit(
        f"[error] Failed to fetch GAIR/lima from HF hub: {type(e).__name__}: {e}\n"
        f"        Did you (1) `huggingface-cli login` and (2) accept the\n"
        f"        dataset's access agreement at\n"
        f"        https://huggingface.co/datasets/GAIR/lima ?"
    )
ds.to_json(out_path)
print(f"[done] Wrote {len(ds)} rows to {out_path}")
PY

echo ""
echo "LIMA ready at $OUT"
echo ""
echo "Next:"
echo "  export LIMA_DATA_FILES=$OUT"
echo "  python -m tads.baselines.lima.train \\"
echo "      --config configs/experiments/main_7b/llama2/lima.yaml \\"
echo "      --tag LIMA"
