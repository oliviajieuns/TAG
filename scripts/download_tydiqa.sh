#!/usr/bin/env bash
# Download TyDiQA Gold-Passage v1.1 dev + train splits.
#
# Usage:
#   bash scripts/download_tydiqa.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${TYDIQA_DATA_DIR:-/group-volume/IT-datasets/tydiqa}
#
# After completion the directory layout will be:
#   <target_dir>/
#     validation-00000-of-00001.parquet   # eval split (used by tads.evals.tydiqa)
#     train-00000-of-00001.parquet        # demo source for 5-shot prompting
#
# Source: the official Google Research mirror on HuggingFace:
#   https://huggingface.co/datasets/google-research-datasets/tydiqa
# The legacy storage.googleapis.com URLs from the v1.1 README now return
# HTTP 403 (the bucket revoked anonymous read), so this script intentionally
# does NOT use them.
set -euo pipefail

TARGET="${1:-${TYDIQA_DATA_DIR:-/group-volume/IT-datasets/tydiqa}}"
mkdir -p "$TARGET"

HF_BASE="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task"

declare -A URLS=(
  [validation-00000-of-00001.parquet]="$HF_BASE/validation-00000-of-00001.parquet"
  [train-00000-of-00001.parquet]="$HF_BASE/train-00000-of-00001.parquet"
)

for fname in "${!URLS[@]}"; do
  dst="$TARGET/$fname"
  if [ -s "$dst" ]; then
    echo "[skip] $fname already present ($(stat -c%s "$dst") bytes)"
    continue
  fi
  echo "[fetch] $fname  →  $dst"
  # -L follows redirects; -f bails out on HTTP error; --retry handles
  # transient 5xx that HF occasionally returns under load.
  if ! curl -fL --retry 3 --retry-delay 2 -o "$dst.part" "${URLS[$fname]}"; then
    echo "[error] curl failed for $fname (URL=${URLS[$fname]})" >&2
    rm -f "$dst.part"
    exit 1
  fi
  mv "$dst.part" "$dst"
  echo "[ok]    $fname  →  $(stat -c%s "$dst") bytes"
done

echo ""
echo "TyDiQA ready at: $TARGET"
echo ""
echo "If TYDIQA_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export TYDIQA_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check (requires pandas + pyarrow):"
echo "    python3 -c \"import pandas as pd; d=pd.read_parquet('$TARGET/validation-00000-of-00001.parquet'); print(d.shape, list(d.columns))\""
