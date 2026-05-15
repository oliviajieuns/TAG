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

# Pre-flight: surface the exact failure cause before falling into curl
# noise. The single most common report is "ran the script but TyDiQA
# still fails" — almost always the parent dir doesn't exist, OR the
# user doesn't have write permission, OR cluster firewall blocks HF.
parent=$(dirname "$TARGET")
if [ ! -d "$parent" ]; then
  echo "[error] parent directory does not exist: $parent" >&2
  echo "        create it first (with proper permissions) or choose a different target." >&2
  exit 2
fi
if ! mkdir -p "$TARGET" 2>/dev/null; then
  echo "[error] cannot create target directory: $TARGET" >&2
  echo "        check permissions on $parent" >&2
  ls -ld "$parent" >&2 || true
  exit 2
fi
if [ ! -w "$TARGET" ]; then
  echo "[error] target directory is not writable: $TARGET" >&2
  ls -ld "$TARGET" >&2 || true
  exit 2
fi

HF_BASE="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task"

# Connectivity probe — if HF is unreachable from this host (cluster
# firewall, DNS, proxy misconfig) bail out NOW with a clear message
# instead of hitting curl four times under --retry.
if ! curl -fLsS -o /dev/null --max-time 10 -I "$HF_BASE/validation-00000-of-00001.parquet" 2>/dev/null; then
  echo "[error] cannot reach huggingface.co from this host. Probed:" >&2
  echo "          $HF_BASE/validation-00000-of-00001.parquet" >&2
  echo "        Common causes on cluster nodes: outbound HTTPS blocked," >&2
  echo "        no DNS, or a corporate proxy that needs HTTP(S)_PROXY env vars." >&2
  echo "        If the cluster has a local HF mirror, set HF_ENDPOINT or" >&2
  echo "        download the two parquet files manually from a login node." >&2
  exit 3
fi

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
