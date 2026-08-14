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
#     validation-00000-of-00001.parquet   # eval split (used by tag.evals.tydiqa)
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

# Two candidate URL patterns per split — HF dataset repos sometimes have
# raw parquet committed to main (#1), and sometimes only the auto-converted
# parquet branch (#2). We try #1 first; if that 404's, fall back to #2.
# (Some weeks one works and the other doesn't; both have been observed.)
#
# refs%2Fconvert%2Fparquet  is the URL-encoded form of refs/convert/parquet
# — the special read-only branch HF maintains with auto-converted parquet
# shards for every dataset.

declare -A CANDIDATES_DEV=(
  [main]="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task/validation-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/refs%2Fconvert%2Fparquet/secondary_task/validation/0000.parquet"
)
declare -A CANDIDATES_TRAIN=(
  [main]="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/main/secondary_task/train-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/tydiqa/resolve/refs%2Fconvert%2Fparquet/secondary_task/train/0000.parquet"
)

# Connectivity probe — try the FIRST candidate of each split. If both
# patterns are unreachable we still bail (no point retrying 4 URLs).
_probe_url=""
for url in "${CANDIDATES_DEV[main]}" "${CANDIDATES_DEV[refs]}"; do
  if curl -fLsS -o /dev/null --max-time 10 -I "$url" 2>/dev/null; then
    _probe_url="$url"
    break
  fi
done
if [ -z "$_probe_url" ]; then
  echo "[error] cannot reach any TyDiQA parquet URL on huggingface.co." >&2
  echo "        Tried:" >&2
  echo "          ${CANDIDATES_DEV[main]}" >&2
  echo "          ${CANDIDATES_DEV[refs]}" >&2
  echo "        Common causes on cluster nodes: outbound HTTPS blocked," >&2
  echo "        no DNS, or a corporate proxy that needs HTTP(S)_PROXY env vars." >&2
  echo "        Manual alternatives:" >&2
  echo "          1) download from a login node + scp to TYDIQA_DATA_DIR" >&2
  echo "          2) python: " >&2
  echo "             from datasets import load_dataset" >&2
  echo "             ds = load_dataset('google-research-datasets/tydiqa', 'secondary_task')" >&2
  echo "             ds['validation'].to_parquet('$TARGET/validation-00000-of-00001.parquet')" >&2
  echo "             ds['train'].to_parquet('$TARGET/train-00000-of-00001.parquet')" >&2
  exit 3
fi
echo "[probe] reachable via: $_probe_url"

# Final filenames (what tag/evals/tydiqa.py looks for via _resolve_split_paths).
# Both candidate URL patterns get saved under the same canonical filename so
# the evaluator doesn't care which one we fetched from.
declare -A DST_FNAMES=(
  [dev]="validation-00000-of-00001.parquet"
  [train]="train-00000-of-00001.parquet"
)

fetch_with_fallback() {
  # Args: split (dev|train), dst_path. Tries main→refs.
  local split="$1" dst="$2"
  local -n urls=$([ "$split" = "dev" ] && echo CANDIDATES_DEV || echo CANDIDATES_TRAIN)
  for tag in main refs; do
    local url="${urls[$tag]}"
    echo "  [try ${tag}] $url"
    if curl -fL --retry 3 --retry-delay 2 -o "$dst.part" "$url" 2>&1 | tail -3; then
      mv "$dst.part" "$dst"
      echo "  [ok ${tag}] $(stat -c%s "$dst") bytes"
      return 0
    fi
    rm -f "$dst.part"
    echo "  [fail ${tag}] trying next candidate..."
  done
  return 1
}

for split in dev train; do
  fname="${DST_FNAMES[$split]}"
  dst="$TARGET/$fname"
  if [ -s "$dst" ]; then
    echo "[skip] $fname already present ($(stat -c%s "$dst") bytes)"
    continue
  fi
  echo "[fetch] $fname  →  $dst"
  if ! fetch_with_fallback "$split" "$dst"; then
    echo "[error] every candidate URL failed for $split split." >&2
    echo "        See manual alternative at the top of this script." >&2
    exit 1
  fi
done

echo ""
echo "TyDiQA ready at: $TARGET"
echo ""
echo "If TYDIQA_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export TYDIQA_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check (requires pandas + pyarrow):"
echo "    python3 -c \"import pandas as pd; d=pd.read_parquet('$TARGET/validation-00000-of-00001.parquet'); print(d.shape, list(d.columns))\""
