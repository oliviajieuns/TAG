#!/usr/bin/env bash
# Download SVAMP (math word problems, 1,000 test items).
#
# Usage:
#   bash scripts/download_svamp.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${SVAMP_DATA_DIR:-/group-volume/IT-datasets/svamp}
#
# After completion the directory layout will be:
#   <target_dir>/
#     test-00000-of-00001.parquet
#
# Source: ChilleD/SVAMP on HuggingFace
#   https://huggingface.co/datasets/ChilleD/SVAMP
# (The Patel et al. 2021 release is also available on GitHub as SVAMP.json
# under arkilpatel/SVAMP, but the HF parquet is the canonical
# eval-time mirror — same 1k items, just parquet'd.)
#
# Schema after download:
#   Body / Question / Equation / Answer / Type / ID
# Our evaluator builds "Body + ' ' + Question" as the math problem text.
set -euo pipefail

TARGET="${1:-${SVAMP_DATA_DIR:-/group-volume/IT-datasets/svamp}}"

parent=$(dirname "$TARGET")
if [ ! -d "$parent" ]; then
  echo "[error] parent directory does not exist: $parent" >&2
  exit 2
fi
if ! mkdir -p "$TARGET" 2>/dev/null; then
  echo "[error] cannot create target directory: $TARGET" >&2
  ls -ld "$parent" >&2 || true
  exit 2
fi
if [ ! -w "$TARGET" ]; then
  echo "[error] target directory is not writable: $TARGET" >&2
  ls -ld "$TARGET" >&2 || true
  exit 2
fi

declare -A CANDIDATES_TEST=(
  [main]="https://huggingface.co/datasets/ChilleD/SVAMP/resolve/main/data/test-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/ChilleD/SVAMP/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)

_probe_url=""
for url in "${CANDIDATES_TEST[main]}" "${CANDIDATES_TEST[refs]}"; do
  if curl -fLsS -o /dev/null --max-time 10 -I "$url" 2>/dev/null; then
    _probe_url="$url"
    break
  fi
done
if [ -z "$_probe_url" ]; then
  echo "[error] cannot reach any SVAMP parquet URL on huggingface.co." >&2
  echo "        Tried:" >&2
  echo "          ${CANDIDATES_TEST[main]}" >&2
  echo "          ${CANDIDATES_TEST[refs]}" >&2
  echo "        Manual fallback (download GitHub JSON instead):" >&2
  echo "          curl -fL https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json \\" >&2
  echo "            -o $TARGET/SVAMP.json" >&2
  echo "        The evaluator auto-detects flat JSON too." >&2
  exit 3
fi
echo "[probe] reachable via: $_probe_url"

DST_FNAME="test-00000-of-00001.parquet"

fetch_with_fallback() {
  local dst="$1"
  for tag in main refs; do
    local url="${CANDIDATES_TEST[$tag]}"
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

dst="$TARGET/$DST_FNAME"
if [ -s "$dst" ]; then
  echo "[skip] $DST_FNAME already present ($(stat -c%s "$dst") bytes)"
else
  echo "[fetch] $DST_FNAME  →  $dst"
  if ! fetch_with_fallback "$dst"; then
    echo "[error] every candidate URL failed for SVAMP test split." >&2
    exit 1
  fi
fi

echo ""
echo "SVAMP ready at: $TARGET"
echo ""
echo "If SVAMP_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export SVAMP_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check (requires pandas + pyarrow):"
echo "    python3 -c \"import pandas as pd; d=pd.read_parquet('$TARGET/$DST_FNAME'); print(d.shape, list(d.columns))\""
