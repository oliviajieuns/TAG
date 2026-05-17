#!/usr/bin/env bash
# Download MMLU-Pro (TIGER-Lab/MMLU-Pro) test + validation parquet.
#
# Usage:
#   bash scripts/download_mmlu_pro.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${MMLU_PRO_DATA_DIR:-/group-volume/IT-datasets/mmlu_pro}
#
# After completion the directory layout will be:
#   <target_dir>/
#     test-00000-of-00001.parquet      # 12,032 questions (eval split)
#     validation-00000-of-00001.parquet  # ~70 questions × 14 categories,
#                                        # used as the 5-shot CoT prompt source
#
# Source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
#
# MMLU-Pro shape (vs vanilla MMLU):
#   - 10 options A..J (NOT 4)
#   - long-form CoT answers; we use NAIT 5-shot CoT generation + extraction
#   - 14 categories (biology, business, chemistry, ..., other)
set -euo pipefail

TARGET="${1:-${MMLU_PRO_DATA_DIR:-/group-volume/IT-datasets/mmlu_pro}}"

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

# Two candidate URL patterns per split — main branch first, refs/convert/parquet
# fallback (same trick TyDiQA uses). MMLU-Pro currently keeps the parquet on
# main, but the auto-converted branch is the safety net if upstream renames.
declare -A CANDIDATES_TEST=(
  [main]="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
declare -A CANDIDATES_VAL=(
  [main]="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/validation-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
)

# Connectivity probe — same pattern as download_tydiqa.sh.
_probe_url=""
for url in "${CANDIDATES_TEST[main]}" "${CANDIDATES_TEST[refs]}"; do
  if curl -fLsS -o /dev/null --max-time 10 -I "$url" 2>/dev/null; then
    _probe_url="$url"
    break
  fi
done
if [ -z "$_probe_url" ]; then
  echo "[error] cannot reach any MMLU-Pro parquet URL on huggingface.co." >&2
  echo "        Tried:" >&2
  echo "          ${CANDIDATES_TEST[main]}" >&2
  echo "          ${CANDIDATES_TEST[refs]}" >&2
  echo "        Manual fallback (run on a node with HTTPS):" >&2
  echo "          python -c \"from datasets import load_dataset" >&2
  echo "          ds = load_dataset('TIGER-Lab/MMLU-Pro')" >&2
  echo "          ds['test'].to_parquet('$TARGET/test-00000-of-00001.parquet')" >&2
  echo "          ds['validation'].to_parquet('$TARGET/validation-00000-of-00001.parquet')\"" >&2
  exit 3
fi
echo "[probe] reachable via: $_probe_url"

declare -A DST_FNAMES=(
  [test]="test-00000-of-00001.parquet"
  [val]="validation-00000-of-00001.parquet"
)

fetch_with_fallback() {
  local split="$1" dst="$2"
  local -n urls=$([ "$split" = "test" ] && echo CANDIDATES_TEST || echo CANDIDATES_VAL)
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

for split in test val; do
  fname="${DST_FNAMES[$split]}"
  dst="$TARGET/$fname"
  if [ -s "$dst" ]; then
    echo "[skip] $fname already present ($(stat -c%s "$dst") bytes)"
    continue
  fi
  echo "[fetch] $fname  →  $dst"
  if ! fetch_with_fallback "$split" "$dst"; then
    echo "[error] every candidate URL failed for $split split." >&2
    exit 1
  fi
done

echo ""
echo "MMLU-Pro ready at: $TARGET"
echo ""
echo "If MMLU_PRO_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export MMLU_PRO_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check (requires pandas + pyarrow):"
echo "    python3 -c \"import pandas as pd; d=pd.read_parquet('$TARGET/test-00000-of-00001.parquet'); print(d.shape, list(d.columns))\""
