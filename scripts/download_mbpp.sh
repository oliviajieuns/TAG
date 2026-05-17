#!/usr/bin/env bash
# Download MBPP (Mostly Basic Python Problems) — sanitized + full splits.
#
# Usage:
#   bash scripts/download_mbpp.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${MBPP_DATA_DIR:-/group-volume/IT-datasets/mbpp}
#
# After completion the directory layout will be:
#   <target_dir>/
#     sanitized/
#       test-00000-of-00001.parquet     # 257 problems (sanitized eval split)
#       prompt-00000-of-00001.parquet   # 90 problems (3-shot demo source)
#     full/                              # optional, for cross-paper checks
#       test-00000-of-00001.parquet     # 500 problems
#       prompt-00000-of-00001.parquet
#
# Source: google-research-datasets/mbpp on HuggingFace
#   https://huggingface.co/datasets/google-research-datasets/mbpp
#
# We default to the SANITIZED config (NAIT Appendix D); the full config is
# downloaded too so cross-paper comparisons aren't blocked. Failure of the
# full split is non-fatal — the evaluator only needs sanitized to run.
#
# Schema (both configs):
#   task_id / text / code / test_list / test_setup_code / challenge_test_list
# The evaluator concatenates the prompt (text + first assert) and uses the
# raw `code` reference solution + `test_list` for pass@1 scoring.
set -euo pipefail

TARGET="${1:-${MBPP_DATA_DIR:-/group-volume/IT-datasets/mbpp}}"

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

# config → (split → main URL → refs URL) candidates.
declare -A URL_SAN_TEST=(
  [main]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/refs%2Fconvert%2Fparquet/sanitized/test/0000.parquet"
)
declare -A URL_SAN_PROMPT=(
  [main]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/prompt-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/refs%2Fconvert%2Fparquet/sanitized/prompt/0000.parquet"
)
declare -A URL_FULL_TEST=(
  [main]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"
)
declare -A URL_FULL_PROMPT=(
  [main]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/full/prompt-00000-of-00001.parquet"
  [refs]="https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/refs%2Fconvert%2Fparquet/full/prompt/0000.parquet"
)

_probe_url=""
for url in "${URL_SAN_TEST[main]}" "${URL_SAN_TEST[refs]}"; do
  if curl -fLsS -o /dev/null --max-time 10 -I "$url" 2>/dev/null; then
    _probe_url="$url"; break
  fi
done
if [ -z "$_probe_url" ]; then
  echo "[error] cannot reach any MBPP parquet URL on huggingface.co." >&2
  echo "        Manual fallback:" >&2
  echo "          python -c \"from datasets import load_dataset" >&2
  echo "          ds = load_dataset('google-research-datasets/mbpp', 'sanitized')" >&2
  echo "          ds['test'].to_parquet('$TARGET/sanitized/test-00000-of-00001.parquet')" >&2
  echo "          ds['prompt'].to_parquet('$TARGET/sanitized/prompt-00000-of-00001.parquet')\"" >&2
  exit 3
fi
echo "[probe] reachable via: $_probe_url"

fetch_one() {
  # Args: var_prefix (e.g. URL_SAN_TEST), dst_path. Tries main→refs.
  local prefix="$1" dst="$2"
  for tag in main refs; do
    local var="${prefix}[$tag]"
    local url="${!var}"
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

fetch_pair() {
  # Args: config (sanitized|full), test_prefix, prompt_prefix, optional_full
  local cfg="$1" test_prefix="$2" prompt_prefix="$3" optional="${4:-0}"
  mkdir -p "$TARGET/$cfg"

  local test_dst="$TARGET/$cfg/test-00000-of-00001.parquet"
  if [ -s "$test_dst" ]; then
    echo "[skip] $cfg/test already present ($(stat -c%s "$test_dst") bytes)"
  else
    echo "[fetch] $cfg/test  →  $test_dst"
    if ! fetch_one "$test_prefix" "$test_dst"; then
      if [ "$optional" = "1" ]; then
        echo "[warn] $cfg/test fetch failed — continuing (optional split)" >&2
        return 0
      fi
      echo "[error] every candidate URL failed for $cfg/test." >&2
      return 1
    fi
  fi

  local prompt_dst="$TARGET/$cfg/prompt-00000-of-00001.parquet"
  if [ -s "$prompt_dst" ]; then
    echo "[skip] $cfg/prompt already present ($(stat -c%s "$prompt_dst") bytes)"
  else
    echo "[fetch] $cfg/prompt  →  $prompt_dst"
    if ! fetch_one "$prompt_prefix" "$prompt_dst"; then
      if [ "$optional" = "1" ]; then
        echo "[warn] $cfg/prompt fetch failed — continuing (optional split)" >&2
        return 0
      fi
      echo "[error] every candidate URL failed for $cfg/prompt." >&2
      return 1
    fi
  fi
}

# sanitized is mandatory (default eval split per NAIT Appendix D).
if ! fetch_pair sanitized URL_SAN_TEST URL_SAN_PROMPT 0; then
  exit 1
fi

# full is best-effort — useful for cross-paper checks but not required by
# our default evaluator. Don't fail the script if it's gone.
fetch_pair full URL_FULL_TEST URL_FULL_PROMPT 1 || true

echo ""
echo "MBPP ready at: $TARGET"
echo ""
echo "If MBPP_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export MBPP_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check (requires pandas + pyarrow):"
echo "    python3 -c \"import pandas as pd; d=pd.read_parquet('$TARGET/sanitized/test-00000-of-00001.parquet'); print(d.shape, list(d.columns))\""
