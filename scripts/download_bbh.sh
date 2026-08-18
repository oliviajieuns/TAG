#!/usr/bin/env bash
# Download BIG-Bench-Hard from the source tag/evals/bbh.py was written for.
#
# Usage:
#   bash scripts/download_bbh.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${BBH_DATA_DIR:-$TAG_EVAL_DATA/bbh}
#
# After completion:
#   <target_dir>/
#     bbh/<task>.json          # 27 tasks, {"examples":[{input,target}]}
#     cot-prompts/<task>.txt   # the 3-shot chain-of-thought prefixes
#
# The cot-prompts are the point. Without them the evaluator falls back to a
# direct-answer few-shot baseline and says so — a legitimate way to run BBH,
# but NOT the one NAIT Table 2 reports, so the column stops being comparable.
# The HF mirror lukaemon/bbh ships only the task data, which is why building
# BBH from that parquet (scripts/prepare_eval_data.py) leaves this gap.
#
# Source: github.com/suzgunmirac/BIG-Bench-Hard (public, no auth)
set -euo pipefail

TARGET="${1:-${BBH_DATA_DIR:-${TAG_EVAL_DATA:-./eval-data}/bbh}}"
REPO="https://github.com/suzgunmirac/BIG-Bench-Hard"
mkdir -p "$TARGET"

_count() { ls -1 $1 2>/dev/null | wc -l; }
_n_task="$(_count "$TARGET/bbh/*.json")"
_n_cot="$(_count "$TARGET/cot-prompts/*.txt")"
if [ "$_n_cot" -ge 27 ]; then
  # Say what is there. "nothing to do" without the counts is the same
  # non-answer as a directory that merely exists.
  echo "[bbh] already present under $TARGET — nothing to do"
  echo "[bbh]   bbh/*.json        : $_n_task"
  echo "[bbh]   cot-prompts/*.txt : $_n_cot"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[bbh] cloning $REPO"
if ! git clone --depth 1 --quiet "$REPO" "$TMP/bbh-src" 2>/dev/null; then
  echo "[bbh] git clone failed — is outbound network available?" >&2
  echo "[bbh] Otherwise copy bbh/ and cot-prompts/ from a checkout by hand." >&2
  exit 1
fi

for sub in bbh cot-prompts; do
  if [ ! -d "$TMP/bbh-src/$sub" ]; then
    echo "[bbh] $sub/ missing from the clone — upstream layout changed?" >&2
    exit 1
  fi
  mkdir -p "$TARGET/$sub"
  cp "$TMP/bbh-src/$sub"/* "$TARGET/$sub/"
done

n_task="$(ls -1 "$TARGET"/bbh/*.json 2>/dev/null | wc -l)"
n_cot="$(ls -1 "$TARGET"/cot-prompts/*.txt 2>/dev/null | wc -l)"
echo "[bbh] $TARGET"
echo "[bbh]   bbh/*.json        : $n_task"
echo "[bbh]   cot-prompts/*.txt : $n_cot"
if [ "$n_task" -lt 27 ] || [ "$n_cot" -lt 27 ]; then
  echo "[bbh] expected 27 of each — the download is incomplete" >&2
  exit 1
fi
echo "[bbh] done. Verify with: python scripts/check_eval_data.py --benchmarks bbh"
