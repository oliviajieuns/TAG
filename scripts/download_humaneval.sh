#!/usr/bin/env bash
# Download the HumanEval problem set (164 problems, 45 KB).
#
# Usage:
#   bash scripts/download_humaneval.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${HUMANEVAL_DATA_DIR:-/group-volume/IT-datasets/human-eval}
#
# After completion:
#   <target_dir>/
#     HumanEval.jsonl.gz   # exactly the file tag.evals.humaneval expects
#
# Source: official OpenAI release on GitHub
#   https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
#
# Note: this only fetches the PROBLEMS. The scoring code (the `human_eval`
# pip package, `evaluate_functional_correctness`) must be installed
# separately: `pip install human-eval`. The evaluator already raises a
# clear RuntimeError if it isn't present.
set -euo pipefail

TARGET="${1:-${HUMANEVAL_DATA_DIR:-/group-volume/IT-datasets/human-eval}}"
mkdir -p "$TARGET"

URL="https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
DST="$TARGET/HumanEval.jsonl.gz"

if [ -s "$DST" ]; then
  echo "[skip] HumanEval.jsonl.gz already present ($(stat -c%s "$DST") bytes)"
else
  echo "[fetch] HumanEval.jsonl.gz  →  $DST"
  # -L follows redirects (github.com → raw.githubusercontent.com),
  # -f bails on HTTP error, --retry handles transient 5xx.
  if ! curl -fL --retry 3 --retry-delay 2 -o "$DST.part" "$URL"; then
    echo "[error] curl failed (URL=$URL)" >&2
    rm -f "$DST.part"
    exit 1
  fi
  mv "$DST.part" "$DST"
  echo "[ok]    HumanEval.jsonl.gz  →  $(stat -c%s "$DST") bytes"
fi

# Sanity check
n=$(zcat "$DST" 2>/dev/null | wc -l)
echo "[verify] problem count: $n  (expected 164)"
if [ "$n" -ne 164 ]; then
  echo "[warn]   unexpected problem count — file may be corrupt; re-download" >&2
fi

echo ""
echo "HumanEval ready at: $TARGET"
echo ""
echo "Required for scoring (separate install):"
echo "    pip install human-eval"
echo ""
echo "If HUMANEVAL_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export HUMANEVAL_DATA_DIR=\"$TARGET\""
