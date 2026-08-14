#!/usr/bin/env bash
# Download Evol-Instruct (Xu et al., WizardLM 2023) — 70K instruction/response
# pairs evolved from Alpaca seeds. Used by Table 5 (instruction-dataset transfer).
#
# Usage:
#   bash scripts/download_evol_instruct.sh [target_dir]
#
# If target_dir is omitted, falls back to the parent of $EVOL_INSTRUCT_DATA_FILES
# (so the path setup_env.sh advertises is created without extra args), else
#   /group-volume/IT-datasets/wizardlm_evol_instruct_70k
#
# After completion:
#   <target_dir>/
#     train.jsonl    # exactly the file tag.train (and every baseline) consumes
#
# The script handles two column conventions automatically:
#   (a) flat (instruction, output) — V1 70k canonical
#   (b) (conversations: [{from:human, value:...}, {from:gpt, value:...}, ...])
#       common in V2 196k; we extract the first human→gpt turn as
#       (instruction, output) so tokenize_alpaca consumes it unchanged.
#
# Pre-requisites:
#   - `huggingface-cli login` (or HF_TOKEN env)  — the WizardLMTeam mirror is
#     public but the request still benefits from auth quota.
#
# Source: huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k
set -euo pipefail

if [ $# -ge 1 ] && [ -n "$1" ]; then
  TARGET="$1"
elif [ -n "${EVOL_INSTRUCT_DATA_FILES:-}" ]; then
  TARGET="$(dirname "$EVOL_INSTRUCT_DATA_FILES")"
else
  TARGET="/group-volume/IT-datasets/wizardlm_evol_instruct_70k"
fi
mkdir -p "$TARGET"
OUT="${TARGET}/train.jsonl"

if [ -f "$OUT" ] && [ -s "$OUT" ]; then
  echo "[skip] $OUT already present ($(wc -l < "$OUT") rows)."
  echo ""
  echo "Next: export EVOL_INSTRUCT_DATA_FILES=$OUT"
  exit 0
fi

# Temporarily allow network access to HF hub (setup_env.sh pins OFFLINE=1
# project-wide). Only this subprocess is affected.
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

DATASET_ID="${EVOL_INSTRUCT_DATASET_ID:-WizardLMTeam/WizardLM_evol_instruct_70k}"
echo "[fetch] ${DATASET_ID} (HF hub) → $OUT"

python3 - "$OUT" "$DATASET_ID" <<'PY'
import json
import sys

out_path, dataset_id = sys.argv[1], sys.argv[2]
try:
    from datasets import load_dataset
except ImportError as e:
    sys.exit(
        f"[error] `datasets` package not installed: {e}\n"
        f"        pip install datasets"
    )
try:
    ds = load_dataset(dataset_id, split="train")
except Exception as e:
    sys.exit(
        f"[error] Failed to fetch {dataset_id} from HF hub: {type(e).__name__}: {e}\n"
        f"        Try (1) `huggingface-cli login`, or override the dataset id\n"
        f"        with EVOL_INSTRUCT_DATASET_ID=<other id>."
    )

cols = set(ds.column_names)
print(f"[info] columns = {sorted(cols)}  | rows = {len(ds)}")

n_written = 0
with open(out_path, "w", encoding="utf-8") as fp:
    if {"instruction", "output"} <= cols:
        # Canonical V1 layout — pass through; preserve optional `input` if any.
        for ex in ds:
            row = {
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    elif "conversations" in cols:
        # V2 196k / ShareGPT-style — extract the first human→gpt turn so the
        # tokenize_alpaca contract (instruction/output) is preserved without
        # changing downstream code.
        for ex in ds:
            conv = ex["conversations"]
            if not conv or len(conv) < 2:
                continue
            human_msg = next((m for m in conv if m.get("from") in ("human", "user")), None)
            gpt_msg = next((m for m in conv if m.get("from") in ("gpt", "assistant")), None)
            if human_msg is None or gpt_msg is None:
                continue
            fp.write(json.dumps({
                "instruction": human_msg.get("value", ""),
                "input": "",
                "output": gpt_msg.get("value", ""),
            }, ensure_ascii=False) + "\n")
            n_written += 1
    else:
        sys.exit(
            f"[error] Unrecognised Evol-Instruct schema. Columns = {sorted(cols)}.\n"
            f"        Expected either (instruction, output) or (conversations)."
        )

print(f"[done] Wrote {n_written} rows to {out_path}")
PY

echo ""
echo "Evol-Instruct ready at $OUT"
echo ""
echo "Next:"
echo "  export EVOL_INSTRUCT_DATA_FILES=$OUT"
echo "  python -m tag.train --config configs/experiments/evol_7b/llama2/legacy_10.yaml"
