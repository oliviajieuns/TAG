#!/usr/bin/env bash
# make_table.sh — collect 9-bench eval results into the paper-style table.
#
# Usage:
#   bash scripts/make_table.sh <eval_results_root> [output_format]
#
#   <eval_results_root> : a directory that contains per-method sub-dirs
#                         (full_100, random_10, lima, alpagasus, q2q_10,
#                          selectit_10, nait_10, data_agent_10, tads_10).
#                         Each sub-dir holds the eval-run layout this codebase
#                         emits (`runs/<tag>/<exp>-eval_summary.json` or a
#                         `_latest/` symlink). Falls back to $EVAL_RESULTS_ROOT
#                         when omitted.
#   [output_format]     : markdown (default) | csv | tsv
#
# How "latest" is resolved per method:
#   1. <method>/_latest/<exp>-eval_summary.json   (if symlink exists)
#   2. <method>/runs/<*>/<exp>-eval_summary.json  (newest mtime)
#   3. <method>/<exp>-eval_summary.json            (flat layout)
#
# Per cell: looks up bench by `.summaries[].benchmark == "<bench>"` and reads
# `.accuracy` (already paper-canonical: pass@1 for code, F1 for QA, etc.).
# Missing cells emit "[??.??]".
#
# W-AVG (Overall column): unweighted mean of the 8 available bench accuracies
# × 100. If a bench is missing the average is over the present ones (count
# reported in the table footer).
#
# Δ vs Full FT: (method_W-AVG − full_100_W-AVG) / full_100_W-AVG · 100.
# Sign included (e.g. +3.06% / −2.54%). Empty if Full FT row missing.
set -euo pipefail

ROOT="${1:-${EVAL_RESULTS_ROOT:-}}"
OUT_FMT="${2:-markdown}"

if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  echo "[make_table] eval_results_root not given or not a directory: ${ROOT:-<empty>}" >&2
  echo "Usage: bash scripts/make_table.sh <eval_results_root> [markdown|csv|tsv]" >&2
  exit 1
fi

python3 - "$ROOT" "$OUT_FMT" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT_FMT = sys.argv[2].lower()

# (ID, display_name, method_dir_name).  method_dir_name == "" → no on-disk
# results expected (e.g. the no-FT base row); cells stay [??.??].
METHODS = [
    ("00",  "Pure LLaMA-2-7B (No FT)",         ""),
    ("01",  "Alpaca-GPT4 (Full FT)",           "full_100"),
    ("R10", "Random (10%, paper baseline)",    "random_10"),
    ("02",  "LIMA",                            "lima"),
    ("03",  "AlpaGasus",                       "alpagasus"),
    ("04",  "Q2Q",                             "q2q_10"),
    ("05",  "SelectIT",                        "selectit_10"),
    ("07",  "NAIT (All)",                      "nait_10"),
    ("08",  "Composite-reward only (λ=0)",     "data_agent_10"),
    ("09",  "TADS (λ=1)",                      "tads_10"),
]

BENCHES = [
    ("mmlu",      "MMLU\nexam"),
    ("bbh",       "BBH\nhard tasks"),
    ("gsm8k",     "GSM8K\ngrade math"),
    ("svamp",     "SVAMP\nrobust math"),
    ("mbpp",      "MBPP\nprogram"),
    ("humaneval", "H-Eval\nfunction"),
    ("tydiqa",    "TydiQA\n9-lang F1"),
    ("xquad",     "XQuAD\n12-lang F1"),
]

MISSING = "[??.??]"


def find_summary(method_dir: Path) -> Path:
    """Return newest *-eval_summary.json under method_dir, honouring the
    `runs/_latest/` symlink convention. None if nothing found."""
    if not method_dir.exists():
        return None
    # (1) explicit _latest symlink/dir
    latest = method_dir / "_latest"
    if latest.exists():
        hits = sorted(latest.glob("*-eval_summary.json"))
        if hits:
            return hits[-1]
    # (2) newest runs/<tag>/...
    hits = list(method_dir.glob("runs/*/*-eval_summary.json"))
    if hits:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0]
    # (3) flat layout
    hits = sorted(method_dir.glob("*-eval_summary.json"))
    if hits:
        return hits[-1]
    return None


def extract_bench(summary_path: Path, bench: str):
    """Return float accuracy (× 100) for bench, or None if missing/non-numeric."""
    try:
        with open(summary_path) as f:
            payload = json.load(f)
    except Exception:
        return None
    for s in payload.get("summaries", []) or []:
        if s.get("benchmark") != bench:
            continue
        v = s.get("accuracy")
        if isinstance(v, (int, float)):
            return float(v) * 100.0
    return None


def fmt_cell(v):
    return MISSING if v is None else f"{v:.2f}"


# ---- gather ----
rows = []  # list of (id, name, [val_or_None×8], avg_or_None)
for mid, mname, mdir in METHODS:
    if not mdir:
        rows.append((mid, mname, [None] * len(BENCHES), None))
        continue
    summary = find_summary(ROOT / mdir)
    if summary is None:
        rows.append((mid, mname, [None] * len(BENCHES), None))
        continue
    vals = [extract_bench(summary, b) for b, _ in BENCHES]
    present = [v for v in vals if v is not None]
    avg = sum(present) / len(present) if present else None
    rows.append((mid, mname, vals, avg))

# Full FT W-AVG for Δ
full_avg = None
for mid, _mname, _vals, avg in rows:
    if mid == "01":
        full_avg = avg
        break


def fmt_delta(method_avg):
    if method_avg is None or full_avg is None:
        return ""
    d = (method_avg - full_avg) / full_avg * 100.0
    return f"{d:+.2f}%"


# ---- emit ----
HEADERS = ["ID", "Method"] + [b for _, b in BENCHES] + ["W-AVG", "Δ vs Full FT"]

if OUT_FMT == "csv" or OUT_FMT == "tsv":
    sep = "," if OUT_FMT == "csv" else "\t"
    # Flatten the 2-line bench headers to single-line for CSV/TSV.
    headers_flat = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG", "Δ"]
    print(sep.join(headers_flat))
    for mid, mname, vals, avg in rows:
        cells = [mid, mname] + [fmt_cell(v) for v in vals]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print(sep.join(cells))
else:  # markdown
    # Use the second line of the bench header as the sub-header for readability.
    line1 = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG", "Δ"]
    line2 = ["", ""] + [b.split("\n", 1)[1] if "\n" in b else "" for _, b in BENCHES] + ["weighted", "vs Full FT"]
    print("| " + " | ".join(line1) + " |")
    print("| " + " | ".join("---" for _ in line1) + " |")
    print("| " + " | ".join(line2) + " |")
    for mid, mname, vals, avg in rows:
        cells = [mid, mname] + [fmt_cell(v) for v in vals]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print("| " + " | ".join(cells) + " |")

# ---- footer: per-method bench presence count + source path ----
sys.stderr.write("\n[make_table] source root: " + str(ROOT) + "\n")
for mid, mname, vals, _avg in rows:
    if not METHODS[[m[0] for m in METHODS].index(mid)][2]:
        continue
    present = sum(v is not None for v in vals)
    summary = find_summary(ROOT / METHODS[[m[0] for m in METHODS].index(mid)][2])
    src = str(summary) if summary else "(not found)"
    sys.stderr.write(f"  [{mid}] {mname:38s}  bench={present}/{len(BENCHES)}  ← {src}\n")
PY
