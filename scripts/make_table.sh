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


def _all_json_files(method_dir: Path):
    """Yield every *.json under method_dir recursively (runs/, _latest/,
    legacy/, flat, any depth). Skips .tmp / .lock side-files."""
    if not method_dir.exists():
        return
    for p in method_dir.rglob("*.json"):
        if not p.is_file():
            continue
        if p.name.endswith(".tmp") or ".lock" in p.name:
            continue
        yield p


def _accs_from_payload(payload, bench: str):
    """Pull every accuracy reading for `bench` out of one decoded JSON
    payload. Handles three shapes:

      (a) Combined summary file emitted by tads.eval — top-level
          {"summaries": [{"benchmark": "...", "accuracy": ...}, ...]}.
      (b) Per-bench file emitted by each evaluator directly —
          {"benchmark": "...", "accuracy": ...}.
      (c) Stray dicts that happen to include "benchmark" + "accuracy" at
          some inner key (defensive — older runs may have nested layouts).

    Returns a list of floats (× 100), possibly empty.
    """
    out = []
    if isinstance(payload, dict):
        if payload.get("benchmark") == bench:
            v = payload.get("accuracy")
            if isinstance(v, (int, float)):
                out.append(float(v) * 100.0)
        for s in payload.get("summaries", []) or []:
            if isinstance(s, dict) and s.get("benchmark") == bench:
                v = s.get("accuracy")
                if isinstance(v, (int, float)):
                    out.append(float(v) * 100.0)
    return out


def collect_bench_max(method_dir: Path):
    """Walk every *.json under method_dir, return (best[bench], sources).

    For each bench we pick the MAX accuracy across all files — multiple
    eval runs (different timestamps, sweep tags, legacy/ dirs, etc.) all
    contribute and the best result wins.

    `sources[bench]` = path of the file that supplied the winning value
    (for the footer log so the user can verify the table came from where).
    """
    best = {b: None for b, _ in BENCHES}
    sources = {b: None for b, _ in BENCHES}
    for jp in _all_json_files(method_dir):
        try:
            payload = json.load(open(jp))
        except Exception:
            continue
        for bench, _ in BENCHES:
            for v in _accs_from_payload(payload, bench):
                if best[bench] is None or v > best[bench]:
                    best[bench] = v
                    sources[bench] = jp
    return best, sources


def fmt_cell(v):
    return MISSING if v is None else f"{v:.2f}"


def best_or_none(vals, bench):
    """Format `vals[idx_of_bench]` for the footer log (or "—" if missing)."""
    idx_by_bench = {b: i for i, (b, _) in enumerate(BENCHES)}
    v = vals[idx_by_bench[bench]]
    return "—" if v is None else f"{v:.2f}"


# ---- gather ----
rows = []  # list of (id, name, [val_or_None×8], avg_or_None, sources_dict)
for mid, mname, mdir in METHODS:
    if not mdir:
        rows.append((mid, mname, [None] * len(BENCHES), None, {}))
        continue
    best, sources = collect_bench_max(ROOT / mdir)
    vals = [best[b] for b, _ in BENCHES]
    present = [v for v in vals if v is not None]
    avg = sum(present) / len(present) if present else None
    rows.append((mid, mname, vals, avg, sources))

# Full FT W-AVG for Δ
full_avg = None
for mid, _mname, _vals, avg, _src in rows:
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
    headers_flat = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG", "Δ"]
    print(sep.join(headers_flat))
    for mid, mname, vals, avg, _src in rows:
        cells = [mid, mname] + [fmt_cell(v) for v in vals]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print(sep.join(cells))
else:  # markdown
    line1 = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG", "Δ"]
    line2 = ["", ""] + [b.split("\n", 1)[1] if "\n" in b else "" for _, b in BENCHES] + ["weighted", "vs Full FT"]
    print("| " + " | ".join(line1) + " |")
    print("| " + " | ".join("---" for _ in line1) + " |")
    print("| " + " | ".join(line2) + " |")
    for mid, mname, vals, avg, _src in rows:
        cells = [mid, mname] + [fmt_cell(v) for v in vals]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print("| " + " | ".join(cells) + " |")

# ---- footer: bench-presence + source-of-MAX per (method, bench) ----
sys.stderr.write("\n[make_table] source root: " + str(ROOT) + "\n")
sys.stderr.write("[make_table] per-bench source = file that supplied the MAX value\n")
mdir_by_id = {m[0]: m[2] for m in METHODS}
for mid, mname, vals, _avg, sources in rows:
    if not mdir_by_id[mid]:
        continue
    present = sum(v is not None for v in vals)
    sys.stderr.write(f"  [{mid}] {mname}  ({present}/{len(BENCHES)} bench)\n")
    for bench, _ in BENCHES:
        src = sources.get(bench)
        if src is None:
            sys.stderr.write(f"      {bench:10s} —  (no value found)\n")
        else:
            try:
                rel = src.relative_to(ROOT)
            except Exception:
                rel = src
            sys.stderr.write(f"      {bench:10s} {best_or_none(vals, bench):>6s}  ← {rel}\n")
PY
