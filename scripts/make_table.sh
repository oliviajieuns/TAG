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


def _all_json_files(root: Path):
    """Yield every *.json under root recursively (regardless of method-dir
    layout). Skips .tmp / .lock side-files."""
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        if not p.is_file():
            continue
        if p.name.endswith(".tmp") or ".lock" in p.name:
            continue
        yield p


def _identify_method(jp: Path, payload):
    """Return one of METHODS[*].mdir if the file looks like a result for
    that method, else None. Identification sources (first match wins):
      (1) payload['experiment']  e.g. "llama2_tads_10" → tads_10
      (2) filename prefix         e.g. "llama2_tads_10-mmlu.json" → tads_10
      (3) any parent dir name     e.g. ".../tads_10/runs/.../*.json" → tads_10

    Matching against METHODS:
      - exact equality, OR
      - candidate ends with "_<mdir>" (model-prefix variant), OR
      - candidate ends with mdir (suffix match — handles misc layouts).
    """
    candidates = []
    if isinstance(payload, dict):
        exp = payload.get("experiment")
        if exp:
            candidates.append(str(exp))
    # filename prefix (before the last '-bench' suffix or whole stem)
    stem = jp.stem
    if "-" in stem:
        candidates.append(stem.rsplit("-", 1)[0])
    candidates.append(stem)
    # parent dirs (closest first)
    for p in jp.parents:
        candidates.append(p.name)

    for cand in candidates:
        for _id, _name, mdir in METHODS:
            if not mdir:
                continue
            if cand == mdir or cand.endswith("_" + mdir) or cand.endswith("/" + mdir):
                return mdir
    return None


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


def collect_all_results(root: Path):
    """Walk every *.json under root, return {mdir: {bench: (max_acc, src_path)}}.

    method 식별은 file path 가 아니라 payload['experiment'] / filename /
    parent dir name 으로 — 사용자가 --out_dir 를 다른 위치로 잘못 지정해
    결과 JSON 이 엉뚱한 폴더에 떨어진 경우에도 (1) experiment label 이나
    (2) <exp>-<bench>.json 파일명 만 우리 method list 와 매칭되면 표에
    포함된다. method dir 위치는 더 이상 신뢰 source 가 아님.
    """
    # by_method[mdir][bench] = list of (acc_float, src_path)
    by_method = {m[2]: {b: [] for b, _ in BENCHES} for m in METHODS if m[2]}
    n_total = 0
    n_parsed = 0
    n_fail = 0
    n_skipped = 0
    fail_log = []  # (path, exc_type, exc_msg) for the footer noti
    for jp in _all_json_files(root):
        n_total += 1
        try:
            with open(jp) as f:
                payload = json.load(f)
        except Exception as e:
            n_fail += 1
            msg = f"{type(e).__name__}: {e}"
            fail_log.append((jp, msg))
            try:
                rel = jp.relative_to(root)
            except Exception:
                rel = jp
            sys.stderr.write(
                f"[make_table] JSON parse fail: {rel}  ←  {msg}\n"
            )
            continue
        n_parsed += 1
        mdir = _identify_method(jp, payload)
        if mdir is None:
            n_skipped += 1
            continue
        for bench, _ in BENCHES:
            for v in _accs_from_payload(payload, bench):
                by_method[mdir][bench].append((v, jp))
    sys.stderr.write(
        f"[make_table] scanned={n_total}  parsed={n_parsed}  "
        f"parse_fail={n_fail}  skipped_unmatched_method={n_skipped}\n"
    )
    if fail_log:
        sys.stderr.write(
            f"[make_table] {len(fail_log)} file(s) failed JSON parse — see "
            "lines above. Common causes: half-written file (kill mid-dump), "
            "trailing junk bytes, BOM, or non-JSON content saved with .json "
            "extension by mistake.\n"
        )
    return by_method


def fmt_cell(v):
    return MISSING if v is None else f"{v:.2f}"


def fmt_list_cell(measurements):
    """Format a list of measurements for one (method, bench) cell.

    Input: list of (acc_float, src_path) tuples (any order, possibly with
    duplicates from multiple eval runs of the same checkpoint).
    Output: descending-sorted, deduplicated (rounded to 2 dp) values joined
    by ", ". Empty list → [??.??].
    """
    if not measurements:
        return MISSING
    # Deduplicate at 2 dp resolution (eval runs of the same ckpt should
    # produce bit-identical scores; tiny diffs come from non-determinism
    # in sampling tasks and aren't meaningful as separate "values").
    seen = []
    seen_keys = set()
    for v, _src in sorted(measurements, key=lambda t: -t[0]):
        key = round(v, 2)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seen.append(f"{v:.2f}")
    return ", ".join(seen)


def best_or_none(vals, bench):
    """Format the MAX of `vals[idx_of_bench]` (a list of (acc, src)) for the
    footer log (or "—" if missing)."""
    idx_by_bench = {b: i for i, (b, _) in enumerate(BENCHES)}
    measurements = vals[idx_by_bench[bench]]
    if not measurements:
        return "—"
    return f"{max(v for v, _ in measurements):.2f}"


# ---- gather (single walk over ENTIRE root, then bucket by identified method) ----
all_results = collect_all_results(ROOT)

rows = []  # list of (id, name, [list_of_measurements_per_bench×8], avg_max, all_lists)
for mid, mname, mdir in METHODS:
    if not mdir:
        rows.append((mid, mname, [[] for _ in BENCHES], None, []))
        continue
    by_bench = all_results.get(mdir, {})
    vals_lists = []      # list of list-of-(acc,src) per bench
    for bench, _ in BENCHES:
        vals_lists.append(by_bench.get(bench, []) or [])
    # W-AVG: average of the MAX value across present benches (사용자가 list
    # 보고 후속 정정할 수 있도록 단지 reference 값).
    maxes = [max(v for v, _ in lst) for lst in vals_lists if lst]
    avg = sum(maxes) / len(maxes) if maxes else None
    rows.append((mid, mname, vals_lists, avg, vals_lists))

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
    # CSV/TSV — quote the multi-value cell so the comma list doesn't break
    # the field separator. TSV is safer for multi-value cells (tabs never
    # appear inside our values) so we recommend that for spreadsheet import.
    sep = "," if OUT_FMT == "csv" else "\t"
    def _esc(s):
        if sep == "," and ("," in s or '"' in s):
            s = s.replace('"', '""')
            return f'"{s}"'
        return s
    headers_flat = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG (max-of-each)", "Δ"]
    print(sep.join(_esc(h) for h in headers_flat))
    for mid, mname, vals_lists, avg, _src in rows:
        cells = [mid, mname] + [fmt_list_cell(lst) for lst in vals_lists]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print(sep.join(_esc(c) for c in cells))
else:  # markdown
    line1 = ["ID", "Method"] + [b.split("\n")[0] for _, b in BENCHES] + ["W-AVG", "Δ"]
    line2 = (
        ["", ""]
        + [b.split("\n", 1)[1] if "\n" in b else "" for _, b in BENCHES]
        + ["(max-of-each)", "vs Full FT"]
    )
    print("| " + " | ".join(line1) + " |")
    print("| " + " | ".join("---" for _ in line1) + " |")
    print("| " + " | ".join(line2) + " |")
    for mid, mname, vals_lists, avg, _src in rows:
        cells = [mid, mname] + [fmt_list_cell(lst) for lst in vals_lists]
        cells.append(fmt_cell(avg))
        cells.append(fmt_delta(avg))
        print("| " + " | ".join(cells) + " |")

# ---- footer: per-method, per-bench full list (value ← source) ----
sys.stderr.write("\n[make_table] source root: " + str(ROOT) + "\n")
sys.stderr.write("[make_table] cell entries are ALL measurements (desc, deduped @ 2dp)\n")
mdir_by_id = {m[0]: m[2] for m in METHODS}
idx_by_bench = {b: i for i, (b, _) in enumerate(BENCHES)}
for mid, mname, vals_lists, _avg, _src in rows:
    if not mdir_by_id[mid]:
        continue
    present = sum(1 for lst in vals_lists if lst)
    sys.stderr.write(f"  [{mid}] {mname}  ({present}/{len(BENCHES)} bench have data)\n")
    for bench, _ in BENCHES:
        lst = vals_lists[idx_by_bench[bench]]
        if not lst:
            sys.stderr.write(f"      {bench:10s} —  (no value found)\n")
            continue
        # Show every measurement with source path, sorted desc.
        for v, src in sorted(lst, key=lambda t: -t[0]):
            try:
                rel = src.relative_to(ROOT)
            except Exception:
                rel = src
            sys.stderr.write(f"      {bench:10s} {v:>6.2f}  ← {rel}\n")
PY
