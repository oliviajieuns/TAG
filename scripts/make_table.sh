#!/usr/bin/env bash
# make_table.sh — collect 8-bench eval results into the paper-style table,
# split by model (llama2 / qwen25 / mistral / deepseek).
#
# Usage:
#   bash scripts/make_table.sh <eval_results_root> [output_format] [--set NAME]
#
#   <eval_results_root> : directory that contains per-model / per-method
#                         sub-dirs. Typical layout:
#                           <root>/main_7b/llama2/<method>/runs/...
#                           <root>/evol_7b/llama2/<method>/runs/...
#                         Either model-prefixed or flat layout works — model
#                         is identified by ANY parent-dir or filename segment
#                         matching one of: llama2 / qwen25 / mistral / deepseek.
#                         Falls back to $EVAL_RESULTS_ROOT when omitted.
#   [output_format]     : markdown (default) | csv | tsv
#   --set NAME          : restrict scan to JSONs under a path-segment
#                         matching NAME (e.g. --set main_7b for Table 1,
#                         --set evol_7b for Table 5). Without it, ALL sets
#                         are merged into one (model, method) cell — fine
#                         when only one set has data; produces incorrect
#                         numbers when both main_7b/ and evol_7b/ live under
#                         the same EVAL_RESULTS_ROOT. A heads-up is printed
#                         to stderr when that collision is detected.
#
# Output: one table PER model that has at least one eval JSON. Models with
# no data are skipped entirely. Missing cells (model has no result for that
# (method, bench)) are rendered as "-".
#
# How "latest" is resolved per (model, method):
#   We walk every *.json under <root> and bucket by (identified model,
#   identified method) via parent-dir / filename / payload.experiment.
#   Every measurement is kept; the cell shows ALL measurements desc-sorted
#   and dedup'd at 2 dp.
#
# W-AVG (Overall column): unweighted mean of the 8 available bench MAX
# accuracies × 100. If a bench is missing the average is over the present
# ones.
#
# Δ vs Full FT: (method_W-AVG − full_100_W-AVG) / full_100_W-AVG · 100,
# computed PER MODEL (each model's own Full FT row is the baseline). Empty
# if Full FT row missing for that model.
set -euo pipefail

ROOT=""
OUT_FMT=""
SET_FILTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --set)    SET_FILTER="$2"; shift 2 ;;
    --set=*)  SET_FILTER="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *)
      if [ -z "$ROOT" ]; then ROOT="$1"
      elif [ -z "$OUT_FMT" ]; then OUT_FMT="$1"
      else echo "[make_table] unexpected arg: $1" >&2; exit 1
      fi
      shift ;;
  esac
done

ROOT="${ROOT:-${EVAL_RESULTS_ROOT:-}}"
OUT_FMT="${OUT_FMT:-markdown}"

if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  echo "[make_table] eval_results_root not given or not a directory: ${ROOT:-<empty>}" >&2
  echo "Usage: bash scripts/make_table.sh <eval_results_root> [markdown|csv|tsv] [--set main_7b|evol_7b|...]" >&2
  exit 1
fi

python3 - "$ROOT" "$OUT_FMT" "$SET_FILTER" <<'PY'
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT_FMT = sys.argv[2].lower()
SET_FILTER = sys.argv[3] if len(sys.argv) > 3 else ""

# Model-axis: (dir_key, human-readable display name).
# `dir_key` must match the model-segment that appears in the file path /
# experiment label (see _identify_model).
MODELS = [
    ("llama2",   "LLaMA-2-7B"),
    ("qwen25",   "Qwen2.5-7B"),
    ("mistral",  "Mistral-7B"),
    ("deepseek", "DeepSeek-7B"),
]
MODEL_KEYS = {m[0] for m in MODELS}
MODEL_ORDER = {m[0]: i for i, m in enumerate(MODELS)}

# Method-axis. `name_tpl` may contain `{model}` to interpolate the model's
# display name (used by the Pure-base row). `mdir == ""` → no on-disk
# results expected; cells stay as the missing marker.
METHODS = [
    ("00",  "Pure {model} (No FT)",            ""),
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

MISSING = "-"

# Known table-set segments — used only for collision-detection in the no-filter
# path. Add new sets here as they're introduced (e.g. main_7b, evol_7b, ...).
KNOWN_SETS = ("main_7b", "evol_7b")


def _path_segments(p: Path):
    """All parent-dir names + filename stem, used for set-filter matching."""
    segs = [p.stem]
    for par in p.parents:
        segs.append(par.name)
    return segs


def _all_json_files(root: Path):
    """Yield every *.json under root recursively. Skips .tmp / .lock side-files.
    When SET_FILTER is set, only files whose path includes that segment are
    yielded.
    """
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        if not p.is_file():
            continue
        if p.name.endswith(".tmp") or ".lock" in p.name:
            continue
        if SET_FILTER and SET_FILTER not in _path_segments(p):
            continue
        yield p


def _segments_from_candidates(candidates):
    """Expand candidate strings into individual segments by splitting on
    /, _, -. Returns the union of the originals + each segment. Used so a
    string like 'main_7b/llama2/tads_10' matches the model 'llama2' AND
    the method 'tads_10'."""
    out = list(candidates)
    for c in candidates:
        for sep in ("/", "_", "-"):
            if sep in c:
                out.extend(c.split(sep))
    return out


def _identify_model(jp: Path, payload):
    """Return one of MODEL_KEYS if any parent-dir / filename / payload
    segment matches, else None."""
    candidates = []
    if isinstance(payload, dict):
        exp = payload.get("experiment")
        if exp:
            candidates.append(str(exp))
    candidates.append(jp.stem)
    for p in jp.parents:
        candidates.append(p.name)
    for cand in _segments_from_candidates(candidates):
        if cand in MODEL_KEYS:
            return cand
    return None


def _identify_method(jp: Path, payload):
    """Return one of METHODS[*].mdir if the file looks like a result for
    that method, else None. Match sources (first match wins):
      (1) payload['experiment'], e.g. "llama2_tads_10" → tads_10
      (2) filename prefix, e.g. "llama2_tads_10-mmlu.json" → tads_10
      (3) any parent dir name, e.g. ".../tads_10/runs/.../*.json" → tads_10
    """
    candidates = []
    if isinstance(payload, dict):
        exp = payload.get("experiment")
        if exp:
            candidates.append(str(exp))
    stem = jp.stem
    if "-" in stem:
        candidates.append(stem.rsplit("-", 1)[0])
    candidates.append(stem)
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
      (a) Combined summary: {"summaries": [{"benchmark":..., "accuracy":...}, ...]}
      (b) Per-bench file: {"benchmark":..., "accuracy":...}
      (c) Stray nested dicts with both fields (defensive — older layouts).
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
    """Walk root once, bucket every parseable JSON by (model, method, bench).

    Returns:
        by_model[mkey][mdir][bench] = list of (acc_float, src_path)
    """
    method_mdirs = [m[2] for m in METHODS if m[2]]
    bench_keys = [b for b, _ in BENCHES]
    by_model: dict = {}

    n_total = 0
    n_parsed = 0
    n_fail = 0
    n_no_method = 0
    n_no_model = 0
    fail_log = []
    sets_seen: set = set()

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
            sys.stderr.write(f"[make_table] JSON parse fail: {rel}  ←  {msg}\n")
            continue
        n_parsed += 1

        mkey = _identify_model(jp, payload)
        mdir = _identify_method(jp, payload)
        if mdir is None:
            n_no_method += 1
            continue
        if mkey is None:
            n_no_model += 1
            continue

        if mkey not in by_model:
            by_model[mkey] = {m: {b: [] for b in bench_keys} for m in method_mdirs}
        for bench in bench_keys:
            for v in _accs_from_payload(payload, bench):
                by_model[mkey][mdir][bench].append((v, jp))

        for s in _path_segments(jp):
            if s in KNOWN_SETS:
                sets_seen.add(s)

    sys.stderr.write(
        f"[make_table] scanned={n_total}  parsed={n_parsed}  parse_fail={n_fail}  "
        f"no_method={n_no_method}  no_model={n_no_model}  "
        f"models_found={','.join(sorted(by_model.keys())) or '(none)'}\n"
    )
    if SET_FILTER:
        sys.stderr.write(f"[make_table] active --set filter: {SET_FILTER}\n")
    elif len(sets_seen) > 1:
        sys.stderr.write(
            f"[make_table] WARNING: multiple table-sets detected under root "
            f"({', '.join(sorted(sets_seen))}). They've been MERGED into the "
            f"same (model, method) cells — numbers will be wrong if you meant "
            f"to report them separately. Re-run with `--set <name>` to filter "
            f"to one set at a time.\n"
        )
    if fail_log:
        sys.stderr.write(
            f"[make_table] {len(fail_log)} file(s) failed JSON parse — see "
            "lines above. Common causes: half-written file (kill mid-dump), "
            "trailing junk bytes, BOM, or non-JSON content saved with .json "
            "extension by mistake.\n"
        )
    return by_model


def fmt_cell(v):
    return MISSING if v is None else f"{v:.2f}"


def fmt_list_cell(measurements):
    """Format a list of measurements for one (method, bench) cell.

    Input: list of (acc_float, src_path) tuples (any order, possibly with
    duplicates from multiple eval runs of the same checkpoint).
    Output: descending-sorted, deduplicated (rounded to 2 dp) values joined
    by ", ". Empty list → MISSING.
    """
    if not measurements:
        return MISSING
    seen_keys = set()
    seen = []
    for v, _src in sorted(measurements, key=lambda t: -t[0]):
        key = round(v, 2)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seen.append(f"{v:.2f}")
    return ", ".join(seen)


def _model_display(mkey):
    for k, disp in MODELS:
        if k == mkey:
            return disp
    return mkey


def build_rows_for_model(model_results):
    """Return list of (mid, mname, vals_lists, avg)."""
    rows = []
    for mid, mname_tpl, mdir in METHODS:
        mname = mname_tpl  # `{model}` is interpolated at emit time
        if not mdir:
            rows.append((mid, mname, [[] for _ in BENCHES], None))
            continue
        by_bench = model_results.get(mdir, {})
        vals_lists = []
        for bench, _ in BENCHES:
            vals_lists.append(by_bench.get(bench, []) or [])
        maxes = [max(v for v, _ in lst) for lst in vals_lists if lst]
        avg = sum(maxes) / len(maxes) if maxes else None
        rows.append((mid, mname, vals_lists, avg))
    return rows


def emit_model_table(mkey, model_results, out_fmt, *, is_first):
    """Emit one table for `mkey`."""
    disp = _model_display(mkey)
    rows = build_rows_for_model(model_results)

    # Per-model Full FT baseline for Δ.
    full_avg = None
    for mid, _mname, _vals, avg in rows:
        if mid == "01":
            full_avg = avg
            break

    def fmt_delta(method_avg):
        if method_avg is None or full_avg is None:
            return MISSING
        d = (method_avg - full_avg) / full_avg * 100.0
        return f"{d:+.2f}%"

    if out_fmt == "csv" or out_fmt == "tsv":
        sep = "," if out_fmt == "csv" else "\t"
        def _esc(s):
            if sep == "," and ("," in s or '"' in s):
                s = s.replace('"', '""')
                return f'"{s}"'
            return s
        if not is_first:
            print()  # blank separator between tables
        print(f"# Model: {disp} ({mkey})")
        headers_flat = (
            ["ID", "Method"]
            + [b.split("\n")[0] for _, b in BENCHES]
            + ["W-AVG (max-of-each)", "Δ"]
        )
        print(sep.join(_esc(h) for h in headers_flat))
        for mid, mname, vals_lists, avg in rows:
            mname_filled = mname.format(model=disp) if "{model}" in mname else mname
            cells = [mid, mname_filled] + [fmt_list_cell(lst) for lst in vals_lists]
            cells.append(fmt_cell(avg))
            cells.append(fmt_delta(avg))
            print(sep.join(_esc(c) for c in cells))
    else:  # markdown
        if not is_first:
            print()  # blank line between tables
        print(f"## Model: {disp} ({mkey})")
        print()
        line1 = (
            ["ID", "Method"]
            + [b.split("\n")[0] for _, b in BENCHES]
            + ["W-AVG", "Δ"]
        )
        line2 = (
            ["", ""]
            + [b.split("\n", 1)[1] if "\n" in b else "" for _, b in BENCHES]
            + ["(max-of-each)", "vs Full FT"]
        )
        print("| " + " | ".join(line1) + " |")
        print("| " + " | ".join("---" for _ in line1) + " |")
        print("| " + " | ".join(line2) + " |")
        for mid, mname, vals_lists, avg in rows:
            mname_filled = mname.format(model=disp) if "{model}" in mname else mname
            cells = [mid, mname_filled] + [fmt_list_cell(lst) for lst in vals_lists]
            cells.append(fmt_cell(avg))
            cells.append(fmt_delta(avg))
            print("| " + " | ".join(cells) + " |")


# ---- collect ----
all_results = collect_all_results(ROOT)

if not all_results:
    sys.stderr.write(
        "[make_table] no model results matched any of: "
        + ", ".join(m[0] for m in MODELS) + "\n"
        "[make_table] file paths or experiment labels must contain a model-name "
        "segment (e.g. .../llama2/tads_10/...).\n"
    )
    sys.exit(1)

# Sort by canonical MODEL order; unknown keys go to the end.
model_keys_sorted = sorted(
    all_results.keys(),
    key=lambda k: (MODEL_ORDER.get(k, 999), k),
)

for i, mkey in enumerate(model_keys_sorted):
    emit_model_table(mkey, all_results[mkey], OUT_FMT, is_first=(i == 0))

# ---- footer: per-model, per-method, per-bench full list (value ← source) ----
sys.stderr.write("\n[make_table] source root: " + str(ROOT) + "\n")
sys.stderr.write("[make_table] cell entries are ALL measurements (desc, deduped @ 2dp)\n")

mdir_by_id = {m[0]: m[2] for m in METHODS}
idx_by_bench = {b: i for i, (b, _) in enumerate(BENCHES)}

for mkey in model_keys_sorted:
    disp = _model_display(mkey)
    sys.stderr.write(f"\n[make_table] ===== Model: {disp} ({mkey}) =====\n")
    rows = build_rows_for_model(all_results[mkey])
    for mid, mname, vals_lists, _avg in rows:
        if not mdir_by_id[mid]:
            continue
        mname_filled = mname.format(model=disp) if "{model}" in mname else mname
        present = sum(1 for lst in vals_lists if lst)
        sys.stderr.write(
            f"  [{mid}] {mname_filled}  ({present}/{len(BENCHES)} bench have data)\n"
        )
        for bench, _ in BENCHES:
            lst = vals_lists[idx_by_bench[bench]]
            if not lst:
                sys.stderr.write(f"      {bench:10s} —  (no value found)\n")
                continue
            for v, src in sorted(lst, key=lambda t: -t[0]):
                try:
                    rel = src.relative_to(ROOT)
                except Exception:
                    rel = src
                sys.stderr.write(f"      {bench:10s} {v:>6.2f}  ← {rel}\n")
PY
