#!/usr/bin/env bash
# make_table.sh — collect 8-bench eval results into the paper-style table.
#
# Output layout: one table per (table-set, model) pair, stacked vertically.
# table-sets currently recognised: main_7b / main_05b / evol_7b.
#
# Usage:
#   bash scripts/make_table.sh <eval_results_root> [output_format]
#
#   <eval_results_root> : directory that contains per-set / per-model /
#                         per-method sub-dirs. Typical layouts:
#                           <root>/main_7b/llama2/<method>/runs/...
#                           <root>/main_05b/qwen25/<method>/runs/...
#                           <root>/evol_7b/llama2/<method>/runs/...
#                         Set + model are identified by ANY parent-dir or
#                         filename segment matching the configured lists.
#                         Falls back to $EVAL_RESULTS_ROOT when omitted.
#   [output_format]     : markdown (default) | csv | tsv
#
# Cell rendering: each (set, model, method, bench) cell shows ALL
# measurements (desc-sorted, dedup'd at 2 dp). Missing cells → "-".
#
# W-AVG: unweighted mean of the per-bench MAX accuracies × 100.
# Δ: (method_W-AVG − full_100_W-AVG) / full_100_W-AVG · 100, computed PER
#    (set, model) — each pair's own Full FT row is the baseline. Empty if
#    Full FT row missing.
#
# Files that can't be assigned a set are still emitted, grouped under the
# `(no-set)` heading at the bottom.
set -euo pipefail

ROOT=""
OUT_FMT=""

while [ $# -gt 0 ]; do
  case "$1" in
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
  echo "Usage: bash scripts/make_table.sh <eval_results_root> [markdown|csv|tsv]" >&2
  exit 1
fi

python3 - "$ROOT" "$OUT_FMT" <<'PY'
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT_FMT = sys.argv[2].lower()

# Table-set axis. Order here = emit order.
# Dataset label per set. Header renders as `<Model> (<dataset>)`. main_05b
# and light both use Alpaca-GPT4; we tag `light` with a variant qualifier
# so the two 0.5B tables don't collide on header text.
KNOWN_SETS = [
    ("main_7b",  "Alpaca-GPT4"),
    ("main_05b", "Alpaca-GPT4"),
    ("evol_7b",  "Evol-Instruct"),
    ("light",    "Alpaca-GPT4 — Light"),
]
SET_KEYS = {s[0] for s in KNOWN_SETS}
SET_ORDER = {s[0]: i for i, s in enumerate(KNOWN_SETS)}
SET_DISPLAY = {k: d for k, d in KNOWN_SETS}

# Model-axis: (dir_key, default display name). `dir_key` must match the
# model-segment that appears in the file path / experiment label
# (see _identify_model).
MODELS = [
    ("llama2",        "LLaMA-2-7B"),
    ("qwen25",        "Qwen2.5-7B"),
    ("mistral",       "Mistral-7B"),
    ("deepseek",      "DeepSeek-7B"),
    ("qwen2.5-0.5b",  "Qwen2.5-0.5B"),
]
MODEL_KEYS = {m[0] for m in MODELS}
MODEL_ORDER = {m[0]: i for i, m in enumerate(MODELS)}

# Model aliases: surface form on disk → canonical MODELS key. Only one
# 0.5B model exists in this codebase (qwen2.5-0.5b), so a bare "05b" /
# "0.5b" token unambiguously identifies it. Used to pick up the
# `light/<method>_05b/` layout whose path never carries a literal
# "qwen2.5-0.5b" segment.
MODEL_ALIASES = {
    "qwen05b":  "qwen2.5-0.5b",
    "0.5b":     "qwen2.5-0.5b",
    "05b":      "qwen2.5-0.5b",
}

# Per-(set, model) display override. Falls back to MODELS[*][1] when no
# entry. Used so e.g. `main_05b/qwen25/` shows "Qwen2.5-0.5B" instead of
# the default 7B label.
MODEL_DISPLAY_BY_SET = {
    ("main_05b", "qwen25"): "Qwen2.5-0.5B",
}

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

# Method aliases: light/<method>_05b/ configs run the same selection
# strategy as the paper-faithful main_*/<method>_{10,100}/ configs
# (selection_ratio + train_epochs equivalent), so they collapse onto the
# same row instead of producing a separate "tads_05b" row.
METHOD_ALIASES = {
    "tads_05b":       "tads_10",
    "random_05b":     "random_10",
    "data_agent_05b": "data_agent_10",
    "full_05b":       "full_100",
}

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
NO_SET = None  # sentinel — files whose path doesn't contain any KNOWN_SETS segment


def _path_segments(p: Path):
    """All parent-dir names + filename stem, used for set / model matching."""
    segs = [p.stem]
    for par in p.parents:
        segs.append(par.name)
    return segs


def _all_json_files(root: Path):
    """Yield every *.json under root recursively. Skips .tmp / .lock side-files."""
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        if not p.is_file():
            continue
        if p.name.endswith(".tmp") or ".lock" in p.name:
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


def _identify_set(jp: Path, payload=None):
    """Return one of SET_KEYS, else None. Priority:
      (1) any parent-dir segment of `jp` matching SET_KEYS — the on-disk
          layout that `--out_dir` produces, e.g. .../main_7b/.../*.json
      (2) any segment of the ckpt path (recorded inside the eval summary)
          matching SET_KEYS — disambiguates flat eval-results layouts
          (`<model>/<method>/...`) where Alpaca-GPT4 (main_*) vs
          Evol-Instruct (evol_*) runs share the same path. eval.py writes
          `ckpt` as the full training output path
          (`${OUTPUT_ROOT}/<output_subdir>/runs/.../epoch_last`) so the
          set marker is preserved there even when the eval-results path
          drops it."""
    for par in jp.parents:
        if par.name in SET_KEYS:
            return par.name
    ckpt = _ckpt_for(jp, payload) if payload is not None else None
    if ckpt:
        # Walk the ckpt path back-to-front so the deepest (most specific)
        # marker wins — pathological case: nested OUTPUT_ROOT containing
        # multiple set names.
        for seg in reversed(Path(ckpt).parts):
            if seg in SET_KEYS:
                return seg
    return None


def _summary_for(jp: Path, payload):
    """Return the eval-summary JSON dict for the run that produced `jp`,
    or None. If `jp` IS the summary (payload has `experiment` and
    `summaries`), return payload. Otherwise look up the sibling
    `<exp_label>-eval_summary.json`. eval.py writes only this file with
    the run-level metadata (ckpt, base_model)."""
    if isinstance(payload, dict) and "summaries" in payload and "experiment" in payload:
        return payload
    stem = jp.stem
    if "-" in stem:
        prefix = stem.rsplit("-", 1)[0]
        sibling = jp.parent / f"{prefix}-eval_summary.json"
        if sibling.is_file():
            try:
                with open(sibling) as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def _base_model_for(jp: Path, payload):
    """Return the `base_model` string of the eval run that produced `jp`,
    or None."""
    if isinstance(payload, dict):
        bm = payload.get("base_model")
        if bm:
            return str(bm)
    summary = _summary_for(jp, payload)
    if summary and summary.get("base_model"):
        return str(summary["base_model"])
    return None


def _ckpt_for(jp: Path, payload):
    """Return the `ckpt` path string of the eval run that produced `jp`,
    or None. Used by `_identify_set` to recover the set marker when the
    eval-results layout is flat (no main_*/evol_*/light parent dir)."""
    if isinstance(payload, dict):
        c = payload.get("ckpt")
        if c:
            return str(c)
    summary = _summary_for(jp, payload)
    if summary and summary.get("ckpt"):
        return str(summary["ckpt"])
    return None


def _model_from_base_model(bm: str):
    """Map a base_model path string (e.g. '/.../qwen2.5-0.5b') to a
    MODEL_KEYS key. Authoritative for 7B-vs-0.5B disambiguation when the
    on-disk path uses the same model dir for both (flat layout where
    `output_subdir` collides between main_7b/qwen25 and main_05b/qwen25
    or between paper-faithful and light variants)."""
    s = bm.lower()
    # 0.5B FIRST — '/qwen2.5-0.5b' otherwise fluke-matches the '/qwen2.5'
    # prefix used by the 7B family below.
    if "0.5b" in s or "-05b" in s or "_05b" in s or s.endswith("05b"):
        return "qwen2.5-0.5b"
    if "qwen2.5" in s and ("7b" in s or "-7-b" in s):
        return "qwen25"
    if "llama" in s and "7b" in s:
        return "llama2"
    if "mistral" in s and "7b" in s:
        return "mistral"
    if "deepseek" in s and "7b" in s:
        return "deepseek"
    return None


def _identify_model(jp: Path, payload):
    """Return one of MODEL_KEYS. base_model from the eval summary payload
    is authoritative when present; otherwise we fall back to matching
    parent-dir / filename / payload-experiment segments."""
    bm = _base_model_for(jp, payload)
    if bm:
        m = _model_from_base_model(bm)
        if m:
            return m
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
        if cand in MODEL_ALIASES:
            return MODEL_ALIASES[cand]
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
        for alias, canonical in METHOD_ALIASES.items():
            if cand == alias or cand.endswith("_" + alias) or cand.endswith("/" + alias):
                return canonical
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
    """Walk root once, bucket every parseable JSON by (set, model, method, bench).

    Returns:
        by_set[set_key][mkey][mdir][bench] = list of (acc_float, src_path)
        set_key is one of SET_KEYS or NO_SET (None) for files outside any
        recognized table-set.
    """
    method_mdirs = [m[2] for m in METHODS if m[2]]
    bench_keys = [b for b, _ in BENCHES]
    by_set: dict = {}

    n_total = 0
    n_parsed = 0
    n_fail = 0
    n_no_method = 0
    n_no_model = 0
    fail_log = []

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

        skey = _identify_set(jp, payload)  # falls back to ckpt-path peek when path lacks a set marker
        # 0.5B fall-through: flat layouts (<model>/<method>/...) carry no
        # `light`/`main_05b` dir hint, so a 0.5B eval would land under
        # (no-set) instead of the Light table. base_model disambiguates
        # 7B from 0.5B at the JSON level, so once mkey resolves to the
        # 0.5B model we anchor it to the Light set unless the path
        # already declared one.
        if skey is None and mkey == "qwen2.5-0.5b":
            skey = "light"

        set_bucket = by_set.setdefault(skey, {})
        if mkey not in set_bucket:
            set_bucket[mkey] = {m: {b: [] for b in bench_keys} for m in method_mdirs}
        for bench in bench_keys:
            for v in _accs_from_payload(payload, bench):
                set_bucket[mkey][mdir][bench].append((v, jp))

    sets_str = ",".join(_set_label(k) for k in _sorted_set_keys(by_set.keys())) or "(none)"
    sys.stderr.write(
        f"[make_table] scanned={n_total}  parsed={n_parsed}  parse_fail={n_fail}  "
        f"no_method={n_no_method}  no_model={n_no_model}  "
        f"sets_found={sets_str}\n"
    )
    if fail_log:
        sys.stderr.write(
            f"[make_table] {len(fail_log)} file(s) failed JSON parse — see "
            "lines above. Common causes: half-written file (kill mid-dump), "
            "trailing junk bytes, BOM, or non-JSON content saved with .json "
            "extension by mistake.\n"
        )
    return by_set


def _set_label(skey):
    if skey is None:
        return "(no-set)"
    return skey


def _sorted_set_keys(keys):
    """Canonical set-key order: KNOWN_SETS order, unknown strings alpha, None last."""
    def _k(s):
        if s is None:
            return (2, "")
        return (0 if s in SET_ORDER else 1, SET_ORDER.get(s, 0), s)
    return sorted(keys, key=_k)


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


def _model_display(skey, mkey):
    override = MODEL_DISPLAY_BY_SET.get((skey, mkey))
    if override:
        return override
    for k, disp in MODELS:
        if k == mkey:
            return disp
    return mkey


def _set_display(skey):
    if skey is None:
        return "unknown dataset"
    return SET_DISPLAY.get(skey, skey)


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


def emit_model_table(skey, mkey, model_results, out_fmt, *, is_first):
    """Emit one table for one (set, model) pair."""
    disp = _model_display(skey, mkey)
    set_disp = _set_display(skey)
    rows = build_rows_for_model(model_results)

    # Per-(set, model) Full FT baseline for Δ.
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
        print(f"# {disp} ({set_disp})")
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
        print(f"## {disp} ({set_disp})")
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
        "segment (e.g. .../main_7b/llama2/tads_10/...).\n"
    )
    sys.exit(1)

# Emit per-(set, model) tables, stacked. Sets in KNOWN_SETS order, then
# unknown strings alphabetical, then `(no-set)` last.
first = True
for skey in _sorted_set_keys(all_results.keys()):
    set_results = all_results[skey]
    mkeys_sorted = sorted(
        set_results.keys(),
        key=lambda k: (MODEL_ORDER.get(k, 999), k),
    )
    for mkey in mkeys_sorted:
        emit_model_table(skey, mkey, set_results[mkey], OUT_FMT, is_first=first)
        first = False

# ---- footer: per-(set, model, method, bench) full list (value ← source) ----
sys.stderr.write("\n[make_table] source root: " + str(ROOT) + "\n")
sys.stderr.write("[make_table] cell entries are ALL measurements (desc, deduped @ 2dp)\n")

mdir_by_id = {m[0]: m[2] for m in METHODS}
idx_by_bench = {b: i for i, (b, _) in enumerate(BENCHES)}

for skey in _sorted_set_keys(all_results.keys()):
    set_disp = _set_display(skey)
    set_results = all_results[skey]
    mkeys_sorted = sorted(
        set_results.keys(),
        key=lambda k: (MODEL_ORDER.get(k, 999), k),
    )
    for mkey in mkeys_sorted:
        disp = _model_display(skey, mkey)
        sys.stderr.write(f"\n[make_table] ===== {disp} ({set_disp}) =====\n")
        rows = build_rows_for_model(set_results[mkey])
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
