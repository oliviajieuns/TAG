#!/usr/bin/env python
"""Materialise a corrupted instruction pool + manifest for the low-quality
MVF experiments (docs/plan_low_quality_multiview.md §3).

Reads one or more Alpaca-schema JSON files (a list of
``{"instruction", "input", "output"}`` records), applies the requested
corruption mix, and writes:

    <out-dir>/pool.json                 corrupted pool (train on this via
                                        ALPACA_DATA_FILES / data_files);
                                        in T7 emit mode the pool is written
                                        as pool_PENDING_fluent_wrong.json
                                        instead — see the T7 workflow below
    <out-dir>/corruption_manifest.json  ground-truth labels for Dirty@K etc.
    <out-dir>/counterfactual.json       index-aligned counterfactual pool
                                        (with --emit-counterfactual; feeds
                                        tads.mvf.counterfactual_data_files)
    <out-dir>/counterfactual_<k>.json   K independent counterfactual pools
                                        (with --num-counterfactuals K > 1;
                                        derived seeds seed+1000*k, k
                                        0-based, so counterfactual_1.json
                                        == counterfactual.json; feeds the
                                        dispersion-discounted reliability
                                        variant)
    <out-dir>/fluent_wrong_targets.json T7 target list for the server-side
                                        generator (with
                                        --emit-fluent-wrong-targets)
    <out-dir>/dedup_clusters.json       near-duplicate cluster id per record
                                        (with --emit-dedup-clusters; feeds
                                        tads.mvf.dedup_clusters_file)

Source imbalance (T6): pass --input several times; each file becomes a
source. --source-scale NAME=FACTOR multiplies the four in-place corruption
fractions for that source only (e.g. a "dirty" source at 2x the rate of a
"clean" one). Swap which source gets the high factor across seeds.

Cross-source mismatch (T1b): --donor-file supplies a DIFFERENT source
dataset (same {instruction,input,output} schema); --xsource-frac F replaces
that fraction of responses with length-bucket-matched donor responses.
Manifest entries record the donor-file index. Neither --source-scale nor
the presets touch T1b/T7 fractions.

Fluent-wrong (T7) is a TWO-STEP workflow — the replacement responses come
from an instruct model on the GPU server while this script stays
model-free and deterministic:

  1. Run with ``--fluent-wrong-frac F --emit-fluent-wrong-targets`` (plus
     all the other flags of the final run). The T7 target indices are drawn
     deterministically and written to <out-dir>/fluent_wrong_targets.json
     (index + instruction/input + target word count); the pool itself is
     NOT yet T7-corrupted, so it is written as
     <out-dir>/pool_PENDING_fluent_wrong.json — never pool.json — and the
     manifest spec carries ``"fluent_wrong_pending": true``. Pointing
     ALPACA_DATA_FILES at this intermediate output is a naming error, not a
     silently-mislabeled training run.
  2. On the server:  python scripts/gen_fluent_wrong.py \
         --targets .../fluent_wrong_targets.json --out fluent_wrong.json
     (use --dry-run to test the plumbing without a model).
  3. Rerun the IDENTICAL command with ``--fluent-wrong-file
     fluent_wrong.json`` instead of ``--emit-fluent-wrong-targets``. The
     same indices are drawn again (same seed/flags; applying T7 consumes no
     randomness) and the pre-generated responses are spliced in. Only this
     step writes the real <out-dir>/pool.json (manifest spec without the
     pending flag). The run errors out if the file lacks any drawn index.

Presets: --preset composite10|composite20|composite40 sets the four
in-place fractions to an even split of the total rate; --preset
pertype-<mismatch|noisy|truncated|wrong_answer>-20 corrupts 20 % with a
single type. Explicit fraction flags override preset values.

Examples:
    python scripts/make_corrupted_pool.py \
        --input data/alpaca_gpt4.json --out-dir data/lowq/composite20 \
        --preset composite20 --duplicate-frac 0.05 --seed 42 \
        --emit-counterfactual --emit-dedup-clusters

    python scripts/make_corrupted_pool.py \
        --input data/alpaca_gpt4.json --input data/dolly15k.json \
        --source-scale alpaca_gpt4=0.5 --source-scale dolly15k=2.0 \
        --preset composite20 --out-dir data/lowq/source_skewed_20 \
        --emit-counterfactual --emit-dedup-clusters
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tads.data.corruption import (  # noqa: E402
    corrupt_pool,
    make_counterfactual,
    response_word_len,
)

IN_PLACE = ("mismatch", "noisy", "truncated", "wrong_answer")


def _load_records(path: str) -> List[Dict[str, Any]]:
    # encoding pinned: pool records are UTF-8 JSON and the process default
    # on Windows is cp949 — a locale-dependent decode must never corrupt or
    # reject a pool.
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of records")
    for i, r in enumerate(data[:5]):
        if "instruction" not in r or "output" not in r:
            raise ValueError(
                f"{path}: record {i} missing 'instruction'/'output' keys "
                f"(got {sorted(r)})"
            )
    return data


def _dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ensure_ascii=False emits raw unicode, so the encoding must be pinned
    # (Windows defaults to cp949, which cannot encode arbitrary pool text).
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=None)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"wrote {path}")


def _preset_fracs(preset: str) -> Dict[str, float]:
    if preset.startswith("composite"):
        total = float(preset.replace("composite", "")) / 100.0
        each = total / len(IN_PLACE)
        return {t: each for t in IN_PLACE}
    if preset.startswith("pertype-"):
        _, type_name, pct = preset.split("-")
        if type_name not in IN_PLACE:
            raise ValueError(f"unknown corruption type in preset: {type_name}")
        fr = {t: 0.0 for t in IN_PLACE}
        fr[type_name] = float(pct) / 100.0
        return fr
    raise ValueError(f"unknown preset: {preset}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", action="append", required=True,
                   help="Alpaca-schema JSON file; repeat for multiple sources.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", default=None,
                   help="composite10|composite20|composite40|pertype-<type>-<pct>")
    for t in IN_PLACE:
        p.add_argument(f"--{t.replace('_', '-')}", type=float, default=None,
                       help=f"fraction of records to corrupt with {t}")
    p.add_argument("--duplicate-frac", type=float, default=0.0)
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--source-scale", action="append", default=[],
                   metavar="NAME=FACTOR",
                   help="per-source multiplier on the in-place fractions (T6)")
    p.add_argument("--donor-file", default=None,
                   help="T1b donor records (different-source Alpaca-schema "
                        "JSON, e.g. Dolly)")
    p.add_argument("--xsource-frac", type=float, default=0.0,
                   help="fraction of records to corrupt with T1b "
                        "mismatch_xsource (requires --donor-file)")
    p.add_argument("--fluent-wrong-frac", type=float, default=0.0,
                   help="fraction of records to corrupt with T7 fluent_wrong")
    p.add_argument("--fluent-wrong-file", default=None,
                   help='T7 replacements JSON {"<pool_index>": "<response>"} '
                        "from scripts/gen_fluent_wrong.py")
    p.add_argument("--emit-fluent-wrong-targets", action="store_true",
                   help="step 1 of the T7 workflow: write "
                        "fluent_wrong_targets.json instead of applying T7")
    p.add_argument("--emit-counterfactual", action="store_true")
    p.add_argument("--num-counterfactuals", type=int, default=1,
                   help="with --emit-counterfactual, also write K "
                        "counterfactual_<k>.json pools at derived seeds "
                        "seed+1000*k (counterfactual_1 == counterfactual)")
    p.add_argument("--emit-dedup-clusters", action="store_true")
    p.add_argument("--dedup-threshold", type=float, default=0.7)
    args = p.parse_args()

    if args.xsource_frac > 0 and not args.donor_file:
        p.error("--xsource-frac requires --donor-file")
    if args.emit_fluent_wrong_targets and args.fluent_wrong_file:
        p.error("--emit-fluent-wrong-targets and --fluent-wrong-file are "
                "mutually exclusive (steps 1 and 3 of the T7 workflow)")
    if args.fluent_wrong_frac > 0 and not (
        args.emit_fluent_wrong_targets or args.fluent_wrong_file
    ):
        p.error("--fluent-wrong-frac needs --emit-fluent-wrong-targets "
                "(step 1) or --fluent-wrong-file (step 3)")
    if args.fluent_wrong_file and args.fluent_wrong_frac <= 0:
        p.error("--fluent-wrong-file without --fluent-wrong-frac > 0")
    if args.emit_fluent_wrong_targets and args.fluent_wrong_frac <= 0:
        p.error("--emit-fluent-wrong-targets without --fluent-wrong-frac > 0")
    if args.num_counterfactuals < 1:
        p.error("--num-counterfactuals must be >= 1")

    fracs = _preset_fracs(args.preset) if args.preset else {t: 0.0 for t in IN_PLACE}
    for t in IN_PLACE:
        override = getattr(args, t)
        if override is not None:
            fracs[t] = override

    scales: Dict[str, float] = {}
    for spec in args.source_scale:
        name, _, factor = spec.partition("=")
        scales[name] = float(factor)

    donor_records = _load_records(args.donor_file) if args.donor_file else None
    fluent_replacements: Dict[int, str] | None = None
    if args.fluent_wrong_file:
        with open(args.fluent_wrong_file, encoding="utf-8") as f:
            fluent_replacements = {int(k): v for k, v in json.load(f).items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-source corruption (T6): each source gets its own corrupt_pool call
    # with scaled fractions, then everything is concatenated with index
    # offsets folded into the merged manifest.
    all_records: List[Dict[str, Any]] = []
    all_sources: List[str] = []
    merged_entries: Dict[str, Any] = {}
    merged_clusters: List[List[int]] = []
    spec_by_source: Dict[str, Any] = {}
    fluent_targets: List[Dict[str, Any]] = []
    offset = 0
    for si, path in enumerate(args.input):
        source = Path(path).stem
        records = _load_records(path)
        scale = scales.get(source, 1.0)
        src_fracs = {t: min(1.0, f * scale) for t, f in fracs.items()}
        local_repl = None
        if fluent_replacements is not None:
            # The replacements file maps GLOBAL pool indices; T7 targets are
            # always original (pre-duplicate) records, so global = local +
            # this source's running offset.
            local_repl = {
                g - offset: text for g, text in fluent_replacements.items()
                if offset <= g < offset + len(records)
            }
        try:
            corrupted, manifest = corrupt_pool(
                records,
                seed=args.seed + 1000 * si,
                duplicate_frac=args.duplicate_frac,
                n_buckets=args.n_buckets,
                xsource_frac=args.xsource_frac,
                donor_records=donor_records,
                fluent_wrong_frac=args.fluent_wrong_frac,
                fluent_wrong_replacements=local_repl,
                **src_fracs,
            )
        except KeyError as e:
            raise SystemExit(
                f"--fluent-wrong-file is missing drawn T7 indices for "
                f"source={source} (global offset {offset}): {e}"
            ) from e
        for li in manifest.get("fluent_wrong_targets", []):
            fluent_targets.append({
                "index": li + offset,
                "instruction": records[li].get("instruction", ""),
                "input": records[li].get("input", ""),
                "target_words": response_word_len(records[li]),
            })
        for k, e in manifest["entries"].items():
            merged_entries[str(int(k) + offset)] = e
        for cluster in manifest["duplicate_clusters"]:
            merged_clusters.append([c + offset for c in cluster])
        spec_by_source[source] = manifest["spec"]
        all_records.extend(corrupted)
        all_sources.extend([source] * len(corrupted))
        offset += len(corrupted)
        n_dirty = len(manifest["entries"])
        print(
            f"source={source}: n={len(corrupted)} dirty={n_dirty} "
            f"({100.0 * n_dirty / max(1, len(corrupted)):.1f}%) scale={scale}"
        )

    global_spec: Dict[str, Any] = {
        "global_fracs": fracs,
        "duplicate_frac": args.duplicate_frac,
        "by_source": spec_by_source,
        "source_scales": scales,
    }
    # Only recorded when used, so pre-T1b/T7 manifests stay byte-identical.
    if args.xsource_frac > 0:
        global_spec["xsource_frac"] = args.xsource_frac
        global_spec["donor_file"] = args.donor_file
    if args.fluent_wrong_frac > 0:
        global_spec["fluent_wrong_frac"] = args.fluent_wrong_frac
        global_spec["fluent_wrong_file"] = args.fluent_wrong_file
    # T7 step 1: the records are NOT yet corrupted even though the spec
    # claims fluent_wrong_frac > 0 — flag it so no consumer can mistake the
    # intermediate manifest for a final one. Step 3 omits the key entirely.
    if args.emit_fluent_wrong_targets:
        global_spec["fluent_wrong_pending"] = True

    manifest_out = {
        "seed": args.seed,
        "n_original": offset - sum(
            len(c) - 1 for c in merged_clusters
        ),
        "n_total": len(all_records),
        "spec": global_spec,
        "entries": merged_entries,
        "duplicate_clusters": merged_clusters,
        "sources": all_sources if len(args.input) > 1 else None,
    }

    # T7 step 1 must not produce a final-looking pool.json: the records are
    # not yet T7-corrupted, so training on them would silently mislabel
    # every fluent_wrong entry. The PENDING name makes that path an error.
    pool_name = (
        "pool_PENDING_fluent_wrong.json" if args.emit_fluent_wrong_targets
        else "pool.json"
    )
    _dump(all_records, out_dir / pool_name)
    _dump(manifest_out, out_dir / "corruption_manifest.json")

    if args.emit_fluent_wrong_targets:
        _dump(fluent_targets, out_dir / "fluent_wrong_targets.json")
        print(
            f"T7 step 1 done: {len(fluent_targets)} targets; generate "
            f"replacements with scripts/gen_fluent_wrong.py, then rerun with "
            f"--fluent-wrong-file to write the real pool.json "
            f"({pool_name} is NOT yet T7-corrupted; do not train on it)"
        )

    if args.emit_counterfactual:
        cf = make_counterfactual(all_records, seed=args.seed, n_buckets=args.n_buckets)
        _dump(cf, out_dir / "counterfactual.json")
        if args.num_counterfactuals > 1:
            # K reliability pools at derived seeds seed+1000*k (k 0-based);
            # counterfactual_1.json == counterfactual.json by construction.
            for k in range(args.num_counterfactuals):
                cf_k = cf if k == 0 else make_counterfactual(
                    all_records, seed=args.seed + 1000 * k,
                    n_buckets=args.n_buckets,
                )
                _dump(cf_k, out_dir / f"counterfactual_{k + 1}.json")

    if args.emit_dedup_clusters:
        from tads.core.dedup import near_duplicate_clusters

        texts = [str(r.get("instruction", "")) for r in all_records]
        cluster_ids = near_duplicate_clusters(
            texts, threshold=args.dedup_threshold, seed=args.seed,
        )
        _dump(cluster_ids, out_dir / "dedup_clusters.json")

    n_dirty_total = len(merged_entries)
    print(
        f"TOTAL: n={len(all_records)} dirty={n_dirty_total} "
        f"({100.0 * n_dirty_total / max(1, len(all_records)):.1f}%)"
    )


if __name__ == "__main__":
    main()
