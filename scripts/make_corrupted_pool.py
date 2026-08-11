#!/usr/bin/env python
"""Materialise a corrupted instruction pool + manifest for the low-quality
MVF experiments (docs/plan_low_quality_multiview.md §3).

Reads one or more Alpaca-schema JSON files (a list of
``{"instruction", "input", "output"}`` records), applies the requested
corruption mix, and writes:

    <out-dir>/pool.json                 corrupted pool (train on this via
                                        ALPACA_DATA_FILES / data_files)
    <out-dir>/corruption_manifest.json  ground-truth labels for Dirty@K etc.
    <out-dir>/counterfactual.json       index-aligned counterfactual pool
                                        (with --emit-counterfactual; feeds
                                        tads.mvf.counterfactual_data_files)
    <out-dir>/dedup_clusters.json       near-duplicate cluster id per record
                                        (with --emit-dedup-clusters; feeds
                                        tads.mvf.dedup_clusters_file)

Source imbalance (T6): pass --input several times; each file becomes a
source. --source-scale NAME=FACTOR multiplies the four in-place corruption
fractions for that source only (e.g. a "dirty" source at 2x the rate of a
"clean" one). Swap which source gets the high factor across seeds.

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

from tads.data.corruption import corrupt_pool, make_counterfactual  # noqa: E402

IN_PLACE = ("mismatch", "noisy", "truncated", "wrong_answer")


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
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
    with open(tmp, "w") as f:
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
    p.add_argument("--emit-counterfactual", action="store_true")
    p.add_argument("--emit-dedup-clusters", action="store_true")
    p.add_argument("--dedup-threshold", type=float, default=0.7)
    args = p.parse_args()

    fracs = _preset_fracs(args.preset) if args.preset else {t: 0.0 for t in IN_PLACE}
    for t in IN_PLACE:
        override = getattr(args, t)
        if override is not None:
            fracs[t] = override

    scales: Dict[str, float] = {}
    for spec in args.source_scale:
        name, _, factor = spec.partition("=")
        scales[name] = float(factor)

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
    offset = 0
    for si, path in enumerate(args.input):
        source = Path(path).stem
        records = _load_records(path)
        scale = scales.get(source, 1.0)
        src_fracs = {t: min(1.0, f * scale) for t, f in fracs.items()}
        corrupted, manifest = corrupt_pool(
            records,
            seed=args.seed + 1000 * si,
            duplicate_frac=args.duplicate_frac,
            n_buckets=args.n_buckets,
            **src_fracs,
        )
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

    manifest_out = {
        "seed": args.seed,
        "n_original": offset - sum(
            len(c) - 1 for c in merged_clusters
        ),
        "n_total": len(all_records),
        "spec": {
            "global_fracs": fracs,
            "duplicate_frac": args.duplicate_frac,
            "by_source": spec_by_source,
            "source_scales": scales,
        },
        "entries": merged_entries,
        "duplicate_clusters": merged_clusters,
        "sources": all_sources if len(args.input) > 1 else None,
    }

    _dump(all_records, out_dir / "pool.json")
    _dump(manifest_out, out_dir / "corruption_manifest.json")

    if args.emit_counterfactual:
        cf = make_counterfactual(all_records, seed=args.seed, n_buckets=args.n_buckets)
        _dump(cf, out_dir / "counterfactual.json")

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
