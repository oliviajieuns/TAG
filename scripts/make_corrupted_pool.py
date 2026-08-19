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
                                        selection.mvf.counterfactual_data_files)
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
                                        selection.mvf.dedup_clusters_file)

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
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _load_module(name: str, rel_path: str):
    """Load a repo module by file path WITHOUT importing the ``tag``
    package __init__ (which pulls in torch). Pool generation is pure
    Python and must stay runnable on nodes whose torch install is broken
    or absent."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_corruption = _load_module("_tag_corruption", "tag/data/corruption.py")
corrupt_pool = _corruption.corrupt_pool
make_counterfactual = _corruption.make_counterfactual
response_word_len = _corruption.response_word_len

IN_PLACE = ("mismatch", "noisy", "truncated", "wrong_answer")


# Only the fields the SFT prompt reads. A parquet corpus can carry extras
# (the vicgalle Alpaca-GPT4 mirror ships a pre-formatted `text` column); a
# stray field surviving into the pool would change the tokenised text
# without appearing in any config.
_POOL_FIELDS = ("instruction", "input", "output")


def _load_parquet_records(path: Path) -> List[Dict[str, Any]]:
    """Read a parquet file, a directory of shards, or a glob of them.

    Corpora consolidated out of an HF arrow cache land as parquet, so the
    pool generator has to read them directly — the arrow cache they came
    from is deleted by then, and requiring a hand-made JSON step in between
    is how a pool ends up built from a corpus nobody can point at.
    """
    import glob as _glob

    import pandas as pd

    if path.is_dir():
        # Prefer the train split; a corpus with only one split names it
        # anything, so fall back to every shard in a stable order.
        files = sorted(path.glob("train-*.parquet")) or sorted(path.glob("*.parquet"))
    elif any(c in str(path) for c in "*?["):
        files = [Path(f) for f in sorted(_glob.glob(str(path)))]
    else:
        files = [path]
    files = [f for f in files if f.is_file()]
    if not files:
        raise ValueError(f"{path}: no parquet file(s) found")
    frames = [pd.read_parquet(f) for f in files]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    missing = [c for c in ("instruction", "output") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: parquet is missing column(s) {missing}; has "
            f"{sorted(df.columns)}. This is not an Alpaca-shaped corpus."
        )
    keep = [c for c in _POOL_FIELDS if c in df.columns]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"[pool] {path.name}: dropping non-prompt column(s) {dropped}")
    recs = df[keep].fillna("").to_dict(orient="records")
    return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in recs]


def _load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if path.endswith(".parquet") or p.is_dir() or "*" in path:
        data = _load_parquet_records(p)
    else:
        # encoding pinned: pool records are UTF-8 JSON and the process default
        # on Windows is cp949 — a locale-dependent decode must never corrupt or
        # reject a pool.
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of records")
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


# The two-operator, literature-grounded recipe. Everything in it traces to a
# published corruption/augmentation class, and the two types are structurally
# disjoint (one rewires the pairing, the other degrades the text), so the
# per-type detection table is unambiguous:
#   mismatch  response derangement within response-length buckets
#             (Honovich et al., ACL 2023 — observed mismatch class)
#   noisy     EDA random_deletion + random_swap applied to the RESPONSE as
#             corruption (Wei & Zou, EMNLP-IJCNLP 2019, adapted from
#             augmentation) — NO foreign-sentence injection, which the
#             composite noisy operator mixes in and which has no literature
#             anchor and blurs the boundary with mismatch.
GROUNDED_CITATIONS = {
    "mismatch": {
        "impl": "response derangement within response-length buckets",
        "reference": "Honovich et al. (ACL 2023), mismatch/incorrect-output classes",
        "relation": "adapted",
    },
    "noisy": {
        "impl": "EDA random_deletion + random_swap on the response (mode=eda)",
        "reference": "Wei & Zou (EMNLP-IJCNLP 2019), EDA",
        "relation": "adapted_from_augmentation",
    },
}


def _preset_fracs(preset: str) -> Dict[str, float]:
    if preset.startswith("grounded"):
        total = float(preset.replace("grounded", "")) / 100.0
        fr = {t: 0.0 for t in IN_PLACE}
        fr["mismatch"] = total / 2.0
        fr["noisy"] = total / 2.0
        return fr
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
    p.add_argument("--noisy-alpha", type=float, default=0.1,
                   help="grounded presets only: EDA alpha for the noisy "
                        "operator (deletion prob per token; swap count "
                        "fraction). Ignored by composite/pertype presets, "
                        "whose noisy operator is frozen for reproducibility.")
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
            _grounded = bool(args.preset and args.preset.startswith("grounded"))
            corrupted, manifest = corrupt_pool(
                records,
                seed=args.seed + 1000 * si,
                duplicate_frac=args.duplicate_frac,
                n_buckets=args.n_buckets,
                xsource_frac=args.xsource_frac,
                donor_records=donor_records,
                fluent_wrong_frac=args.fluent_wrong_frac,
                fluent_wrong_replacements=local_repl,
                noisy_mode="eda" if _grounded else "legacy",
                noisy_alpha=args.noisy_alpha,
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

    # Record WHICH corpus this pool came from. The candidate pool and the
    # clean reference pool used to calibrate the gate must come from the same
    # corpus: s is a quantile of Delta_hat measured on the reference, so a
    # reference drawn from a different distribution mis-scales every gate
    # value in the run — with no symptom other than a wrong zero-weight rate. Only
    # the manifest can catch that later, so it has to be written here.
    def _corpus_id(path):
        # --input now also takes a parquet DIRECTORY (a consolidated corpus),
        # which cannot be opened as one file. Hash the shards it is actually
        # built from, in a fixed order, with their names — so a reshard or a
        # renamed file changes the digest, as it should.
        src = Path(path)
        h = hashlib.sha256()
        if src.is_dir():
            files = sorted(src.glob("train-*.parquet")) or sorted(src.glob("*.parquet"))
            if not files:
                files = sorted(f for f in src.rglob("*") if f.is_file())
            for f in files:
                h.update(f.name.encode("utf-8"))
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            return {
                "path": str(src.resolve()),
                "sha256": h.hexdigest(),
                "n_files": len(files),
                "files": [f.name for f in files],
            }
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"path": str(src.resolve()), "sha256": h.hexdigest()}

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
        "inputs": [_corpus_id(p) for p in args.input],
    }

    # T7 step 1 must not produce a final-looking pool.json: the records are
    # not yet T7-corrupted, so training on them would silently mislabel
    # every fluent_wrong entry. The PENDING name makes that path an error.
    pool_name = (
        "pool_PENDING_fluent_wrong.json" if args.emit_fluent_wrong_targets
        else "pool.json"
    )
    if args.preset and args.preset.startswith("grounded"):
        # A reviewer asking "why this corruption mix" gets the answer from
        # the manifest itself, next to the fractions and seeds.
        manifest_out["operators"] = {
            t: {**GROUNDED_CITATIONS[t],
                "params": ({"n_buckets": args.n_buckets} if t == "mismatch"
                           else {"alpha": args.noisy_alpha})}
            for t in ("mismatch", "noisy")
        }
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
        near_duplicate_clusters = _load_module(
            "_tag_dedup", "tag/core/dedup.py",
        ).near_duplicate_clusters

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
