"""Emit the clean-only subset of a corrupted pool (Oracle-clean arm).

The Oracle-clean ceiling arm (plan §5.1 arm 3) trains on manifest-verified
clean samples only. This script filters a corrupted pool by its ground-truth
manifest and writes an index-aligned clean pool + the kept-index list, so the
arm runs through the ordinary training path via ALPACA_DATA_FILES with no
pipeline changes.

    python scripts/make_oracle_pool.py \
        --pool pools/composite20/pool.json \
        --manifest pools/composite20/corruption_manifest.json \
        --out-dir pools/composite20/oracle

Outputs:
    <out-dir>/pool_clean.json     clean records only
    <out-dir>/kept_indices.json   original pool indices of the kept records
                                  (for mapping selections back to the manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tag.data.corruption import dirty_labels_from_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pool", required=True, help="corrupted pool.json")
    p.add_argument("--manifest", required=True, help="corruption_manifest.json")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    with open(args.pool, encoding="utf-8") as f:
        records = json.load(f)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    labels = dirty_labels_from_manifest(manifest)
    if len(labels) != len(records):
        sys.exit(
            f"manifest n_total {len(labels)} != pool size {len(records)} — "
            f"wrong manifest for this pool?"
        )

    kept = [i for i, dirty in enumerate(labels) if not dirty]
    clean = [records[i] for i in kept]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool_clean.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False)
    with open(out / "kept_indices.json", "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(
        f"kept {len(kept)}/{len(records)} clean records "
        f"({100.0 * len(kept) / max(1, len(records)):.1f}%) -> {out}"
    )


if __name__ == "__main__":
    main()
