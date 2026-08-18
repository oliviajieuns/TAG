#!/usr/bin/env python
"""Move HF `datasets` arrow caches into a plain per-corpus directory tree.

    # look first — writes nothing
    python scripts/consolidate_hf_datasets.py \\
        --hf-cache /group-volume/data/hf_home/datasets \\
        --dest /group-volume/datasets

    # then materialise
    python scripts/consolidate_hf_datasets.py ... --apply

    # then, only for the ones that verified, remove the cache copy
    python scripts/consolidate_hf_datasets.py ... --apply --delete-source

An HF cache path is unreadable by anything that is not `datasets`
(`liangxin___alpaca_gpt4/default/0.0.0/c361907.../`), which is why the corpus
had to be exported by hand to be usable at all. This writes each cached
dataset out as parquet under a flat, guessable name, next to the corpora
that are already laid out that way.

Safety, in order:

  * Dry run is the default. Nothing is written until --apply.
  * Only ``<hf-cache>`` is read. A sibling ``hub/`` holding model weights is
    never touched — the two live under the same HF_HOME and deleting the
    parent would take both.
  * A destination that already exists is left alone unless --force. The
    destination tree here belongs to someone else; silently overwriting a
    colleague's corpus is worse than doing nothing.
  * --delete-source removes a cache entry only after its destination has
    been written AND re-read with a matching row count. A dataset that
    failed for any reason keeps its source copy.

It also groups datasets whose names normalise to the same thing —
``liangxin___alpaca_gpt4`` and ``vicgalle___alpaca-gpt4`` are two mirrors of
one corpus, and which one a run used is exactly the kind of detail that
becomes unanswerable later.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SPLIT_RE = re.compile(r"-(train|test|validation|dev|eval)(?:-\d+)?\.arrow$", re.I)


def _slug(cache_name: str) -> str:
    """``vicgalle___alpaca-gpt4`` -> ``alpaca-gpt4`` (namespace dropped)."""
    return cache_name.split("___", 1)[-1] if "___" in cache_name else cache_name


def _norm(slug: str) -> str:
    """Mirror-insensitive key: alpaca_gpt4 and alpaca-gpt4 are one corpus."""
    return re.sub(r"[^a-z0-9]", "", slug.lower())


def _dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.1f}{unit}"
        n /= 1024.0
    return f"{n}B"


def _arrow_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.arrow"))


def _split_of(f: Path) -> str:
    m = _SPLIT_RE.search(f.name)
    if m:
        return m.group(1).lower()
    # save_to_disk layout: data-00000-of-00001.arrow under <...>/<split>/
    for part in reversed(f.parts[:-1]):
        if part.lower() in ("train", "test", "validation", "dev", "eval"):
            return part.lower()
    return "train"


def _load_splits(files: List[Path]) -> Dict[str, Any]:
    from datasets import Dataset, concatenate_datasets

    by_split: Dict[str, List[Any]] = {}
    for f in files:
        try:
            by_split.setdefault(_split_of(f), []).append(Dataset.from_file(str(f)))
        except Exception as e:  # noqa: BLE001 — one bad shard is not fatal
            print(f"      ! unreadable {f.name}: {e}", file=sys.stderr)
    return {
        s: (parts[0] if len(parts) == 1 else concatenate_datasets(parts))
        for s, parts in by_split.items() if parts
    }


def scan(hf_cache: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in sorted(hf_cache.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name in ("downloads", "_downloads"):
            continue
        files = _arrow_files(d)
        if not files:
            continue
        out.append({
            "cache_name": d.name,
            "path": d,
            "slug": _slug(d.name),
            "arrow": files,
            "size": _dir_size(d),
        })
    return out


def _already_materialised(target: Path, entry: Dict[str, Any]) -> bool:
    """Did an earlier run of THIS tool write ``target`` from ``entry``?

    Requires the provenance file to name the same cache entry AND every
    recorded split to still be on disk with the recorded row count. Anything
    less and the destination is treated as a stranger's directory.
    """
    src = target / "SOURCE.json"
    if not src.is_file():
        return False
    try:
        meta = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("hf_cache_entry") != entry["cache_name"]:
        return False
    rows = meta.get("rows_by_split") or {}
    if not rows:
        return False
    try:
        import pandas as pd
        for split, n in rows.items():
            f = target / f"{split}-00000-of-00001.parquet"
            if not f.is_file() or len(pd.read_parquet(f)) != n:
                return False
    except Exception:  # noqa: BLE001 — unreadable means "not verified"
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache", required=True,
                    help="the datasets/ directory inside HF_HOME (NOT HF_HOME itself)")
    ap.add_argument("--dest", required=True, help="flat per-corpus destination tree")
    ap.add_argument("--apply", action="store_true", help="actually write")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a destination that already exists")
    ap.add_argument("--delete-source", action="store_true",
                    help="with --apply: remove each cache entry AFTER its "
                         "destination verifies")
    ap.add_argument("--only", default="",
                    help="comma-separated cache names or slugs to act on")
    args = ap.parse_args()

    hf_cache, dest = Path(args.hf_cache), Path(args.dest)
    if not hf_cache.is_dir():
        print(f"no such directory: {hf_cache}", file=sys.stderr)
        return 2
    if hf_cache.name != "datasets":
        print(f"WARNING: --hf-cache is {hf_cache.name!r}, expected 'datasets'.")
        print(f"         Point it at HF_HOME/datasets — a sibling hub/ holds")
        print(f"         model weights and must not be swept up in this.")
        print()
    if not dest.is_dir():
        print(f"no such destination: {dest}", file=sys.stderr)
        return 2

    entries = scan(hf_cache)
    if not entries:
        print(f"no arrow-backed datasets under {hf_cache}")
        return 0

    wanted = {w.strip() for w in args.only.split(",") if w.strip()}
    if wanted:
        entries = [e for e in entries
                   if e["cache_name"] in wanted or e["slug"] in wanted]
        if not entries:
            print(f"--only matched nothing", file=sys.stderr)
            return 2

    existing = {p.name for p in dest.iterdir() if p.is_dir()} if dest.is_dir() else set()

    # ---- mirrors of the same corpus ----
    groups: Dict[str, List[str]] = {}
    for e in entries:
        groups.setdefault(_norm(e["slug"]), []).append(e["cache_name"])
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if dupes:
        print("Mirrors of the same corpus under different names:")
        for k, names in dupes.items():
            print(f"  {k}: {', '.join(names)}")
        print("  Pick ONE for any run and record which — they differ in record")
        print("  count and cleaning, and a table row cannot be reproduced")
        print("  without knowing which was used.")
        print()

    print(f"{'cache entry':<34}{'-> dest':<24}{'size':>9}  status")
    print("-" * 82)
    plan: List[Tuple[Dict[str, Any], Path]] = []
    already: List[Dict[str, Any]] = []
    for e in entries:
        target = dest / e["slug"]
        if e["slug"] in existing and not args.force:
            # A destination this tool wrote from THIS cache entry is not a
            # collision, it is work already done — otherwise the documented
            # two-step (--apply, then --apply --delete-source) would skip
            # everything on the second pass and delete nothing.
            if _already_materialised(target, e):
                already.append(e)
                status = "already materialised here (source removable)"
            else:
                status = "SKIP — destination exists (use --force to replace)"
        else:
            status = "will write" if args.apply else "would write"
            plan.append((e, target))
        print(f"{e['cache_name'][:33]:<34}{e['slug'][:23]:<24}"
              f"{_human(e['size']):>9}  {status}")
    print()

    if not args.apply:
        print("Dry run — nothing written. Re-run with --apply.")
        if already:
            print(f"{len(already)} source(s) already materialised; "
                  f"--apply --delete-source would remove those cache copies.")
        if args.delete_source:
            print("(--delete-source does nothing without --apply.)")
        return 0

    if not plan and not already:
        print("Nothing to do.")
        return 0

    ok: List[Dict[str, Any]] = []
    for e, target in plan:
        print(f"[{e['cache_name']}]")
        splits = _load_splits(e["arrow"])
        if not splits:
            print("   no readable split — leaving the source alone")
            continue
        tmp = target.with_name(target.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        written: Dict[str, int] = {}
        try:
            for split, ds in splits.items():
                f = tmp / f"{split}-00000-of-00001.parquet"
                ds.to_parquet(str(f))
                written[split] = ds.num_rows
                print(f"   {split:<11} {ds.num_rows:>8} rows  "
                      f"{sorted(ds.column_names)}")
            (tmp / "SOURCE.json").write_text(json.dumps({
                "hf_cache_entry": e["cache_name"],
                "hf_cache_path": str(e["path"]),
                "rows_by_split": written,
                "note": "materialised by scripts/consolidate_hf_datasets.py",
            }, indent=2))
        except Exception as ex:  # noqa: BLE001
            print(f"   FAILED: {ex} — source left in place", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            continue

        # ---- verify by re-reading what was written, before anything is removed
        try:
            import pandas as pd
            for split, n in written.items():
                got = len(pd.read_parquet(tmp / f"{split}-00000-of-00001.parquet"))
                if got != n:
                    raise ValueError(f"{split}: wrote {n} rows, read back {got}")
        except Exception as ex:  # noqa: BLE001
            print(f"   VERIFY FAILED: {ex} — source left in place", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            continue

        if target.exists():
            backup = target.with_name(target.name + ".replaced")
            shutil.rmtree(backup, ignore_errors=True)
            target.rename(backup)
            print(f"   previous {target.name}/ moved aside to {backup.name}/")
        tmp.rename(target)
        print(f"   -> {target}")
        ok.append(e)

    print()
    print(f"{len(ok)}/{len(plan)} materialised and verified this run.")
    if already:
        print(f"{len(already)} already materialised by an earlier run.")
    ok = ok + already

    if not args.delete_source:
        if ok:
            print()
            print("Source copies kept. To remove the ones that verified:")
            print("  re-run the same command with --delete-source")
        return 0 if len(ok) == len(plan) else 1

    print()
    for e in ok:
        try:
            shutil.rmtree(e["path"])
            print(f"removed {e['path']}")
        except OSError as ex:
            print(f"could NOT remove {e['path']}: {ex}", file=sys.stderr)
    ok_names = {e["cache_name"] for e in ok}
    skipped = [e["cache_name"] for e in entries if e["cache_name"] not in ok_names]
    if skipped:
        print()
        print(f"Kept (not verified this run): {', '.join(skipped)}")
    print()
    print("Lock files and downloads/ under the cache root are not touched.")
    print("Review and remove them by hand once you are satisfied.")
    return 0 if len(ok) == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
