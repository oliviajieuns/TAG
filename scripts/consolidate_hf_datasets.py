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

# Bumped when the on-disk layout changes. A destination written by an older
# version is not "already materialised" — the pre-config-aware layout merged
# subsets into one file and has to be rewritten.
_LAYOUT_VERSION = 2

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


def _config_of(f: Path, entry_dir: Path) -> str:
    """The HF *config* (subset) a shard belongs to.

    The cache nests as ``<ns>___<name>/<config>/<version>/<hash>/*.arrow``.
    Ignoring that level merges subsets that are separate datasets with
    separate schemas — jinzhuoran/rwku alone ships forget_level1,
    neighbor_level2 and friends, and concatenating them fails on mismatched
    features. The existing corpora in the destination keep the level too
    (gsm8k/main/test.parquet), so preserving it also matches what the
    evaluators already expect.
    """
    try:
        rel = f.relative_to(entry_dir)
    except ValueError:
        return "default"
    return rel.parts[0] if len(rel.parts) > 1 else "default"


def _load_groups(files: List[Path], entry_dir: Path) -> Dict[Tuple[str, str], Any]:
    """``{(config, split): Dataset}``, skipping groups that cannot be built."""
    from datasets import Dataset, concatenate_datasets

    buckets: Dict[Tuple[str, str], List[Any]] = {}
    for f in files:
        key = (_config_of(f, entry_dir), _split_of(f))
        try:
            buckets.setdefault(key, []).append(Dataset.from_file(str(f)))
        except Exception as e:  # noqa: BLE001 — one bad shard is not fatal
            print(f"      ! unreadable {f.name}: {e}", file=sys.stderr)
    out: Dict[Tuple[str, str], Any] = {}
    for key, parts in buckets.items():
        if not parts:
            continue
        try:
            out[key] = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        except Exception as e:  # noqa: BLE001 — a subset with mixed shard
            # schemas is one subset's problem, not the dataset's.
            print(f"      ! cannot merge {key[0]}/{key[1]}: {e}", file=sys.stderr)
    return out


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


def _prior_version(target: Path, entry: Dict[str, Any]) -> Optional[int]:
    """Layout version of a destination THIS tool wrote from ``entry``, else None."""
    src = target / "SOURCE.json"
    if not src.is_file():
        return None
    try:
        meta = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("hf_cache_entry") != entry["cache_name"]:
        return None
    return int(meta.get("layout_version", 1))


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
    if int(meta.get("layout_version", 1)) != _LAYOUT_VERSION:
        return False
    rows = meta.get("rows") or {}
    if not rows:
        return False
    try:
        import pandas as pd
        for rel, n in rows.items():
            f = target / rel
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
            prior = _prior_version(target, e)
            if _already_materialised(target, e):
                already.append(e)
                status = "already materialised here (source removable)"
            elif prior is not None:
                # Ours, but written before the layout changed — the older one
                # merged a dataset's subsets into a single file. Say so
                # instead of reporting a generic collision.
                status = (f"REWRITE NEEDED — written by layout v{prior}, now v"
                          f"{_LAYOUT_VERSION}: --force --only {e['cache_name']}")
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
        try:
            groups = _load_groups(e["arrow"], e["path"])
        except Exception as ex:  # noqa: BLE001 — one dataset must not abort
            # the run; a crash halfway leaves the rest of the cache in limbo.
            print(f"   FAILED to read: {ex} — source left in place", file=sys.stderr)
            continue
        if not groups:
            print("   nothing readable — leaving the source alone")
            continue
        tmp = target.with_name(target.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        written: Dict[str, int] = {}
        try:
            for (config, split), ds in sorted(groups.items()):
                # A single "default" config flattens; anything else keeps its
                # own directory, matching gsm8k/main/test.parquet.
                sub = tmp if config == "default" else tmp / config
                sub.mkdir(parents=True, exist_ok=True)
                rel = (f"{split}-00000-of-00001.parquet" if config == "default"
                       else f"{config}/{split}-00000-of-00001.parquet")
                ds.to_parquet(str(tmp / rel))
                written[rel] = ds.num_rows
                label = split if config == "default" else f"{config}/{split}"
                print(f"   {label:<28} {ds.num_rows:>8} rows  "
                      f"{sorted(ds.column_names)}")
            (tmp / "SOURCE.json").write_text(json.dumps({
                "hf_cache_entry": e["cache_name"],
                "hf_cache_path": str(e["path"]),
                "layout_version": _LAYOUT_VERSION,
                "rows": written,
                "note": "materialised by scripts/consolidate_hf_datasets.py",
            }, indent=2))
        except Exception as ex:  # noqa: BLE001
            print(f"   FAILED: {ex} — source left in place", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            continue

        # ---- verify by re-reading what was written, before anything is removed
        try:
            import pandas as pd
            for rel, n in written.items():
                got = len(pd.read_parquet(tmp / rel))
                if got != n:
                    raise ValueError(f"{rel}: wrote {n} rows, read back {got}")
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
