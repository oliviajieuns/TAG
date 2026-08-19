#!/usr/bin/env python3
"""Aggregate the Table 2 seed batch: discover sealed runs, pin them in a
manifest, run make_table_v2.

    git pull && python3 scripts/t2_table.py

Stdlib only — works from any node, no venv. Refuses to aggregate unless all
six (arm, seed) cells are sealed with the full bench set, so a partial batch
cannot produce a table that looks final.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

W = Path(os.environ.get(
    "TAG_WORKSPACE", "/group-volume/jieuns.shin/tads/tests/tag/workspace"))
ARMS = ("legacy_10", "tag_10")
SEEDS = (42, 1, 7)
BENCHES = "mmlu,bbh,svamp,gsm8k,mbpp,tydiqa,xquad"
N_BENCH = len(BENCHES.split(","))
PINNED_42 = {"legacy_10": "20260818_075532", "tag_10": "20260818_083139"}

rows = []
missing = []
for arm in ARMS:
    edir = W / "eval-results/main_7b/llama2" / arm / "runs"
    if not edir.is_dir():
        sys.exit(f"no eval results at {edir} — wrong TAG_WORKSPACE, or wrong machine")
    best = {}  # seed -> (run_dir, n_bench)
    for rd in sorted(edir.iterdir()):
        cfg = rd / "cfg.json"
        if cfg.is_file():
            seed = json.load(open(cfg)).get("seed")
        elif rd.name == PINNED_42[arm]:
            seed = 42
        else:
            continue
        if seed not in SEEDS or not (rd / "_complete").is_file():
            continue
        n = len([p for p in rd.glob("*-*.json") if "eval_summary" not in p.name])
        if n >= N_BENCH and (seed not in best or rd.name > best[seed][0].name):
            best[seed] = (rd, n)
    for s in SEEDS:
        if s in best:
            rows.append({"set": "main_7b", "model": "llama2", "method": arm,
                         "seed": s, "run_dir": str(best[s][0])})
        else:
            missing.append(f"{arm} seed={s}")

if missing:
    sys.exit(f"refusing to aggregate — missing sealed full-bench runs: "
             f"{', '.join(missing)}\n(run scripts/t2_status.py for the recovery plan)")

man = Path("/tmp/t2_3seed_manifest.json")
man.write_text(json.dumps(rows, indent=2))
print(f"manifest: {len(rows)} rows -> {man}\n")
sys.exit(subprocess.call([
    sys.executable, str(Path(__file__).parent / "make_table_v2.py"),
    "--manifest", str(man), "--benches", BENCHES,
    "--pairs", "tag_10:legacy_10", "--tsv", "/tmp/t2_3seed.tsv",
]))
