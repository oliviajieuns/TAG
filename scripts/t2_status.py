#!/usr/bin/env python3
"""Table 2 seed-batch status — which runs exist, which are sealed, what's left.

    git pull && python3 scripts/t2_status.py

Reads only /group-volume (works from any node, no venv, no GPU). Prints the
training and eval state per (arm, seed) and finishes with the exact next
command for whatever is missing.
"""
import json
import os
from pathlib import Path

W = Path(os.environ.get(
    "TAG_WORKSPACE", "/group-volume/jieuns.shin/tads/tests/tag/workspace"))
ARMS = ("legacy_10", "tag_10")
SEEDS = (42, 1, 7)
BENCHES = 7  # mmlu bbh svamp gsm8k mbpp tydiqa xquad (humaneval excluded)

# seed -> arm -> state
train = {s: {} for s in SEEDS}
evald = {s: {} for s in SEEDS}

for arm in ARMS:
    for rd in sorted((W / "runs/main_7b/llama2" / arm / "runs").glob("*")):
        seed = next((s for s in SEEDS if rd.name.endswith(f"seed{s}")), None)
        if seed is None:
            continue
        c = rd / "epoch_last" / "_complete"
        ep = c.read_text().strip() if c.is_file() else "0"
        # keep the best run per seed (a re-run supersedes a dead one)
        if ep >= train[seed].get(arm, ("", "0"))[1]:
            train[seed][arm] = (rd.name, ep)

    edir = W / "eval-results/main_7b/llama2" / arm / "runs"
    if edir.is_dir():
        for rd in sorted(edir.iterdir()):
            cfg = rd / "cfg.json"
            if cfg.is_file():
                seed = json.load(open(cfg)).get("seed")
            elif rd.name in ("20260818_075532", "20260818_083139"):
                seed = 42  # the two pre-stamp seed-42 runs, pinned by hand
            else:
                seed = None
            if seed not in SEEDS:
                continue
            sealed = (rd / "_complete").is_file()
            n = len([p for p in rd.glob("*-*.json")
                     if "eval_summary" not in p.name])
            prev = evald[seed].get(arm)
            if prev is None or (sealed, n) > (prev[1], prev[2]):
                evald[seed][arm] = (rd.name, sealed, n)

print(f"{'seed':<6}{'arm':<12}{'train':<34}{'eval':<26}state")
print("-" * 92)
todo_train, todo_eval = [], []
for s in SEEDS:
    for arm in ARMS:
        t = train[s].get(arm)
        e = evald[s].get(arm)
        t_str = f"{t[0]} ep={t[1]}" if t else "MISSING"
        e_str = (f"{e[0]} sealed={e[1]} b={e[2]}/{BENCHES}" if e else "missing")
        t_ok = bool(t and t[1] == "3")
        e_ok = bool(e and e[1] and e[2] >= BENCHES)
        state = "DONE" if (t_ok and e_ok) else (
            "needs EVAL" if t_ok else "needs TRAIN+EVAL")
        print(f"{s:<6}{arm:<12}{t_str:<34}{e_str:<26}{state}")
        if not t_ok:
            todo_train.append((s, arm))
        elif not e_ok:
            todo_eval.append((s, arm))

print()
if not todo_train and not todo_eval:
    print("All 6 cells complete. Aggregate with the 3-seed manifest snippet")
    print("(docs: make_table_v2 --manifest).")
else:
    seeds_t = sorted({s for s, _ in todo_train})
    seeds_e = sorted({s for s, _ in todo_eval})
    print("Remaining — run on a GPU node, after:")
    print("  source /group-volume/jieuns.shin/venvs/exp/bin/activate")
    print("  source scripts/gpu_cloud/env.sh")
    for s in seeds_t:
        print(f"  # seed {s}: train both arms, then eval")
        print(f"  ARM_DIR=configs/experiments/main_7b/llama2 SCALE=7b "
              f"OVERRIDES=grad_accum=16 bash scripts/run_lowq_all_arms.sh {s} "
              + " ".join(a for x, a in todo_train if x == s))
    for s in seeds_e + seeds_t:
        arms = [a for x, a in todo_eval if x == s] or list(ARMS)
        print(f"  # seed {s}: eval (checkpoint must be the seed-{s} run; "
              f"_latest points at the NEWEST training run,")
        print(f"  #   so evaluate a seed right after training it)")
        print(f"  SET=main_7b MODELS=llama2 METHODS=\"{' '.join(arms)}\" "
              f"BENCHMARKS=mmlu,bbh,svamp,gsm8k,mbpp,tydiqa,xquad "
              f"bash scripts/run_eval_main_7b.sh --gpus 0,1 --parallel")
