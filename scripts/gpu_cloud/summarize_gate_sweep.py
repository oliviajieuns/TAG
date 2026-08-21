#!/usr/bin/env python3
"""Summarise completed TAG, historical R x A, and gate-sweep Table-2 runs.

Reads only validated all-eight-task evaluation directories.  Partial tasks are
reported as missing rather than silently entering a macro average.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path


BENCHES = ("mmlu", "bbh", "svamp", "gsm8k", "mbpp", "humaneval", "tydiqa", "xquad")
SEEDS = (1, 7, 42)
ARMS = {
    "strong": "tag_10_schedfloor_bs64_seed{seed}",
    # Historical Table-2 R x A rerun: effective batch 128 and a zero LR
    # floor.  Keep it separate from the matched batch-64 control below so a
    # schedule difference is never mistaken for a gate-only comparison.
    "ra_repeat": "legacy_10_repeat_seed{seed}",
    "weak": "tag_10_weakpower50_bs64_seed{seed}",
    "soft": "tag_10_softmix50_bs64_seed{seed}",
    "control": "legacy_10_schedfloor_bs64_seed{seed}",
}


def workspace_roots(args: list[str]) -> list[Path]:
    candidates = [Path(value) for value in args]
    if not candidates:
        ambient = os.environ.get("TAG_WORKSPACE")
        if ambient:
            candidates.append(Path(ambient))
        candidates.extend(
            [
                Path("/group-volume/jieuns.shin/tagx/workspace"),
                Path("/group-volume/jieuns.shin/tag2/workspace"),
            ]
        )

    roots: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in roots:
            roots.append(candidate)
    return roots


def complete_runs(workspaces: list[Path], pattern: str) -> list[Path]:
    runs: list[Path] = []
    for workspace in workspaces:
        base = workspace / "eval-results/main_7b/llama2" / pattern
        runs.extend(
            p for p in (base / "runs").glob("all8_*")
            if (p / "_complete").is_file()
        )
    return runs


def newest_complete(workspaces: list[Path], pattern: str) -> tuple[Path | None, int]:
    runs = complete_runs(workspaces, pattern)
    if not runs:
        return None, 0
    return max(runs, key=lambda p: (p / "_complete").stat().st_mtime_ns), len(runs)


def scores_for(run: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for bench in BENCHES:
        paths = list(run.glob(f"*-{bench}.json"))
        if len(paths) != 1:
            raise RuntimeError(f"{run}: expected one {bench} result, found {paths}")
        payload = json.loads(paths[0].read_text())
        if payload.get("benchmark") != bench:
            raise RuntimeError(f"{paths[0]}: benchmark mismatch")
        score = 100.0 * float(payload["accuracy"])
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise RuntimeError(f"{paths[0]}: invalid score {score}")
        scores[bench] = score
    return scores


def main() -> int:
    workspaces = workspace_roots(sys.argv[1:])
    data: dict[str, dict[int, dict[str, float]]] = {}

    print("WORKSPACES")
    for workspace in workspaces:
        state = "FOUND" if workspace.is_dir() else "MISSING"
        print(f"  {state:7s} {workspace}")
    print()
    print("===== GATE SWEEP: VALIDATED ALL-8 RUNS =====")
    for arm, pattern in ARMS.items():
        data[arm] = {}
        for seed in SEEDS:
            run, candidates = newest_complete(
                workspaces, pattern.format(seed=seed)
            )
            if run is None:
                print(f"{arm:8s} seed={seed:2d} MISSING")
                continue
            scores = scores_for(run)
            data[arm][seed] = scores
            macro = statistics.fmean(scores.values())
            duplicate = f" candidates={candidates}" if candidates > 1 else ""
            print(
                f"{arm:8s} seed={seed:2d} AVG={macro:6.2f}{duplicate}  {run}"
            )

    print("\n===== CROSS-SEED AGGREGATE =====")
    header = "arm       n  " + " ".join(f"{b[:6].upper():>6s}" for b in BENCHES) + "    AVG     SD              95%CI"
    print(header)
    for arm in ARMS:
        rows = data[arm]
        if not rows:
            print(f"{arm:8s}  0  MISSING")
            continue
        means = {
            bench: statistics.fmean(rows[s][bench] for s in rows)
            for bench in BENCHES
        }
        run_macros = [statistics.fmean(rows[s].values()) for s in rows]
        avg = statistics.fmean(run_macros)
        if len(run_macros) == 3:
            sd = statistics.stdev(run_macros)
            half = 4.3026527299 * sd / math.sqrt(3.0)
            ci = f"[{avg-half:.2f},{avg+half:.2f}]"
            sd_text = f"{sd:.2f}"
        else:
            sd_text = "--"
            ci = "-- (need seeds 1,7,42)"
        cols = " ".join(f"{means[b]:6.2f}" for b in BENCHES)
        print(f"{arm:8s} {len(rows):2d}  {cols}  {avg:6.2f}  {sd_text:>5s}  {ci:>18s}")

    if len(data["control"]) == 3:
        ctl = statistics.fmean(
            statistics.fmean(data["control"][s].values()) for s in SEEDS
        )
        print("\n===== AVG DIFFERENCE VS MATCHED R x A =====")
        for arm in ("strong", "weak", "soft"):
            if len(data[arm]) == 3:
                avg = statistics.fmean(
                    statistics.fmean(data[arm][s].values()) for s in SEEDS
                )
                print(f"{arm:8s} {avg-ctl:+.2f} points")

    if len(data["strong"]) == 3 and len(data["ra_repeat"]) == 3:
        strong = statistics.fmean(
            statistics.fmean(data["strong"][s].values()) for s in SEEDS
        )
        ra_repeat = statistics.fmean(
            statistics.fmean(data["ra_repeat"][s].values()) for s in SEEDS
        )
        print("\n===== STRONG TAG VS HISTORICAL R x A REPEAT =====")
        print(f"strong - ra_repeat = {strong-ra_repeat:+.2f} points")
        print(
            "NOTE: this contrast is schedule-confounded "
            "(batch 64/floor .10 vs batch 128/floor 0)."
        )
    print("\nOnly _complete all8 directories are included; partial evals never enter these means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
