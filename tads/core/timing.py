"""Phase-level timing for cross-method comparison.

Each training script (``tads.train``, ``baselines.{nait,selectit,
data_agent,lima,alpagasus,q2q}.train``) brackets its expensive sections
with ``timer.phase(name, category=...)``. At end-of-training we dump
a JSON report and log a comparison table so different selection
algorithms can be compared apples-to-apples on wall-clock cost.

Categories
----------
    setup       — model load, tokenizer load, env init.
    data        — dataset build, dataloader prep, seed file IO.
    selection   — method-specific sample-scoring work (anchor.update,
                  NAIT PCA, SelectIT token/sentence scoring, Data Agent
                  PPO episode + actor update, etc.).
    sft         — pure SFT loop (forward + backward + optimizer.step).
    checkpoint  — save_pretrained + metrics.json writes.
    misc        — uncategorised bracketed sections.

The category split is what enables "pure training time vs. algorithm
overhead" comparison across methods that have very different
selection costs.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


VALID_CATEGORIES = ("setup", "data", "selection", "sft", "checkpoint", "misc")


class PhaseTimer:
    def __init__(self, log: Optional[logging.Logger] = None, method: str = "?"):
        self.log = log
        self.method = method
        self._totals: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._start_overall = time.time()

    @contextmanager
    def phase(self, name: str, category: str = "misc") -> Iterator[None]:
        if category not in VALID_CATEGORIES:
            category = "misc"
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            entry = self._totals.setdefault(
                name, {"category": category, "seconds": 0.0, "count": 0},
            )
            entry["seconds"] += dt
            entry["count"] += 1
            self._events.append(
                {"phase": name, "category": category, "seconds": dt,
                 "t_end": time.time()},
            )
            if self.log:
                self.log.info(
                    "PHASE %s | cat=%s | %.2fs (count=%d, cum=%.1fs)",
                    name, category, dt, entry["count"], entry["seconds"],
                )

    def report(self) -> Dict[str, Any]:
        elapsed_total = time.time() - self._start_overall
        by_phase = []
        for name, info in sorted(
            self._totals.items(), key=lambda x: -x[1]["seconds"],
        ):
            by_phase.append({
                "phase": name,
                "category": info["category"],
                "seconds": round(info["seconds"], 3),
                "count": info["count"],
                "avg_seconds": round(info["seconds"] / max(1, info["count"]), 3),
                "pct_of_total": round(
                    100.0 * info["seconds"] / max(1e-8, elapsed_total), 2,
                ),
            })
        by_cat: Dict[str, float] = {}
        for info in self._totals.values():
            by_cat[info["category"]] = by_cat.get(info["category"], 0.0) + info["seconds"]
        by_category = [
            {
                "category": cat,
                "seconds": round(s, 3),
                "pct_of_total": round(100.0 * s / max(1e-8, elapsed_total), 2),
            }
            for cat, s in sorted(by_cat.items(), key=lambda x: -x[1])
        ]
        # "Tracked" = sum of bracketed phases; "untracked" is wall-clock minus
        # tracked — covers anything outside `with timer.phase(...)` blocks
        # (logger setup, top-level config parsing, etc.).
        tracked = sum(info["seconds"] for info in self._totals.values())
        return {
            "method": self.method,
            "elapsed_total_s": round(elapsed_total, 3),
            "tracked_s": round(tracked, 3),
            "untracked_s": round(elapsed_total - tracked, 3),
            "by_phase": by_phase,
            "by_category": by_category,
        }

    def save_report(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rep = self.report()
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=2)
            f.flush()
        tmp.replace(p)

    def log_table(self) -> None:
        if not self.log:
            return
        rep = self.report()
        self.log.info("=" * 88)
        self.log.info(
            "TIMING BREAKDOWN | method=%s | total=%.1fmin (%.1fs) | "
            "tracked=%.1fs | untracked=%.1fs",
            rep["method"],
            rep["elapsed_total_s"] / 60,
            rep["elapsed_total_s"],
            rep["tracked_s"],
            rep["untracked_s"],
        )
        self.log.info(
            "%-32s %-12s %8s %8s %10s",
            "phase", "category", "count", "sec", "pct",
        )
        self.log.info("-" * 88)
        for row in rep["by_phase"]:
            self.log.info(
                "%-32s %-12s %8d %8.2f %9.2f%%",
                row["phase"], row["category"],
                row["count"], row["seconds"], row["pct_of_total"],
            )
        self.log.info("-" * 88)
        self.log.info("%-32s %-12s %8s %8s %10s", "", "BY CATEGORY", "", "sec", "pct")
        for row in rep["by_category"]:
            self.log.info(
                "%-32s %-12s %8s %8.2f %9.2f%%",
                "", row["category"], "", row["seconds"], row["pct_of_total"],
            )
        self.log.info("=" * 88)
