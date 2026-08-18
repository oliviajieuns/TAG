#!/usr/bin/env python
"""make_table_v2 — one table row = one sealed eval run = one final checkpoint.

Replaces ``scripts/make_table.sh``, whose aggregation took the MAX accuracy
per benchmark across ALL historical eval runs ("W-AVG (max-of-each)") and
averaged those maxima. Because the eval layout is history-preserving
(``runs/<timestamp>/`` snapshots, AUTO_EVAL_AGENT.md §0-3(b)), every
re-evaluation of a cell adds another candidate for the max — so cells that
were re-run more often silently drifted upward, mixing scores from
*different checkpoints* into a single row. docs/cikm-review-revision-audit.md
§2.1 classifies this as an integrity problem, not a stylistic choice, and
bans max-over-runs in any form. This script is the enforcement.

HARD RULE (checked, not just documented):
  * every table row reads benchmark JSONs from exactly ONE sealed run dir —
    a dir carrying the ``_complete`` sentinel written when eval sealing
    finished (AUTO_EVAL_AGENT.md §0-3(b): no sentinel = eval died mid-way);
  * two DIFFERENT accuracy readings for the same benchmark inside that one
    run dir abort the whole table with exit code 2 (identical duplicates are
    fine — the per-bench file and the eval summary legitimately repeat the
    same value);
  * values are NEVER combined across run dirs — no max, no mean, nothing.
    Cross-run aggregation is structurally impossible here because each row
    object holds a single ``run_dir`` and readers never see a second one;
  * unknown seeds are no escape hatch: two seed=None rows in one
    (set, model, method) cell — or a seed=None row mixed with seeded rows —
    abort, because indistinguishable runs aggregated together would
    fabricate replication. A single seed=None row may stand alone, with its
    n_seeds rendered as "1?" to flag the unverified seed;
  * the same resolved run dir may back at most ONE table row — listing one
    run under two pinned seeds (or two cells) counts a single checkpoint
    twice and aborts.

Bench list is an explicit, pre-fixed argument (audit §2.2): benches found on
disk but not requested are ignored, and requested benches missing from a run
render "-" and drop that run from macro aggregation (unless
``--allow-partial``, which macros over the present benches and marks the
value with an asterisk).

Statistics (audit §2.5): per-seed macro (unweighted mean over the fixed
bench list) first, then across seeds: mean, sample SD, and a 95% CI using
Student's t with df = n_seeds − 1 (computed from the regularized incomplete
beta function — stdlib only, no scipy). Paired method comparisons
(``--pairs``) use seed-matched differences: t-based 95% CI, an exact
two-sided sign-flip permutation p-value (all 2^n assignments when n ≤ 12),
and Holm-Bonferroni adjustment across the requested pairs within each
(set, model) cell. Comparing methods whose seed sets do not overlap is
refused outright — unpaired comparison is exactly the apples-to-oranges
mixing this script exists to prevent.

Input modes (mutually exclusive, one required):
  --manifest manifest.json
      Explicit pinned list: [{"set":..., "model":..., "method":...,
      "seed":..., "run_dir":...}, ...]. Preferred — full provenance is
      spelled out. Relative run_dir paths resolve against the manifest's
      own directory. If the run dir's cfg.json records a seed that
      contradicts the manifest's pinned seed, that is an error.
  --results-root DIR
      Walk the AUTO_EVAL_AGENT.md §0-3(b) layout
      ``<root>/[<set>/]<model>/<method>/runs/<timestamp>/``. For each cell
      use ONLY the run dir the ``_latest`` pointer names (symlink, or
      ``_latest.txt`` containing the run tag), require the ``_complete``
      sentinel, and read the seed from the run dir's cfg.json ("seed" key;
      fallback: a path segment matching ``seed(\\d+)``; else seed=None with
      a warning). Runs not named by ``_latest`` — including unsealed ones —
      are never read.

Output: GitHub-markdown tables (per-run scores, per-method aggregate,
paired comparisons) on stdout, a provenance footer listing every row's run
dir and git SHA (from cfg.json, when recorded), and optionally the same
data machine-readably via ``--tsv out.tsv``.

Usage:
    python scripts/make_table_v2.py --manifest results/manifest.json \
        --pairs "legacy_10:random_10,legacy_10:data_agent_10" --tsv table.tsv
    python scripts/make_table_v2.py --results-root $EVAL_RESULTS_ROOT \
        --benches mmlu,bbh,gsm8k --allow-partial
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("make_table_v2")

# Fixed default bench list (audit §2.2: decide the macro's bench set BEFORE
# looking at numbers; deviations must be an explicit --benches argument).
DEFAULT_BENCHES = "mmlu,bbh,gsm8k,svamp,mbpp,humaneval,tydiqa,xquad,ifeval"

CONFIDENCE = 0.95           # 95% CI everywhere, per audit §2.5.
EXACT_PERM_MAX_N = 12       # all 2^n sign flips up to here; Monte Carlo above.
MC_PERM_DRAWS = 20_000
MC_PERM_SEED = 20260812     # fixed so re-runs of the script agree.

# Two readings for the same bench in one run dir count as "the same" only if
# they agree to this tolerance; anything larger is a conflict → abort.
ACC_CONFLICT_TOL = 1e-9

# cfg.json keys that may carry the code revision, in lookup order.
GIT_SHA_KEYS = ("git_sha", "git_commit", "commit", "git_rev")

_SEED_SEG_PAT = re.compile(r"seed(\d+)", re.IGNORECASE)
# Mirrors tag/core/run_layout.py: run tags are a portable charset, so a
# _latest.txt payload that fails this is corruption, not a valid pointer.
_RUN_TAG_PAT = re.compile(r"^[A-Za-z0-9._-]+$")

MISSING = "-"


class TableIntegrityError(RuntimeError):
    """A condition that would make the table lie. Always fatal (exit 2)."""


# --------------------------------------------------------------------------
# Row model
# --------------------------------------------------------------------------

@dataclass
class Row:
    """One table row == one sealed eval run dir == one final checkpoint."""

    set_key: Optional[str]
    model: str
    method: str
    seed: Optional[int]
    run_dir: Path
    accs: Dict[str, float]          # bench -> accuracy × 100, from run_dir ONLY
    git_sha: Optional[str] = None
    macro: Optional[float] = None   # filled by compute_macros()
    partial: bool = False           # macro over a subset of benches (--allow-partial)

    @property
    def cell(self) -> Tuple[Optional[str], str, str]:
        return (self.set_key, self.model, self.method)


# --------------------------------------------------------------------------
# Reading one run dir
# --------------------------------------------------------------------------

def _readings_from_payload(payload: object, benches: Sequence[str]):
    """Yield ``(bench, acc×100)`` for every reading of a requested bench.

    Handles the two payload shapes eval.py writes (see make_table.sh, which
    parsed the same files):
      (a) combined summary: {"summaries": [{"benchmark":..., "accuracy":...}]}
      (b) per-bench file:   {"benchmark":..., "accuracy":...}
    Benches not in ``benches`` are ignored — the macro's bench set is fixed
    up front (audit §2.2), so stray extra benches must not affect anything.
    """
    wanted = set(benches)
    if not isinstance(payload, dict):
        return
    if payload.get("benchmark") in wanted:
        v = payload.get("accuracy")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            yield str(payload["benchmark"]), float(v) * 100.0
    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        for s in summaries:
            if isinstance(s, dict) and s.get("benchmark") in wanted:
                v = s.get("accuracy")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    yield str(s["benchmark"]), float(v) * 100.0


def read_run_dir(run_dir: Path, benches: Sequence[str]) -> Dict[str, float]:
    """Collect per-bench accuracies from ONE run dir, refusing ambiguity.

    Non-recursive on purpose: the sealed layout keeps result JSONs at the
    run dir's top level (logs/ holds no scores), and recursing would be the
    first step back toward reading more than one run per row.

    Raises TableIntegrityError when the same benchmark appears with two
    different accuracy values — that means the run dir does not describe a
    single self-consistent evaluation and no row may be built from it.
    """
    if not run_dir.is_dir():
        raise TableIntegrityError(f"run dir does not exist: {run_dir}")
    accs: Dict[str, float] = {}
    src: Dict[str, Path] = {}
    for jp in sorted(run_dir.glob("*.json")):
        if jp.name == "cfg.json" or ".lock" in jp.name:
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            # A malformed JSON inside a *sealed* run dir means the seal lied;
            # we cannot certify the row's readings are complete + unique.
            raise TableIntegrityError(
                f"unreadable JSON in sealed run dir {run_dir}: "
                f"{jp.name} ({type(e).__name__}: {e})"
            ) from e
        for bench, acc in _readings_from_payload(payload, benches):
            if bench in accs and abs(accs[bench] - acc) > ACC_CONFLICT_TOL:
                raise TableIntegrityError(
                    f"conflicting accuracies for bench '{bench}' inside one "
                    f"run dir {run_dir}: {accs[bench]:.6f} (from "
                    f"{src[bench].name}) vs {acc:.6f} (from {jp.name}) — "
                    "a sealed run must contain exactly one reading per bench"
                )
            accs.setdefault(bench, acc)
            src.setdefault(bench, jp)
    return accs


def _require_sealed(run_dir: Path) -> None:
    """`_complete` sentinel or the run never finished sealing — no row."""
    if not (run_dir / "_complete").is_file():
        raise TableIntegrityError(
            f"run dir is not sealed (missing _complete sentinel): {run_dir} "
            "— unsealed runs are evals that died mid-way "
            "(AUTO_EVAL_AGENT.md §0-3(b)) and must not enter the table"
        )
    _require_unlimited(run_dir)


def _require_unlimited(run_dir: Path) -> None:
    """Refuse a run that scored only the first N examples per task.

    ``tag.eval --limit`` is how the pipeline is rehearsed before a real run:
    8 examples per BBH task instead of 250, three minutes instead of an
    hour. The run seals normally, because it completed everything it was
    asked to do — and it is sealed, complete, self-consistent, and utterly
    unpublishable. Nothing else in this file could tell it apart from the
    real thing, so a rehearsal that reached the ``_latest`` pointer would
    have gone straight into the table as a full benchmark.

    ``tag.eval`` no longer moves ``_latest`` for a limited run. This is the
    second lock, for the pointers that were written before that, and for
    ``--run-dir`` which bypasses the pointer entirely.
    """
    for jp in sorted(run_dir.glob("*eval_summary.json")):
        try:
            with open(jp, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue  # _read_run_accuracies reports the unreadable JSON
        if not isinstance(payload, dict):
            continue
        lim = payload.get("limit")
        if lim is not None:
            raise TableIntegrityError(
                f"run dir {run_dir} was evaluated with --limit {lim} — it "
                f"scored the first {lim} example(s) per task, not the "
                f"benchmark. A rehearsal is not a table row. Re-run without "
                f"--limit (scripts/run_eval_main_7b.sh without LIMIT=)."
            )


def _load_cfg(run_dir: Path) -> Optional[dict]:
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else None
    except (OSError, ValueError) as e:
        logger.warning("unreadable cfg.json in %s (%s) — ignored", run_dir, e)
        return None


def _seed_from_cfg(cfg: Optional[dict]) -> Optional[int]:
    if cfg is None:
        return None
    v = cfg.get("seed")
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _seed_from_path(run_dir: Path) -> Optional[int]:
    """Fallback: deepest path segment matching ``seed(\\d+)`` wins."""
    for seg in reversed(run_dir.parts):
        m = _SEED_SEG_PAT.search(seg)
        if m:
            return int(m.group(1))
    return None


def _git_sha_from_cfg(cfg: Optional[dict]) -> Optional[str]:
    if cfg is None:
        return None
    for key in GIT_SHA_KEYS:
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# --------------------------------------------------------------------------
# Input mode 1: manifest
# --------------------------------------------------------------------------

def load_manifest(manifest_path: Path, benches: Sequence[str]) -> List[Row]:
    """Build rows from an explicit pinned manifest (preferred input mode)."""
    try:
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, ValueError) as e:
        raise TableIntegrityError(f"cannot read manifest {manifest_path}: {e}") from e
    if not isinstance(entries, list):
        raise TableIntegrityError(
            f"manifest must be a JSON list of row objects, got "
            f"{type(entries).__name__}: {manifest_path}"
        )

    rows: List[Row] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TableIntegrityError(f"manifest entry #{i} is not an object")
        missing = [k for k in ("model", "method", "run_dir") if not entry.get(k)]
        if missing:
            raise TableIntegrityError(
                f"manifest entry #{i} missing required key(s): {missing}"
            )
        run_dir = Path(str(entry["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = (manifest_path.parent / run_dir).resolve()
        _require_sealed(run_dir)

        cfg = _load_cfg(run_dir)
        cfg_seed = _seed_from_cfg(cfg)
        seed = entry.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as e:
                raise TableIntegrityError(
                    f"manifest entry #{i} has non-integer seed: {entry['seed']!r}"
                ) from e
            # A pinned seed that contradicts the run's own cfg snapshot means
            # the manifest points at the wrong checkpoint — the exact mix-up
            # this script exists to prevent.
            if cfg_seed is not None and cfg_seed != seed:
                raise TableIntegrityError(
                    f"manifest entry #{i} pins seed={seed} but "
                    f"{run_dir / 'cfg.json'} records seed={cfg_seed}"
                )
        else:
            seed = cfg_seed if cfg_seed is not None else _seed_from_path(run_dir)
            if seed is None:
                logger.warning(
                    "manifest entry #%d (%s/%s): no seed in manifest, cfg.json "
                    "or path — row is excluded from paired comparisons",
                    i, entry["model"], entry["method"],
                )

        rows.append(
            Row(
                set_key=entry.get("set"),
                model=str(entry["model"]),
                method=str(entry["method"]),
                seed=seed,
                run_dir=run_dir,
                accs=read_run_dir(run_dir, benches),
                git_sha=_git_sha_from_cfg(cfg),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Input mode 2: results-root walk
# --------------------------------------------------------------------------

def _resolve_latest(cell: Path) -> Optional[Path]:
    """Resolve the cell's ``_latest`` pointer to a run dir, or None.

    Same transparent symlink-or-textfile contract as
    tag.core.run_layout.resolve_latest (not imported: that module pulls in
    pyyaml and this script must stay stdlib-only). A pointer that exists but
    names a missing run dir is an integrity error, not a skip — the layout's
    invariant ("_latest names the newest sealed run") is broken.
    """
    link = cell / "_latest"
    if link.is_symlink():
        target = link.resolve()
        if not target.is_dir():
            raise TableIntegrityError(f"dangling _latest symlink in {cell}")
        return target
    if link.is_dir():
        # Materialised as a real dir (some mounts) — it IS the latest run.
        return link
    txt = cell / "_latest.txt"
    if txt.is_file():
        tag = txt.read_text(encoding="utf-8").strip()
        # Accept both a bare run tag and a "runs/<tag>" relative form.
        tag = tag.split("/")[-1].split("\\")[-1]
        if not tag or not _RUN_TAG_PAT.match(tag):
            raise TableIntegrityError(
                f"malformed _latest.txt in {cell}: {tag!r}"
            )
        run_dir = cell / "runs" / tag
        if not run_dir.is_dir():
            raise TableIntegrityError(
                f"_latest.txt in {cell} names missing run dir: {run_dir}"
            )
        return run_dir
    return None


def discover_results_root(root: Path, benches: Sequence[str]) -> List[Row]:
    """Walk ``<root>/[<set>/]<model>/<method>/runs/<tag>/`` cells.

    Exactly one run dir per cell is ever read: the one ``_latest`` names.
    All other runs — older snapshots, unsealed crashes — are invisible, so
    max-of-each style mixing cannot happen even by accident. Cells with a
    ``runs/`` dir but no ``_latest`` pointer are skipped with a warning
    (nothing was promoted, so there is nothing sealed to report).
    """
    if not root.is_dir():
        raise TableIntegrityError(f"results root is not a directory: {root}")
    rows: List[Row] = []
    for runs_dir in sorted(root.rglob("runs")):
        if not runs_dir.is_dir():
            continue
        cell = runs_dir.parent
        latest = _resolve_latest(cell)
        if latest is None:
            logger.warning(
                "cell %s has runs/ but no _latest pointer — skipped "
                "(no sealed run was ever promoted)", cell
            )
            continue
        _require_sealed(latest)

        rel = cell.relative_to(root)
        parts = rel.parts
        method = parts[-1] if parts else cell.name
        model = parts[-2] if len(parts) >= 2 else "(unknown)"
        set_key = "/".join(parts[:-2]) if len(parts) >= 3 else None

        cfg = _load_cfg(latest)
        seed = _seed_from_cfg(cfg)
        if seed is None:
            seed = _seed_from_path(latest)
            if seed is None:
                logger.warning(
                    "cell %s: no seed in cfg.json or any path segment — "
                    "row is excluded from paired comparisons", cell
                )

        rows.append(
            Row(
                set_key=set_key,
                model=model,
                method=method,
                seed=seed,
                run_dir=latest,
                accs=read_run_dir(latest, benches),
                git_sha=_git_sha_from_cfg(cfg),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Statistics (stdlib-only Student t via regularized incomplete beta)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(q: float, df: int) -> float:
    """Inverse CDF (quantile) of Student's t, by bisection on t_cdf.

    Accurate to ~1e-10 — far below the 2-decimal table precision. Written
    here because the repo's no-scipy rule leaves stdlib math as the only
    dependency this script is allowed.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    if q == 0.5:
        return 0.0
    sign = 1.0 if q > 0.5 else -1.0
    target = q if q > 0.5 else 1.0 - q
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover — unreachable for sane q
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return sign * (lo + hi) / 2.0


def mean_sd_ci(
    values: Sequence[float], confidence: float = CONFIDENCE
) -> Tuple[float, Optional[float], Optional[Tuple[float, float]]]:
    """(mean, sample SD, t-based CI). SD/CI are None when n < 2 — a single
    seed has no dispersion estimate and pretending otherwise is audit-§2.5
    territory."""
    n = len(values)
    if n == 0:
        raise ValueError("mean_sd_ci needs at least one value")
    mean = statistics.fmean(values)
    if n < 2:
        return mean, None, None
    sd = statistics.stdev(values)  # ddof=1
    half = t_ppf(0.5 + confidence / 2.0, n - 1) * sd / math.sqrt(n)
    return mean, sd, (mean - half, mean + half)


def perm_pvalue(diffs: Sequence[float]) -> Tuple[float, str]:
    """Two-sided sign-flip permutation p-value for paired differences.

    Convention (asserted by tests/test_make_table_v2.py): the test statistic
    is the mean difference; the p-value counts sign assignments s ∈ {−1,+1}^n
    with |mean(s·d)| ≥ |mean(d)|, divided by 2^n. The identity assignment is
    always counted, and the all-flipped mirror ties it, so the smallest
    attainable p for nonzero diffs is 2/2^n — e.g. n=3 all-positive diffs
    give p = 2/8 = 0.25 (the one-sided sign-test analogue would be 1/8; we
    report two-sided because the audit's claim standard is two-sided).

    Exact enumeration for n ≤ EXACT_PERM_MAX_N; above that, Monte Carlo with
    a fixed RNG seed and the standard (1 + hits) / (1 + draws) estimator.
    Returns (p, "exact" | "monte-carlo").
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("perm_pvalue needs at least one difference")
    obs = abs(statistics.fmean(diffs))
    tol = 1e-12 * max(1.0, obs)
    if n <= EXACT_PERM_MAX_N:
        hits = 0
        for mask in range(1 << n):
            s = sum(d if (mask >> i) & 1 == 0 else -d for i, d in enumerate(diffs))
            if abs(s) / n >= obs - tol:
                hits += 1
        return hits / float(1 << n), "exact"
    rng = random.Random(MC_PERM_SEED)
    hits = 0
    for _ in range(MC_PERM_DRAWS):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) / n >= obs - tol:
            hits += 1
    return (1 + hits) / float(1 + MC_PERM_DRAWS), "monte-carlo"


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment, order-preserving output."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class Aggregate:
    set_key: Optional[str]
    model: str
    method: str
    n_seeds: int
    seeds: List[Optional[int]]
    mean: Optional[float]
    sd: Optional[float]
    ci: Optional[Tuple[float, float]]
    any_partial: bool = False
    # True when the cell aggregates an unknown-seed row (after
    # check_no_duplicate_cells that can only be a single seed=None row on
    # its own); n_seeds then renders with a "?" so the table never claims
    # a verified seed count it does not have.
    unknown_seed: bool = False


@dataclass
class PairResult:
    set_key: Optional[str]
    model: str
    method_a: str
    method_b: str
    seeds: List[int]
    mean_diff: float
    ci: Optional[Tuple[float, float]]
    p_perm: float
    p_kind: str
    p_holm: float = field(default=math.nan)  # filled after the cell's family is known


def compute_macros(rows: List[Row], benches: Sequence[str], allow_partial: bool) -> None:
    """Fill each row's macro (mean over the fixed bench list) in place.

    A run missing any requested bench gets macro=None (excluded from
    aggregation) unless --allow-partial, in which case the macro covers only
    the present benches and the row is flagged partial for the asterisk
    annotation. Silently macro-ing over whatever happens to exist would
    reintroduce non-comparable rows through the back door.
    """
    for row in rows:
        present = [b for b in benches if b in row.accs]
        if len(present) == len(benches):
            row.macro = statistics.fmean(row.accs[b] for b in benches)
        elif allow_partial and present:
            row.macro = statistics.fmean(row.accs[b] for b in present)
            row.partial = True
            logger.warning(
                "%s/%s/%s seed=%s: partial macro over %d/%d benches (missing: %s)",
                row.set_key or MISSING, row.model, row.method, row.seed,
                len(present), len(benches),
                ",".join(b for b in benches if b not in row.accs),
            )
        else:
            row.macro = None
            if present:
                logger.warning(
                    "%s/%s/%s seed=%s: missing bench(es) %s — row excluded "
                    "from macro aggregation (use --allow-partial to include)",
                    row.set_key or MISSING, row.model, row.method, row.seed,
                    ",".join(b for b in benches if b not in row.accs),
                )


def check_no_duplicate_cells(rows: List[Row]) -> None:
    """Refuse two sealed runs claiming the same (set, model, method, seed).

    Two rows for one cell+seed would mean two different checkpoints compete
    for the same table slot — resolving that by picking either (let alone
    the max) is the checkpoint mixing this script bans.

    seed=None rows are NOT exempt: two unknown-seed rows in one cell are
    indistinguishable from two sealed runs of the same configuration, and
    aggregating them would fabricate an "n_seeds=2" replication claim out
    of runs nobody can tell apart — exactly the cross-run mixing this
    module exists to prevent. Likewise, an unknown-seed row sharing a cell
    with seeded rows cannot be proven distinct from any of them, so the
    mix aborts. A SINGLE seed=None row alone in its cell is legal (it
    aggregates by itself, flagged unknown-seed downstream).
    """
    seen: Dict[Tuple, Path] = {}
    none_seed: Dict[Tuple, Path] = {}
    has_seeded: Dict[Tuple, bool] = {}
    for row in rows:
        if row.seed is None:
            if row.cell in none_seed:
                raise TableIntegrityError(
                    f"two unknown-seed rows for (set={row.set_key}, "
                    f"model={row.model}, method={row.method}): "
                    f"{none_seed[row.cell]} vs {row.run_dir} — without seeds "
                    "they cannot be told apart, so aggregating them would "
                    "fabricate replication; pin seeds or drop one run"
                )
            none_seed[row.cell] = row.run_dir
            continue
        has_seeded[row.cell] = True
        key = (*row.cell, row.seed)
        if key in seen:
            detail = (
                f"{seen[key]} vs {row.run_dir}"
                if seen[key] != row.run_dir
                else f"{row.run_dir} listed twice"
            )
            raise TableIntegrityError(
                f"duplicate row for (set={row.set_key}, model={row.model}, "
                f"method={row.method}, seed={row.seed}): {detail} — "
                "pin exactly one sealed run per seed"
            )
        seen[key] = row.run_dir
    for cell, run_dir in none_seed.items():
        if has_seeded.get(cell):
            raise TableIntegrityError(
                f"unknown-seed row {run_dir} shares cell (set={cell[0]}, "
                f"model={cell[1]}, method={cell[2]}) with seeded rows — it "
                "cannot be proven distinct from any of them; pin its seed "
                "or drop it"
            )


def check_unique_run_dirs(rows: List[Row]) -> None:
    """Refuse the same resolved run dir appearing in more than one row.

    Independent of the cell/seed check: a manifest listing one run dir
    under two different pinned seeds (or two cells) would count a single
    checkpoint twice, fabricating replication from one eval. One sealed
    run dir = at most one table row, ever.
    """
    seen: Dict[Path, Row] = {}
    for row in rows:
        resolved = row.run_dir.resolve()
        if resolved in seen:
            prev = seen[resolved]
            raise TableIntegrityError(
                f"run dir {resolved} appears in more than one row: "
                f"(set={prev.set_key}, model={prev.model}, "
                f"method={prev.method}, seed={prev.seed}) and "
                f"(set={row.set_key}, model={row.model}, "
                f"method={row.method}, seed={row.seed}) — one sealed run is "
                "one checkpoint and may back at most one table row"
            )
        seen[resolved] = row


def aggregate_rows(rows: List[Row]) -> List[Aggregate]:
    """Per (set, model, method): macro mean / SD / 95% CI across seeds."""
    groups: Dict[Tuple, List[Row]] = {}
    for row in rows:
        groups.setdefault(row.cell, []).append(row)

    out: List[Aggregate] = []
    for cell in sorted(groups, key=lambda c: (c[0] or "", c[1], c[2])):
        usable = [r for r in groups[cell] if r.macro is not None]
        if not usable:
            out.append(Aggregate(*cell, 0, [], None, None, None))
            continue
        macros = [r.macro for r in usable]
        mean, sd, ci = mean_sd_ci(macros)
        out.append(
            Aggregate(
                *cell,
                n_seeds=len(usable),
                seeds=sorted((r.seed for r in usable), key=lambda s: (s is None, s)),
                mean=mean,
                sd=sd,
                ci=ci,
                any_partial=any(r.partial for r in usable),
                unknown_seed=any(r.seed is None for r in usable),
            )
        )
    return out


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise TableIntegrityError(
                f"malformed --pairs entry {chunk!r}; expected 'methodA:methodB'"
            )
        pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise TableIntegrityError("--pairs given but no valid 'a:b' entries found")
    return pairs


def compute_pairs(rows: List[Row], pairs: List[Tuple[str, str]]) -> List[PairResult]:
    """Seed-matched paired comparisons (A − B) per (set, model) cell.

    Only seeds present in BOTH methods enter; zero overlap is a hard error
    because an unpaired difference of means across disjoint seeds is exactly
    the uncontrolled comparison audit §2.5 forbids. Holm-Bonferroni is
    applied across the requested pairs WITHIN each (set, model) cell (the
    family the paper reports together).
    """
    # (set, model) -> method -> seed -> macro
    cells: Dict[Tuple[Optional[str], str], Dict[str, Dict[int, float]]] = {}
    for row in rows:
        if row.macro is None or row.seed is None:
            continue
        cells.setdefault((row.set_key, row.model), {}).setdefault(
            row.method, {}
        )[row.seed] = row.macro

    results: List[PairResult] = []
    matched: Dict[Tuple[str, str], int] = {p: 0 for p in pairs}
    for cell_key in sorted(cells, key=lambda c: (c[0] or "", c[1])):
        by_method = cells[cell_key]
        cell_results: List[PairResult] = []
        for method_a, method_b in pairs:
            if method_a not in by_method or method_b not in by_method:
                continue
            seeds = sorted(set(by_method[method_a]) & set(by_method[method_b]))
            if not seeds:
                raise TableIntegrityError(
                    f"pair {method_a}:{method_b} in cell "
                    f"(set={cell_key[0]}, model={cell_key[1]}) has NO "
                    f"overlapping seeds ({sorted(by_method[method_a])} vs "
                    f"{sorted(by_method[method_b])}) — a paired comparison "
                    "requires seed-matched runs; refusing an unpaired one"
                )
            matched[(method_a, method_b)] += 1
            diffs = [by_method[method_a][s] - by_method[method_b][s] for s in seeds]
            mean, _sd, ci = mean_sd_ci(diffs)
            p, kind = perm_pvalue(diffs)
            cell_results.append(
                PairResult(
                    set_key=cell_key[0],
                    model=cell_key[1],
                    method_a=method_a,
                    method_b=method_b,
                    seeds=seeds,
                    mean_diff=mean,
                    ci=ci,
                    p_perm=p,
                    p_kind=kind,
                )
            )
        if cell_results:
            for pr, p_adj in zip(
                cell_results, holm_adjust([pr.p_perm for pr in cell_results])
            ):
                pr.p_holm = p_adj
            results.extend(cell_results)

    unmatched = [p for p, n in matched.items() if n == 0]
    if unmatched:
        raise TableIntegrityError(
            "pair(s) matched no (set, model) cell with both methods present: "
            + ", ".join(f"{a}:{b}" for a, b in unmatched)
        )
    return results


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt(v: Optional[float], nd: int = 2) -> str:
    return MISSING if v is None else f"{v:.{nd}f}"


def _fmt_ci(ci: Optional[Tuple[float, float]]) -> str:
    return MISSING if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _fmt_seed(seed: Optional[int]) -> str:
    return MISSING if seed is None else str(seed)


def _fmt_n_seeds(a: Aggregate) -> str:
    """Unknown-seed cells render "N?" — the count is of RUNS, not verified
    seeds, and the table must say so."""
    return f"{a.n_seeds}?" if a.unknown_seed else str(a.n_seeds)


def _fmt_set(set_key: Optional[str]) -> str:
    return set_key if set_key else MISSING


def _md_table(headers: Sequence[str], rows_out: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows_out:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _sorted_rows(rows: List[Row]) -> List[Row]:
    return sorted(
        rows,
        key=lambda r: (
            r.set_key or "", r.model, r.method,
            r.seed is None, r.seed if r.seed is not None else 0,
            str(r.run_dir),
        ),
    )


def render_markdown(
    rows: List[Row],
    aggregates: List[Aggregate],
    pair_results: List[PairResult],
    benches: Sequence[str],
) -> str:
    rows = _sorted_rows(rows)
    parts: List[str] = []

    parts.append("## Per-run scores (one row = one sealed run dir = one checkpoint)")
    parts.append("")
    per_run = []
    for r in rows:
        macro = _fmt(r.macro) + ("*" if r.partial else "")
        per_run.append(
            [_fmt_set(r.set_key), r.model, r.method, _fmt_seed(r.seed)]
            + [_fmt(r.accs.get(b)) for b in benches]
            + [macro, r.run_dir.name]
        )
    parts.append(
        _md_table(
            ["Set", "Model", "Method", "Seed", *benches, "Macro", "Run"], per_run
        )
    )
    if any(r.partial for r in rows):
        parts.append("")
        parts.append(
            "\\* macro over present benches only (`--allow-partial`); "
            "not comparable to full-bench macros."
        )

    parts.append("")
    parts.append("## Aggregate per (set, model, method) across seeds")
    parts.append("")
    agg_rows = []
    for a in aggregates:
        agg_rows.append(
            [
                _fmt_set(a.set_key), a.model, a.method, _fmt_n_seeds(a),
                ",".join(_fmt_seed(s) for s in a.seeds) or MISSING,
                _fmt(a.mean) + ("*" if a.any_partial else ""),
                _fmt(a.sd), _fmt_ci(a.ci),
            ]
        )
    parts.append(
        _md_table(
            ["Set", "Model", "Method", "n_seeds", "Seeds",
             "Macro mean", "SD", "95% CI (t)"],
            agg_rows,
        )
    )

    if pair_results:
        parts.append("")
        parts.append("## Paired comparisons (A - B, seed-matched)")
        parts.append("")
        pr_rows = []
        for pr in pair_results:
            pr_rows.append(
                [
                    _fmt_set(pr.set_key), pr.model,
                    f"{pr.method_a} - {pr.method_b}",
                    str(len(pr.seeds)),
                    ",".join(str(s) for s in pr.seeds),
                    _fmt(pr.mean_diff), _fmt_ci(pr.ci),
                    f"{pr.p_perm:.4g} ({pr.p_kind})",
                    f"{pr.p_holm:.4g}",
                ]
            )
        parts.append(
            _md_table(
                ["Set", "Model", "Pair", "n", "Seeds", "Mean diff",
                 "95% CI (t)", "p (perm, two-sided)", "p (Holm)"],
                pr_rows,
            )
        )

    parts.append("")
    parts.append("## Provenance (every row's single source run dir)")
    parts.append("")
    for r in rows:
        sha = f"git {r.git_sha}" if r.git_sha else "git sha not recorded"
        parts.append(
            f"- {_fmt_set(r.set_key)}/{r.model}/{r.method} "
            f"seed={_fmt_seed(r.seed)}: `{r.run_dir}` ({sha})"
        )
    parts.append("")
    return "\n".join(parts)


def write_tsv(
    path: Path,
    rows: List[Row],
    aggregates: List[Aggregate],
    pair_results: List[PairResult],
    benches: Sequence[str],
) -> None:
    """Machine-readable mirror of the markdown output, in commented sections."""
    def f4(v: Optional[float]) -> str:
        return MISSING if v is None else f"{v:.4f}"

    lines: List[str] = []
    lines.append("# per_run")
    lines.append(
        "\t".join(
            ["set", "model", "method", "seed", "run_tag", "run_dir", "git_sha"]
            + list(benches) + ["macro", "partial"]
        )
    )
    for r in _sorted_rows(rows):
        lines.append(
            "\t".join(
                [
                    _fmt_set(r.set_key), r.model, r.method, _fmt_seed(r.seed),
                    r.run_dir.name, str(r.run_dir), r.git_sha or MISSING,
                ]
                + [f4(r.accs.get(b)) for b in benches]
                + [f4(r.macro), "yes" if r.partial else "no"]
            )
        )
    lines.append("")
    lines.append("# aggregate")
    lines.append(
        "\t".join(
            ["set", "model", "method", "n_seeds", "seeds", "macro_mean",
             "macro_sd", "ci_lo", "ci_hi", "any_partial"]
        )
    )
    for a in aggregates:
        lines.append(
            "\t".join(
                [
                    _fmt_set(a.set_key), a.model, a.method, _fmt_n_seeds(a),
                    ",".join(_fmt_seed(s) for s in a.seeds) or MISSING,
                    f4(a.mean), f4(a.sd),
                    f4(a.ci[0]) if a.ci else MISSING,
                    f4(a.ci[1]) if a.ci else MISSING,
                    "yes" if a.any_partial else "no",
                ]
            )
        )
    if pair_results:
        lines.append("")
        lines.append("# pairs")
        lines.append(
            "\t".join(
                ["set", "model", "pair", "n_seeds", "seeds", "mean_diff",
                 "ci_lo", "ci_hi", "p_perm", "p_kind", "p_holm"]
            )
        )
        for pr in pair_results:
            lines.append(
                "\t".join(
                    [
                        _fmt_set(pr.set_key), pr.model,
                        f"{pr.method_a}:{pr.method_b}",
                        str(len(pr.seeds)),
                        ",".join(str(s) for s in pr.seeds),
                        f4(pr.mean_diff),
                        f4(pr.ci[0]) if pr.ci else MISSING,
                        f4(pr.ci[1]) if pr.ci else MISSING,
                        f"{pr.p_perm:.6g}", pr.p_kind, f"{pr.p_holm:.6g}",
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote TSV: %s", path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_table_v2",
        description=(
            "Paper-table builder with the one-row-one-checkpoint integrity "
            "rule (docs/cikm-review-revision-audit.md §2.1). Never mixes "
            "eval runs; refuses ambiguous inputs with exit code 2."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest", type=Path,
        help="JSON list of pinned rows: "
             '[{"set","model","method","seed","run_dir"}, ...]',
    )
    src.add_argument(
        "--results-root", type=Path,
        help="walk <root>/[<set>/]<model>/<method>/runs/ and use each cell's "
             "_latest sealed run only",
    )
    p.add_argument(
        "--benches", default=DEFAULT_BENCHES,
        help=f"comma-separated bench list for the macro (default: {DEFAULT_BENCHES})",
    )
    p.add_argument(
        "--allow-partial", action="store_true",
        help="macro over present benches (asterisked) instead of excluding "
             "rows with missing benches",
    )
    p.add_argument(
        "--pairs", default=None,
        help="paired comparisons, e.g. 'legacy_10:random_10,legacy_10:full_100' "
             "(A:B reports A − B)",
    )
    p.add_argument("--tsv", type=Path, default=None, help="also write a TSV file")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows consoles default to legacy codepages (cp949 here) that cannot
    # encode the markdown output; force UTF-8 where the stream supports it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):  # pragma: no cover — exotic streams
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="[make_table_v2] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    if not benches:
        logger.error("--benches resolved to an empty list")
        raise SystemExit(2)

    try:
        if args.manifest is not None:
            rows = load_manifest(args.manifest, benches)
        else:
            rows = discover_results_root(args.results_root, benches)
        if not rows:
            raise TableIntegrityError(
                "no sealed runs found — nothing to tabulate"
            )
        compute_macros(rows, benches, args.allow_partial)
        check_no_duplicate_cells(rows)
        check_unique_run_dirs(rows)
        aggregates = aggregate_rows(rows)
        pair_results = (
            compute_pairs(rows, parse_pairs(args.pairs)) if args.pairs else []
        )
    except TableIntegrityError as e:
        logger.error("INTEGRITY: %s", e)
        raise SystemExit(2) from e

    sys.stdout.write(render_markdown(rows, aggregates, pair_results, benches))
    if args.tsv is not None:
        write_tsv(args.tsv, rows, aggregates, pair_results, benches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
