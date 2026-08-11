"""Tests for scripts/make_table_v2.py — the one-row-one-checkpoint table.

Every fixture here builds the AUTO_EVAL_AGENT.md §0-3(b) layout (or a
manifest over it) under tmp_path and drives the script through its public
``main()``; numeric assertions parse the ``--tsv`` output because it is the
machine-readable mirror of the markdown. The point of the suite is the
integrity contract from docs/cikm-review-revision-audit.md §2.1: no value
may ever be aggregated across run dirs, and ambiguous inputs must abort
loudly instead of producing a plausible-looking number.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_table_v2.py"
_spec = importlib.util.spec_from_file_location("make_table_v2", _SCRIPT)
mtv2 = importlib.util.module_from_spec(_spec)
# dataclasses resolves string annotations via sys.modules[cls.__module__],
# so the module must be registered BEFORE exec_module runs its @dataclass
# definitions (scripts/ is not a package, hence the by-path import).
sys.modules[_spec.name] = mtv2
_spec.loader.exec_module(mtv2)

BENCHES3 = "mmlu,bbh,gsm8k"


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------

def write_run(
    run_dir: Path,
    label: str,
    accs: Dict[str, float],
    seed: Optional[int] = None,
    sealed: bool = True,
    git_sha: Optional[str] = None,
    summary_accs: Optional[Dict[str, float]] = None,
) -> Path:
    """Create one eval run dir: per-bench JSONs + combined summary + cfg.json.

    ``summary_accs`` lets a test inject a summary that disagrees with the
    per-bench files (the conflict case); by default the summary repeats the
    per-bench values, which is the normal, legal duplication.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for bench, acc in accs.items():
        (run_dir / f"{label}-{bench}.json").write_text(
            json.dumps({"benchmark": bench, "accuracy": acc}), encoding="utf-8"
        )
    summary = summary_accs if summary_accs is not None else accs
    (run_dir / f"{label}-eval_summary.json").write_text(
        json.dumps(
            {
                "experiment": label,
                "summaries": [
                    {"benchmark": b, "accuracy": a} for b, a in summary.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg: Dict[str, object] = {}
    if seed is not None:
        cfg["seed"] = seed
    if git_sha is not None:
        cfg["git_sha"] = git_sha
    if cfg:
        (run_dir / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    if sealed:
        (run_dir / "_complete").write_text("ok", encoding="utf-8")
    return run_dir


def read_tsv_sections(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Parse the script's sectioned TSV into {section: [row-dict, ...]}."""
    sections: Dict[str, List[Dict[str, str]]] = {}
    header: List[str] = []
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            current = line[2:].strip()
            sections[current] = []
            header = []
            continue
        cells = line.split("\t")
        if not header:
            header = cells
            continue
        sections[current].append(dict(zip(header, cells)))
    return sections


def build_2x3x3_manifest(tmp_path: Path) -> Path:
    """2 methods × 3 seeds × 3 benches with hand-computable macros.

    tads_10 per-seed macros: 60, 62, 64  → mean 62, SD 2
    random_10 = tads − 10 everywhere     → mean 52, SD 2
    paired diff tads − random per seed   → +10, +10, +10
    """
    entries = []
    for method, delta in (("tads_10", 0.0), ("random_10", -0.10)):
        for seed, bump in ((1, 0.0), (2, 0.02), (3, 0.04)):
            run = tmp_path / "runs_store" / method / f"seed{seed}"
            write_run(
                run,
                f"llama2_{method}",
                {
                    "mmlu": 0.50 + bump + delta,
                    "bbh": 0.60 + bump + delta,
                    "gsm8k": 0.70 + bump + delta,
                },
                seed=seed,
                git_sha=f"abc{seed}",
            )
            entries.append(
                {
                    "set": "main_7b",
                    "model": "llama2",
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run),
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------
# (a) manifest mode happy path: macro means / SD / CI / paired diff sign
# --------------------------------------------------------------------------

def test_manifest_happy_path_macros_and_pairs(tmp_path: Path, capsys):
    manifest = build_2x3x3_manifest(tmp_path)
    tsv = tmp_path / "out.tsv"
    rc = mtv2.main(
        [
            "--manifest", str(manifest),
            "--benches", BENCHES3,
            "--pairs", "tads_10:random_10",
            "--tsv", str(tsv),
        ]
    )
    assert rc == 0
    md = capsys.readouterr().out
    assert "one sealed run dir" in md          # per-run table is present
    assert "Provenance" in md and "abc1" in md  # git SHA surfaced

    sections = read_tsv_sections(tsv)
    per_run = sections["per_run"]
    assert len(per_run) == 6                    # 2 methods × 3 seeds, no more
    agg = {r["method"]: r for r in sections["aggregate"]}

    tads = agg["tads_10"]
    assert tads["n_seeds"] == "3"
    assert float(tads["macro_mean"]) == pytest.approx(62.0, abs=1e-6)
    assert float(tads["macro_sd"]) == pytest.approx(2.0, abs=1e-6)
    # 95% CI: 62 ± t(0.975, df=2) · 2/√3, with t = 4.302652729911275.
    half = 4.302652729911275 * 2.0 / math.sqrt(3.0)
    assert float(tads["ci_lo"]) == pytest.approx(62.0 - half, abs=1e-3)
    assert float(tads["ci_hi"]) == pytest.approx(62.0 + half, abs=1e-3)

    rand = agg["random_10"]
    assert float(rand["macro_mean"]) == pytest.approx(52.0, abs=1e-6)
    assert float(rand["macro_sd"]) == pytest.approx(2.0, abs=1e-6)

    (pair,) = sections["pairs"]
    assert pair["pair"] == "tads_10:random_10"
    assert pair["n_seeds"] == "3" and pair["seeds"] == "1,2,3"
    # A:B reports A − B; tads is uniformly +10 above random → positive sign.
    assert float(pair["mean_diff"]) == pytest.approx(10.0, abs=1e-6)
    assert float(pair["ci_lo"]) == pytest.approx(10.0, abs=1e-6)  # SD of diffs = 0
    # All-positive n=3 diffs: two-sided exact sign-flip p = 2/8 = 0.25.
    assert float(pair["p_perm"]) == pytest.approx(0.25, abs=1e-12)
    assert pair["p_kind"] == "exact"
    assert float(pair["p_holm"]) == pytest.approx(0.25, abs=1e-12)  # single pair


def test_t_ppf_matches_reference():
    """Stdlib t quantile must match published tables (scipy is banned here)."""
    assert mtv2.t_ppf(0.975, 2) == pytest.approx(4.302652729911275, abs=1e-6)
    assert mtv2.t_ppf(0.975, 1) == pytest.approx(12.706204736432095, abs=1e-5)
    assert mtv2.t_ppf(0.975, 30) == pytest.approx(2.0422724563012373, abs=1e-6)


# --------------------------------------------------------------------------
# (b) conflicting accuracies for one bench inside one run dir → abort
# --------------------------------------------------------------------------

def test_conflicting_accuracy_in_one_run_aborts(tmp_path: Path):
    run = tmp_path / "store" / "tads_10" / "seed1"
    # Per-bench file says 0.50 mmlu; summary claims 0.60 → the run dir does
    # not describe one self-consistent eval, so the whole table must die.
    write_run(
        run,
        "llama2_tads_10",
        {"mmlu": 0.50, "bbh": 0.60, "gsm8k": 0.70},
        seed=1,
        summary_accs={"mmlu": 0.60, "bbh": 0.60, "gsm8k": 0.70},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"set": "main_7b", "model": "llama2", "method": "tads_10",
              "seed": 1, "run_dir": str(run)}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", BENCHES3])
    assert ei.value.code == 2


# --------------------------------------------------------------------------
# (c) results-root mode: _latest.txt + _complete honored, unsealed ignored
# --------------------------------------------------------------------------

def test_results_root_latest_txt_and_unsealed_ignored(tmp_path: Path, capsys):
    root = tmp_path / "eval_results"
    cell = root / "llama2" / "tads_10"
    # Sealed run pointed to by _latest.txt — the ONLY legitimate source.
    write_run(
        cell / "runs" / "20260101_000000",
        "llama2_tads_10",
        {"mmlu": 0.50},
        seed=7,
        git_sha="deadbeef",
    )
    # Newer, higher-scoring but UNSEALED run: make_table.sh's max-of-each
    # would have grabbed this 0.90; v2 must never even look at it.
    write_run(
        cell / "runs" / "20260202_000000",
        "llama2_tads_10",
        {"mmlu": 0.90},
        seed=7,
        sealed=False,
    )
    (cell / "_latest.txt").write_text("20260101_000000", encoding="utf-8")

    tsv = tmp_path / "out.tsv"
    rc = mtv2.main(
        ["--results-root", str(root), "--benches", "mmlu", "--tsv", str(tsv)]
    )
    assert rc == 0
    md = capsys.readouterr().out
    assert "50.00" in md
    assert "90.00" not in md  # the unsealed run leaked → integrity broken

    sections = read_tsv_sections(tsv)
    (row,) = sections["per_run"]  # exactly ONE row for the cell
    assert row["model"] == "llama2" and row["method"] == "tads_10"
    assert row["seed"] == "7"                     # from cfg.json
    assert row["run_tag"] == "20260101_000000"    # the _latest.txt target
    assert row["git_sha"] == "deadbeef"
    assert float(row["macro"]) == pytest.approx(50.0, abs=1e-6)
    (agg,) = sections["aggregate"]
    assert agg["n_seeds"] == "1"
    assert float(agg["macro_mean"]) == pytest.approx(50.0, abs=1e-6)
    assert agg["macro_sd"] == "-"  # n=1: no dispersion estimate, no fake SD


def test_results_root_seed_fallback_from_path_segment(tmp_path: Path, capsys):
    """No cfg.json seed → a ``seed(\\d+)`` path segment must fill in."""
    root = tmp_path / "eval_results"
    cell = root / "llama2" / "tads_10_seed42"
    write_run(cell / "runs" / "20260101_000000", "l", {"mmlu": 0.5})
    (cell / "_latest.txt").write_text("20260101_000000", encoding="utf-8")
    tsv = tmp_path / "out.tsv"
    assert mtv2.main(
        ["--results-root", str(root), "--benches", "mmlu", "--tsv", str(tsv)]
    ) == 0
    capsys.readouterr()
    (row,) = read_tsv_sections(tsv)["per_run"]
    assert row["seed"] == "42"


# --------------------------------------------------------------------------
# (d) permutation p-value convention, hand-computable
# --------------------------------------------------------------------------

def test_perm_pvalue_exact_convention():
    """Two-sided sign-flip test, statistic = mean diff, identity included.

    diffs = [1, 2, 3]: of the 2^3 = 8 sign assignments only +++ (sum 6) and
    −−− (sum −6) reach |mean| ≥ 2, so two-sided p = 2/8 = 0.25 exactly.
    (The one-sided sign-test analogue would be 1/8; the script deliberately
    reports two-sided, matching the audit's two-sided claim standard.)
    """
    p, kind = mtv2.perm_pvalue([1.0, 2.0, 3.0])
    assert kind == "exact"
    assert p == pytest.approx(0.25, abs=1e-12)

    # n=1: both assignments tie |mean| → p = 1. Degenerate but well-defined.
    p1, _ = mtv2.perm_pvalue([5.0])
    assert p1 == pytest.approx(1.0, abs=1e-12)

    # All-zero diffs: every assignment ties → p = 1, never a division blowup.
    p0, _ = mtv2.perm_pvalue([0.0, 0.0, 0.0])
    assert p0 == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# (e) missing bench: row excluded from macro unless --allow-partial
# --------------------------------------------------------------------------

def _missing_bench_manifest(tmp_path: Path) -> Path:
    r1 = write_run(
        tmp_path / "s" / "seed1", "l",
        {"mmlu": 0.50, "bbh": 0.60, "gsm8k": 0.70}, seed=1,
    )
    r2 = write_run(  # gsm8k missing
        tmp_path / "s" / "seed2", "l", {"mmlu": 0.52, "bbh": 0.62}, seed=2,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"set": "m", "model": "llama2", "method": "tads_10",
                 "seed": 1, "run_dir": str(r1)},
                {"set": "m", "model": "llama2", "method": "tads_10",
                 "seed": 2, "run_dir": str(r2)},
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def test_missing_bench_excludes_row_by_default(tmp_path: Path, capsys):
    manifest = _missing_bench_manifest(tmp_path)
    tsv = tmp_path / "out.tsv"
    assert mtv2.main(
        ["--manifest", str(manifest), "--benches", BENCHES3, "--tsv", str(tsv)]
    ) == 0
    capsys.readouterr()
    sections = read_tsv_sections(tsv)
    by_seed = {r["seed"]: r for r in sections["per_run"]}
    assert by_seed["2"]["gsm8k"] == "-"   # missing bench renders "-"
    assert by_seed["2"]["macro"] == "-"   # and kills the row's macro
    (agg,) = sections["aggregate"]
    assert agg["n_seeds"] == "1"          # seed 2 EXCLUDED from aggregation
    assert float(agg["macro_mean"]) == pytest.approx(60.0, abs=1e-6)


def test_missing_bench_allow_partial_includes_with_asterisk(tmp_path: Path, capsys):
    manifest = _missing_bench_manifest(tmp_path)
    tsv = tmp_path / "out.tsv"
    assert mtv2.main(
        ["--manifest", str(manifest), "--benches", BENCHES3,
         "--allow-partial", "--tsv", str(tsv)]
    ) == 0
    md = capsys.readouterr().out
    assert "57.00*" in md                 # partial macro is asterisked in markdown
    assert "--allow-partial" in md        # with the explanatory footnote
    sections = read_tsv_sections(tsv)
    by_seed = {r["seed"]: r for r in sections["per_run"]}
    assert by_seed["2"]["partial"] == "yes"
    assert float(by_seed["2"]["macro"]) == pytest.approx(57.0, abs=1e-6)
    (agg,) = sections["aggregate"]
    assert agg["n_seeds"] == "2"
    assert float(agg["macro_mean"]) == pytest.approx(58.5, abs=1e-6)
    assert agg["any_partial"] == "yes"


# --------------------------------------------------------------------------
# Refusals: non-overlapping seeds, seed pin contradiction, duplicate rows
# --------------------------------------------------------------------------

def test_pairs_refuse_non_overlapping_seeds(tmp_path: Path):
    ra = write_run(tmp_path / "a", "l", {"mmlu": 0.5}, seed=1)
    rb = write_run(tmp_path / "b", "l", {"mmlu": 0.4}, seed=2)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"set": "m", "model": "llama2", "method": "tads_10",
                 "seed": 1, "run_dir": str(ra)},
                {"set": "m", "model": "llama2", "method": "random_10",
                 "seed": 2, "run_dir": str(rb)},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(
            ["--manifest", str(manifest), "--benches", "mmlu",
             "--pairs", "tads_10:random_10"]
        )
    assert ei.value.code == 2


def test_manifest_seed_contradicting_cfg_aborts(tmp_path: Path):
    """A pinned seed that disagrees with the run's cfg.json = wrong checkpoint."""
    run = write_run(tmp_path / "r", "l", {"mmlu": 0.5}, seed=3)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"set": "m", "model": "llama2", "method": "tads_10",
              "seed": 1, "run_dir": str(run)}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def test_duplicate_cell_seed_rows_abort(tmp_path: Path):
    """Two sealed runs for one (set, model, method, seed) → ambiguous → die."""
    r1 = write_run(tmp_path / "r1", "l", {"mmlu": 0.5}, seed=1)
    r2 = write_run(tmp_path / "r2", "l", {"mmlu": 0.6}, seed=1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"set": "m", "model": "llama2", "method": "tads_10",
                 "seed": 1, "run_dir": str(r1)},
                {"set": "m", "model": "llama2", "method": "tads_10",
                 "seed": 1, "run_dir": str(r2)},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def _manifest_for(tmp_path: Path, entries: list) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return manifest


def test_two_unseeded_rows_same_cell_abort(tmp_path: Path):
    """Two seed=None rows in one (set, model, method) cell must die: they are
    indistinguishable sealed runs, and aggregating them would report
    n_seeds=2 — replication fabricated from runs nobody can tell apart
    (the exact bypass an adversarial review reproduced)."""
    r1 = write_run(tmp_path / "r1", "l", {"mmlu": 0.5})  # no cfg.json seed,
    r2 = write_run(tmp_path / "r2", "l", {"mmlu": 0.6})  # no path segment
    manifest = _manifest_for(
        tmp_path,
        [
            {"set": "m", "model": "llama2", "method": "tads_10",
             "run_dir": str(r1)},
            {"set": "m", "model": "llama2", "method": "tads_10",
             "run_dir": str(r2)},
        ],
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def test_unseeded_row_mixed_with_seeded_rows_aborts(tmp_path: Path):
    """A seed=None row sharing a cell with seeded rows cannot be proven
    distinct from any of them → abort, never aggregate."""
    r1 = write_run(tmp_path / "r1", "l", {"mmlu": 0.5}, seed=1)
    r2 = write_run(tmp_path / "r2", "l", {"mmlu": 0.6})
    manifest = _manifest_for(
        tmp_path,
        [
            {"set": "m", "model": "llama2", "method": "tads_10",
             "seed": 1, "run_dir": str(r1)},
            {"set": "m", "model": "llama2", "method": "tads_10",
             "run_dir": str(r2)},
        ],
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def test_single_unseeded_row_aggregates_alone_with_marker(tmp_path: Path, capsys):
    """One seed=None row per cell stays legal, but its n_seeds must carry the
    unknown-seed marker instead of claiming a verified count of 1."""
    run = write_run(tmp_path / "r", "l", {"mmlu": 0.5})
    manifest = _manifest_for(
        tmp_path,
        [{"set": "m", "model": "llama2", "method": "tads_10",
          "run_dir": str(run)}],
    )
    tsv = tmp_path / "out.tsv"
    assert mtv2.main(
        ["--manifest", str(manifest), "--benches", "mmlu", "--tsv", str(tsv)]
    ) == 0
    md = capsys.readouterr().out
    assert "1?" in md
    (agg,) = read_tsv_sections(tsv)["aggregate"]
    assert agg["n_seeds"] == "1?"       # a run count, not a verified seed
    assert agg["seeds"] == "-"
    assert float(agg["macro_mean"]) == pytest.approx(50.0, abs=1e-6)


def test_same_run_dir_under_two_pinned_seeds_aborts(tmp_path: Path):
    """One sealed run dir listed under two different pinned seeds counts a
    single checkpoint twice — fabricated replication → abort. (The run
    records no seed of its own, so the pin-vs-cfg check cannot catch it;
    run_dir uniqueness must.)"""
    run = write_run(tmp_path / "r", "l", {"mmlu": 0.5})
    manifest = _manifest_for(
        tmp_path,
        [
            {"set": "m", "model": "llama2", "method": "tads_10",
             "seed": 1, "run_dir": str(run)},
            {"set": "m", "model": "llama2", "method": "tads_10",
             "seed": 2, "run_dir": str(run)},
        ],
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def test_same_run_dir_across_methods_aborts(tmp_path: Path):
    """run_dir uniqueness is independent of the cell key: the same sealed run
    backing rows of two different methods is still one checkpoint counted
    twice."""
    run = write_run(tmp_path / "r", "l", {"mmlu": 0.5}, seed=1)
    manifest = _manifest_for(
        tmp_path,
        [
            {"set": "m", "model": "llama2", "method": "tads_10",
             "seed": 1, "run_dir": str(run)},
            {"set": "m", "model": "llama2", "method": "random_10",
             "seed": 1, "run_dir": str(run)},
        ],
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2


def test_unsealed_run_in_manifest_aborts(tmp_path: Path):
    """Manifest mode enforces the _complete sentinel too — a pinned but
    unsealed run is still an eval that died mid-way."""
    run = write_run(tmp_path / "r", "l", {"mmlu": 0.5}, seed=1, sealed=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"set": "m", "model": "llama2", "method": "tads_10",
              "seed": 1, "run_dir": str(run)}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        mtv2.main(["--manifest", str(manifest), "--benches", "mmlu"])
    assert ei.value.code == 2
