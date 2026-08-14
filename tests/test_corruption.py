"""Unit tests for tag.data.corruption — synthetic low-quality pool
generation (plan §3): determinism, manifest ground truth, per-type
behaviour, and the counterfactual derangement used by the reliability view.
Also covers T1b cross-source mismatch, T7 fluent-wrong (two-step CLI
workflow via --dry-run plumbing), K counterfactual pools, and a
byte-identical regression against the pre-T1b/T7 code."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from tag.data.corruption import (
    append_duplicates,
    corrupt_cross_source,
    corrupt_fluent_wrong,
    corrupt_pool,
    derange_within_buckets,
    dirty_labels_from_manifest,
    has_numeric_answer,
    length_bucket_ids,
    make_counterfactual,
    noisy_text,
    perturb_final_number,
    truncate_text,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _pool(n=60, seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        n_words = rng.randint(5, 60)
        out = " ".join(f"w{i}t{j}" for j in range(n_words)) + "."
        recs.append({
            "instruction": f"Do task number {i} carefully.",
            "input": "",
            "output": out if i % 3 else out + f" The answer is {i * 7}.",
        })
    return recs


def test_corrupt_pool_deterministic():
    recs = _pool()
    a_recs, a_man = corrupt_pool(recs, seed=7, mismatch=0.1, noisy=0.1,
                                 truncated=0.1, wrong_answer=0.05,
                                 duplicate_frac=0.05)
    b_recs, b_man = corrupt_pool(recs, seed=7, mismatch=0.1, noisy=0.1,
                                 truncated=0.1, wrong_answer=0.05,
                                 duplicate_frac=0.05)
    assert a_recs == b_recs
    assert a_man == b_man
    # Different seed changes the outcome.
    c_recs, _ = corrupt_pool(recs, seed=8, mismatch=0.1, noisy=0.1,
                             truncated=0.1, wrong_answer=0.05)
    assert c_recs != a_recs


def test_corrupt_pool_input_not_mutated():
    recs = _pool()
    snapshot = copy.deepcopy(recs)
    corrupt_pool(recs, seed=1, mismatch=0.2, noisy=0.2, truncated=0.2)
    assert recs == snapshot


def test_manifest_counts_and_disjoint_types():
    recs = _pool(n=100)
    out, man = corrupt_pool(recs, seed=3, mismatch=0.1, noisy=0.1,
                            truncated=0.1, wrong_answer=0.1)
    assert man["n_original"] == 100
    assert man["n_total"] == len(out) == 100  # no duplicates requested
    # ~40 % dirty, each sample at most one type (dict keys are unique by
    # construction; verify counts add up).
    types = [e["type"] for e in man["entries"].values()]
    assert 30 <= len(types) <= 45
    labels = dirty_labels_from_manifest(man)
    assert sum(labels) == len(types)


def test_mismatch_swaps_within_pool_and_marks_partner():
    recs = _pool(n=50)
    out, man = corrupt_pool(recs, seed=5, mismatch=0.2)
    originals = {r["output"] for r in recs}
    for idx_str, entry in man["entries"].items():
        i = int(idx_str)
        assert entry["type"] == "mismatch"
        j = entry["partner"]
        assert j != i
        # The corrupted response is the partner's ORIGINAL response.
        assert out[i]["output"] == recs[j]["output"]
        assert out[i]["output"] in originals


def test_truncation_shortens_and_strips_final_punctuation():
    rng = random.Random(0)
    text = " ".join(f"word{i}" for i in range(40)) + "."
    cut = truncate_text(text, rng)
    assert len(cut.split()) < len(text.split())
    assert not cut.endswith(".")


def test_noisy_text_changes_content():
    rng = random.Random(0)
    text = " ".join(f"tok{i}" for i in range(50))
    noised = noisy_text(text, rng, inject_pool=["Alien sentence one."])
    assert noised != text


def test_wrong_answer_perturbs_last_number():
    rng = random.Random(0)
    text = "First 3 apples, then 4 pears. The answer is 21."
    out = perturb_final_number(text, rng)
    assert out is not None and out != text
    assert out.startswith("First 3 apples, then 4 pears.")
    assert perturb_final_number("no numbers here", rng) is None
    assert has_numeric_answer({"output": "x = 5"})
    assert not has_numeric_answer({"output": "none"})


def test_duplicates_appended_with_clusters():
    recs = _pool(n=40)
    entries, clusters = append_duplicates(recs, random.Random(0), frac=0.1,
                                          copies_lo=2, copies_hi=3)
    assert len(recs) > 40
    for cluster in clusters:
        src = cluster[0]
        assert src < 40
        for dup in cluster[1:]:
            assert dup >= 40
            assert entries[dup]["source_index"] == src
            # Jitter preserves the response.
            assert recs[dup]["output"] == recs[src]["output"]


def test_derangement_has_no_fixed_points():
    recs = _pool(n=30)
    idxs = list(range(30))
    mapping = derange_within_buckets(recs, idxs, random.Random(0), n_buckets=5)
    assert set(mapping) == set(idxs)
    assert all(i != j for i, j in mapping.items())


def test_counterfactual_alignment_and_swap():
    recs = _pool(n=30)
    cf = make_counterfactual(recs, seed=11)
    assert len(cf) == len(recs)
    n_swapped = 0
    for i, (orig, c) in enumerate(zip(recs, cf)):
        # Response is preserved; instruction comes from someone else.
        assert c["output"] == orig["output"]
        if c["instruction"] != orig["instruction"]:
            n_swapped += 1
    assert n_swapped == len(recs)
    # Deterministic.
    assert make_counterfactual(recs, seed=11) == cf


def _donor_pool(n=180, seed=1):
    """A different-source donor pool (distinct token vocabulary, so donor
    responses can never collide with _pool responses)."""
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        n_words = rng.randint(5, 60)
        out = " ".join(f"d{i}x{j}" for j in range(n_words)) + "."
        recs.append({
            "instruction": f"Donor task {i} from another dataset.",
            "input": "",
            "output": out,
        })
    return recs


def _digest(obj):
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_no_new_args_regression_byte_identical():
    """T1b/T7/K-counterfactual support must not change any output when the
    new options are unused. Expected hashes were generated by running the
    PRE-change code on these exact fixtures (same seed, no new args)."""
    recs = _pool()
    out_a, man_a = corrupt_pool(recs, seed=7, mismatch=0.1, noisy=0.1,
                                truncated=0.1, wrong_answer=0.05,
                                duplicate_frac=0.05)
    assert _digest(out_a) == (
        "8e4c6d6b6b5f318304386991e86bbeeae49c45affbad1f3367e0c5e98c0c6fab")
    assert _digest(man_a) == (
        "242e7468a3e671a9b381f500021f8d1ed865ef745f997779b0eb79c6fb85967f")
    recs50 = _pool(n=50)
    out_b, man_b = corrupt_pool(recs50, seed=3, mismatch=0.2,
                                sources=["src_a"] * 25 + ["src_b"] * 25)
    assert _digest(out_b) == (
        "a7dcecf998ff76f8aeaaf006c3e02a219bb1b2381c55dbac79cb2d1d7b40ab25")
    assert _digest(man_b) == (
        "68a8c6f25b1b7023762c3c905887f927f4b7c18e64d11862be4e740a03873eb4")
    # Legacy K=1 counterfactual pool (the k=0 derived seed is the base seed,
    # so counterfactual_1 must keep matching this forever).
    assert _digest(make_counterfactual(recs, seed=42)) == (
        "9565c7c5063bcd38c7aa27fafb7c279efd101520b27b0f2673bde9ea7074292c")


def test_xsource_deterministic_manifest_and_donor_uniqueness():
    recs = _pool(n=60)
    donors = _donor_pool(n=180)
    a_recs, a_man = corrupt_pool(recs, seed=9, xsource_frac=0.2,
                                 donor_records=donors)
    b_recs, b_man = corrupt_pool(recs, seed=9, xsource_frac=0.2,
                                 donor_records=donors)
    assert a_recs == b_recs
    assert a_man == b_man
    assert a_man["spec"]["xsource_frac"] == 0.2
    assert len(a_man["entries"]) == 12  # round(60 * 0.2)
    donors_used = []
    for idx_str, entry in a_man["entries"].items():
        i = int(idx_str)
        assert entry["type"] == "mismatch_xsource"
        j = entry["donor"]
        # The corrupted response is the donor's response, not any pool text.
        assert a_recs[i]["output"] == donors[j]["output"]
        assert a_recs[i]["output"] != recs[i]["output"]
        assert a_recs[i]["instruction"] == recs[i]["instruction"]
        donors_used.append(j)
    # Donors plentiful: each donor used at most once.
    assert len(set(donors_used)) == len(donors_used)


def test_xsource_donor_matches_target_length_bucket():
    recs = _pool(n=60)
    donors = _donor_pool(n=180)
    out, man = corrupt_pool(recs, seed=13, xsource_frac=0.25,
                            donor_records=donors, n_buckets=5)
    targets = sorted(int(k) for k in man["entries"])
    t_bucket = {i: b
                for b, grp in length_bucket_ids(recs, targets, 5).items()
                for i in grp}
    d_bucket = {j: b
                for b, grp in length_bucket_ids(
                    donors, list(range(len(donors))), 5).items()
                for j in grp}
    # 36 donors per bucket vs ~3 targets per bucket: no borrowing, so every
    # donor comes from the target's own length-quantile bucket.
    for idx_str, entry in man["entries"].items():
        assert d_bucket[entry["donor"]] == t_bucket[int(idx_str)]


def test_xsource_reuses_donors_only_when_exhausted():
    recs = _pool(n=40)
    donors = _donor_pool(n=5)
    _, man = corrupt_pool(recs, seed=3, xsource_frac=0.5,
                          donor_records=donors)
    used = [e["donor"] for e in man["entries"].values()]
    assert len(used) == 20
    # Every donor is consumed once before any donor is reused.
    assert set(used) == set(range(5))
    # Requiring donors without providing them is an error.
    with pytest.raises(ValueError):
        corrupt_pool(recs, seed=3, xsource_frac=0.1)
    with pytest.raises(ValueError):
        corrupt_cross_source([dict(r) for r in recs], [0, 1], [],
                             random.Random(0))


def test_fluent_wrong_two_step_draw_and_application():
    recs = _pool(n=30)
    # Step 1 (emit): frac > 0 without replacements draws targets but leaves
    # the records and entries untouched.
    emit_recs, emit_man = corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2)
    targets = emit_man["fluent_wrong_targets"]
    assert len(targets) == 6  # round(30 * 0.2)
    assert emit_man["entries"] == {}
    assert emit_recs == recs
    assert emit_man["spec"]["fluent_wrong_frac"] == 0.2
    # Step 3 (apply): identical args + replacements draws the SAME indices.
    repl = {str(i): f"Confidently wrong answer number {i}." for i in targets}
    out, man = corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2,
                            fluent_wrong_replacements=repl)
    assert "fluent_wrong_targets" not in man
    assert sorted(int(k) for k in man["entries"]) == sorted(targets)
    for idx_str, entry in man["entries"].items():
        i = int(idx_str)
        assert entry == {"type": "fluent_wrong"}
        assert out[i]["output"] == repl[idx_str]
        assert out[i]["instruction"] == recs[i]["instruction"]
    # Deterministic.
    out2, man2 = corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2,
                              fluent_wrong_replacements=repl)
    assert out2 == out and man2 == man


def test_fluent_wrong_missing_replacement_errors():
    recs = _pool(n=30)
    _, emit_man = corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2)
    targets = emit_man["fluent_wrong_targets"]
    incomplete = {str(i): "x" for i in targets[:-1]}  # drop one index
    with pytest.raises(KeyError):
        corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2,
                     fluent_wrong_replacements=incomplete)
    with pytest.raises(KeyError):
        corrupt_fluent_wrong([copy.deepcopy(r) for r in recs],
                             list(targets), {})
    # Int keys are accepted too (JSON round-trips produce str keys).
    out, _ = corrupt_pool(recs, seed=21, fluent_wrong_frac=0.2,
                          fluent_wrong_replacements={i: "y" for i in targets})
    assert all(out[i]["output"] == "y" for i in targets)


def test_k_counterfactuals_derived_seeds_and_legacy_equality():
    recs = _pool(n=40)
    legacy = make_counterfactual(recs, seed=17)
    ks = [make_counterfactual(recs, seed=17 + 1000 * k) for k in range(3)]
    # counterfactual_1 (k=0) is byte-identical to the legacy K=1 output.
    assert ks[0] == legacy
    # The K pools are genuinely different pairings...
    assert ks[1] != ks[0] and ks[2] != ks[1] and ks[2] != ks[0]
    # ...but each preserves the reliability-view invariants.
    for cf in ks:
        for orig, c in zip(recs, cf):
            assert c["output"] == orig["output"]
            assert c["instruction"] != orig["instruction"]
    # Deterministic.
    assert [make_counterfactual(recs, seed=17 + 1000 * k)
            for k in range(3)] == ks


def _run_cli(script, *argv, expect_ok=True):
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / script),
         *map(str, argv)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc


def test_fluent_wrong_cli_end_to_end_dry_run(tmp_path):
    """Full two-step T7 workflow (plus T1b and K counterfactuals) through
    the CLIs, with gen_fluent_wrong.py --dry-run standing in for the
    server-side model."""
    pool_file = tmp_path / "input.json"
    pool_file.write_text(json.dumps(_pool(n=40)), encoding="utf-8")
    donor_file = tmp_path / "donors.json"
    donor_file.write_text(json.dumps(_donor_pool(n=120)), encoding="utf-8")
    out_dir = tmp_path / "out"
    common = ["--input", pool_file, "--out-dir", out_dir, "--seed", 5,
              "--fluent-wrong-frac", 0.2,
              "--donor-file", donor_file, "--xsource-frac", 0.1]
    # Step 1: emit targets. The pool is NOT yet T7-corrupted, so no
    # final-looking pool.json may exist — only the PENDING-named pool, and
    # the manifest spec must carry the pending flag (pointing
    # ALPACA_DATA_FILES at step-1 output must be a loud naming error, not a
    # silently-mislabeled training run).
    _run_cli("make_corrupted_pool.py", *common, "--emit-fluent-wrong-targets")
    assert not (out_dir / "pool.json").exists()
    pending = json.loads((out_dir / "pool_PENDING_fluent_wrong.json")
                         .read_text(encoding="utf-8"))
    assert len(pending) == 40
    pending_man = json.loads((out_dir / "corruption_manifest.json")
                             .read_text(encoding="utf-8"))
    assert pending_man["spec"]["fluent_wrong_pending"] is True
    targets = json.loads((out_dir / "fluent_wrong_targets.json")
                         .read_text(encoding="utf-8"))
    assert len(targets) == 8  # round(40 * 0.2)
    for t in targets:
        assert {"index", "instruction", "input", "target_words"} <= set(t)
    # Step 2: dry-run generator (no model).
    repl_file = tmp_path / "fluent_wrong.json"
    _run_cli("gen_fluent_wrong.py", "--targets",
             out_dir / "fluent_wrong_targets.json", "--out", repl_file,
             "--dry-run")
    repl = json.loads(repl_file.read_text(encoding="utf-8"))
    assert set(repl) == {str(t["index"]) for t in targets}
    # Step 3: identical flags + the replacements file; same indices drawn.
    # Only THIS step writes the real pool.json, without the pending flag.
    _run_cli("make_corrupted_pool.py", *common,
             "--fluent-wrong-file", repl_file,
             "--emit-counterfactual", "--num-counterfactuals", 2)
    man = json.loads((out_dir / "corruption_manifest.json")
                     .read_text(encoding="utf-8"))
    assert not man["spec"].get("fluent_wrong_pending", False)
    fw = {int(k) for k, e in man["entries"].items()
          if e["type"] == "fluent_wrong"}
    assert fw == {t["index"] for t in targets}
    xs = [e for e in man["entries"].values()
          if e["type"] == "mismatch_xsource"]
    assert len(xs) == 4  # round(40 * 0.1)
    pool = json.loads((out_dir / "pool.json").read_text(encoding="utf-8"))
    donors = json.loads(donor_file.read_text(encoding="utf-8"))
    for t in targets:
        assert pool[t["index"]]["output"] == repl[str(t["index"])]
    for k, e in man["entries"].items():
        if e["type"] == "mismatch_xsource":
            assert pool[int(k)]["output"] == donors[e["donor"]]["output"]
    # K counterfactual pools: counterfactual_1 == legacy counterfactual.
    cf = json.loads((out_dir / "counterfactual.json")
                    .read_text(encoding="utf-8"))
    cf1 = json.loads((out_dir / "counterfactual_1.json")
                     .read_text(encoding="utf-8"))
    cf2 = json.loads((out_dir / "counterfactual_2.json")
                     .read_text(encoding="utf-8"))
    assert cf1 == cf
    assert cf2 != cf1 and len(cf2) == len(pool)
    # Missing drawn index in the replacements file fails loudly.
    bad = dict(repl)
    bad.pop(str(targets[0]["index"]))
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad), encoding="utf-8")
    proc = _run_cli("make_corrupted_pool.py", *common,
                    "--fluent-wrong-file", bad_file, expect_ok=False)
    assert proc.returncode != 0
    assert "missing" in (proc.stderr + proc.stdout).lower()


def test_sources_tracked_through_duplicates():
    recs = _pool(n=20)
    sources = ["src_a"] * 10 + ["src_b"] * 10
    out, man = corrupt_pool(recs, seed=2, noisy=0.1, duplicate_frac=0.2,
                            sources=sources)
    assert man["sources"] is not None
    assert len(man["sources"]) == man["n_total"] == len(out)
    for cluster in man["duplicate_clusters"]:
        src_tag = man["sources"][cluster[0]]
        for dup in cluster[1:]:
            assert man["sources"][dup] == src_tag
