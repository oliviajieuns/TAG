"""Unit tests for tads.data.corruption — synthetic low-quality pool
generation (plan §3): determinism, manifest ground truth, per-type
behaviour, and the counterfactual derangement used by the reliability view."""
from __future__ import annotations

import copy
import random

from tads.data.corruption import (
    append_duplicates,
    corrupt_pool,
    derange_within_buckets,
    dirty_labels_from_manifest,
    has_numeric_answer,
    make_counterfactual,
    noisy_text,
    perturb_final_number,
    truncate_text,
)


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
