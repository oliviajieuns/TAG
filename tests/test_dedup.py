"""Unit tests for tag.core.dedup — near-duplicate clustering and the
cluster-constrained top-K used by the MVF selection (plan §1.5)."""
from __future__ import annotations

import torch

from tag.core.dedup import constrained_topk, near_duplicate_clusters


def test_exact_and_jittered_duplicates_clustered():
    base = "Write a short story about a robot who learns to paint landscapes."
    texts = [
        base,
        base + " ",                       # whitespace jitter
        base.lower(),                     # case jitter
        "Explain the difference between TCP and UDP protocols in detail.",
        "Summarize the causes of the French Revolution for a history class.",
    ]
    cids = near_duplicate_clusters(texts)
    assert cids[0] >= 0
    assert cids[0] == cids[1] == cids[2]
    assert cids[3] == -1
    assert cids[4] == -1


def test_distinct_texts_stay_singletons():
    texts = [
        f"Task {i}: compute the sum of the first {i + 2} prime numbers only."
        for i in range(20)
    ]
    assert all(c == -1 for c in near_duplicate_clusters(texts))


def test_clustering_deterministic():
    texts = ["alpha beta gamma delta epsilon"] * 3 + ["one two three four five"]
    assert near_duplicate_clusters(texts) == near_duplicate_clusters(texts)


def test_constrained_topk_one_per_cluster():
    # Scores descending by index; indices 0,1,2 share cluster 0.
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    cids = [0, 0, 0, -1, -1]
    picked = constrained_topk(scores, 3, cids).tolist()
    assert picked == [0, 3, 4]  # 1 and 2 skipped (same cluster as 0)


def test_constrained_topk_none_matches_plain_topk():
    scores = torch.rand(50)
    assert torch.equal(
        constrained_topk(scores, 10, None), scores.topk(10).indices,
    )


def test_constrained_topk_fills_when_exhausted():
    # k=4 but only 2 clusters + 0 singletons exist → must fill from skipped.
    scores = torch.tensor([4.0, 3.0, 2.0, 1.0])
    cids = [0, 0, 1, 1]
    picked = constrained_topk(scores, 4, cids).tolist()
    assert len(picked) == 4
    assert set(picked) == {0, 1, 2, 3}
    assert picked[:2] == [0, 2]  # constrained picks first, then fills
