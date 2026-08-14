"""Near-duplicate detection and cluster-constrained top-K selection.

Duplicated instructions are handled OUTSIDE the reward (design decision in
``docs/plan_low_quality_multiview.md`` §1.5): folding duplication into the
score would blur what each view measures, so instead

  1. instructions are clustered offline with MinHash-LSH over word
     shingles (:func:`near_duplicate_clusters`, pure Python — runnable
     without torch at pool-build time), and
  2. the final top-K selection admits at most ONE sample per cluster
     (:func:`constrained_topk`).

Cluster files are produced by ``scripts/make_corrupted_pool.py``
(``--emit-dedup-clusters``) and loaded at training time via
``selection.mvf.dedup_clusters_file``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")

# 64 permutations in 16 bands of 4 rows: candidate-pair probability is
# ~0.5 at Jaccard ≈ 0.55 and >0.97 at Jaccard ≈ 0.8 — catches the
# whitespace/case-jittered duplicates injected by corruption.py while
# leaving ordinary distinct instructions alone (verified by the exact
# Jaccard check below either way).
_NUM_PERM = 64
_BANDS = 16
_ROWS = _NUM_PERM // _BANDS
_MERSENNE = (1 << 61) - 1


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _shingles(text: str, k: int = 3) -> List[int]:
    """Word k-shingles hashed to ints; falls back to the whole string for
    very short texts."""
    words = _normalize(text).split()
    if len(words) < k:
        grams = [" ".join(words)] if words else [""]
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    out = []
    for g in grams:
        h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
        out.append(int.from_bytes(h, "big"))
    return out


def _minhash_params(seed: int) -> List[Tuple[int, int]]:
    import random

    rng = random.Random(seed)
    return [
        (rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE))
        for _ in range(_NUM_PERM)
    ]


def _signature(shingle_set: Sequence[int], params) -> List[int]:
    sig = []
    for a, b in params:
        sig.append(min(((a * s + b) % _MERSENNE) for s in shingle_set))
    return sig


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_clusters(
    texts: Sequence[str],
    *,
    threshold: float = 0.7,
    seed: int = 42,
) -> List[int]:
    """Cluster near-duplicate texts. Returns ``cluster_id`` per text:
    ``-1`` for singletons (no near-duplicate found), otherwise a
    non-negative id shared by all members of the duplicate group.

    MinHash-LSH proposes candidate pairs; every candidate pair is verified
    with the exact Jaccard similarity over word 3-shingles before merging,
    so the ``threshold`` semantics are exact, not probabilistic.
    """
    n = len(texts)
    if n == 0:
        return []
    params = _minhash_params(seed)
    shingle_sets = [set(_shingles(t)) for t in texts]
    signatures = [_signature(s or {0}, params) for s in shingle_sets]

    uf = _UnionFind(n)
    for band in range(_BANDS):
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        lo = band * _ROWS
        for i in range(n):
            key = tuple(signatures[i][lo : lo + _ROWS])
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            head = members[0]
            for other in members[1:]:
                if uf.find(head) == uf.find(other):
                    continue
                if _jaccard(shingle_sets[head], shingle_sets[other]) >= threshold:
                    uf.union(head, other)

    root_to_cluster: Dict[int, int] = {}
    counts: Dict[int, int] = {}
    for i in range(n):
        counts[uf.find(i)] = counts.get(uf.find(i), 0) + 1
    cluster_ids: List[int] = []
    next_id = 0
    for i in range(n):
        r = uf.find(i)
        if counts[r] < 2:
            cluster_ids.append(-1)
            continue
        if r not in root_to_cluster:
            root_to_cluster[r] = next_id
            next_id += 1
        cluster_ids.append(root_to_cluster[r])
    n_grouped = sum(1 for c in cluster_ids if c >= 0)
    if n_grouped:
        logger.info(
            "near_duplicate_clusters: %d/%d texts in %d duplicate groups "
            "(threshold=%.2f)",
            n_grouped, n, next_id, threshold,
        )
    return cluster_ids


def save_clusters(cluster_ids: Sequence[int], path: str) -> None:
    with open(path, "w") as f:
        json.dump(list(cluster_ids), f)


def load_clusters(path: str) -> List[int]:
    with open(path) as f:
        out = json.load(f)
    if not isinstance(out, list):
        raise ValueError(f"load_clusters: {path} does not contain a list")
    return [int(x) for x in out]


def constrained_topk(scores, k: int, cluster_ids: Optional[Sequence[int]]):
    """Top-K indices by score with at most one sample per duplicate cluster.

    Args:
        scores: 1-D torch tensor of shape (N,).
        k: number of indices to return.
        cluster_ids: per-sample cluster id (``-1`` = unconstrained), or
            None to fall back to the plain top-K.

    Returns:
        LongTensor of ``k`` indices in descending-score order. If the
        cluster constraint exhausts the pool before ``k`` picks (only
        possible when k > #clusters + #singletons), the remaining slots
        are filled with the best skipped samples and a warning is logged.
    """
    import torch

    if cluster_ids is None:
        return scores.topk(k).indices
    if len(cluster_ids) != scores.numel():
        raise ValueError(
            f"constrained_topk: cluster_ids length {len(cluster_ids)} != "
            f"scores length {scores.numel()}"
        )
    order = torch.argsort(scores, descending=True)
    picked: List[int] = []
    skipped: List[int] = []
    used_clusters: set = set()
    for idx_t in order.tolist():
        cid = cluster_ids[idx_t]
        if cid >= 0:
            if cid in used_clusters:
                skipped.append(idx_t)
                continue
            used_clusters.add(cid)
        picked.append(idx_t)
        if len(picked) >= k:
            break
    if len(picked) < k:
        fill = skipped[: k - len(picked)]
        logger.warning(
            "constrained_topk: cluster constraint left only %d/%d picks; "
            "filling %d slots with best skipped duplicates.",
            len(picked), k, len(fill),
        )
        picked.extend(fill)
    return torch.tensor(picked[:k], dtype=torch.long)
