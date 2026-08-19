"""Synthetic low-quality instruction-data generation (MVF experiments §4).

Deterministic, seeded transforms over raw Alpaca-schema records
(``{"instruction", "input", "output"}``) that inject the corruption types
studied in the low-quality-pool experiments (T1–T7 plus the cross-source
variant T1b), plus the counterfactual pool used by the reliability view:

    T1 ``mismatch``      instruction–response mismatch (response derangement
                         swap within response-length buckets)
    T2 ``noisy``         noisy response (word dropout / adjacent swaps /
                         foreign-sentence injection)
    T3 ``truncated``     incomplete response (cut at 30–70 % of words)
    T4 ``duplicate``     duplicated instructions (exact + whitespace/case
                         jitter), appended to the pool
    T5 ``wrong_answer``  answer errors on the numeric-verifiable subset
                         (final number perturbed)
    T6 source imbalance  handled at the CLI level (`make_corrupted_pool.py`
                         merges several source files with per-source rates
                         and records a ``source`` tag per record)
    T1b ``mismatch_xsource``  cross-source mismatch: the replacement
                         response comes from a DIFFERENT source dataset
                         (length-bucket matched), so detecting it is not
                         tautological for the counterfactual detector,
                         whose derangement operation generates T1
    T7 ``fluent_wrong``  fluent-but-wrong response: plausible, on-topic,
                         confidently incorrect/vacuous text pre-generated
                         off-line by ``scripts/gen_fluent_wrong.py`` and
                         applied here by pool index (defeats trivial
                         perplexity filters)

Every transform records ground truth in a **manifest** so that selection
quality (Dirty@K, AUPRC, per-type recall) can be measured exactly.

Length-bucket derangements: swaps and counterfactual pairings are made
within response-length quantile buckets so that corrupted samples cannot
be detected from response length alone (the length-bias failure mode that
RECOST reports for entropy/NLL-based filters).

All functions are pure Python (no torch/numpy) and deterministic given the
seed. The tokenised/training side never imports this module — corrupted
pools are materialised to JSON by ``scripts/make_corrupted_pool.py`` and
loaded through the ordinary ``ALPACA_DATA_FILES`` path.
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Record = Dict[str, Any]

IN_PLACE_TYPES = (
    "mismatch", "noisy", "truncated", "wrong_answer",
    "mismatch_xsource", "fluent_wrong",
)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    return text.split()


def response_word_len(rec: Record) -> int:
    return len(_words(str(rec.get("output", ""))))


def length_bucket_ids(
    records: Sequence[Record],
    indices: Sequence[int],
    n_buckets: int = 10,
) -> Dict[int, List[int]]:
    """Partition ``indices`` into ``n_buckets`` response-length quantile
    buckets. Returns {bucket_id: [record_index, ...]} with every bucket
    non-empty (empty tails are dropped)."""
    if not indices:
        return {}
    order = sorted(indices, key=lambda i: (response_word_len(records[i]), i))
    n = len(order)
    n_buckets = max(1, min(n_buckets, n))
    buckets: Dict[int, List[int]] = {}
    for pos, idx in enumerate(order):
        b = min(pos * n_buckets // n, n_buckets - 1)
        buckets.setdefault(b, []).append(idx)
    return buckets


def derange(items: List[int], rng: random.Random, max_tries: int = 100) -> Dict[int, int]:
    """Return a derangement mapping i -> j (j != i) over ``items``.

    For a single-element list a derangement is impossible; the caller is
    expected to merge such buckets (see :func:`derange_within_buckets`).
    """
    if len(items) < 2:
        raise ValueError("derange needs at least 2 items")
    for _ in range(max_tries):
        perm = items[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))
    # Deterministic fallback: rotate by one (always a valid derangement).
    return {a: items[(k + 1) % len(items)] for k, a in enumerate(items)}


def derange_within_buckets(
    records: Sequence[Record],
    indices: Sequence[int],
    rng: random.Random,
    n_buckets: int = 10,
) -> Dict[int, int]:
    """Derangement mapping over ``indices``, restricted (as far as possible)
    to response-length quantile buckets. Buckets with fewer than 2 members
    are merged into the following bucket."""
    buckets = length_bucket_ids(records, indices, n_buckets)
    mapping: Dict[int, int] = {}
    pending: List[int] = []
    for b in sorted(buckets):
        group = pending + buckets[b]
        pending = []
        if len(group) < 2:
            pending = group
            continue
        mapping.update(derange(group, rng))
    if pending:
        # A single leftover index (or a global pool of size 1): pair it with
        # any other index so it still gets a mismatched partner.
        leftover = pending[0]
        others = [i for i in indices if i != leftover]
        if others:
            mapping[leftover] = rng.choice(others)
    return mapping


# ---------------------------------------------------------------------------
# T1 — instruction–response mismatch
# ---------------------------------------------------------------------------

def corrupt_mismatch(
    records: List[Record],
    target_idxs: List[int],
    rng: random.Random,
    n_buckets: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """Replace each target's response with a length-matched partner's
    response (derangement within response-length buckets). Only the targets'
    responses change; partners keep their own records."""
    originals = {i: records[i].get("output", "") for i in target_idxs}
    mapping = derange_within_buckets(records, target_idxs, rng, n_buckets)
    entries: Dict[int, Dict[str, Any]] = {}
    for i, j in mapping.items():
        records[i]["output"] = originals.get(j, records[j].get("output", ""))
        entries[i] = {"type": "mismatch", "partner": j}
    return entries


# ---------------------------------------------------------------------------
# T1b — cross-source instruction–response mismatch
# ---------------------------------------------------------------------------

def corrupt_cross_source(
    records: List[Record],
    target_idxs: List[int],
    donor_records: Sequence[Record],
    rng: random.Random,
    n_buckets: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """Replace each target's response with one drawn from ``donor_records``
    (a DIFFERENT source dataset, e.g. Dolly), matched by response-length
    quantile bucket: target bucket b draws from donor bucket b, so corrupted
    samples cannot be detected from response length alone.

    Unlike T1 ``mismatch`` (a within-pool derangement — the very operation
    the counterfactual detector performs), the donor text appears nowhere
    else in the pool, so T1b detection is not tautological for
    counterfactual-based scorers.

    Each donor is used at most once while any unused donor remains (nearest
    non-empty bucket as fallback); reuse only happens once targets
    outnumber donors. Manifest entries record the index into
    ``donor_records`` under ``"donor"``.
    """
    if not donor_records:
        raise ValueError("corrupt_cross_source: empty donor_records")
    target_buckets = length_bucket_ids(records, target_idxs, n_buckets)
    donor_buckets = length_bucket_ids(
        donor_records, list(range(len(donor_records))), n_buckets,
    )
    # bucket_id -> shuffled stack of not-yet-used donor indices.
    unused: Dict[int, List[int]] = {}
    for b, group in donor_buckets.items():
        stack = group[:]
        rng.shuffle(stack)
        unused[b] = stack

    def _pop_nearest(b: int) -> Optional[int]:
        live = [k for k, stack in unused.items() if stack]
        if not live:
            return None
        return unused[min(live, key=lambda k: (abs(k - b), k))].pop()

    entries: Dict[int, Dict[str, Any]] = {}
    for b in sorted(target_buckets):
        for i in target_buckets[b]:
            j = _pop_nearest(b)
            if j is None:
                # Every donor already used once: sample with replacement
                # from the matching (or nearest) donor bucket.
                nearest = min(donor_buckets, key=lambda k: (abs(k - b), k))
                j = rng.choice(donor_buckets[nearest])
            records[i]["output"] = donor_records[j].get("output", "")
            entries[i] = {"type": "mismatch_xsource", "donor": j}
    return entries


# ---------------------------------------------------------------------------
# T2 — noisy response
# ---------------------------------------------------------------------------

def noisy_text(
    text: str,
    rng: random.Random,
    *,
    drop_p: float = 0.15,
    swap_p: float = 0.10,
    inject_pool: Optional[Sequence[str]] = None,
    inject_p: float = 0.30,
) -> str:
    """Word-level noise: dropout, adjacent swaps, and (optionally) one
    foreign sentence spliced in from another response."""
    words = _words(text)
    if len(words) >= 4:
        kept = [w for w in words if rng.random() >= drop_p]
        if len(kept) >= 2:
            words = kept
        k = 0
        while k < len(words) - 1:
            if rng.random() < swap_p:
                words[k], words[k + 1] = words[k + 1], words[k]
                k += 2
            else:
                k += 1
    out = " ".join(words)
    if inject_pool and rng.random() < inject_p:
        foreign = rng.choice(list(inject_pool))
        sentences = re.split(r"(?<=[.!?])\s+", foreign)
        fragment = rng.choice([s for s in sentences if s.strip()] or [foreign])
        pos = rng.randint(0, 1)
        out = f"{fragment} {out}" if pos == 0 else f"{out} {fragment}"
    return out


def eda_noisy_text(text: str, rng: random.Random, *, alpha: float = 0.1) -> str:
    """EDA random_deletion + random_swap, per Wei & Zou (EMNLP-IJCNLP 2019).

    This exists next to :func:`noisy_text` rather than replacing it because
    the two answer different critiques. ``noisy_text`` is the composite20
    operator and must stay byte-reproducible; it also splices in a sentence
    from ANOTHER response 30% of the time, which has no literature anchor
    and — worse — blurs the type boundary with ``mismatch``, so a per-type
    detection table cannot say which class the gate actually caught. This
    function is the literature-faithful replacement used by the
    ``grounded*`` presets: two published augmentation operators, applied as
    corruption, nothing spliced in from anywhere.

    Semantics follow the NoiseBench operator spec exactly:
      * deletion: each whitespace token dropped independently w.p. ``alpha``,
        always keeping at least one token; if the draw deletes nothing, one
        token is deleted so every selected record changes;
      * swap: ``max(1, round(alpha * n_tokens))`` random PAIR swaps (any two
        positions, not adjacent-only).

    Texts with fewer than two tokens are returned unchanged — the caller
    should treat them as ineligible.
    """
    words = _words(text)
    if len(words) < 2:
        return text
    kept = [w for w in words if rng.random() >= alpha]
    if len(kept) == len(words):
        drop = rng.randrange(len(words))
        kept = [w for k, w in enumerate(words) if k != drop]
    if not kept:
        kept = [rng.choice(words)]
    words = kept
    if len(words) >= 2:
        for _ in range(max(1, round(alpha * len(words)))):
            a, b = rng.sample(range(len(words)), 2)
            words[a], words[b] = words[b], words[a]
    return " ".join(words)


def corrupt_noisy(
    records: List[Record],
    target_idxs: List[int],
    rng: random.Random,
    **noise_kwargs: Any,
) -> Dict[int, Dict[str, Any]]:
    mode = noise_kwargs.pop("mode", "legacy")
    if mode == "eda":
        alpha = float(noise_kwargs.pop("alpha", 0.1))
        if noise_kwargs:
            raise ValueError(
                f"corrupt_noisy(mode='eda') takes only alpha; got "
                f"{sorted(noise_kwargs)}"
            )
        entries_eda: Dict[int, Dict[str, Any]] = {}
        for i in target_idxs:
            records[i]["output"] = eda_noisy_text(
                str(records[i].get("output", "")), rng, alpha=alpha,
            )
            entries_eda[i] = {"type": "noisy", "mode": "eda"}
        return entries_eda
    if mode != "legacy":
        raise ValueError(f"corrupt_noisy: unknown mode {mode!r}")

    inject_pool = noise_kwargs.pop("inject_pool", None)
    if inject_pool is None:
        # Sample foreign sentences from non-target responses.
        target_set = set(target_idxs)
        donors = [
            str(r.get("output", "")) for k, r in enumerate(records)
            if k not in target_set and str(r.get("output", "")).strip()
        ]
        inject_pool = donors[:2000] if donors else None
    entries: Dict[int, Dict[str, Any]] = {}
    for i in target_idxs:
        records[i]["output"] = noisy_text(
            str(records[i].get("output", "")), rng,
            inject_pool=inject_pool, **noise_kwargs,
        )
        entries[i] = {"type": "noisy"}
    return entries


# ---------------------------------------------------------------------------
# T3 — truncated response
# ---------------------------------------------------------------------------

def truncate_text(
    text: str,
    rng: random.Random,
    lo: float = 0.30,
    hi: float = 0.70,
) -> str:
    """Cut at a uniform point in [lo, hi] of the word count and strip any
    trailing sentence-final punctuation so the cut reads as incomplete."""
    words = _words(text)
    if len(words) < 3:
        return text
    frac = rng.uniform(lo, hi)
    cut = max(1, int(len(words) * frac))
    out = " ".join(words[:cut])
    return out.rstrip(".!?\"')]} ").rstrip()


def corrupt_truncated(
    records: List[Record],
    target_idxs: List[int],
    rng: random.Random,
    lo: float = 0.30,
    hi: float = 0.70,
) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for i in target_idxs:
        records[i]["output"] = truncate_text(
            str(records[i].get("output", "")), rng, lo, hi,
        )
        entries[i] = {"type": "truncated"}
    return entries


# ---------------------------------------------------------------------------
# T5 — wrong answer (numeric-verifiable subset)
# ---------------------------------------------------------------------------

def has_numeric_answer(rec: Record) -> bool:
    return bool(_NUM_RE.search(str(rec.get("output", ""))))


def perturb_final_number(text: str, rng: random.Random) -> Optional[str]:
    """Replace the LAST number in ``text`` with a perturbed value.
    Returns None when the text contains no number."""
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    raw = m.group(0)
    try:
        if "." in raw:
            val = float(raw)
            delta = max(1.0, abs(val) * rng.uniform(0.1, 0.5))
            new_val = val + rng.choice([-1.0, 1.0]) * delta
            new_raw = f"{new_val:.2f}"
        else:
            val = int(raw)
            delta = max(1, int(abs(val) * rng.uniform(0.1, 0.5)))
            new_val = val + rng.choice([-1, 1]) * delta
            if new_val == val:
                new_val = val + 1
            new_raw = str(new_val)
    except ValueError:
        return None
    if new_raw == raw:
        new_raw = str(raw) + "0"
    return text[: m.start()] + new_raw + text[m.end():]


def corrupt_wrong_answer(
    records: List[Record],
    target_idxs: List[int],
    rng: random.Random,
) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for i in target_idxs:
        perturbed = perturb_final_number(str(records[i].get("output", "")), rng)
        if perturbed is None:
            continue  # caller pre-filters via has_numeric_answer; belt & braces
        records[i]["output"] = perturbed
        entries[i] = {"type": "wrong_answer"}
    return entries


# ---------------------------------------------------------------------------
# T7 — fluent-but-wrong response (pre-generated off-line)
# ---------------------------------------------------------------------------

def corrupt_fluent_wrong(
    records: List[Record],
    target_idxs: List[int],
    replacements: Dict[Any, str],
) -> Dict[int, Dict[str, Any]]:
    """Apply pre-generated fluent-but-wrong replacement responses (T7).

    ``replacements`` maps pool index -> replacement text; int and string
    keys are both accepted (JSON object keys arrive as strings). The texts
    are produced off-line by ``scripts/gen_fluent_wrong.py`` so this module
    stays model-free and deterministic. A target index without a
    replacement is a hard error — a silent skip would corrupt the manifest
    ground truth.
    """
    lookup = {int(k): str(v) for k, v in (replacements or {}).items()}
    missing = [i for i in target_idxs if i not in lookup]
    if missing:
        raise KeyError(
            f"corrupt_fluent_wrong: no replacement for indices "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''} "
            f"({len(missing)} of {len(target_idxs)} targets)"
        )
    entries: Dict[int, Dict[str, Any]] = {}
    for i in target_idxs:
        records[i]["output"] = lookup[i]
        entries[i] = {"type": "fluent_wrong"}
    return entries


# ---------------------------------------------------------------------------
# T4 — duplicate instructions (appended records)
# ---------------------------------------------------------------------------

def _jitter_text(text: str, rng: random.Random) -> str:
    """Surface-level jitter that preserves meaning: leading/trailing
    whitespace, sentence-case flip of the first character, double spaces."""
    out = text
    choice = rng.randint(0, 3)
    if choice == 0:
        out = " " + out
    elif choice == 1:
        out = out + " "
    elif choice == 2 and out:
        out = (out[0].lower() if out[0].isupper() else out[0].upper()) + out[1:]
    else:
        out = out.replace(" ", "  ", 1)
    return out


def append_duplicates(
    records: List[Record],
    rng: random.Random,
    frac: float = 0.05,
    copies_lo: int = 2,
    copies_hi: int = 4,
    jitter_p: float = 0.5,
) -> Tuple[Dict[int, Dict[str, Any]], List[List[int]]]:
    """Append near-duplicate copies of a random ``frac`` of the ORIGINAL
    records. Returns (manifest_entries_for_new_records, clusters) where each
    cluster lists [original_idx, dup_idx, ...]."""
    n_original = len(records)
    n_seed = max(1, int(n_original * frac)) if frac > 0 else 0
    if n_seed == 0:
        return {}, []
    seeds = rng.sample(range(n_original), n_seed)
    entries: Dict[int, Dict[str, Any]] = {}
    clusters: List[List[int]] = []
    for s in seeds:
        cluster = [s]
        n_copies = rng.randint(copies_lo, copies_hi)
        for _ in range(n_copies):
            dup = copy.deepcopy(records[s])
            if rng.random() < jitter_p:
                dup["instruction"] = _jitter_text(str(dup.get("instruction", "")), rng)
            new_idx = len(records)
            records.append(dup)
            entries[new_idx] = {"type": "duplicate", "source_index": s}
            cluster.append(new_idx)
        clusters.append(cluster)
    return entries, clusters


# ---------------------------------------------------------------------------
# Counterfactual pool (reliability view)
# ---------------------------------------------------------------------------

def make_counterfactual(
    records: Sequence[Record],
    seed: int = 42,
    n_buckets: int = 10,
) -> List[Record]:
    """Index-aligned counterfactual pool: record i keeps its own response
    y_i but receives the instruction (and input) of a semantically unrelated
    partner from the same response-length bucket.

    Used by the reliability view: Q_i = rank[ L(y_i|x_i^-) - L(y_i|x_i) ].
    The bucket restriction keeps prompt statistics comparable so that Q is
    not a length detector in disguise.
    """
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    if len(idxs) < 2:
        return [copy.deepcopy(r) for r in records]
    mapping = derange_within_buckets(records, idxs, rng, n_buckets)
    out: List[Record] = []
    for i, rec in enumerate(records):
        j = mapping.get(i, (i + 1) % len(records))
        cf = copy.deepcopy(rec)
        cf["instruction"] = records[j].get("instruction", "")
        cf["input"] = records[j].get("input", "")
        out.append(cf)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def corrupt_pool(
    records: Sequence[Record],
    *,
    seed: int = 42,
    mismatch: float = 0.0,
    noisy: float = 0.0,
    truncated: float = 0.0,
    wrong_answer: float = 0.0,
    duplicate_frac: float = 0.0,
    duplicate_copies: Tuple[int, int] = (2, 4),
    n_buckets: int = 10,
    sources: Optional[Sequence[str]] = None,
    xsource_frac: float = 0.0,
    donor_records: Optional[Sequence[Record]] = None,
    fluent_wrong_frac: float = 0.0,
    fluent_wrong_replacements: Optional[Dict[Any, str]] = None,
    noisy_mode: str = "legacy",
    noisy_alpha: float = 0.1,
) -> Tuple[List[Record], Dict[str, Any]]:
    """Apply the requested corruption mix and return (new_records, manifest).

    Fractions are relative to the ORIGINAL pool size and the in-place
    types are applied to DISJOINT index sets (a sample carries at most one
    corruption type, so per-type metrics stay unambiguous). ``wrong_answer``
    targets are drawn from the numeric-verifiable subset only.

    T1b (``xsource_frac`` > 0) requires ``donor_records`` — a different
    source dataset whose responses replace the targets' (length-bucket
    matched, see :func:`corrupt_cross_source`).

    T7 (``fluent_wrong_frac`` > 0) draws its targets deterministically; with
    ``fluent_wrong_replacements`` provided the pre-generated texts are
    applied (missing index -> hard error), without them nothing is modified
    and the drawn indices are reported under ``manifest["fluent_wrong_targets"]``
    (the emit step of the two-step workflow in
    ``scripts/make_corrupted_pool.py``). Rerunning with identical arguments
    plus the replacements draws the SAME indices, since applying T7 consumes
    no randomness.

    Back-compat: with the T1b/T7 kwargs at their defaults the output —
    records AND manifest — is byte-identical to the pre-T1b/T7 code (the
    new spec keys are only emitted when the fractions are nonzero).

    The manifest schema:
        {
          "seed": int, "n_original": int, "n_total": int,
          "spec": {...fractions...},
          "entries": {str(idx): {"type": ..., ...}},   # corrupted only
          "duplicate_clusters": [[orig, dup, ...], ...],
          "sources": [str, ...] | None,                # per-record tag
          "fluent_wrong_targets": [idx, ...],          # emit mode only
        }
    """
    records = [copy.deepcopy(r) for r in records]
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        raise ValueError("corrupt_pool: empty record list")

    fracs = {
        "mismatch": mismatch,
        "noisy": noisy,
        "truncated": truncated,
        "wrong_answer": wrong_answer,
    }
    for k, v in {**fracs, "xsource_frac": xsource_frac,
                 "fluent_wrong_frac": fluent_wrong_frac}.items():
        if v < 0 or v > 1:
            raise ValueError(f"corrupt_pool: fraction {k}={v} outside [0,1]")
    total_in_place = sum(fracs.values()) + xsource_frac + fluent_wrong_frac
    if total_in_place > 1.0 + 1e-9:
        raise ValueError(
            f"corrupt_pool: in-place fractions sum to {total_in_place:.3f} > 1"
        )
    if xsource_frac > 0 and not donor_records:
        raise ValueError("corrupt_pool: xsource_frac > 0 requires donor_records")

    # Wrong-answer candidates first (constrained subset), then the rest from
    # the remaining pool. All four sets are disjoint.
    entries: Dict[int, Dict[str, Any]] = {}
    available = set(range(n))

    n_wrong = int(round(n * fracs["wrong_answer"]))
    if n_wrong > 0:
        numeric = [i for i in available if has_numeric_answer(records[i])]
        rng.shuffle(numeric)
        chosen = numeric[:n_wrong]
        if len(chosen) < n_wrong:
            # Not enough numeric-bearing samples; corrupt what exists.
            pass
        entries.update(corrupt_wrong_answer(records, sorted(chosen), rng))
        available -= set(chosen)

    def _draw(frac: float) -> List[int]:
        k = int(round(n * frac))
        k = min(k, len(available))
        chosen = rng.sample(sorted(available), k) if k > 0 else []
        available.difference_update(chosen)
        return sorted(chosen)

    t1 = _draw(fracs["mismatch"])
    if t1:
        entries.update(corrupt_mismatch(records, t1, rng, n_buckets))
    t2 = _draw(fracs["noisy"])
    if t2:
        if noisy_mode == "legacy":
            # No kwargs: the composite20 rng stream and output stay
            # byte-identical to every pool already built with it.
            entries.update(corrupt_noisy(records, t2, rng))
        else:
            entries.update(corrupt_noisy(records, t2, rng,
                                         mode=noisy_mode, alpha=noisy_alpha))
    t3 = _draw(fracs["truncated"])
    if t3:
        entries.update(corrupt_truncated(records, t3, rng))
    # T1b/T7 draws happen last so that _draw(0.0) consumes no randomness and
    # the pre-T1b/T7 rng stream is preserved exactly when they are unused.
    t1b = _draw(xsource_frac)
    if t1b:
        entries.update(
            corrupt_cross_source(records, t1b, donor_records, rng, n_buckets)
        )
    t7 = _draw(fluent_wrong_frac)
    if t7 and fluent_wrong_replacements is not None:
        entries.update(corrupt_fluent_wrong(records, t7, fluent_wrong_replacements))

    dup_entries, clusters = append_duplicates(
        records, rng, frac=duplicate_frac,
        copies_lo=duplicate_copies[0], copies_hi=duplicate_copies[1],
    )
    entries.update(dup_entries)

    source_list: Optional[List[str]] = None
    if sources is not None:
        if len(sources) != n:
            raise ValueError(
                f"corrupt_pool: sources length {len(sources)} != n_original {n}"
            )
        source_list = list(sources)
        # Appended duplicates inherit their seed record's source tag.
        for new_idx in sorted(dup_entries):
            source_list.append(source_list[dup_entries[new_idx]["source_index"]])

    spec: Dict[str, Any] = {
        **fracs,
        "duplicate_frac": duplicate_frac,
        "duplicate_copies": list(duplicate_copies),
        "n_buckets": n_buckets,
    }
    # Only emitted when used, so pre-T1b/T7 manifests stay byte-identical.
    if noisy_mode != "legacy":
        spec["noisy_mode"] = noisy_mode
        spec["noisy_alpha"] = noisy_alpha
    if xsource_frac > 0:
        spec["xsource_frac"] = xsource_frac
    if fluent_wrong_frac > 0:
        spec["fluent_wrong_frac"] = fluent_wrong_frac

    manifest: Dict[str, Any] = {
        "seed": seed,
        "n_original": n,
        "n_total": len(records),
        "spec": spec,
        "entries": {str(i): e for i, e in sorted(entries.items())},
        "duplicate_clusters": clusters,
        "sources": source_list,
    }
    if fluent_wrong_frac > 0 and fluent_wrong_replacements is None:
        manifest["fluent_wrong_targets"] = t7
    return records, manifest


def dirty_labels_from_manifest(manifest: Dict[str, Any]) -> List[int]:
    """Return a 0/1 dirty label per record (length n_total) from a manifest."""
    n_total = int(manifest["n_total"])
    labels = [0] * n_total
    for k in manifest.get("entries", {}):
        labels[int(k)] = 1
    return labels
