#!/usr/bin/env python
"""Measure the completeness heuristic against a pool manifest — no GPU, seconds.

    python scripts/audit_completeness.py \\
        --pool  $POOLS/composite20/pool.json \\
        --manifest $POOLS/composite20/manifest.json

``c_i`` multiplies every candidate's score by ``c_trunc`` (0.2) when
``tads.data.sft_prompts.text_is_complete`` says the response reads as cut off.
That is a five-fold demotion decided by a string heuristic, so its error rate
is not a detail: a false positive rate of ten percent on a clean pool demotes
more good data than the T3 corruption it is there to catch.

The manifest records which records were actually corrupted and how, so the
heuristic can be scored properly rather than eyeballed:

    precision  of the flagged set, how much really was T3-truncated
    recall     of the T3 set, how much the heuristic caught
    FP rate    of the UNCORRUPTED set, how much clean data gets demoted

Read it as a trade, not a score to maximise. Recall protects the gate's
purpose; the FP rate is the collateral. ``--ablate`` reruns the audit with
individual rules disabled so the contribution of each is visible.

Corruptions other than T3 (mismatch, noisy, wrong-answer, fluent-wrong) are
NOT truncation and are reported separately: flagging them is neither a hit nor
a false positive, it is a different view's job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _rate(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f}%  ({num}/{den})" if den else "    n/a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pool", required=True, help="pool.json (list of records)")
    p.add_argument("--manifest", default=None,
                   help="corruption_manifest.json. Defaults to the one beside "
                        "--pool; without either, only the overall flag rate "
                        "can be reported.")
    p.add_argument("--response-key", default="output")
    p.add_argument("--ablate", action="store_true",
                   help="also report the heuristic with each added rule off")
    p.add_argument("--show", type=int, default=0,
                   help="print this many example false positives")
    args = p.parse_args()

    from tads.data import sft_prompts as sp

    records: List[Dict[str, Any]] = _load_json(args.pool)
    if not isinstance(records, list):
        sys.exit(f"{args.pool} is not a list of records")
    texts = [str(r.get(args.response_key, "")) for r in records]
    n = len(texts)

    # make_corrupted_pool.py writes corruption_manifest.json beside the pool.
    # Requiring the exact name buys nothing and is easy to get wrong, and
    # getting it wrong silently downgrades this to a bare flag-rate report
    # with no precision or recall — the numbers that decide the question.
    if not args.manifest:
        guess = Path(args.pool).with_name("corruption_manifest.json")
        if guess.exists():
            args.manifest = str(guess)
            print(f"(using the manifest beside the pool: {guess})")

    types: List[str] = ["clean"] * n
    if args.manifest:
        man = _load_json(args.manifest)
        entries = man.get("entries") or {}
        for k, e in entries.items():
            i = int(k)
            if 0 <= i < n:
                types[i] = str(e.get("type", "unknown"))
        print(f"manifest  : {args.manifest}")
    print(f"pool      : {args.pool}   n={n}")
    print()

    variants = [("shipped", {})]
    if args.ablate:
        variants += [
            ("no list rule", {"_LIST_MARKERS": ()}),
            ("no short rule", {"_SHORT_ANSWER_WORDS": 0}),
            ("punctuation only", {"_LIST_MARKERS": (), "_SHORT_ANSWER_WORDS": 0}),
        ]

    first_fps: List[str] = []
    for name, patch in variants:
        saved = {k: getattr(sp, k) for k in patch}
        for k, v in patch.items():
            setattr(sp, k, v)
        try:
            flagged = [not sp.text_is_complete(t) for t in texts]
        finally:
            for k, v in saved.items():
                setattr(sp, k, v)

        n_flag = sum(flagged)
        idx_t3 = [i for i in range(n) if types[i] == "truncated"]
        idx_clean = [i for i in range(n) if types[i] == "clean"]
        idx_other = [i for i in range(n) if types[i] not in ("clean", "truncated")]
        tp = sum(flagged[i] for i in idx_t3)
        fp_clean = sum(flagged[i] for i in idx_clean)
        fp_other = sum(flagged[i] for i in idx_other)

        print(f"--- {name} ---")
        print(f"  flagged incomplete : {_rate(n_flag, n)}")
        if args.manifest:
            print(f"  recall on T3       : {_rate(tp, len(idx_t3))}")
            print(f"  FP rate on CLEAN   : {_rate(fp_clean, len(idx_clean))}")
            print(f"  precision (vs T3)  : {_rate(tp, n_flag)}")
            print(f"  flagged among other corruptions (neither hit nor FP): "
                  f"{_rate(fp_other, len(idx_other))}")
            if len(idx_clean):
                # The number that decides whether the view helps or hurts: how
                # much clean data is demoted for every truly-truncated one
                # caught. Above 1.0 the c_i view costs more than it buys.
                ratio = fp_clean / max(1, tp)
                print(f"  clean demoted per T3 caught: {ratio:.2f}"
                      + ("   <-- costs more than it buys" if ratio > 1.0 else ""))
        print()
        if name == "shipped" and args.show:
            first_fps = [texts[i] for i in idx_clean if flagged[i]][: args.show]

    if first_fps:
        print(f"--- {len(first_fps)} example false positives (clean, flagged) ---")
        for t in first_fps:
            body = t if len(t) <= 300 else t[:297] + "..."
            print("  " + body.replace("\n", "\n  "))
            print("  " + "-" * 60)

    if not args.manifest:
        print("No --manifest: only the flag rate is meaningful. Point it at the "
              "manifest.json that make_corrupted_pool.py wrote beside the pool.")


if __name__ == "__main__":
    main()
