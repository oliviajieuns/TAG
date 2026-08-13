#!/usr/bin/env python
"""How much of a pool does max_seq_len cut off, and what does it cost the gate?

    python scripts/check_truncation.py --config <cfg> [--max-seq-len 1024]

Budget truncation drops the appended EOS, so the completeness check marks a
clean-but-long response incomplete and hands it c_trunc — a 5x gate cut at
the default. It hits exactly the long responses that Delta^min already
penalises, so the two effects stack. Run this before committing to a
max_seq_len; it needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--pool", default=None, help="defaults to the config's data_files")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--sample", type=int, default=4000)
    args = p.parse_args()

    from tads.core.utils import load_config
    from tads.modeling.loader import load_tokenizer

    cfg = load_config(args.config)
    tok = load_tokenizer(cfg["model_path"])
    pool_path = args.pool or cfg["data_files"]
    recs = json.load(open(pool_path))
    step = max(1, len(recs) // args.sample)
    sample = recs[::step]

    lens = []
    for r in sample:
        n_p = len(tok(str(r.get("instruction", "")) + str(r.get("input", "")),
                      add_special_tokens=False)["input_ids"])
        n_r = len(tok(str(r.get("output", "")), add_special_tokens=False)["input_ids"])
        lens.append((n_p, n_r))

    print(f"pool     : {pool_path}")
    print(f"sampled  : {len(sample)} of {len(recs)}")
    resp = sorted(n_r for _, n_r in lens)
    tot = sorted(n_p + n_r + 9 for n_p, n_r in lens)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    print(f"response tokens  p50={pct(resp,.5)}  p90={pct(resp,.9)}  "
          f"p99={pct(resp,.99)}  max={resp[-1]}")
    print(f"total tokens     p50={pct(tot,.5)}  p90={pct(tot,.9)}  "
          f"p99={pct(tot,.99)}  max={tot[-1]}")
    print()
    print(f"{'max_seq_len':>12} {'truncated':>10}   {'gate cost':<50}")
    candidates = [args.max_seq_len] if args.max_seq_len else [512, 768, 1024, 1536, 2048]
    c_trunc = float(((cfg.get("tads") or {}).get("tag") or {}).get("c_trunc", 0.2))
    for m in candidates:
        n = sum(1 for t in tot if t > m)
        rate = n / len(tot)
        note = ""
        if rate > 0.15:
            note = f"~{100*rate:.0f}% of CLEAN samples cut to c_trunc={c_trunc}"
        elif rate > 0.05:
            note = "moderate; report the rate"
        else:
            note = "negligible"
        print(f"{m:>12} {100*rate:>9.1f}%   {note:<50}")
    print()
    print("The truncated set is the LONG responses, which Delta^min already")
    print("pushes down (docs/tag-paper-deltas.md B2) — the two effects stack,")
    print("so a high rate here makes the gate partly a length filter.")


if __name__ == "__main__":
    main()
