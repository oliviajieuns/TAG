#!/usr/bin/env python
"""Does the gate separate corrupted data from clean? — CPU, seconds, no forward.

    python scripts/gate_report.py \\
        --gate $POOLS/composite20/tag_gate_qwen2.5-7b.pt \\
        --manifest $POOLS/composite20/corruption_manifest.json

Everything needed is already on disk after scripts/precompute_gate.sh: the
cache holds G for every pool record, and the manifest holds the ground-truth
corruption labels. So the first and most decisive question — *is G predictive
of dirtiness at all* — costs nothing and should be answered BEFORE spending
GPU hours on training arms.

This answers the gate in isolation. It does NOT answer "does G . R beat R",
which needs the dynamic reward R and therefore a forward pass
(scripts/score_pool.py). But if G shows no separation here, that comparison
cannot come out well either, and the cheap check has saved the expensive one.

What to read:

  AP(dirty)     average precision of -G as a dirty detector, against the
                base rate (= the AP a random detector gets). Above the base
                rate means G carries signal; at the base rate it carries none.
  floor block   the G == 0 samples. This is the set the gate refuses outright,
                so its purity is the sharpest statement the gate makes: a
                floor block that is mostly dirty is the gate working.
  dirty@K       corrupted fraction of the top-K by G alone, versus the pool's
                base rate. Selection in training ranks by G . R, not G, so
                this is an upper-bound-ish sanity check on the gate's own
                contribution rather than a prediction of the arm's result.
  by type       mean G per corruption type. The gate is built to catch
                instruction-response mismatch; it is NOT built to catch every
                corruption (a fluent wrong answer can still be well explained
                by its instruction). Types it misses are a finding to report,
                not necessarily a bug.
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


def average_precision(detector, labels) -> float:
    """AP for binary ``labels`` (1 = dirty) ranked by ``detector`` desc."""
    import torch

    order = torch.argsort(detector, descending=True)
    y = labels[order].float()
    n_pos = float(y.sum().item())
    if n_pos == 0:
        return float("nan")
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float(((precision * y).sum() / n_pos).item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gate", required=True, help="tag_gate_*.pt from precompute_gate.sh")
    p.add_argument("--config", default=None,
                   help="arm YAML. Re-derives G from the cache's token losses "
                        "under THAT config before reporting, which is what the "
                        "arm itself does when the cached config differs. "
                        "Without it you are reading whatever config the cache "
                        "was stamped with, which may not be the one training "
                        "uses. CPU only.")
    p.add_argument("--manifest", default=None,
                   help="corruption_manifest.json (default: beside the pool "
                        "recorded in the cache identity)")
    p.add_argument("--ks", default="0.1",
                   help="comma-separated selection ratios for dirty@K")
    p.add_argument("--clean-ref", default=None,
                   help="clean-reference .pt — enables the distribution-shift "
                        "check, which compares the pool's CLEAN records "
                        "against the pool s and mu(M) were calibrated on")
    args = p.parse_args()

    import torch

    # An unset environment variable arrives here as an empty string, and
    # `FileNotFoundError: ''` says nothing about which variable was empty or
    # why. Every gate cache path in the runbook comes from one.
    if not str(args.gate).strip():
        sys.exit(
            "--gate is empty. It was probably an unset environment variable "
            "(TAG_GATE_CACHE_Q7, _PREFIX, _LLAMA2, ...). A new container "
            "starts with none of them set:\n"
            "  git pull origin main\n"
            "  TAG_ENV_RESET=1 source scripts/gpu_cloud/env.sh\n"
            "then check it with:  echo \"$TAG_GATE_CACHE_Q7\""
        )
    if not Path(args.gate).is_file():
        sys.exit(
            f"no gate cache at {args.gate}\n"
            f"Build it with:  bash scripts/precompute_gate.sh <arm config> "
            f"{args.gate}"
        )
    cache = torch.load(args.gate, map_location="cpu", weights_only=False)

    if args.config:
        # The cache and the arm can disagree — a gate cache stamped at
        # prefix_tokens=0 while the arm gates on 32 reports a distribution
        # for a statistic nothing trains on. Re-derive the way the arm does.
        from tag.core import gate as _g
        from tag.core.utils import load_config as _lc
        from tag.pipelines.selection import (
            _build_gate_config, _resolve_gate_calibration,
        )
        params = (_lc(args.config).get("selection") or {}).get("tag") or {}
        _scale, _null = _resolve_gate_calibration(params)
        gcfg = _build_gate_config(params, _scale, _null)
        cached_cfg = cache.get("config") or {}
        now = gcfg.identity()
        diff = {k: (cached_cfg.get(k), now[k]) for k in now
                if k in cached_cfg and cached_cfg[k] != now[k]}
        print(f"(re-deriving under {args.config}; differs from the cache on "
              f"{diff or 'nothing'})")
        redone = _g.recompute_gate_from_cache(cache, gcfg)
        if redone is None:
            sys.exit(
                "cannot re-derive from this cache (no token losses, or the "
                "change is forward-bound). Rebuild it:\n"
                f"  bash scripts/precompute_gate.sh {args.config} {args.gate}"
            )
        cache = {**cache, **redone, "config": now}
    G = cache["gate"].float().view(-1)
    n = G.numel()

    manifest_path = args.manifest
    if not manifest_path:
        ident = cache.get("identity") or {}
        pool_files = str(ident.get("pool_files") or "")
        if pool_files:
            guess = Path(pool_files.split(",")[0]).with_name("corruption_manifest.json")
            if guess.exists():
                manifest_path = str(guess)
                print(f"(manifest from the cache's pool identity: {guess})")
    entries: Dict[str, Any] = {}
    if manifest_path:
        with open(manifest_path) as f:
            entries = json.load(f).get("entries") or {}
    else:
        print("(no manifest — reporting the G distribution only; the "
              "separation analysis below needs corruption labels)")

    types: List[str] = ["clean"] * n
    for k, e in entries.items():
        i = int(k)
        if 0 <= i < n:
            types[i] = str(e.get("type", "unknown"))
    dirty = torch.tensor([t != "clean" for t in types], dtype=torch.bool)
    base = float(dirty.float().mean().item())

    print(f"gate     : {args.gate}")
    print(f"manifest : {manifest_path}")
    print(f"n        : {n}   dirty {100*base:.1f}%  ({int(dirty.sum())})")
    cfg = cache.get("config") or {}
    print(f"config   : W={cfg.get('span_tokens')} tau={cfg.get('tau')} "
          f"s={cfg.get('scale')} null={'on' if cfg.get('null_correction') else 'OFF'}")
    print()

    # ---- weight distribution -------------------------------------------
    at_floor = G == 0
    print("G distribution")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  q{q:<5.2f} = {float(torch.quantile(G, q)):.4f}")
    print(f"  G == 0        : {100*float(at_floor.float().mean()):5.1f}%  "
          f"({int(at_floor.sum())})")
    print(f"  0 < G < 0.99  : {100*float(((G>0)&(G<0.99)).float().mean()):5.1f}%   "
          f"<- graded, not a mask")
    print(f"  G >= 0.99     : {100*float((G>=0.99).float().mean()):5.1f}%")
    print()

    if base <= 0:
        # A CLEAN pool — Table 2's is one. Every section below divides by the
        # dirty rate or ranks against it, so they are not merely uninformative
        # here, they are undefined. What the distribution above already says
        # is the whole answer for a clean pool: how much of it the gate
        # floors, and how much it grades.
        print("This pool carries no corruption labels (dirty 0.0%), so there "
              "is nothing to separate.")
        print("Read the distribution above instead:")
        print(f"  - G == 0 should land near the configured target_zero_rate; "
              f"here {100*float((G <= 0).float().mean()):.1f}%.")
        print("  - a large graded band (0 < G < 0.99) means the gate is "
              "reweighting rather than acting as a mask.")
        print("  - if almost everything sat at G >= 0.99 the gate would be "
              "doing nothing, whatever a corrupted pool showed.")
        return

    # ---- the headline: does G know what is dirty? -----------------------
    ap = average_precision(-G, dirty)
    print("Separation")
    print(f"  AP(dirty | -G)      : {ap:.4f}   (base rate {base:.4f}"
          f" = a detector with no signal)")
    lift = ap / base if base > 0 else float("nan")
    print(f"  lift over base      : {lift:.2f}x")
    print(f"  mean G  clean       : {float(G[~dirty].mean()):.4f}")
    print(f"  mean G  dirty       : {float(G[dirty].mean()):.4f}")
    print()

    # The floor block is the gate's sharpest claim: these are refused outright.
    if int(at_floor.sum()) > 0:
        pur = float(dirty[at_floor].float().mean())
        print(f"  floor block (G==0)  : n={int(at_floor.sum())}, "
              f"{100*pur:.1f}% dirty  (pool base {100*base:.1f}%)")
        if base > 0:
            print(f"                        -> {pur/base:.2f}x enriched in corruption")
            rec = float(dirty[at_floor].float().sum() / dirty.float().sum())
            print(f"                        catches {100*rec:.1f}% of all dirty samples")
    else:
        print("  floor block (G==0)  : EMPTY — the gate refuses nothing.")
    print()

    # ---- which statistic is actually carrying the gate? -----------------
    # Delta_hat is what Eq. 6 consumes, but it is min(Delta_bar, Delta^min),
    # so if the tail term is noise the minimum ACTIVELY DESTROYS whatever
    # Delta_bar had. Comparing the three as whole-pool dirty detectors says
    # whether that is happening, and it is the number that decides whether
    # tail_mode should be "min" or "none".
    _db = cache.get("delta_bar")
    _dm = cache.get("delta_min")
    _dh = cache.get("delta_hat")
    if _db is not None and _dm is not None and _dh is not None:
        print("Which statistic carries the signal (whole pool, AP of -stat)")
        rows = [
            ("Delta_bar  (Eq. 3)", _db), ("Delta^min  (Eq. 5)", _dm),
            ("Delta_hat  (Eq. 6 input)", _dh),
        ]
        best_name, best_ap = None, -1.0
        for name, v in rows:
            a = average_precision(-v.float().view(-1), dirty)
            if a > best_ap:
                best_name, best_ap = name, a
            print(f"  {name:<26} AP={a:.4f}   lift {a/base:.2f}x")
        print(f"  base rate                  AP={base:.4f}   (no signal)")
        print(f"  strongest: {best_name}")
        ap_db = average_precision(-_db.float().view(-1), dirty)
        ap_dh = average_precision(-_dh.float().view(-1), dirty)
        if ap_db > ap_dh + 0.02:
            print()
            print(f"  ** Delta_bar ALONE beats Delta_hat ({ap_db:.4f} vs "
                  f"{ap_dh:.4f}). Delta_hat is a MINIMUM, so the tail term is")
            print("     not merely useless here — it is destroying signal.")
            print("     Set selection.tag.tail_mode: none to gate on Eq. 3 alone.")
        # Caveat worth stating: delta_hat in the cache is already null-centred
        # per span count while the other two are raw, and AP is not invariant
        # to a per-bin shift. The comparison is still informative when the gap
        # is this large, but it is not exact.
        print()
        print("  (note: delta_hat is null-centred per span count, the other two")
        print("   are raw. AP is not invariant to a per-bin shift, so read this")
        print("   as a large-effect comparison, not a precise one.)")
        print()

    # ---- what a G-only selection would pick -----------------------------
    print("Selecting by G alone (training ranks by G . R, so this is the gate's"
          " own contribution, not the arm's result)")
    for ks in args.ks.split(","):
        k = float(ks.strip())
        kk = max(1, int(round(k * n)))
        top = torch.topk(G, kk).indices
        d = float(dirty[top].float().mean())
        print(f"  dirty@{k:<5.2f} = {100*d:5.1f}%   (base {100*base:.1f}%, "
              f"{'-' if d < base else '+'}{abs(100*(d-base)):.1f}pp)")
    print()

    # ---- per corruption type -------------------------------------------
    # The gate targets instruction-response mismatch. Types it does not move
    # are a reportable limitation, not automatically a defect.
    by_type: Dict[str, List[int]] = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    comp = cache.get("completeness")
    have_comp = comp is not None
    if have_comp:
        comp = comp.float().view(-1)
        # G = c * sigma_part, so sigma_part isolates what the COUNTERFACTUAL
        # CONTRAST (Eqs. 2-6) contributes, with the completeness heuristic's
        # contribution divided out. A type whose mean G is low but whose
        # sigma_part is ~clean is being caught by the string heuristic, not by
        # the likelihood contrast the method is about.
        sigma_part = torch.where(comp > 0, G / comp.clamp(min=1e-6), G)
    hdr = f"{'type':<16} {'n':>7} {'mean G':>8} {'G==0':>8}"
    if have_comp:
        hdr += f" {'mean c':>8} {'G/c':>8}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    for t in sorted(by_type, key=lambda x: (x != "clean", x)):
        idx = torch.tensor(by_type[t], dtype=torch.long)
        g = G[idx]
        row = (f"{t:<16} {len(idx):>7} {float(g.mean()):>8.4f} "
               f"{100*float((g==0).float().mean()):>7.1f}%")
        if have_comp:
            row += f" {float(comp[idx].mean()):>8.4f} {float(sigma_part[idx].mean()):>8.4f}"
        print(row)
    if have_comp:
        print("  mean c  = completeness (the c_trunc string heuristic)")
        print("  G/c     = the counterfactual contrast alone. Compare it to")
        print("            clean's: a type that only separates in 'mean G' but")
        print("            not in 'G/c' is caught by the heuristic, not Eqs. 2-6.")

    d_bar = cache.get("delta_bar")
    d_min = cache.get("delta_min")
    d_bar = d_bar.float().view(-1) if d_bar is not None else None
    d_min = d_min.float().view(-1) if d_min is not None else None

    # ---- is the calibration even describing this pool? ------------------
    # s and mu(M) are quantiles of the CLEAN REFERENCE's Delta_hat. They only
    # mean anything on the candidate pool if the two pools' clean records are
    # drawn from the same distribution. If they are not, every gate value is
    # mis-scaled while every rank-based diagnostic above still looks fine —
    # which is exactly the kind of error that survives review. preflight
    # cannot check this when the manifests predate corpus recording, so it is
    # checked here, against the statistic itself rather than against a
    # provenance string.
    if args.clean_ref and d_bar is not None:
        ref = torch.load(args.clean_ref, map_location="cpu", weights_only=False)
        rb = ref.get("delta_bar")
        if rb is None:
            print("(--clean-ref has no delta_bar; skipping the shift check)\n")
        else:
            rb = rb.float().view(-1)
            pb = d_bar[~dirty]  # the pool's CLEAN records only
            print("Calibration validity — pool's CLEAN records vs the reference")
            print(f"{'':<12} {'n':>7} {'mean':>9} {'P10':>9} {'P50':>9} {'P90':>9}")
            for nm, v in (("reference", rb), ("pool clean", pb)):
                print(f"{nm:<12} {v.numel():>7} {float(v.mean()):>9.4f} "
                      f"{float(torch.quantile(v, 0.10)):>9.4f} "
                      f"{float(torch.quantile(v, 0.50)):>9.4f} "
                      f"{float(torch.quantile(v, 0.90)):>9.4f}")
            # MEDIAN and IQR, not mean and SD: delta_bar has a heavy left
            # tail (the eps_den clamp sends 1 - L+/L- far negative whenever
            # the counterfactual sum is tiny), which drags the mean below
            # even the 10th percentile and inflates the SD enough to hide
            # any real shift. The quantiles are what describe the bulk.
            med_r = float(torch.quantile(rb, 0.50))
            med_p = float(torch.quantile(pb, 0.50))
            iqr_r = float(torch.quantile(rb, 0.75) - torch.quantile(rb, 0.25))
            shift = med_p - med_r
            print(f"  median shift = {shift:+.4f}  "
                  f"({abs(shift)/max(iqr_r,1e-9):.2f} reference IQRs)")
            if abs(shift) > 0.25 * iqr_r:
                print()
                print("  ** The two pools' CLEAN records do not match. s and mu(M)")
                print("     were fit on the reference, so on this pool the gate is")
                print("     calibrated to the wrong distribution: the intended")
                print("     target_zero_rate does not hold and G's absolute values")
                print("     are not interpretable. Rank-based numbers above are")
                print("     unaffected; the gate the ARMS use is.")
                print("     Check both pools came from the same corpus and the same")
                print("     counterfactual construction:")
                print(f"       reference pool: {ref.get('pool')}")
                print(f"       reference cf  : {ref.get('counterfactual')}")
                ident = cache.get("identity") or {}
                print(f"       gate pool     : {ident.get('pool_files')}")
            print()

    # ---- span ablation: does Delta^min earn its place over Delta_bar? ----
    # This IS the paper's justification for Eqs. 4-5. The stated argument is
    # that a localized corruption (a wrong final answer in an otherwise
    # correct response) leaves Delta_bar healthy while Delta^min collapses.
    # That is a testable claim and the cache already holds both statistics,
    # so it gets measured rather than assumed. Per corruption type, AP of
    # each statistic as a detector of THAT type against clean only.
    if d_bar is None or d_min is None:
        return
    clean_idx = by_type.get("clean") or []
    if not clean_idx:
        return
    print()
    print("Span ablation — AP for detecting each type against CLEAN only")
    print("(the paper's reason for Eqs. 4-5: Delta^min should beat Delta_bar")
    print(" on LOCALIZED corruption. Where it does not, spans are not earning")
    print(" their place on that type.)")
    print(f"{'type':<16} {'base':>7} {'AP dbar':>9} {'AP dmin':>9} {'AP dhat':>9}  verdict")
    print("-" * 66)
    d_hat = cache.get("delta_hat")
    d_hat = d_hat.float().view(-1) if d_hat is not None else None
    for t in sorted(by_type):
        if t == "clean":
            continue
        idx = by_type[t]
        sel = torch.tensor(list(idx) + list(clean_idx), dtype=torch.long)
        lab = torch.zeros(sel.numel(), dtype=torch.bool)
        lab[: len(idx)] = True
        b = float(lab.float().mean())
        ap_bar = average_precision(-d_bar[sel], lab)
        ap_min = average_precision(-d_min[sel], lab)
        ap_hat = average_precision(-d_hat[sel], lab) if d_hat is not None else float("nan")
        if ap_min > ap_bar + 0.01:
            verdict = "spans help"
        elif ap_min < ap_bar - 0.01:
            verdict = "spans HURT"
        else:
            verdict = "no difference"
        print(f"{t:<16} {b:>7.3f} {ap_bar:>9.4f} {ap_min:>9.4f} {ap_hat:>9.4f}  {verdict}")


if __name__ == "__main__":
    main()
