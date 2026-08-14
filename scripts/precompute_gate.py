#!/usr/bin/env python
"""Compute the TAG reliability gate once, sharded across GPUs, and share it.

Why this exists as a separate step
----------------------------------
Selection runs on rank 0 only — the module docstring in
``tads/pipelines/selection.py`` explains why (an NCCL barrier there deadlocked
earlier versions, because rank 0 sits inside a 30-90 minute scoring pass while
the other ranks wait in the collective and trip the watchdog). That design is
right for selection, but it means the gate's 1+K pool forwards also run on one
GPU while the rest of a big box idles.

The gate does not have to live inside that constraint. ``G`` is a function of
(pool, base checkpoint, gate config) and **nothing else** — not the seed, not
the arm, not the epoch. So it can be computed once, in parallel, before any
training starts, and reused by every arm and every seed. On the paper's grid
(8 arms x 3 seeds) that is 24 redundant computations collapsed into one.

How the parallelism works
-------------------------
Deliberately NOT torchrun/NCCL. Each shard is an independent process pinned to
one GPU with ``CUDA_VISIBLE_DEVICES``, writing its own file; a final merge step
concatenates them. No process group, no collectives, no rendezvous — so a
single dead shard is re-runnable on its own instead of taking the job down,
and the whole thing restarts cleanly after a pre-emption.

    # 4 GPUs, one command:
    bash scripts/precompute_gate.sh --config <cfg> --out <cache.pt>

    # or by hand:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/precompute_gate.py \
        --config <cfg> --out <cache.pt> --shard $i --num-shards 4 &
    done; wait
    python scripts/precompute_gate.py --config <cfg> --out <cache.pt> --merge \
        --num-shards 4

Point the training runs at the result with ``tads.tag.gate_cache_file``
(env: ``TADS_GATE_CACHE``). Every arm then starts with the gate already in
hand and spends its GPU time on training.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("precompute_gate")


def _shard_path(out: Path, shard: int, num_shards: int) -> Path:
    return out.parent / f"{out.stem}.shard{shard}of{num_shards}.pt"


def _build_gate_cfg(params, scale, null=None):
    from tads.core.gate import GateConfig

    return GateConfig(
        span_tokens=int(params.get("span_tokens", 16)),
        tau=float(params.get("tau", 0.5)),
        tau_mode=str(params.get("tau_mode", "per_token")),
        min_span_tokens=int(params.get("min_span_tokens", 4)),
        tail_mode=str(params.get("tail_mode", "min")),
        tail_quantile=float(params.get("tail_quantile", 0.0)),
        include_eos=bool(params.get("include_eos", False)),
        c_trunc=float(params.get("c_trunc", 0.2)),
        eps_den=float(params.get("eps_den", 1e-3)),
        min_common_tokens=int(params.get("min_common_tokens", 8)),
        undefined_policy=str(params.get("undefined_policy", "neutral")),
        undefined_gate_value=float(params.get("undefined_gate_value", 0.6)),
        null_correction=bool(params.get("null_correction", True)),
        target_veto=float(params.get("target_veto", 0.05)),
        null=null,
        scale=scale,
        dispersion_discount=bool(params.get("dispersion_discount", True)),
    )


def _load_everything(cfg, *, want_model: bool):
    """Resolve pool + counterfactual datasets (and optionally the model)."""
    import torch  # noqa: F401  (import cost is real; keep it lazy)
    from tads.data.alpaca import build_alpaca_dataset
    from tads.modeling.loader import load_model, load_tokenizer

    tokenizer = load_tokenizer(cfg["model_path"])
    cache_dir = str(Path(cfg["data_cache"]) / str(cfg.get("model_key", "m"))
                    / str(cfg.get("prompt_style") or "alpaca_default"))
    style = str(cfg.get("prompt_style") or "alpaca_default")

    pool = build_alpaca_dataset(
        tokenizer=tokenizer, cache_dir=cache_dir,
        max_seq_len=int(cfg["max_seq_len"]),
        dataset_name=cfg.get("dataset_name"),
        data_files=cfg.get("data_files"),
        prompt_style=style,
    )

    tag_cfg = (cfg.get("tads") or {}).get("tag") or {}
    cf_spec = tag_cfg.get("counterfactual_data_files") or ""
    if not isinstance(cf_spec, (list, tuple)):
        cf_spec = [s.strip() for s in str(cf_spec).split(",") if s.strip()]
    if not cf_spec:
        sys.exit(
            "tads.tag.counterfactual_data_files is empty — set TADS_CF_FILES "
            "to the counterfactual pool(s)."
        )
    cf_datasets = []
    for k, one in enumerate(cf_spec, start=1):
        ds = build_alpaca_dataset(
            tokenizer=tokenizer,
            cache_dir=str(Path(cache_dir) / ("counterfactual" if k == 1
                                             else f"counterfactual_{k}")),
            max_seq_len=int(cfg["max_seq_len"]),
            dataset_name=None, data_files=str(one), prompt_style=style,
        )
        if len(ds) != len(pool):
            sys.exit(
                f"counterfactual pool #{k} has {len(ds)} records but the pool "
                f"has {len(pool)} — they must be index-aligned."
            )
        cf_datasets.append(ds)

    model = None
    if want_model:
        model = load_model(
            cfg["model_path"], training_mode="full", lora_cfg=None,
            use_ddp=False, local_rank=0, gradient_checkpointing=False,
            attn_implementation=cfg.get("attn_implementation"),
        )
        model.eval()
    return tokenizer, pool, cf_datasets, model, tag_cfg


def preflight_scale(cfg) -> None:
    """Resolve the calibration BEFORE spending any forward passes.

    The scale is only needed at merge time, so the first version resolved it
    there — after every shard had run. A missing or wrong-backbone reference
    therefore cost the full 1+K pool forwards on every GPU before failing,
    which is exactly the mistake this whole script exists to avoid paying
    twice. Resolving it up front is free and turns an 18-minute failure into
    an instant one.
    """
    from tads.pipelines.selection import _resolve_gate_calibration

    tag_cfg = (cfg.get("tads") or {}).get("tag") or {}
    # Raises with the exact fix when the reference is missing, unreadable, or
    # was fit at a different W / target_veto.
    scale, null = _resolve_gate_calibration(tag_cfg)
    if null is not None:
        logger.info(
            "Eq. 5' null curve: %d bin(s), target_veto=%.3f, fit on n=%d at "
            "W=%d (%s)", len(null.bin_edges), null.target_veto, null.n_ref,
            null.span_tokens, null.digest(),
        )
    if scale is None:
        logger.warning(
            "no gate_scale and no gate_ref_file: the gate will self-calibrate "
            "IN-POOL, which makes G depend on how dirty its neighbours are. "
            "Diagnostics only — not for a reported run.",
        )
    else:
        logger.info("calibration resolves to s=%.6g", scale)


def run_shard(args, cfg) -> None:
    import torch
    from torch.utils.data import Subset
    from tads.core import gate as gatelib

    tokenizer, pool, cf_datasets, model, tag_cfg = _load_everything(
        cfg, want_model=True,
    )
    n = len(pool)
    # Contiguous shards keep each worker's sequence lengths correlated, which
    # makes padding waste lower than a strided split would.
    bounds = [round(i * n / args.num_shards) for i in range(args.num_shards + 1)]
    lo, hi = bounds[args.shard], bounds[args.shard + 1]
    idx = list(range(lo, hi))
    logger.info(
        "shard %d/%d -> records [%d, %d) of %d",
        args.shard, args.num_shards, lo, hi, n,
    )
    if not idx:
        logger.warning("empty shard — writing a stub so the merge can proceed")

    gcfg = _build_gate_cfg(tag_cfg, 1.0)   # scale is applied at merge time
    bs = int(args.batch_size or cfg.get("episode_batch_size", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tok_true, n_true = gatelib.compute_pool_token_losses(
        model, Subset(pool, idx), batch_size=bs, device=device,
        tag=f"true[{args.shard}]", eos_token_id=tokenizer.eos_token_id,
        drop_trailing_eos=not gcfg.include_eos,
    )
    tok_cf, n_cf = [], []
    for k, ds in enumerate(cf_datasets, start=1):
        tc, nc = gatelib.compute_pool_token_losses(
            model, Subset(ds, idx), batch_size=bs, device=device,
            tag=f"cf{k}[{args.shard}]", eos_token_id=tokenizer.eos_token_id,
            drop_trailing_eos=not gcfg.include_eos,
        )
        tok_cf.append(tc)
        n_cf.append(nc)

    out = Path(args.out)
    sp = _shard_path(out, args.shard, args.num_shards)
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_suffix(".pt.tmp")
    torch.save(
        {
            "shard": args.shard, "num_shards": args.num_shards,
            "lo": lo, "hi": hi, "n_pool": n,
            # fp16 halves a tensor that is already the biggest thing here;
            # a per-token NLL of 0-20 keeps ~3 decimal digits, far below the
            # precision the span ratios need.
            "token_true": tok_true.to(torch.float16),
            "n_true": n_true,
            "token_cf": [t.to(torch.float16) for t in tok_cf],
            "n_cf": n_cf,
            "gate_config": gcfg.identity(),
        },
        tmp,
    )
    tmp.replace(sp)
    logger.info(
        "shard %d done in %.1f min -> %s",
        args.shard, (time.time() - t0) / 60, sp,
    )


def run_merge(args, cfg) -> None:
    import torch
    import torch.nn.functional as F
    from tads.core import gate as gatelib
    from tads.core.reliability import completeness_from_dataset

    out = Path(args.out)
    shards = []
    for i in range(args.num_shards):
        sp = _shard_path(out, i, args.num_shards)
        if not sp.exists():
            sys.exit(
                f"missing {sp} — shard {i} did not finish. Re-run just that "
                f"shard:\n  CUDA_VISIBLE_DEVICES=<gpu> python "
                f"scripts/precompute_gate.py --config {args.config} "
                f"--out {args.out} --shard {i} --num-shards {args.num_shards}"
            )
        shards.append(torch.load(sp, map_location="cpu", weights_only=False))
    shards.sort(key=lambda d: d["shard"])

    n_pool = shards[0]["n_pool"]
    covered = sum(d["hi"] - d["lo"] for d in shards)
    if covered != n_pool:
        sys.exit(
            f"shards cover {covered} records but the pool has {n_pool} — "
            f"they were produced against different pools or --num-shards "
            f"changed between runs. Delete the shard files and redo."
        )
    cfgs = {json.dumps(d["gate_config"], sort_keys=True) for d in shards}
    if len(cfgs) > 1:
        sys.exit(
            "shards were computed under DIFFERENT gate configs — delete them "
            "and re-run all shards against one config."
        )

    def _cat(key):
        """Concatenate padded (n_i, T_i) blocks with differing widths."""
        mats = [d[key] for d in shards]
        width = max(m.size(1) for m in mats)
        return torch.cat(
            [F.pad(m.float(), (0, width - m.size(1))) for m in mats], dim=0,
        )

    tok_true = _cat("token_true")
    n_true = torch.cat([d["n_true"] for d in shards], dim=0)
    k_cf = len(shards[0]["token_cf"])
    tok_cf, n_cf = [], []
    for k in range(k_cf):
        mats = [d["token_cf"][k] for d in shards]
        width = max(m.size(1) for m in mats)
        tok_cf.append(torch.cat(
            [F.pad(m.float(), (0, width - m.size(1))) for m in mats], dim=0,
        ))
        n_cf.append(torch.cat([d["n_cf"][k] for d in shards], dim=0))
    logger.info("merged %d shards -> %d records", len(shards), tok_true.size(0))

    # Completeness is a data-level property (no forward), so the merge step
    # computes it directly rather than shipping it through every shard.
    tokenizer, pool, _cf_ds, _m, tag_cfg = _load_everything(cfg, want_model=False)
    completeness = completeness_from_dataset(
        pool, eos_token_id=tokenizer.eos_token_id,
        c_trunc=float(tag_cfg.get("c_trunc", 0.2)),
    )

    from tads.pipelines.selection import _resolve_gate_calibration
    scale, null = _resolve_gate_calibration(tag_cfg)
    gcfg = _build_gate_cfg(tag_cfg, scale, null)
    if gcfg.scale is None:
        probe = gatelib.gate_components(tok_true, n_true, tok_cf[0], n_cf[0], cfg=gcfg)
        gcfg = _build_gate_cfg(
            tag_cfg, gatelib.resolve_scale(gcfg, probe["delta_hat"]), null,
        )

    result = gatelib.compute_gate(
        tok_true, n_true, tok_cf, n_cf, completeness, cfg=gcfg,
    )
    identity = gatelib.cache_identity(
        model_path=cfg["model_path"],
        pool_files=cfg.get("data_files"),
        n_pool=n_pool,
    )
    gatelib.save_gate_cache(
        None, result=result, cfg=gcfg, epoch=1,
        token_true=tok_true, n_true=n_true, token_cf=tok_cf, n_cf=n_cf,
        store_token_losses=bool(tag_cfg.get("store_token_losses", False)),
        identity=identity, path=out,
    )
    print(f"\nGate cache written to {out}")
    print(f"  identity : {identity}")
    print(f"  scale s  : {gcfg.scale:.6g}")
    print(f"  G mean   : {float(result['gate'].mean()):.4f}")
    print(f"  G == 0   : {int((result['gate'] == 0).sum())}/{n_pool} "
          f"({100.0 * float((result['gate'] == 0).float().mean()):.1f}%)")
    if float((result["gate"] == 0).float().mean()) > 0.9:
        print("  WARNING: over 90% vetoed — the scale is almost certainly "
              "wrong (uncalibrated in-pool fallback?).")
    print(f"\nPoint runs at it:\n  export TADS_GATE_CACHE={out}")
    if not args.keep_shards:
        for i in range(args.num_shards):
            _shard_path(out, i, args.num_shards).unlink(missing_ok=True)
        print("  (shard files removed; --keep-shards to retain them)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="path of the shared gate cache")
    p.add_argument("--shard", type=int, default=None, help="this shard's index")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--merge", action="store_true", help="combine finished shards")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--keep-shards", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [shard {args.shard}] %(message)s"
        if args.shard is not None else "%(asctime)s [merge] %(message)s",
    )
    from tads.core.utils import load_config
    cfg = load_config(args.config)

    # Both paths need the calibration to exist; check before any GPU work.
    preflight_scale(cfg)

    if args.merge:
        run_merge(args, cfg)
    elif args.shard is not None:
        if not (0 <= args.shard < args.num_shards):
            sys.exit(f"--shard must be in [0, {args.num_shards})")
        run_shard(args, cfg)
    else:
        sys.exit("pass either --shard N or --merge")


if __name__ == "__main__":
    main()
