"""Per-epoch sample selection dispatch.

Wraps the three selection methods (random / full / selection) handled by
``tag.train``. Comparison baselines (data_agent / nait / selectit /
lima / alpagasus / q2q) have their own entrypoints under
``baselines.<method>.train`` and bypass this dispatcher.

For ``method=selection`` the heavy collect_episode runs on rank 0 only;
other ranks share the resulting indices through a filesystem sentinel
+ poll, NOT through an NCCL barrier — that was the deadlock that
crashed runs after epoch 1.

Why polling, not dist.barrier:
    Rank 0 spends 30+ minutes inside collect_episode (52K samples × 32
    decoder layers × chunked rewards). While that runs, the other DDP
    ranks would be stuck inside dist.barrier() inside _broadcast_selection,
    and any of them hitting the NCCL collective watchdog (120 min default
    now, less previously) tears down the communicator. The next forward
    pass then fails on every rank, and rank 0 — which never reached the
    barrier — exits before saving any checkpoint.

    The fix is to remove the NCCL barriers from this path entirely. Rank 0
    writes the selection atomically (tmp + fsync + rename), then writes
    a separate `.ready` sentinel; workers poll on disk for the sentinel
    and read once it appears. The only collective in this module is a
    single barrier at the very end, after everyone has the data — so it
    always completes immediately.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..core.selector import collect_episode
from ..core.trajectory_anchor import TrajectoryAnchor
from ..core.utils import is_main_process, local_rank, rank, world_size

logger = logging.getLogger(__name__)


# Workers poll this often while waiting for rank-0's collect_episode.
_POLL_INTERVAL_SEC = 2.0
# Hard ceiling on how long workers will wait. Set generously — episodes
# can legitimately take an hour at the 7B scale.
_POLL_TIMEOUT_SEC = 6 * 60 * 60  # 6 hours


def _random_indices(n_total, ratio, seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)
    perm = torch.randperm(n_total, generator=g).tolist()
    k = max(1, int(n_total * ratio))
    return perm[:k]


# ---------------------------------------------------------------------------
# MVF support: loss history (learnability view) + reliability cache plumbing
# ---------------------------------------------------------------------------

def _loss_history_path(output_dir, epoch: int) -> Path:
    return Path(output_dir) / f"loss_history_epoch{epoch}.pt"


def _save_loss_history(output_dir, epoch: int, r_loss: torch.Tensor) -> None:
    p = _loss_history_path(output_dir, epoch)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pt.tmp")
    torch.save(r_loss.detach().cpu(), tmp)
    os.replace(tmp, p)
    logger.info("Saved loss history for epoch %d to %s", epoch, p)


def _load_loss_history(output_dir, epoch: int):
    p = _loss_history_path(output_dir, epoch)
    if epoch < 1 or not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning("Could not load loss history at %s (%s)", p, e)
        return None


def _load_selected_prev(output_dir, epoch: int):
    """Previous refresh's selected indices (gradient-evidence set for the
    v3 split-progress D view). None when absent (t=1, or history lost)."""
    if epoch < 1 or output_dir is None:
        return None
    p = Path(output_dir) / f"selected_indices_epoch{epoch}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            sel = json.load(f)
        if isinstance(sel, list) and sel:
            return [int(x) for x in sel]
    except Exception as e:
        logger.warning("Could not load previous selection at %s (%s)", p, e)
    return None


def _resolve_reliability_scale(params: Dict[str, Any]) -> Optional[float]:
    """Resolve the calibrated sigmoid scale s (plan §2.1).

    Priority: explicit ``reliability_scale`` > calibration from
    ``reliability_ref_file`` (a .pt holding the clean-reference ΔL tensor,
    or a dict with key 'delta') > None (in-pool fallback inside
    reliability_from_losses, which warns loudly).
    """
    from ..core import reliability as rel

    scale = params.get("reliability_scale")
    # ${oc.env:TAG_RELIABILITY_SCALE,} resolves to the EMPTY STRING when
    # the env var is unset (utils._resolve_env) — float('') would crash
    # every MVF run at epoch-1 selection. Treat empty/whitespace as unset.
    if scale is not None and str(scale).strip() != "":
        return float(scale)
    ref_file = params.get("reliability_ref_file")
    if ref_file:
        ref = torch.load(str(ref_file), map_location="cpu", weights_only=True)
        if isinstance(ref, dict):
            ref = ref["delta"]
        return rel.calibrate_reliability_scale(
            ref,
            target_pct=float(params.get("calibration_target_pct", 0.10)),
            target_q=float(params.get("calibration_target_q", 0.8)),
        )
    return None


def _build_gate_config(params: Dict[str, Any], scale: Optional[float], null=None):
    """Assemble a :class:`tag.core.gate.GateConfig` from the YAML subtree."""
    from ..core.gate import GateConfig

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
        target_zero_rate=float(params.get("target_zero_rate", 0.05)),
        null=null,
        scale=scale,
        dispersion_discount=bool(params.get("dispersion_discount", True)),
    )


def _resolve_gate_calibration(params: Dict[str, Any]):
    """Resolve BOTH halves of the TAG calibration from one reference file.

    Returns ``(scale, null)``:

      * ``scale`` — the sigmoid scale s of Eq. 6. Priority: explicit
        ``gate_scale`` > derived from ``gate_ref_file`` > None (the in-pool
        fallback inside ``gate.resolve_scale``, which warns loudly and is
        diagnostic-only).
      * ``null`` — the length-conditional null curve mu(M) of Eq. 5', read
        from the same ``gate_ref_file``. None when ``null_correction`` is off.

    They resolve together because they are two quantiles of the SAME clean
    reference and must come from the same fit: s is derived from the
    already-centred statistic, so pairing a scale from one reference with a
    null curve from another silently mis-scales every gate value.

    The reference statistic MUST be ``Delta_hat``, not the MVF module's raw
    ``ΔL``: they are different quantities (a ratio in [-inf, 1] versus a
    difference in nats), so ``reliability_ref_file`` is NOT interchangeable
    with ``gate_ref_file``.
    """
    from ..core import gate as gatelib

    want_null = bool(params.get("null_correction", True))
    target_zero_rate = float(params.get("target_zero_rate", 0.05))
    explicit = params.get("gate_scale")
    # ${oc.env:TAG_GATE_SCALE,} resolves to the EMPTY STRING when unset —
    # float('') would crash every TAG run at epoch-1 selection.
    has_explicit = explicit is not None and str(explicit).strip() != ""

    ref_file = params.get("gate_ref_file")
    if (not ref_file or str(ref_file).strip() == "") and want_null:
        raise FileNotFoundError(
            "selection.tag.null_correction is on but no selection.tag.gate_ref_file is "
            "set. The null curve mu(M) is measured on a CLEAN reference pool "
            "and cannot be derived from the candidate pool — self-calibrating "
            "it there would absorb the very corruption the gate is meant to "
            "find. Point gate_ref_file at a calibration artifact (bash "
            "scripts/gpu_cloud/bootstrap.sh calibrate7b), or set "
            "selection.tag.null_correction: false to run the uncorrected ablation."
        )
    if has_explicit and not want_null:
        return float(explicit), None

    if ref_file and str(ref_file).strip() != "":
        if not Path(str(ref_file)).exists():
            # Deliberately NOT a silent fall-back to in-pool calibration: the
            # user asked for a calibrated gate, and quietly substituting the
            # pool-dependent one would produce a reported run whose gate means
            # something different from what the config says.
            raise FileNotFoundError(
                f"selection.tag.gate_ref_file points to {ref_file}, which does not "
                f"exist. Generate it on a CLEAN reference pool with:\n"
                f"    python scripts/calibrate_reliability.py --mode tag \\\n"
                f"        --config <this config> \\\n"
                f"        --pool <clean>/pool.json \\\n"
                f"        --counterfactual <clean>/counterfactual.json \\\n"
                f"        --out {ref_file}\n"
                f"(on a GPU box: bash scripts/gpu_cloud/bootstrap.sh calibrate)\n"
                f"To run without calibration anyway — diagnostics only — unset "
                f"TAG_GATE_REF so the gate self-calibrates in-pool with a warning."
            )
        # weights_only=False: the artifact carries the null curve as a plain
        # dict alongside its tensors, which the weights-only unpickler rejects.
        ref = torch.load(str(ref_file), map_location="cpu", weights_only=False)
        if not isinstance(ref, dict):
            # A bare tensor is the pre-null artifact format.
            ref = {"delta_hat": ref}
        if "delta_hat" not in ref:
            raise ValueError(
                f"gate_ref_file {ref_file} is a dict without a 'delta_hat' key "
                f"(found {sorted(ref.keys())}). The TAG gate calibrates on "
                f"Delta_hat; an MVF reliability reference (raw ΔL) is not "
                f"interchangeable — regenerate with "
                f"scripts/calibrate_reliability.py --mode tag.",
            )
        _warn_gate_ref_config_mismatch(ref.get("gate_config"), params, ref_file)

        null = None
        if want_null:
            nd = ref.get("null")
            if nd is None:
                raise ValueError(
                    f"selection.tag.null_correction is on but {ref_file} carries no "
                    f"null curve — it predates Eq. 5'. Regenerate it:\n"
                    f"    bash scripts/gpu_cloud/bootstrap.sh calibrate7b\n"
                    f"(or scripts/calibrate_reliability.py --mode tag directly; "
                    f"if the artifact already has token losses, "
                    f"scripts/sweep_gate_config.py --refit-out re-fits it with "
                    f"no GPU)."
                )
            null = gatelib.NullCalibration.from_dict(nd)
            if abs(null.target_zero_rate - target_zero_rate) > 1e-9:
                raise ValueError(
                    f"{ref_file} was fit for target_zero_rate={null.target_zero_rate} but "
                    f"this config asks for {target_zero_rate}. mu(M) IS that "
                    f"quantile, so it does not transfer — refit with "
                    f"scripts/sweep_gate_config.py --target-zero-rate {target_zero_rate} "
                    f"--refit-out {ref_file}."
                )
            if null.span_tokens != int(params.get("span_tokens", 16)):
                raise ValueError(
                    f"{ref_file} fit its null curve at span_tokens="
                    f"{null.span_tokens} but this config uses "
                    f"{params.get('span_tokens', 16)}. M is a span COUNT, so "
                    f"mu(M) means something different at a different W."
                )

        if has_explicit:
            return float(explicit), null
        # s is derived from the statistic Eq. 6 actually sees, so it must be
        # the CENTRED one whenever the correction is on. Deriving it from the
        # uncentred reference and then centring at gate time would shift every
        # value by mu without shifting the scale that interprets them.
        stat = ref["delta_hat"]
        if null is not None:
            n_spans = ref.get("n_spans")
            if n_spans is None:
                raise ValueError(
                    f"{ref_file} carries a null curve but no 'n_spans' — the "
                    f"centring cannot be reproduced. Regenerate the reference."
                )
            stat = null.apply(stat, n_spans)
        return (
            gatelib.calibrate_gate_scale(
                stat,
                target_pct=float(params.get("calibration_target_pct", 0.10)),
                target_q=float(params.get("calibration_target_q", 0.8)),
                null_corrected=null is not None,
            ),
            null,
        )
    return None, None


def _resolve_gate_scale(params: Dict[str, Any]) -> Optional[float]:
    """Back-compat shim: the scale half of :func:`_resolve_gate_calibration`."""
    return _resolve_gate_calibration(params)[0]


# Fields whose value changes the Delta_hat DISTRIBUTION, and therefore make a
# calibration derived under one setting invalid under another.
_CALIBRATION_BOUND_FIELDS = (
    "span_tokens", "tau", "tau_mode", "min_span_tokens",
    "tail_mode", "tail_quantile", "include_eos",
)


def _warn_gate_ref_config_mismatch(ref_cfg, params, ref_file) -> None:
    """The reference records the gate config it was computed under; check it.

    ``s`` is a quantile of the clean reference's Delta_hat, and Delta_hat's
    distribution depends on how the response was partitioned. Calibrating at
    W=16 and gating at W=32 silently mis-scales every gate value in the run,
    with no symptom other than a wrong zero-weight rate — which is exactly the
    quantity the paper reports.
    """
    if not isinstance(ref_cfg, dict):
        return
    diffs = {
        f: (ref_cfg.get(f), params.get(f))
        for f in _CALIBRATION_BOUND_FIELDS
        if f in ref_cfg and f in params and ref_cfg[f] != params[f]
    }
    if diffs:
        raise ValueError(
            f"gate_ref_file {ref_file} was calibrated under a different span "
            f"configuration: {diffs} (ref -> requested). The scale s is a "
            f"quantile of Delta_hat, whose distribution depends on the span "
            f"partition, so this calibration does not transfer. Regenerate "
            f"with scripts/calibrate_reliability.py --mode tag using the SAME "
            f"config, or pin selection.tag.gate_scale explicitly."
        )


def _prepare_tag(
    tag_ctx: Dict[str, Any],
    *,
    model,
    cfg,
    epoch: int,
    device,
    n_pool: int,
):
    """Assemble the ``tag`` dict consumed by ``collect_episode`` (paper Eq. 1).

    G is a property of the DATA, so it is computed once at the base
    checkpoint and cached in ``tag_gate_cache.pt`` (a SEPARATE file from the
    MVF ``reliability_cache.pt`` — the two store different statistics under
    similar names and would silently cross-validate).

    Three paths, cheapest first:
      1. cache hit with a matching :class:`GateConfig` -> reuse G directly;
      2. cache hit with cached per-token losses but a different config ->
         re-derive G with NO forward pass (this is what
         ``store_token_losses`` buys: sweeping W / tau / s becomes free);
      3. no usable cache -> run the token-level forwards. At epoch > 1 this
         is a hard error instead, because G computed at a later checkpoint
         is a different quantity than the one the paper defines.
    """
    from ..core import gate as gatelib

    output_dir = cfg["output_dir"]
    params = tag_ctx.get("params", {}) or {}
    # A SHARED gate cache (scripts/precompute_gate.py) lets every arm and
    # seed reuse one computation, since G depends only on (pool, base
    # checkpoint, gate config). Absent that, the cache is per-run.
    shared_cache = str(params.get("gate_cache_file") or "").strip() or None
    cache_path = Path(shared_cache) if shared_cache else None
    scale, null = _resolve_gate_calibration(params)
    gcfg = _build_gate_config(params, scale, null)
    completeness = tag_ctx["completeness"]

    tag: Dict[str, Any] = {
        "gate": None,
        # Orders the zero-weight block if a backfill is ever forced — see
        # scorer.gated_selection_key.
        "delta_hat": None,
        "cluster_ids": tag_ctx.get("cluster_ids"),
        "gate_config": gcfg,
        "allow_late_gate": bool(params.get("allow_late_gate", False)),
        "store_token_losses": bool(params.get("store_token_losses", False)),
        "components": None,
        "cache_path": cache_path,
        "token_true": None,
        "n_true": None,
        "token_cf": None,
        "n_cf": None,
        "scale_used": scale,
    }

    cache = gatelib.load_gate_cache(output_dir, path=cache_path)
    if cache is not None:
        # A shared cache is reachable by runs it was never meant for, so the
        # producer records what it is valid FOR and we check it here. Shape
        # alone would not catch a cache from a different backbone.
        want_id = gatelib.cache_identity(
            model_path=cfg.get("model_path"),
            pool_files=cfg.get("data_files"),
            n_pool=n_pool,
        )
        why = gatelib.check_cache_identity(cache, want_id)
        if why is not None:
            raise RuntimeError(
                f"TAG gate cache at {cache_path or gatelib.cache_path_for(output_dir)} "
                f"does not belong to this run — {why}. G is only valid for the "
                f"(pool, base checkpoint) it was computed on. Point "
                f"selection.tag.gate_cache_file at the right cache, or regenerate "
                f"with scripts/precompute_gate.sh."
            )
    if cache is not None and cache["gate"].numel() == n_pool:
        cached_cfg = cache.get("config") or {}
        if scale is None and cached_cfg.get("scale") is not None:
            # An UNPINNED scale is derived once, at the base checkpoint, from
            # the pool itself — so the cached value IS this run's scale, not a
            # different configuration. Comparing None against it would miss
            # the cache at epoch 2 and then hit the base-checkpoint hard
            # error, killing every run that does not pin gate_scale (i.e. the
            # shipped default). Adopt the cached scale instead.
            gcfg = _build_gate_config(params, float(cached_cfg["scale"]), null)
            tag["gate_config"] = gcfg
            tag["scale_used"] = gcfg.scale
            logger.info(
                "TAG gate: adopting the scale derived at the base checkpoint "
                "(s=%.6g) from the cache; no gate_scale/gate_ref_file was "
                "pinned for this run.", gcfg.scale,
            )
        if cached_cfg == gcfg.identity():
            tag["gate"] = cache["gate"]
            tag["delta_hat"] = cache.get("delta_hat")
            logger.info("TAG gate: cache hit (config unchanged) — no forward pass.")
        else:
            redone = gatelib.recompute_gate_from_cache(cache, gcfg)
            if redone is not None:
                tag["gate"] = redone["gate"]
                tag["delta_hat"] = redone["delta_hat"]
                tag["components"] = redone
                gatelib.save_gate_cache(
                    output_dir, result=redone, cfg=gcfg,
                    epoch=int(cache.get("epoch", epoch)),
                    token_true=cache.get("token_true"),
                    n_true=cache.get("n_true"),
                    token_cf=cache.get("token_cf"),
                    n_cf=cache.get("n_cf"),
                    store_token_losses=True,
                    identity=cache.get("identity"),
                    path=cache_path,
                )
            else:
                logger.warning(
                    "TAG gate cache config %s != requested %s and the cache has "
                    "no per-token losses to re-derive from — recomputing. Set "
                    "selection.tag.store_token_losses: true to make config sweeps free.",
                    cache.get("config"), gcfg.identity(),
                )
    elif cache is not None:
        logger.warning(
            "TAG gate cache size %d != pool size %d — recomputing.",
            cache["gate"].numel(), n_pool,
        )

    if tag["gate"] is None:
        cf_datasets = tag_ctx.get("cf_datasets") or []
        if not cf_datasets:
            raise ValueError(
                "TAG score_mode requires a counterfactual pool: set "
                "selection.tag.counterfactual_data_files (generate it with "
                "scripts/make_corrupted_pool.py --emit-counterfactual).",
            )
        for k, cf_dataset in enumerate(cf_datasets, start=1):
            if len(cf_dataset) != n_pool:
                raise ValueError(
                    f"Counterfactual pool #{k} size {len(cf_dataset)} != "
                    f"candidate pool size {n_pool} — pools must be "
                    f"index-aligned.",
                )
        if epoch > 1 and not tag["allow_late_gate"]:
            # Fail BEFORE burning 1+K pool forwards.
            raise RuntimeError(
                f"_prepare_tag: no usable gate cache at epoch {epoch} (> 1). "
                f"G is defined at the BASE checkpoint (paper Eq. 6) — restore "
                f"tag_gate_cache.pt from the run's output dir, or set "
                f"selection.tag.allow_late_gate: true to explicitly accept a "
                f"wrong-checkpoint G.",
            )
        eos_id = tag_ctx.get("eos_token_id")
        drop_eos = not gcfg.include_eos
        bs = int(cfg.get("episode_batch_size", 1))
        logger.info(
            "TAG gate: computing per-token losses over %d pools (1 true + %d "
            "counterfactual) at the base checkpoint.", 1 + len(cf_datasets),
            len(cf_datasets),
        )
        token_true, n_true = gatelib.compute_pool_token_losses(
            model, tag_ctx["dataset"], batch_size=bs, device=str(device),
            tag="true", eos_token_id=eos_id, drop_trailing_eos=drop_eos,
        )
        token_cf: List[torch.Tensor] = []
        n_cf: List[torch.Tensor] = []
        for k, cf_dataset in enumerate(cf_datasets, start=1):
            tc, nc = gatelib.compute_pool_token_losses(
                model, cf_dataset, batch_size=bs, device=str(device),
                tag=f"counterfactual_{k}" if len(cf_datasets) > 1 else "counterfactual",
                eos_token_id=eos_id, drop_trailing_eos=drop_eos,
            )
            token_cf.append(tc)
            n_cf.append(nc)

        if gcfg.scale is None:
            # Calibrate on THIS pool only as an explicitly diagnostic
            # fallback; resolve_scale warns loudly.
            probe = gatelib.gate_components(
                token_true, n_true, token_cf[0], n_cf[0], cfg=gcfg,
            )
            gcfg = _build_gate_config(
                params, gatelib.resolve_scale(gcfg, probe["delta_hat"]), null,
            )
            tag["gate_config"] = gcfg
            tag["scale_used"] = gcfg.scale
        result = gatelib.compute_gate(
            token_true, n_true, token_cf, n_cf, completeness, cfg=gcfg,
        )
        tag["gate"] = result["gate"]
        tag["delta_hat"] = result["delta_hat"]
        tag["components"] = result
        tag["token_true"] = token_true
        tag["n_true"] = n_true
        tag["token_cf"] = token_cf
        tag["n_cf"] = n_cf
    return tag


def _finalize_tag(tag, episode, *, cfg, epoch: int) -> Dict[str, Any]:
    """Persist the gate cache (when freshly computed) and return extras."""
    from ..core import gate as gatelib

    output_dir = cfg["output_dir"]
    _save_loss_history(output_dir, epoch, episode["r_loss"])
    comp = tag.get("components")
    if comp is not None and tag.get("token_true") is not None:
        gatelib.save_gate_cache(
            output_dir,
            result=comp,
            cfg=tag["gate_config"],
            epoch=epoch,
            token_true=tag["token_true"],
            n_true=tag["n_true"],
            token_cf=tag["token_cf"],
            n_cf=tag["n_cf"],
            store_token_losses=bool(tag.get("store_token_losses", False)),
            identity=gatelib.cache_identity(
                model_path=cfg.get("model_path"),
                pool_files=cfg.get("data_files"),
                n_pool=int(episode["gate"].numel()),
            ),
            path=tag.get("cache_path"),
        )
    gate_t = episode.get("gate")
    extras: Dict[str, Any] = {
        "score_mode": "tag",
        "gate_scale": tag.get("scale_used"),
        # The realised zero-weight accounting — the only evidence in the run
        # artifacts that the non-compensation claim held for THIS run.
        "n_admissible": episode.get("n_admissible"),
        "n_zero_weight_selected": episode.get("n_zero_weight_selected"),
        "selection_budget": episode.get("selection_budget"),
    }
    if gate_t is not None:
        extras.update({
            "gate_mean": float(gate_t.float().mean().item()),
            "gate_zero_frac": float((gate_t == 0).float().mean().item()),
        })
    if comp is not None:
        extras.update({
            "delta_bar_mean": float(comp["delta_bar"].mean().item()),
            "delta_min_mean": float(comp["delta_min"].mean().item()),
            "delta_hat_mean": float(comp["delta_hat"].mean().item()),
            "gate_undefined": int(comp["undefined"].sum().item()),
            "gate_empty_c": int(comp["empty_c"].sum().item()),
        })
    return extras


def _prepare_mvf(
    mvf_ctx: Dict[str, Any],
    *,
    model,
    cfg,
    epoch: int,
    device,
    n_pool: int,
):
    """Assemble the ``mvf`` dict consumed by ``collect_episode``.

    - Reliability Q: loaded from the run's cache when the cached gate
      configuration matches; recomputed FROM CACHED LOSSES (no forward)
      when only the gate config changed; computed fresh (counterfactual
      forward pass) only at epoch 1 — at later epochs collect_episode
      hard-errors instead of silently recomputing Q at the wrong
      checkpoint.
    - Learnability: previous refresh's loss vector + previous selection
      (gradient-evidence set) for the v3 split-progress D.
    """
    from ..core import reliability as rel

    output_dir = cfg["output_dir"]
    params = mvf_ctx.get("params", {}) or {}
    rmode = str(params.get("reliability_mode", "sigmoid"))
    rezero = bool(params.get("reliability_rezero", True))
    rscale = _resolve_reliability_scale(params)
    mvf: Dict[str, Any] = {
        "completeness": mvf_ctx["completeness"],
        "cluster_ids": mvf_ctx.get("cluster_ids"),
        "eta": float(params.get("eta", 0.5)),
        "gamma": float(params.get("gamma", 1.0)),
        "eps": float(params.get("eps", 0.01)),
        "d_floor": float(params.get("d_floor", 0.5)),
        "progress_mode": str(params.get("progress_mode", "split")),
        "reliability_mode": rmode,
        "reliability_scale": rscale,
        "reliability_rezero": rezero,
        "allow_late_reliability": bool(params.get("allow_late_reliability", False)),
        "lam_scale": float(mvf_ctx.get("lam_scale", 1.0)),
        "reliability": None,
        "loss_cf": None,
        "loss_prev": None,
        "selected_prev": None,
    }

    cache = rel.load_reliability_cache(output_dir)
    if cache is not None and cache["q"].numel() == n_pool:
        cache_cfg = (
            cache.get("mode", "rank"),  # pre-v3 caches were rank-transformed
            cache.get("scale"),
            cache.get("rezero", False),
        )
        want_cfg = (rmode, rscale, rezero)
        if cache_cfg == want_cfg:
            mvf["reliability"] = cache["q"]
        elif cache.get("loss_orig") is not None and cache.get("loss_cf") is not None:
            logger.info(
                "Reliability cache gate config %s != requested %s — "
                "recomputing Q from cached losses (no forward pass).",
                cache_cfg, want_cfg,
            )
            mvf["reliability"] = rel.reliability_from_losses(
                cache["loss_orig"], cache["loss_cf"],
                mode=rmode, scale=rscale, rezero=rezero,
            )
            rel.save_reliability_cache(
                output_dir,
                q=mvf["reliability"],
                completeness=mvf_ctx["completeness"],
                loss_orig=cache["loss_orig"],
                loss_cf=cache["loss_cf"],
                epoch=int(cache.get("epoch", epoch)),
                mode=rmode, scale=rscale, rezero=rezero,
            )
        else:
            logger.warning(
                "Reliability cache lacks raw losses — cannot re-gate under "
                "the requested config %s; recomputing from scratch.",
                (rmode, rscale, rezero),
            )
    if mvf["reliability"] is None:
        if cache is not None and cache["q"].numel() != n_pool:
            logger.warning(
                "Reliability cache size %d != pool size %d — recomputing.",
                cache["q"].numel(), n_pool,
            )
        cf_datasets = mvf_ctx.get("cf_datasets") or (
            [mvf_ctx["cf_dataset"]] if mvf_ctx.get("cf_dataset") is not None else []
        )
        if not cf_datasets:
            raise ValueError(
                "MVF score_mode requires a counterfactual pool: set "
                "selection.mvf.counterfactual_data_files (generate it with "
                "scripts/make_corrupted_pool.py --emit-counterfactual).",
            )
        for k, cf_dataset in enumerate(cf_datasets, start=1):
            if len(cf_dataset) != n_pool:
                raise ValueError(
                    f"Counterfactual pool #{k} size {len(cf_dataset)} != "
                    f"candidate pool size {n_pool} — pools must be "
                    f"index-aligned.",
                )
        if epoch > 1 and not bool(params.get("allow_late_reliability", False)):
            # Fail BEFORE burning the counterfactual forward pass(es) —
            # collect_episode enforces the same contract, but by then a
            # full pool forward would already have been spent (adversarial
            # review 2026-08).
            raise RuntimeError(
                f"_prepare_mvf: no usable reliability cache at epoch {epoch} "
                f"(> 1). Q must come from the base checkpoint — restore "
                f"reliability_cache.pt from the run's output dir, or set "
                f"selection.mvf.allow_late_reliability: true to explicitly accept "
                f"a wrong-checkpoint Q.",
            )
        losses = [
            rel.compute_pool_loss(
                model, cf_dataset,
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(device),
                tag=f"counterfactual_{k}" if len(cf_datasets) > 1 else "counterfactual",
            )
            for k, cf_dataset in enumerate(cf_datasets, start=1)
        ]
        mvf["loss_cf"] = losses[0] if len(losses) == 1 else torch.stack(losses, dim=0)

    mvf["loss_prev"] = _load_loss_history(output_dir, epoch - 1)
    if mvf["loss_prev"] is not None and mvf["loss_prev"].numel() != n_pool:
        logger.warning(
            "Loss history size %d != pool size %d — ignoring history.",
            mvf["loss_prev"].numel(), n_pool,
        )
        mvf["loss_prev"] = None
    sel_prev = _load_selected_prev(output_dir, epoch - 1)
    if sel_prev is not None and max(sel_prev) >= n_pool:
        logger.warning(
            "Previous selection references index %d >= pool size %d — "
            "ignoring (pool changed between epochs?).", max(sel_prev), n_pool,
        )
        sel_prev = None
    mvf["selected_prev"] = sel_prev
    return mvf


def _finalize_mvf(mvf, episode, *, cfg, epoch: int) -> Dict[str, Any]:
    """Persist per-epoch MVF state (loss history + reliability cache) and
    return metric extras."""
    from ..core import reliability as rel

    output_dir = cfg["output_dir"]
    _save_loss_history(output_dir, epoch, episode["r_loss"])
    if mvf["reliability"] is None and episode.get("reliability") is not None:
        rel.save_reliability_cache(
            output_dir,
            q=episode["reliability"],
            completeness=mvf["completeness"],
            loss_orig=episode["r_loss"],
            loss_cf=mvf["loss_cf"],
            epoch=epoch,
            mode=mvf["reliability_mode"],
            scale=mvf["reliability_scale"],
            rezero=mvf["reliability_rezero"],
        )
    extras: Dict[str, Any] = {
        "score_mode": "mvf",
        "q_mean": float(episode["reliability"].mean().item()),
        "d_mean": float(episode["difficulty"].mean().item()),
        "completeness_mean": float(mvf["completeness"].float().mean().item()),
        "progress_active": mvf["loss_prev"] is not None,
        "progress_mode": mvf["progress_mode"],
        "progress_evidence": mvf["selected_prev"] is not None,
        "lam_scale": mvf["lam_scale"],
    }
    return extras


def _broadcast_selection(selected, *, epoch=0, output_dir=None, device=None):
    """File-poll selection share — no inter-write/read NCCL barrier.

    Every rank takes the same code path:
      - rank 0 atomically writes the indices, then atomically touches a
        `.ready` sentinel.
      - all other ranks poll for the sentinel on disk and read once it
        exists. They never call a collective while rank 0 is busy.
    A single dist.barrier() at the very end keeps the SFT phase in step
    even if some worker reads a few milliseconds before rank 0 exits its
    write — and it always completes immediately because everyone has
    already converged here.
    """
    r = dist.get_rank() if dist.is_initialized() else 0

    if not dist.is_initialized():
        if hasattr(selected, "tolist"):
            return selected.tolist()
        return list(selected) if not isinstance(selected, list) else selected

    SRC = 0
    base = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    sel_path = base / f"_selection_epoch{epoch}.json"
    ready_path = base / f"_selection_epoch{epoch}.ready"
    ready_tmp = base / f"_selection_epoch{epoch}.ready.tmp"

    if r == SRC:
        # Clean up any stale sentinel from a previous run before writing.
        # Also sweep PRIOR epochs' broadcast files (epoch-1, epoch-2, ...) —
        # we deferred cleanup of those from each epoch's exit (see the
        # NOTE at the bottom of this function) to avoid racing workers
        # that hadn't finished reading the broadcast yet. By the time
        # we re-enter for the next epoch, every worker has definitely
        # moved past the read, so the prior epoch's files are safe to
        # remove now. Limit the sweep to 4 prior epochs to keep the
        # syscall cost bounded.
        prior_stale = [ready_path, ready_tmp]
        for prior_epoch in range(max(0, epoch - 4), epoch):
            prior_stale.append(base / f"_selection_epoch{prior_epoch}.json")
            prior_stale.append(base / f"_selection_epoch{prior_epoch}.ready")
        for stale in prior_stale:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

        if hasattr(selected, "tolist"):
            selected = selected.tolist()
        elif not isinstance(selected, list):
            selected = list(selected)
        selected = [int(x) for x in selected]
        logger.info(
            "[sel-share] rank=0 normalized selection | len=%d | first5=%s",
            len(selected), selected[:5],
        )

        # 1) atomic write of the selection itself.
        tmp_path = sel_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(selected, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, sel_path)

        # 2) atomic write of the `.ready` sentinel — workers only start
        # reading once this exists, so we never race a half-written
        # selection file.
        with open(ready_tmp, "w") as f:
            f.write(str(epoch))
            f.flush()
            os.fsync(f.fileno())
        os.replace(ready_tmp, ready_path)

        logger.info(
            "[sel-share] rank=0 WROTE %s (%d ids) + ready sentinel",
            sel_path, len(selected),
        )
        result = selected
    else:
        # Workers poll on disk for the ready sentinel. NO NCCL collective is
        # called in this loop — earlier heartbeat experiments introduced a
        # race where the worker's all_reduce could fire AFTER rank 0 had
        # already left _broadcast_selection (rank 0's matching call lives
        # inside collect_episode, not after), and the unmatched collective
        # would then hang forever.
        # NCCL idle protection relies entirely on TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
        # being raised (set in train.main's environment); the collective
        # itself never fires here.
        t_start = time.time()
        last_log = t_start
        while not ready_path.exists():
            now = time.time()
            if now - t_start > _POLL_TIMEOUT_SEC:
                raise RuntimeError(
                    f"[rank {r}] timed out after "
                    f"{int(now - t_start)}s waiting for "
                    f"{ready_path}. Rank 0 likely crashed before writing.",
                )
            if now - last_log > 60.0:
                logger.info(
                    "[sel-share] rank=%d polling for selection (%.0fs elapsed)",
                    r, now - t_start,
                )
                last_log = now
            time.sleep(_POLL_INTERVAL_SEC)

        if not sel_path.exists():
            raise RuntimeError(
                f"[rank {r}] ready sentinel present but selection file "
                f"{sel_path} missing.",
            )
        with open(sel_path, "r") as f:
            result = json.load(f)
        if not isinstance(result, list):
            raise RuntimeError(
                f"[rank {r}] selection file has wrong shape: "
                f"type={type(result).__name__}",
            )
        logger.info(
            "[sel-share] rank=%d READ %s len=%d", r, sel_path, len(result),
        )

    # NO dist.barrier here. After rank 0's 30+ minute solo collect_episode
    # the NCCL communicator can be in a state where the next collective
    # hangs even when every rank reaches it — the communicator's
    # background socket has effectively died. Removing the barrier means
    # ranks proceed straight to SFT, and the very first DDP all_reduce
    # inside backward() doubles as the alignment point.
    print(
        f"[sel-share] rank={r} EXIT _broadcast_selection (no barrier)",
        flush=True,
    )

    # NOTE: we used to unlink sel_path + ready_path here on rank 0, but that
    # raced with workers reading the same files — rank 0's cleanup could fire
    # in the ~1 ms between the worker's `ready_path.exists()` check and its
    # subsequent `sel_path.exists()` / json.load(), surfacing as
    # "ready sentinel present but selection file missing" or a JSONDecodeError
    # several minutes into SFT. Without a barrier (intentionally removed; see
    # comment above) we can't safely cleanup until everyone has moved on.
    # The next epoch's entry sweeps prior epochs' files instead.

    return result


def _anchor_stability(anchor, prev_v) -> float:
    """Eigengap-weighted mean |cos| between the previous and current anchor
    directions — the per-refresh reliability of the Geometry view.

    Used for quality-adaptive fusion (plan §2.4 v3): λ_t = λ0 · stability_t,
    so the fusion trusts the alignment view only insofar as its anchor is
    stable between refreshes. |cos| because a PCA direction's sign is
    arbitrary. Falls back to 1.0 (no discount) when there is nothing to
    compare.
    """
    gaps = getattr(anchor, "gap_by_layer", None)
    if not isinstance(gaps, dict):
        gaps = {}
    cosines, weights = [], []
    for li, v_new in getattr(anchor, "v_by_layer", {}).items():
        v_old = prev_v.get(li)
        if v_old is None:
            continue
        c = float(torch.abs(torch.nn.functional.cosine_similarity(
            v_new.detach().float().view(1, -1),
            v_old.detach().float().view(1, -1),
        )).item())
        cosines.append(min(max(c, 0.0), 1.0))
        weights.append(max(float(gaps.get(li, 1.0)), 0.0))
    if not cosines:
        return 1.0
    total_w = sum(weights)
    if total_w <= 0:
        return sum(cosines) / len(cosines)
    return sum(c * w for c, w in zip(cosines, weights)) / total_w


def select_indices(
    method, *, model, anchor, dataset, cfg, epoch, seed, device, mvf_ctx=None,
    tag_ctx=None,
):
    """Return (selected_indices, extras) for the given epoch.

    ``mvf_ctx`` — optional context for the multi-view-fusion score
    (built by ``tag.train`` when ``selection.score_mode == "mvf"``):
    ``{"completeness": (N,) tensor, "cf_dataset(s)": Dataset(s) | None,
    "cluster_ids": list[int] | None, "params": {eta, gamma, eps, d_floor,
    progress_mode, static, adaptive_lam, reliability_*}}``.
    None keeps the legacy scoring path untouched.

    ``tag_ctx`` — optional context for the TAG score (built when
    ``selection.score_mode == "tag"``): ``{"completeness": (N,) tensor,
    "dataset": Dataset, "cf_datasets": [Dataset, ...], "cluster_ids":
    list[int] | None, "eos_token_id": int, "params": {span_tokens, tau,
    tail_mode, gate_scale, ...}}``. Mutually exclusive with ``mvf_ctx``.
    """
    n_total = len(dataset)
    ratio = float(cfg["selection_ratio"])
    extras = {}

    if mvf_ctx is not None and tag_ctx is not None:
        raise ValueError(
            "select_indices: mvf_ctx and tag_ctx are mutually exclusive — "
            "selection.score_mode selects exactly one of 'mvf' / 'tag'.",
        )

    if method == "full":
        selected = list(range(n_total))
        logger.info("Full dataset selection | k=%d", len(selected))
        return selected, extras

    if method == "random":
        selected = _random_indices(n_total, ratio, seed, epoch)
        logger.info("Random selection | k=%d/%d", len(selected), n_total)
        return selected, extras

    _BASELINE_METHODS = {
        "data_agent", "lima", "nait", "selectit", "alpagasus", "q2q",
    }
    if method in _BASELINE_METHODS:
        raise ValueError(
            f"method={method!r} is a comparison baseline — `tag.train` only "
            f"handles random / full / tag.\n"
            f"Use the dedicated entrypoint instead:\n"
            f"    python -m baselines.{method}.train \\\n"
            f"        --config <experiment_yaml> --tag <variant_tag>\n"
            f"See baselines/{method}/train.py docstring for the exact "
            f"command + any required env vars (e.g. ALPAGASUS_FILTERED_FILE, "
            f"LIMA_DATA_FILES)."
        )
    if method != "selection":
        raise ValueError(
            f"Unknown method: {method!r}. Valid in `tag.train`: random, "
            f"full, tag. Baseline methods (data_agent/lima/nait/selectit/"
            f"alpagasus/q2q) have their own entrypoints in baselines/."
        )

    # ---------- selection cache: skip collect_episode if a prior run
    # ---------- already produced selected_indices_epoch{N}.json
    # collect_episode for method=selection takes 30+ min on 7B. If a previous
    # run made it through scoring but hung in the post-broadcast NCCL step,
    # the indices already exist on disk and we can reuse them directly. This
    # path is ONLY hit when the file is present; a fresh start still runs
    # the full episode.
    _output_dir_raw = cfg.get("output_dir") or cfg.get("output_root")
    # The reuse shortcut returns BEFORE any gate/reliability computation, and
    # its key is the epoch number alone. For TAG that is a trap: reusing an
    # epoch-1 selection skips priming tag_gate_cache.pt, and epoch 2 then hits
    # the base-checkpoint hard error. Only take the shortcut when the gate
    # cache this mode depends on already exists.
    _skip_selection_cache = False
    if tag_ctx is not None and _output_dir_raw is not None:
        from ..core import gate as gatelib
        _shared = str(((tag_ctx or {}).get("params") or {}).get("gate_cache_file") or "").strip()
        _gc = Path(_shared) if _shared else gatelib.cache_path_for(_output_dir_raw)
        if not _gc.exists():
            _skip_selection_cache = True
            logger.info(
                "TAG: a cached selection for epoch %d exists but "
                "tag_gate_cache.pt does not — running the full episode so the "
                "gate cache is primed (reusing the selection would strand "
                "every later epoch on the base-checkpoint error).", epoch,
            )
    if _output_dir_raw is not None and not _skip_selection_cache:
        _cached_path = Path(_output_dir_raw) / f"selected_indices_epoch{epoch}.json"
        if _cached_path.exists():
            try:
                with open(_cached_path) as _f:
                    _cached = json.load(_f)
                if isinstance(_cached, list) and len(_cached) > 0:
                    logger.info(
                        "REUSING cached selection from %s (%d indices) — "
                        "skipping collect_episode for epoch %d.",
                        _cached_path, len(_cached), epoch,
                    )
                    # Broadcast the cached indices to all ranks via the same
                    # file-polling mechanism so workers also get them.
                    if is_main_process():
                        selected = [int(x) for x in _cached]
                    else:
                        selected = []
                    selected = _broadcast_selection(
                        selected, epoch=epoch,
                        output_dir=_output_dir_raw,
                        device=device,
                    )
                    extras["selection_cache_reused"] = True
                    return selected, extras
            except Exception as _e:
                logger.warning(
                    "Could not reuse cached selection at %s (%s); "
                    "running full collect_episode.",
                    _cached_path, _e,
                )

    # ---------- MVF-static arm: freeze the epoch-1 selection ----------
    # The static control (plan §5.1 arm 6) scores the pool ONCE at epoch 1
    # and reuses that selection for every later refresh — it isolates the
    # training-adaptive component (per-refresh D/A recomputation) from the
    # score design itself. Without this arm, "adaptive beats static" is
    # unfalsifiable.
    # `static` is a property of the ARM, not of the score design, so both
    # score modes honour it (TAG-static is the paper's adaptive-vs-static
    # control just as MVF-static was). `adaptive_lam` stays MVF-only: TAG's
    # Eq. 1 uses a fixed λ.
    _mvf_params = (mvf_ctx or {}).get("params") or {}
    _mode_params = _mvf_params or ((tag_ctx or {}).get("params") or {})
    if bool(_mode_params.get("static")) and epoch > 1:
        _static_key = "tag" if (tag_ctx is not None and not _mvf_params) else "mvf"
        if _output_dir_raw is None:
            raise ValueError(
                f"{_static_key}.static requires output_dir to locate the "
                f"epoch-1 selection."
            )
        _frozen_path = Path(_output_dir_raw) / "selected_indices_epoch1.json"
        if not _frozen_path.exists():
            raise RuntimeError(
                f"{_static_key}.static=true but {_frozen_path} does not exist — "
                f"the epoch-1 selection must complete (and be saved) first.",
            )
        with open(_frozen_path) as _f:
            _frozen = json.load(_f)
        if not isinstance(_frozen, list) or not _frozen:
            raise RuntimeError(
                f"{_static_key}.static: {_frozen_path} is empty or malformed."
            )
        logger.info(
            "%s-static: reusing frozen epoch-1 selection (%d indices) at "
            "epoch %d.", _static_key.upper(), len(_frozen), epoch,
        )
        selected = [int(x) for x in _frozen] if is_main_process() else []
        selected = _broadcast_selection(
            selected, epoch=epoch, output_dir=_output_dir_raw, device=device,
        )
        extras[f"{_static_key}_static_reuse"] = True
        return selected, extras

    if is_main_process():
        print("[trace] rank=0 ENTER main branch | method=" + method
              + " | anchor=" + ("set" if anchor is not None else "None"), flush=True)
        import traceback as _tb
        try:
            _lam_scale = 1.0
            if method == "selection" and anchor is not None:
                logger.info("Updating trajectory anchor ...")
                # Quality-adaptive fusion (plan §2.4 v3): snapshot the
                # anchor directions BEFORE the refresh so the post-update
                # drift can discount λ for this epoch.
                _adaptive_lam = bool(_mvf_params.get("adaptive_lam", False))
                _prev_v = None
                if _adaptive_lam and getattr(anchor, "is_fitted", False):
                    _prev_v = {
                        li: v.detach().clone()
                        for li, v in getattr(anchor, "v_by_layer", {}).items()
                    }
                print("[trace] rank=0 BEFORE anchor.update", flush=True)
                anchor_stats = anchor.update(
                    model=model, dataset=dataset, seed=seed, epoch=epoch,
                )
                _akeys = list(anchor_stats.keys()) if anchor_stats else None
                print("[trace] rank=0 AFTER anchor.update | stats_keys="
                      + str(_akeys), flush=True)
                extras["anchor_stats"] = anchor_stats
                if _prev_v:
                    _lam_scale = _anchor_stability(anchor, _prev_v)
                    logger.info(
                        "Adaptive λ: anchor stability=%.4f → λ_t = λ0 · %.4f",
                        _lam_scale, _lam_scale,
                    )
                    extras["anchor_stability"] = _lam_scale

            selection_cfg = cfg.get("selection", {}) or {}
            exp_tag = str(cfg.get("model_key", "?")) + "/alpaca/" + method

            mvf = None
            if mvf_ctx is not None:
                mvf_ctx["lam_scale"] = _lam_scale
                mvf = _prepare_mvf(
                    mvf_ctx, model=model, cfg=cfg, epoch=epoch,
                    device=device, n_pool=n_total,
                )

            tag = None
            if tag_ctx is not None:
                tag_ctx.setdefault("dataset", dataset)
                tag = _prepare_tag(
                    tag_ctx, model=model, cfg=cfg, epoch=epoch,
                    device=device, n_pool=n_total,
                )

            print("[trace] rank=0 BEFORE collect_episode", flush=True)
            episode = collect_episode(
                model=model,
                dataset=dataset,
                selection_ratio=ratio,
                trajectory_anchor=anchor if method == "selection" else None,
                lam=float(selection_cfg.get("lam", 0.0)),
                use_anchor=bool(selection_cfg.get("use_anchor", False)) and method == "selection",
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(device),
                seed=seed,
                epoch=epoch,
                exp_tag=exp_tag,
                mvf=mvf,
                tag=tag,
            )
            print("[trace] rank=0 AFTER collect_episode | episode_keys="
                  + str(list(episode.keys())), flush=True)
            selected = episode["selected_indices"]
            _slen = len(selected) if hasattr(selected, "__len__") else "?"
            print("[trace] rank=0 selected=" + type(selected).__name__
                  + " len=" + str(_slen), flush=True)

            extras.update({
                "r_loss_mean": episode["r_loss_mean"],
                "r_entropy_mean": episode["r_entropy_mean"],
                "r_weight": episode["r_weight"],
                "rdiff_mean": episode["rdiff_mean"],
                "rconf_mean": episode["rconf_mean"],
                "lam": episode["lam"],
                "use_anchor": episode["use_anchor"],
                "align_mean": episode["align_mean"],
                "align_std": episode["align_std"],
            })
            if mvf is not None:
                extras.update(
                    _finalize_mvf(mvf, episode, cfg=cfg, epoch=epoch),
                )
            if tag is not None:
                extras.update(
                    _finalize_tag(tag, episode, cfg=cfg, epoch=epoch),
                )
        except Exception as _e:
            print("[trace] rank=0 EXCEPTION in main branch: "
                  + type(_e).__name__ + ": " + str(_e), flush=True)
            _tb.print_exc()
            import sys as _sys
            _sys.stdout.flush()
            _sys.stderr.flush()
            raise
    else:
        selected = []

    _output_dir = (
        cfg.get("output_dir")
        or cfg.get("output_root")
        or "/tmp/tag_selection_share"
    )
    selected = _broadcast_selection(
        selected, epoch=epoch, output_dir=_output_dir, device=device,
    )
    return selected, extras


def save_selection(output_dir, epoch, selected):
    """Persist the per-epoch selection for resume-time cache reuse.

    Atomic tmp + fsync + rename so a crash mid-write does not leave a
    truncated JSON behind — the cache-reuse path in ``select_indices``
    would otherwise hit json.JSONDecodeError on the next run and fall
    back to the 30-min collect_episode unnecessarily.
    """
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"selected_indices_epoch{epoch}.json"
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(selected, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
