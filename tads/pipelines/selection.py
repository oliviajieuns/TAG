"""Per-epoch sample selection dispatch.

Wraps the three selection methods (random / full / tads) handled by
``tads.train``. Comparison baselines (data_agent / nait / selectit /
lima / alpagasus / q2q) have their own entrypoints under
``baselines.<method>.train`` and bypass this dispatcher.

For ``method=tads`` the heavy collect_episode runs on rank 0 only;
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

    - Reliability Q: loaded from the run's cache when present; otherwise
      the counterfactual pool loss is computed here (one forward pass) and
      Q is derived inside collect_episode from this epoch's pool loss —
      then persisted by :func:`_finalize_mvf`.
    - Learnability: previous refresh's loss vector, if this run saved one.
      (A refresh that reused a cached selection skips collect_episode and
      saves no history — the next epoch then falls back to rank(L) only.)
    """
    from ..core import reliability as rel

    output_dir = cfg["output_dir"]
    params = mvf_ctx.get("params", {}) or {}
    mvf: Dict[str, Any] = {
        "completeness": mvf_ctx["completeness"],
        "cluster_ids": mvf_ctx.get("cluster_ids"),
        "eta": float(params.get("eta", 0.5)),
        "gamma": float(params.get("gamma", 1.0)),
        "eps": float(params.get("eps", 0.01)),
        "reliability": None,
        "loss_cf": None,
        "loss_prev": None,
    }

    cache = rel.load_reliability_cache(output_dir)
    if cache is not None and cache["q"].numel() == n_pool:
        mvf["reliability"] = cache["q"]
    else:
        if cache is not None:
            logger.warning(
                "Reliability cache size %d != pool size %d — recomputing.",
                cache["q"].numel(), n_pool,
            )
        cf_dataset = mvf_ctx.get("cf_dataset")
        if cf_dataset is None:
            raise ValueError(
                "MVF score_mode requires a counterfactual pool: set "
                "tads.mvf.counterfactual_data_files (generate it with "
                "scripts/make_corrupted_pool.py --emit-counterfactual).",
            )
        if len(cf_dataset) != n_pool:
            raise ValueError(
                f"Counterfactual pool size {len(cf_dataset)} != candidate "
                f"pool size {n_pool} — pools must be index-aligned.",
            )
        if epoch > 1:
            logger.warning(
                "Reliability is being computed at epoch %d (not the base "
                "checkpoint) — resuming without reliability_cache.pt? Q will "
                "reflect the current checkpoint.", epoch,
            )
        mvf["loss_cf"] = rel.compute_pool_loss(
            model, cf_dataset,
            batch_size=int(cfg.get("episode_batch_size", 1)),
            device=str(device),
            tag="counterfactual",
        )

    mvf["loss_prev"] = _load_loss_history(output_dir, epoch - 1)
    if mvf["loss_prev"] is not None and mvf["loss_prev"].numel() != n_pool:
        logger.warning(
            "Loss history size %d != pool size %d — ignoring history.",
            mvf["loss_prev"].numel(), n_pool,
        )
        mvf["loss_prev"] = None
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
        )
    extras: Dict[str, Any] = {
        "score_mode": "mvf",
        "q_mean": float(episode["reliability"].mean().item()),
        "d_mean": float(episode["difficulty"].mean().item()),
        "completeness_mean": float(mvf["completeness"].float().mean().item()),
        "progress_active": mvf["loss_prev"] is not None,
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


def select_indices(
    method, *, model, anchor, dataset, cfg, epoch, seed, device, mvf_ctx=None,
):
    """Return (selected_indices, extras) for the given epoch.

    ``mvf_ctx`` — optional context for the multi-view-fusion score
    (built by ``tads.train`` when ``tads.score_mode == "mvf"``):
    ``{"completeness": (N,) tensor, "cf_dataset": Dataset | None,
    "cluster_ids": list[int] | None, "params": {eta, gamma, eps}}``.
    None keeps the legacy scoring path untouched.
    """
    n_total = len(dataset)
    ratio = float(cfg["selection_ratio"])
    extras = {}

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
            f"method={method!r} is a comparison baseline — `tads.train` only "
            f"handles random / full / tads.\n"
            f"Use the dedicated entrypoint instead:\n"
            f"    python -m baselines.{method}.train \\\n"
            f"        --config <experiment_yaml> --tag <variant_tag>\n"
            f"See baselines/{method}/train.py docstring for the exact "
            f"command + any required env vars (e.g. ALPAGASUS_FILTERED_FILE, "
            f"LIMA_DATA_FILES)."
        )
    if method != "tads":
        raise ValueError(
            f"Unknown method: {method!r}. Valid in `tads.train`: random, "
            f"full, tads. Baseline methods (data_agent/lima/nait/selectit/"
            f"alpagasus/q2q) have their own entrypoints in baselines/."
        )

    # ---------- selection cache: skip collect_episode if a prior run
    # ---------- already produced selected_indices_epoch{N}.json
    # collect_episode for tads takes 30+ min on 7B. If a previous
    # run made it through scoring but hung in the post-broadcast NCCL step,
    # the indices already exist on disk and we can reuse them directly. This
    # path is ONLY hit when the file is present; a fresh start still runs
    # the full episode.
    _output_dir_raw = cfg.get("output_dir") or cfg.get("output_root")
    if _output_dir_raw is not None:
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

    if is_main_process():
        print("[trace] rank=0 ENTER main branch | method=" + method
              + " | anchor=" + ("set" if anchor is not None else "None"), flush=True)
        import traceback as _tb
        try:
            if method == "tads" and anchor is not None:
                logger.info("Updating trajectory anchor ...")
                print("[trace] rank=0 BEFORE anchor.update", flush=True)
                anchor_stats = anchor.update(
                    model=model, dataset=dataset, seed=seed, epoch=epoch,
                )
                _akeys = list(anchor_stats.keys()) if anchor_stats else None
                print("[trace] rank=0 AFTER anchor.update | stats_keys="
                      + str(_akeys), flush=True)
                extras["anchor_stats"] = anchor_stats

            tads_cfg = cfg.get("tads", {}) or {}
            exp_tag = str(cfg.get("model_key", "?")) + "/alpaca/" + method

            mvf = None
            if mvf_ctx is not None:
                mvf = _prepare_mvf(
                    mvf_ctx, model=model, cfg=cfg, epoch=epoch,
                    device=device, n_pool=n_total,
                )

            print("[trace] rank=0 BEFORE collect_episode", flush=True)
            episode = collect_episode(
                model=model,
                dataset=dataset,
                selection_ratio=ratio,
                trajectory_anchor=anchor if method == "tads" else None,
                lam=float(tads_cfg.get("lam", 0.0)),
                use_anchor=bool(tads_cfg.get("use_anchor", False)) and method == "tads",
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(device),
                seed=seed,
                epoch=epoch,
                exp_tag=exp_tag,
                mvf=mvf,
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
        or "/tmp/tads_selection_share"
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
