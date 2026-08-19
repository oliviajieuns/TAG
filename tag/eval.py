"""Evaluation entrypoint.

Usage:
    python -m tag.eval \\
        --config configs/experiments/light_legacy_05b.yaml \\
        --ckpt /path/to/epoch_last \\
        --benchmarks mmlu,gsm8k,humaneval,tydiqa \\
        --out_dir results/light_legacy_05b/

``tag.train`` writes ``epoch_last/`` only (final epoch); comparison
baselines under ``baselines.<method>`` write ``epoch_N/`` per
epoch — pass whichever the run dir actually contains.

The benchmark list is split by commas; each name is looked up in the
:mod:`tag.evals` registry. Data paths for each benchmark come from the
config (``<benchmark>_data_dir`` keys) or from ``--<benchmark>_data_dir``
on the CLI.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tag.core.utils import (
    clear_runtime_caches,
    disable_coredumps,
    load_config,
    quiet_repeated_warnings,
    setup_logger,
)
from tag.evals import get_evaluator, list_evaluators
from tag.modeling.loader import load_for_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config (model_path, prompt_style).")
    p.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Checkpoint directory (LoRA adapter or full). If omitted, resolves "
            "to <output_dir>/_latest/<epoch_last or largest epoch_N>; pass "
            "--epoch N to select a specific epoch_N within _latest (baselines "
            "only — tag.train only writes epoch_last)."
        ),
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number inside the resolved run dir to evaluate. Default: largest sealed.",
    )
    p.add_argument(
        "--run_tag",
        default=None,
        help=(
            "Run tag under <output_dir>/runs/ to evaluate. Mutually exclusive "
            "with --ckpt. If neither --ckpt nor --run_tag is given, falls "
            "back to the _latest pointer."
        ),
    )
    p.add_argument(
        "--list_runs",
        action="store_true",
        help="List runs/ history under <output_dir> and exit (no eval).",
    )
    p.add_argument(
        "--benchmarks", default="mmlu",
        help=f"Comma-separated. Registered: {list_evaluators()}",
    )
    p.add_argument(
        "--out_dir", required=False, default=None,
        help=(
            "BASE output dir for JSON summaries. Default: <ckpt>/eval/. "
            "Results land under <out_dir>/runs/<eval_tag>/ and <out_dir>/_latest "
            "is updated to point at this run. Pass --flat to disable the "
            "history layout and write directly into <out_dir>/."
        ),
    )
    p.add_argument(
        "--eval_tag",
        default=None,
        help=(
            "Folder name for this eval run under <out_dir>/runs/. Defaults to "
            "an auto timestamp YYYYMMDD_HHMMSS so a re-eval never overwrites "
            "prior results. Pass --eval_tag=latest to reuse whatever the "
            "<out_dir>/_latest pointer currently selects (useful for adding "
            "a benchmark to a recent run without spawning a new dated dir)."
        ),
    )
    p.add_argument(
        "--eval_suffix",
        default="",
        help=(
            "Optional suffix appended to the auto timestamp eval_tag, e.g. "
            "--eval_suffix=retry produces runs/20260516_180000_retry/. Ignored "
            "if --eval_tag is also given."
        ),
    )
    p.add_argument(
        "--list_eval_runs",
        action="store_true",
        help=(
            "Print the eval runs/ history under <out_dir> (resolved from "
            "--ckpt / --run_tag / _latest as usual) and exit without running "
            "any benchmark. Analogous to --list_runs for the train side."
        ),
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Disable the runs/<tag>/ + _latest history layout. Writes results "
            "directly into <out_dir>/, overwriting any prior eval there. Use "
            "only for one-off ad-hoc evals; the score-board agent assumes the "
            "history layout."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Per-benchmark sample cap.")
    p.add_argument(
        "--training_mode", default=None, choices=[None, "full", "lora"],
        help="Override checkpoint type detection.",
    )
    p.add_argument(
        "--cuda_device", type=int, default=0,
        help="CUDA device index for evaluation.",
    )
    # Per-benchmark data dir overrides.
    p.add_argument("--mmlu_data_dir", default=None)
    p.add_argument("--mmlu_pro_data_dir", default=None,
                   help="MMLU-Pro root containing test-*.parquet + validation-*.parquet.")
    p.add_argument("--gsm8k_data_dir", default=None)
    p.add_argument("--svamp_data_dir", default=None,
                   help="SVAMP root containing test-*.parquet.")
    p.add_argument("--humaneval_data_dir", default=None)
    p.add_argument("--mbpp_data_dir", default=None,
                   help="MBPP root containing sanitized/ (+ optional full/) subdirs.")
    p.add_argument("--tydiqa_data_dir", default=None)
    p.add_argument("--xquad_data_dir", default=None,
                   help="XQuAD root containing xquad.<lang>.json files.")
    p.add_argument("--bbh_data_dir", default=None,
                   help="BBH root containing per-task .json + cot-prompts/.")
    # lm_harness extras.
    p.add_argument("--harness_task", default="mmlu", help="lm_harness `task` kwarg.")
    p.add_argument("--lm_eval_path", default=None, help="PYTHONPATH addition for lm-eval-harness fork.")
    return p.parse_args()


def _data_dir_for(
    name: str,
    cli: Dict[str, Optional[str]],
    cfg: Dict[str, Any],
) -> Optional[str]:
    return cli.get(f"{name}_data_dir") or cfg.get(f"{name}_data_dir")


def main() -> None:
    # Cap RLIMIT_CORE on this process and forks — see tag.train.main for
    # rationale. Eval rarely segfaults but if it does (CUDA OOM, model
    # load mismatch) the dump is still ~250 GB worth of bf16 weights.
    disable_coredumps()

    # Clean GC / CUDA allocator / IPC-handle state before loading the
    # model. Eval routinely re-runs against the same checkpoint set
    # (auto_eval_7b_fullft.sh polls in a loop), and stale handles from
    # a crashed prior iteration can otherwise pin VRAM that the new
    # load_for_eval can't allocate.
    clear_runtime_caches()

    # OFFLINE BY DEFAULT — see tag.train.main for rationale.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    quiet_repeated_warnings()

    # Silence the per-generate() warnings that fire 1.3K–6.5K times per
    # benchmark and bury real progress lines. Defense-in-depth alongside the
    # generation_config null-out in load_for_eval — some transformers versions
    # also emit a top-level UserWarning via warnings.warn (not the dedup'd
    # logger path), so we filter by message pattern here.
    import warnings as _warnings
    _warnings.filterwarnings(
        "ignore", message=r".*max_new_tokens.*max_length.*",
    )
    _warnings.filterwarnings(
        "ignore", message=r".*do_sample.*set to `False`.*",
    )
    _warnings.filterwarnings(
        "ignore", message=r".*`temperature`.*will be ignored.*",
    )

    # Eval is designed to run on a single GPU. If invoked under torchrun
    # (which sets RANK / LOCAL_RANK / WORLD_SIZE), only the rank-0 process
    # should run the evaluation — otherwise every worker re-runs the full
    # benchmark and they collide on output files. Workers exit cleanly.
    #
    # The previous predicate `int(os.environ.get("RANK","0")) != 0` was
    # too aggressive: a stale RANK left in the shell (from a prior torchrun
    # session, or a SLURM env that aliases SLURM_PROCID → RANK) would
    # silently return without ever parsing args, giving the appearance
    # of "eval terminates immediately" with no log line.
    #
    # New gate: require BOTH a real torchrun signature (WORLD_SIZE > 1
    # AND LOCAL_RANK present) AND RANK != 0 before exiting. A noisy
    # log line on entry makes "silent immediate exit" impossible going
    # forward — if you see no log at all, the process is being killed
    # by something external (oom-killer, SIGTERM, etc.).
    _rank_env = os.environ.get("RANK")
    _local_rank_env = os.environ.get("LOCAL_RANK")
    _world_size_env = os.environ.get("WORLD_SIZE")
    print(
        f"[eval] entry | pid={os.getpid()} | RANK={_rank_env!r} | "
        f"LOCAL_RANK={_local_rank_env!r} | WORLD_SIZE={_world_size_env!r}",
        flush=True,
    )
    _is_torchrun_child = (
        _local_rank_env is not None
        and int(_world_size_env or "1") > 1
    )
    if _is_torchrun_child and int(_rank_env or "0") != 0:
        print(
            f"[eval] non-rank-0 torchrun worker (rank={_rank_env}) — exiting cleanly",
            flush=True,
        )
        return

    args = parse_args()
    cfg = load_config(args.config)

    # Resolve experiment_dir = OUTPUT_ROOT/output_subdir so we can find the
    # _latest pointer / runs/ history that the new run-layout produced.
    experiment_dir = Path(cfg["output_root"]) / cfg["output_subdir"]

    if args.list_runs:
        from tag.core.run_layout import list_runs as _list_runs
        from tag.core.run_layout import resolve_latest as _resolve_latest
        runs = _list_runs(experiment_dir)
        if not runs:
            print(f"No runs/ directory under {experiment_dir}.")
        else:
            latest = _resolve_latest(experiment_dir)
            latest_name = latest.name if latest is not None else "(unset)"
            print(f"Runs under {experiment_dir}:")
            for tag, _ in runs:
                marker = "  <- _latest" if (latest and tag == latest.name) else ""
                print(f"  {tag}{marker}")
            print(f"_latest -> {latest_name}")
        return

    # Ckpt resolution priority:
    #   1. --ckpt explicit (must exist)
    #   2. --run_tag → <experiment_dir>/runs/<run_tag>/<epoch>
    #   3. _latest pointer → <experiment_dir>/_latest/<epoch>
    if args.ckpt and args.run_tag:
        raise SystemExit("--ckpt and --run_tag are mutually exclusive.")
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"--ckpt {args.ckpt!r} does not exist.")
    else:
        from tag.core.run_layout import (
            find_latest_complete_epoch,
            resolve_latest,
            run_dir_for,
        )
        if args.run_tag:
            target_run = run_dir_for(experiment_dir, args.run_tag)
            if not target_run.is_dir():
                raise SystemExit(
                    f"--run_tag {args.run_tag!r} not found at {target_run}.",
                )
        else:
            target_run = resolve_latest(experiment_dir)
            if target_run is None:
                raise SystemExit(
                    f"No --ckpt / --run_tag given and no _latest pointer "
                    f"under {experiment_dir}. Train first or pass --ckpt.",
                )
        if args.epoch is not None:
            # New epoch_last/ layout: only one ckpt dir per run. Honor
            # --epoch=N only if it matches the sentinel-recorded epoch
            # of epoch_last (or a legacy epoch_<N>/ still exists).
            last_dir = target_run / "epoch_last"
            if (last_dir / "_complete").exists():
                try:
                    saved_n = int((last_dir / "_complete").read_text().strip() or 0)
                except (OSError, ValueError):
                    saved_n = 0
                if saved_n == args.epoch:
                    ckpt_path = last_dir
                elif (target_run / f"epoch_{args.epoch}").exists():
                    ckpt_path = target_run / f"epoch_{args.epoch}"  # legacy fall-back
                else:
                    raise SystemExit(
                        f"--epoch {args.epoch} requested but {target_run}/epoch_last "
                        f"records epoch={saved_n} and no legacy epoch_{args.epoch}/ "
                        f"directory exists. With the epoch_last/ layout there is "
                        f"only one ckpt per run — pass --epoch={saved_n} or omit "
                        f"--epoch to use it.",
                    )
            else:
                ckpt_path = target_run / f"epoch_{args.epoch}"
                if not ckpt_path.exists():
                    raise SystemExit(
                        f"epoch_{args.epoch} not found in {target_run}. "
                        f"Available: {sorted(p.name for p in target_run.glob('epoch_*'))}",
                    )
        else:
            n, ckpt_path = find_latest_complete_epoch(target_run)
            if ckpt_path is None:
                raise SystemExit(
                    f"No sealed checkpoint (epoch_last/ or epoch_N/) "
                    f"inside {target_run}.",
                )

    args.ckpt = str(ckpt_path)

    # ---------- output layout: history-preserving runs/<eval_tag>/ + _latest ----------
    # Mirrors the train-side layout (see tag/core/run_layout.py). Every eval
    # invocation gets its own dated subdir so re-evaluating the same epoch (or
    # the same eval pass with different `--limit`/data-dir overrides) NEVER
    # overwrites prior results. The _latest pointer is what every downstream
    # consumer reads — the auto-eval score-board agent, the user's own
    # `cat <out_dir>/_latest/<exp_label>-eval_summary.json` quick-check, etc.
    #
    # BASE out_dir resolution priority:
    #   1. --out_dir explicit
    #   2. <ckpt>/eval/   (default — keeps results bundled with the ckpt)
    if args.out_dir:
        base_out_dir = Path(args.out_dir)
    else:
        base_out_dir = ckpt_path / "eval"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # `--list_eval_runs` operates on the BASE dir and exits before any model
    # load. Place after base_out_dir resolution so it inherits the same
    # --out_dir / --ckpt fall-through.
    if args.list_eval_runs:
        from tag.core.run_layout import list_runs as _list_runs
        from tag.core.run_layout import resolve_latest as _resolve_latest
        runs = _list_runs(base_out_dir)
        if not runs:
            print(f"No eval runs/ directory under {base_out_dir}.")
        else:
            latest = _resolve_latest(base_out_dir)
            latest_name = latest.name if latest is not None else "(unset)"
            print(f"Eval runs under {base_out_dir}:")
            for tag, _ in runs:
                marker = "  <- _latest" if (latest and tag == latest.name) else ""
                print(f"  {tag}{marker}")
            print(f"_latest -> {latest_name}")
        return

    # Resolve the per-run output dir.
    from tag.core.run_layout import (
        make_run_tag,
        resolve_latest as _resolve_latest_eval,
        run_dir_for,
        update_latest,
    )
    if args.flat:
        # Legacy mode — write directly into base_out_dir. No history, no
        # _latest pointer. Reserved for one-off ad-hoc evals.
        out_dir = base_out_dir
        eval_tag: Optional[str] = None
    else:
        if args.eval_tag == "latest":
            latest_run = _resolve_latest_eval(base_out_dir)
            if latest_run is None:
                raise SystemExit(
                    f"--eval_tag=latest requested but no _latest pointer under "
                    f"{base_out_dir}. Run eval without --eval_tag first.",
                )
            eval_tag = latest_run.name
        elif args.eval_tag:
            eval_tag = args.eval_tag
        else:
            eval_tag = make_run_tag(args.eval_suffix)
        out_dir = run_dir_for(base_out_dir, eval_tag)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stamp provenance into the eval run dir. make_table_v2 pairs rows
        # by the seed in cfg.json, and the eval tag is a bare timestamp —
        # without this stamp a sealed eval run cannot say which training
        # seed produced it, and every row silently falls out of the paired
        # comparison. Seed source, most authoritative first: the training
        # run's own cfg.json (sitting one level above epoch_last/), then a
        # seedNN segment in the checkpoint path.
        _prov: Dict[str, Any] = {"ckpt": str(args.ckpt),
                                 "config": str(args.config)}
        _train_cfg: Dict[str, Any] = {}
        _train_cfg_path = Path(args.ckpt).parent / "cfg.json"
        if _train_cfg_path.is_file():
            try:
                with open(_train_cfg_path) as _f:
                    _train_cfg = json.load(_f) or {}
            except (OSError, ValueError) as _exc:
                print(f"[eval] unreadable training cfg.json "
                      f"({_train_cfg_path}): {_exc}", file=sys.stderr)
        _seed = _train_cfg.get("seed")
        if _seed is None:
            _m = re.search(r"seed(\d+)", str(args.ckpt))
            _seed = int(_m.group(1)) if _m else None
        if _seed is not None:
            _prov["seed"] = int(_seed)
        if _train_cfg.get("git_sha"):
            _prov["git_sha"] = _train_cfg["git_sha"]
        try:
            with open(out_dir / "cfg.json", "w") as _f:
                json.dump(_prov, _f, indent=2)
        except OSError as _exc:
            print(f"[eval] could not write provenance cfg.json: {_exc}",
                  file=sys.stderr)

    log_dir = out_dir / "logs"
    logger = setup_logger(str(log_dir), name="eval")
    if eval_tag is not None:
        logger.info(
            "Eval run layout | base=%s | eval_tag=%s | out_dir=%s",
            base_out_dir, eval_tag, out_dir,
        )
    else:
        logger.info(
            "Eval run layout | flat mode (no history) | out_dir=%s", out_dir,
        )

    # Output files used to be plain `mmlu.json` / `eval_summary.json` —
    # opaque once you copied a few of them into a shared results folder
    # and couldn't tell `eval_summary.json` from `legacy_10`'s vs
    # `data_agent_10`'s. Prefix every artifact with an experiment label
    # so a flat directory listing tells you which (model, method) the
    # numbers belong to. Source of truth, in order:
    #   1. cfg["experiment_name"]  — explicit override in YAML
    #   2. <config-parent-dir>_<config-stem>  — e.g.
    #      configs/experiments/main_7b/llama2/legacy_10.yaml
    #      → "llama2_legacy_10"
    #   3. <config-stem>  — last-resort for ad-hoc configs.
    _cfg_path = Path(args.config)
    _parent = _cfg_path.parent.name
    if cfg.get("experiment_name"):
        experiment_label = str(cfg["experiment_name"])
    elif _parent and _parent not in ("configs", ".", ""):
        experiment_label = f"{_parent}_{_cfg_path.stem}"
    else:
        experiment_label = _cfg_path.stem

    logger.info(
        "Eval start | exp=%s | ckpt=%s | base=%s",
        experiment_label, args.ckpt, cfg.get("model_path"),
    )

    benchmarks: List[str] = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks:
        raise SystemExit(
            f"--benchmarks must list at least one benchmark. "
            f"Available: {list_evaluators()}"
        )
    unknown = [b for b in benchmarks if b not in list_evaluators()]
    if unknown:
        raise SystemExit(f"Unknown benchmarks: {unknown}. Available: {list_evaluators()}")

    model, tokenizer, device = load_for_eval(
        base_model=cfg["model_path"],
        ckpt_dir=args.ckpt,
        training_mode=args.training_mode,
        device=f"cuda:{args.cuda_device}",
    )

    prompt_style = cfg.get("prompt_style", "alpaca_default")
    cli_paths = {
        "mmlu_data_dir": args.mmlu_data_dir,
        "mmlu_pro_data_dir": args.mmlu_pro_data_dir,
        "gsm8k_data_dir": args.gsm8k_data_dir,
        "svamp_data_dir": args.svamp_data_dir,
        "humaneval_data_dir": args.humaneval_data_dir,
        "mbpp_data_dir": args.mbpp_data_dir,
        "bbh_data_dir": args.bbh_data_dir,
        "tydiqa_data_dir": args.tydiqa_data_dir,
        "xquad_data_dir": args.xquad_data_dir,
    }

    summaries = []
    failures: List[Dict[str, str]] = []
    for bench in benchmarks:
        evaluator = get_evaluator(bench)
        output_file = out_dir / f"{experiment_label}-{bench}.json"
        kw: Dict[str, Any] = {}
        if bench == "lm_harness":
            kw.update(
                task=args.harness_task,
                base_model=cfg["model_path"],
                ckpt_dir=args.ckpt,
                training_mode=args.training_mode,
                lm_eval_path=args.lm_eval_path,
            )
        # Per-benchmark try/except: a single benchmark failure (missing data
        # dir, OOM during generation, corrupted parquet) used to abort the
        # entire eval and lose results for the other 3-4 benchmarks that
        # already finished. Capture the error and keep going so partial
        # metrics still land in eval_summary.json.
        try:
            summary = evaluator.evaluate(
                model, tokenizer, device,
                output_file=str(output_file),
                limit=args.limit,
                prompt_style=prompt_style,
                data_dir=_data_dir_for(bench, cli_paths, cfg),
                **kw,
            )
            summaries.append(summary)
        except Exception as e:
            logger.exception("Benchmark %s failed; continuing with remaining benchmarks", bench)
            failures.append({"benchmark": bench, "error": f"{type(e).__name__}: {e}"})
        finally:
            # Free transformer KV caches / generate buffers between
            # benchmarks. Without this the high-water mark accumulates
            # across the 5-bench sequence (HumanEval's n_samples=20 +
            # BBH's 3072-token prompts especially) and a 7B run that
            # would steady-state at ~30 GB peaks past 80 GB by the time
            # we hit the last benchmark.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "experiment": experiment_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt,
        "base_model": cfg.get("model_path"),
        "limit": args.limit,
        "prompt_style": prompt_style,
        "summaries": summaries,
        "failures": failures,
    }
    summary_path = out_dir / f"{experiment_label}-eval_summary.json"
    # Atomic write so a crash mid-dump doesn't leave a half-written JSON
    # behind that the score-board agent would then try to parse.
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(summary_tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(summary_tmp, summary_path)

    # ---------- seal the run + update _latest ----------
    # `_complete` sentinel is written only AFTER eval_summary.json lands on
    # disk atomically. If we crash before this point, the run dir survives
    # without a sentinel and the score-board agent's "is this eval done?"
    # check (sentinel + summary mtime) fails closed — no false "done" claim.
    # Partial bench failures are recorded inside payload["failures"] but
    # still count as a sealed run (the convention matches §0-5 / §0-6 of
    # AUTO_EVAL_AGENT.md, which treats per-bench failure as a Status-column
    # event, not an unsealed run).
    if eval_tag is not None:
        sentinel = out_dir / "_complete"
        sentinel_tmp = out_dir / "_complete.tmp"
        try:
            with open(sentinel_tmp, "w") as f:
                f.write(eval_tag)
                f.flush()
                os.fsync(f.fileno())
            os.replace(sentinel_tmp, sentinel)
            logger.info("Eval run sealed: %s", sentinel)
        except Exception as exc:
            logger.warning(
                "Could not write _complete sentinel (%s); _latest will not "
                "be updated and the next eval will land in a fresh run dir.",
                exc,
            )
        else:
            # A --limit run is a rehearsal, not a result: it scores a few
            # examples per task to prove the pipeline runs. make_table_v2
            # reads exactly the run `_latest` names, so letting a limited run
            # claim that pointer is how 216 examples end up in the table as
            # if they were 6,511. Seal it — it IS a complete run of what it
            # was asked to do — but leave `_latest` on the last full one.
            if args.limit is not None:
                logger.warning(
                    "Run used --limit %s, so _latest was NOT moved: this is "
                    "a rehearsal and must not be picked up as a table row. "
                    "The results are in %s.", args.limit, out_dir,
                )
            else:
                try:
                    mech = update_latest(base_out_dir, eval_tag)
                    logger.info(
                        "_latest -> runs/%s (%s) under %s",
                        eval_tag, mech, base_out_dir,
                    )
                except Exception as exc:
                    logger.warning("Failed to update _latest pointer: %s", exc)

    if failures:
        logger.warning(
            "Eval finished with %d/%d benchmark failure(s): %s",
            len(failures), len(benchmarks), [f["benchmark"] for f in failures],
        )
    logger.info("Eval done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
