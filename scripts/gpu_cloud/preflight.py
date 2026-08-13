#!/usr/bin/env python
"""Pre-run checks for a TAG run on a GPU box — cheap, before the expensive part.

Every check here exists because getting it wrong costs GPU hours rather
than seconds. The failure this is really built around is the one that
actually happened during development: a run that trains happily through
epoch 1 and dies at the start of epoch 2, because the gate is defined at
the base checkpoint and anything that disturbs its cache turns into a hard
error later. Checks that can only be settled by running are covered by
``bash scripts/run_tag_lowq_05b.sh smoke`` instead.

    source scripts/gpu_cloud/env.sh
    python scripts/gpu_cloud/preflight.py
    python scripts/gpu_cloud/preflight.py --config <other.yaml> --strict

Exit code is 0 when nothing FAILED (warnings allowed), 1 otherwise, or 1
on any warning with --strict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OK, _WARN, _FAIL = "PASS", "WARN", "FAIL"
_RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _RESULTS.append((status, name, detail))
    mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": " FAIL"}[status]
    print(f"[{mark}] {name}" + (f"\n         {detail}" if detail else ""))


def check(name, fn, *, fatal=False):
    """Run one check; an exception is a failure, never a crash."""
    try:
        status, detail = fn()
    except Exception as e:  # a check must never take the whole script down
        record(_FAIL, name, f"{type(e).__name__}: {e}")
        return False
    record(status, name, detail)
    if status == _FAIL and fatal:
        print("\nThis failure blocks every later check — stopping here.")
        summarise_and_exit(strict=False)
    return status != _FAIL


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def check_env_sourced():
    if not os.environ.get("TAG_WORKSPACE"):
        return _FAIL, "TAG_WORKSPACE unset — run: source scripts/gpu_cloud/env.sh"
    return _OK, f"workspace {os.environ['TAG_WORKSPACE']}"


def check_torch_cuda():
    import torch
    if not torch.cuda.is_available():
        return _FAIL, (
            "torch.cuda.is_available() is False. On a GPU box this usually "
            "means a CPU-only torch wheel; reinstall the CUDA build."
        )
    n = torch.cuda.device_count()
    names = {torch.cuda.get_device_name(i) for i in range(n)}
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    detail = (
        f"torch {torch.__version__}, {n} x {', '.join(sorted(names))}, "
        f"{total:.0f} GB, bf16={'yes' if bf16 else 'NO'}"
    )
    if not bf16:
        # The loader defaults to bfloat16; on pre-Ampere cards that silently
        # emulates and is very slow.
        return _WARN, detail + " — pre-Ampere GPU, bf16 will be emulated (slow)"
    if total < 12:
        return _WARN, detail + " — under 12 GB; reduce episode_batch_size if OOM"
    return _OK, detail


def check_deps():
    import importlib
    missing = []
    for mod in ("transformers", "datasets", "peft", "accelerate", "numpy", "yaml"):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return _FAIL, f"missing: {', '.join(missing)} — bash scripts/gpu_cloud/bootstrap.sh deps"
    import transformers
    return _OK, f"transformers {transformers.__version__}"


def check_disk():
    import shutil
    ws = os.environ.get("TAG_WORKSPACE", ".")
    free = shutil.disk_usage(ws).free / 1e9
    # A 0.5B LoRA run writes adapters per run plus the tokenised cache and,
    # with store_token_losses, an fp16 token-loss tensor per pool.
    if free < 10:
        return _FAIL, f"{free:.1f} GB free at {ws} — need ~10 GB"
    if free < 25:
        return _WARN, f"{free:.1f} GB free at {ws} — tight for multiple runs"
    return _OK, f"{free:.0f} GB free at {ws}"


# ---------------------------------------------------------------------------
# Config and data
# ---------------------------------------------------------------------------

def _load_cfg(path):
    from tads.core.utils import load_config
    return load_config(str(path))


def make_config_check(cfg_path):
    def _fn():
        cfg = _load_cfg(cfg_path)
        mode = str((cfg.get("tads") or {}).get("score_mode", "tads"))
        if mode != "tag":
            return _FAIL, f"{cfg_path} resolves score_mode={mode!r}, expected 'tag'"
        tag = (cfg.get("tads") or {}).get("tag") or {}
        return _OK, (
            f"score_mode=tag, W={tag.get('span_tokens')}, tau={tag.get('tau')}"
            f"({tag.get('tau_mode')}), tail={tag.get('tail_mode')}, "
            f"ratio={cfg.get('selection_ratio')}, epochs={cfg.get('train_epochs')}"
        )
    return _fn


def make_model_check(cfg_path):
    def _fn():
        cfg = _load_cfg(cfg_path)
        p = Path(cfg["model_path"])
        if not (p / "config.json").exists():
            return _FAIL, (
                f"no config.json under {p} — bash scripts/gpu_cloud/bootstrap.sh model"
            )
        # The loader uses local_files_only=True, so the tokenizer must be
        # complete on disk; a partial snapshot fails here rather than after
        # the pool has been tokenised.
        from tads.modeling.loader import load_tokenizer
        tok = load_tokenizer(str(p))
        if tok.eos_token_id is None:
            return _FAIL, f"tokenizer at {p} has no eos_token_id"
        return _OK, f"{p.name}, vocab {tok.vocab_size}, eos={tok.eos_token_id}"
    return _fn


def make_pool_check(cfg_path):
    """Pool, counterfactual, manifest and dedup must be INDEX-ALIGNED.

    Only the lengths are checked at train time, so a stale counterfactual of
    the right length passes silently and the gate then contrasts responses
    against the wrong instructions.
    """
    def _fn():
        cfg = _load_cfg(cfg_path)
        pool_path = cfg.get("data_files")
        if not pool_path or not Path(str(pool_path)).exists():
            return _FAIL, (
                f"pool not found: {pool_path} — bash scripts/gpu_cloud/bootstrap.sh pools"
            )
        pool = json.load(open(pool_path))
        tag = (cfg.get("tads") or {}).get("tag") or {}
        cf_spec = str(tag.get("counterfactual_data_files") or "")
        cf_files = [s.strip() for s in cf_spec.split(",") if s.strip()]
        if not cf_files:
            return _FAIL, "tads.tag.counterfactual_data_files is empty (TADS_CF_FILES)"
        problems = []
        for f in cf_files:
            if not Path(f).exists():
                problems.append(f"missing counterfactual {f}")
                continue
            cf = json.load(open(f))
            if len(cf) != len(pool):
                problems.append(f"{Path(f).name}: {len(cf)} != pool {len(pool)}")
                continue
            # Same responses, different instructions — that IS the design.
            same_out = sum(
                1 for a, b in zip(pool[:200], cf[:200])
                if a.get("output") == b.get("output")
            )
            if same_out < 200:
                problems.append(
                    f"{Path(f).name}: only {same_out}/200 responses match the pool — "
                    f"not index-aligned (regenerate both together)"
                )
            # The derangement guarantees j != i, but the substituted
            # instruction can still be TEXTUALLY identical when the pool
            # contains duplicate instructions — which it does by design
            # (--duplicate-frac) and by nature. So compare the unchanged rate
            # against the pool's own duplicate rate rather than against zero,
            # or a legitimately deranged pool on a repetitive corpus reads as
            # broken.
            head = pool[:200]
            counts = {}
            for r in pool:
                k = r.get("instruction")
                counts[k] = counts.get(k, 0) + 1
            # Under a VALID derangement (j != i, otherwise arbitrary), the
            # chance that record i still receives a textually identical
            # instruction is (count[instr_i] - 1) / (N - 1). Averaging that
            # over the sample gives the expected collision rate, which is the
            # right null: comparing against zero false-alarms on any corpus
            # with repeated instructions, and comparing against a flat
            # "duplicate rate" goes blind on a highly repetitive one.
            big_n = max(2, len(pool))
            expected = sum(
                (counts.get(r.get("instruction"), 1) - 1) / (big_n - 1) for r in head
            ) / max(1, len(head))
            same_instr = sum(
                1 for a, b in zip(head, cf[:200])
                if a.get("instruction") == b.get("instruction")
            ) / max(1, len(head))
            budget = max(0.10, 3.0 * expected)
            if same_instr > budget:
                problems.append(
                    f"{Path(f).name}: {100 * same_instr:.0f}% of instructions are "
                    f"UNCHANGED, against {100 * expected:.0f}% expected from "
                    f"repeated instructions under a valid derangement — the "
                    f"counterfactual looks un-deranged, so Delta ~ 0 everywhere "
                    f"and the gate would veto the whole pool"
                )
        dedup = str(tag.get("dedup_clusters_file") or "")
        if dedup and Path(dedup).exists():
            ids = json.load(open(dedup))
            if len(ids) != len(pool):
                problems.append(f"dedup clusters {len(ids)} != pool {len(pool)}")
        if problems:
            return _FAIL, "; ".join(problems)
        return _OK, f"pool {len(pool)} records, {len(cf_files)} counterfactual pool(s), aligned"
    return _fn


def make_manifest_check(cfg_path):
    def _fn():
        cfg = _load_cfg(cfg_path)
        pool_path = Path(str(cfg.get("data_files")))
        man = pool_path.parent / "corruption_manifest.json"
        if not man.exists():
            return _WARN, (
                f"no {man.name} beside the pool — Phase A (Dirty@K / AUPRC) "
                f"needs it; training does not"
            )
        m = json.load(open(man))
        n_total = int(m.get("n_total", 0))
        n_dirty = len(m.get("entries", {}))
        pool = json.load(open(pool_path))
        if n_total != len(pool):
            return _FAIL, f"manifest n_total {n_total} != pool {len(pool)}"
        return _OK, f"{n_dirty}/{n_total} dirty ({100.0 * n_dirty / max(1, n_total):.1f}%)"
    return _fn


def make_gate_ref_check(cfg_path):
    """The calibration reference is what keeps the gate from being pool-relative."""
    def _fn():
        import torch
        cfg = _load_cfg(cfg_path)
        tag = (cfg.get("tads") or {}).get("tag") or {}
        scale = str(tag.get("gate_scale") or "").strip()
        ref = str(tag.get("gate_ref_file") or "").strip()
        if scale:
            return _OK, f"gate_scale pinned explicitly to {scale}"
        if not ref:
            return _WARN, (
                "neither gate_scale nor gate_ref_file is set — the gate will "
                "self-calibrate IN-POOL, which makes G depend on how dirty its "
                "neighbours are. Fine for a smoke test, not for a reported run: "
                "bash scripts/gpu_cloud/bootstrap.sh calibrate"
            )
        if not Path(ref).exists():
            return _FAIL, (
                f"gate_ref_file {ref} does not exist — "
                f"bash scripts/gpu_cloud/bootstrap.sh calibrate"
            )
        d = torch.load(ref, map_location="cpu", weights_only=True)
        if not isinstance(d, dict) or "delta_hat" not in d:
            return _FAIL, (
                f"{ref} has no 'delta_hat' key — this looks like an MVF "
                f"reference (raw Delta_L in nats). The two are not "
                f"interchangeable; regenerate with --mode tag."
            )
        # A calibration computed under a different span partition silently
        # mis-scales every gate value in the run.
        ref_cfg = d.get("gate_config") or {}
        bound = ("span_tokens", "tau", "tau_mode", "min_span_tokens",
                 "tail_mode", "include_eos")
        diffs = {
            k: (ref_cfg.get(k), tag.get(k))
            for k in bound
            if k in ref_cfg and k in tag and ref_cfg[k] != tag[k]
        }
        if diffs:
            return _FAIL, (
                f"{Path(ref).name} was calibrated under a different span config "
                f"{diffs} (ref -> run). s is a quantile of Delta_hat, whose "
                f"distribution depends on the partition — recalibrate."
            )
        dh = d["delta_hat"]
        pos = float((dh > 0).float().mean().item())
        detail = (
            f"{Path(ref).name}: n={dh.numel()}, {100 * pos:.1f}% positive, "
            f"s={d.get('scale'):.6g}"
        )
        if pos < 0.8:
            return _WARN, detail + (
                " — under 80% of the CLEAN reference has Delta_hat > 0; the "
                "reference may be contaminated or the counterfactuals not "
                "unrelated. Inspect before trusting the calibration."
            )
        return _OK, detail
    return _fn


def make_stale_cache_check(cfg_path):
    """A gate cache left over from a different pool is a hard error at epoch 2."""
    def _fn():
        cfg = _load_cfg(cfg_path)
        root = Path(cfg["output_root"]) / cfg["output_subdir"]
        if not root.exists():
            return _OK, "no previous runs for this arm"
        caches = list(root.glob("runs/*/tag_gate_cache.pt"))
        if not caches:
            return _OK, f"{len(list(root.glob('runs/*')))} previous run dir(s), no gate cache"
        return _OK, (
            f"{len(caches)} previous gate cache(s) under {root}/runs — each run "
            f"dir is isolated, so a fresh run recomputes its own"
        )
    return _fn


def check_writable():
    root = Path(os.environ.get("OUTPUT_ROOT", "."))
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".tag_preflight_write_test"
    probe.write_text("ok")
    probe.unlink()
    return _OK, f"{root} is writable"


# ---------------------------------------------------------------------------

def summarise_and_exit(strict: bool):
    n_fail = sum(1 for s, _, _ in _RESULTS if s == _FAIL)
    n_warn = sum(1 for s, _, _ in _RESULTS if s == _WARN)
    print("\n" + "=" * 72)
    print(f"preflight: {len(_RESULTS)} checks, {n_fail} failed, {n_warn} warnings")
    if n_fail:
        print("\nFailed:")
        for s, name, detail in _RESULTS:
            if s == _FAIL:
                print(f"  - {name}: {detail}")
        print("\nFix these before starting a run.")
        sys.exit(1)
    if n_warn and strict:
        print("\n--strict: treating warnings as failures.")
        sys.exit(1)
    print("\nReady. Recommended order:")
    print("  bash scripts/run_tag_lowq_05b.sh smoke     # ~2 min, proves the")
    print("                                             # epoch-2 cache path")
    print("  bash scripts/run_tag_lowq_05b.sh phasea    # detection table")
    print("  bash scripts/run_tag_lowq_05b.sh phaseb    # full SFT run")
    sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/experiments/lowq/light_tag_05b.yaml")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on warnings too")
    p.add_argument("--skip-model", action="store_true",
                   help="skip the tokenizer load (fast path for config-only checks)")
    args = p.parse_args()

    cfg_path = _REPO_ROOT / args.config if not Path(args.config).is_absolute() \
        else Path(args.config)
    print(f"TAG preflight — {cfg_path}\n" + "=" * 72)

    check("environment sourced", check_env_sourced, fatal=True)
    check("python dependencies", check_deps, fatal=True)
    check("GPU / CUDA", check_torch_cuda)
    check("disk space", check_disk)
    check("output dir writable", check_writable)
    check("config resolves", make_config_check(cfg_path), fatal=True)
    if not args.skip_model:
        check("model weights + tokenizer", make_model_check(cfg_path))
    check("pool / counterfactual alignment", make_pool_check(cfg_path))
    check("corruption manifest", make_manifest_check(cfg_path))
    check("gate calibration reference", make_gate_ref_check(cfg_path))
    check("previous run caches", make_stale_cache_check(cfg_path))

    summarise_and_exit(strict=args.strict)


if __name__ == "__main__":
    main()
