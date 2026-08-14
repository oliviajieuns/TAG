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
    from tag.core.utils import load_config
    return load_config(str(path))


def make_config_check(cfg_path):
    def _fn():
        cfg = _load_cfg(cfg_path)
        mode = str((cfg.get("selection") or {}).get("score_mode", "legacy"))
        if mode != "tag":
            return _FAIL, f"{cfg_path} resolves score_mode={mode!r}, expected 'tag'"
        tag = (cfg.get("selection") or {}).get("tag") or {}
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
        from tag.modeling.loader import load_tokenizer
        tok = load_tokenizer(str(p))
        if tok.eos_token_id is None:
            return _FAIL, f"tokenizer at {p} has no eos_token_id"
        detail = f"{p.name}, vocab {tok.vocab_size}, eos={tok.eos_token_id}"
        if "instruct" in p.name.lower() or "-it" in p.name.lower():
            # Not a failure — on some clusters base weights simply are not
            # available — but it changes what the numbers mean twice over,
            # so it must not pass silently.
            return _WARN, detail + (
                " — this is an INSTRUCT checkpoint, not a base model. Two "
                "consequences: (a) the paper's setup is SFT from a base "
                "checkpoint, so these numbers are not directly comparable to "
                "base-model runs; (b) an instruction-tuned model follows "
                "instructions better, so Delta_hat separates clean from "
                "corrupted MORE easily than it would from base — the gate "
                "looks better than the base-model setting would show. State "
                "the deviation explicitly."
            )
        return _OK, detail
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
        tag = (cfg.get("selection") or {}).get("tag") or {}
        cf_spec = str(tag.get("counterfactual_data_files") or "")
        cf_files = [s.strip() for s in cf_spec.split(",") if s.strip()]
        if not cf_files:
            return _FAIL, "selection.tag.counterfactual_data_files is empty (TAG_CF_FILES)"
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
                    f"and the gate would zero the whole pool"
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


def make_seq_len_check(cfg_path):
    """How much of the pool does ``max_seq_len`` cut off, and what it costs.

    Budget truncation is not neutral for TAG, for two compounding reasons:

    1. ``c_i``. Tokenisation appends EOS to every response, and the
       completeness check reads the LABEL's last token. When the budget cuts
       the response, that EOS is what gets dropped — so a perfectly clean but
       long response is marked incomplete and takes ``c_trunc`` (0.2 by
       default), a 5x cut to its gate.
    2. It lands on long responses only, which are already the ones penalised
       by the order-statistic drift of Delta^min (docs/tag-paper-deltas.md
       B2). The two effects push the same samples down.

    So a high truncation rate turns the gate into a length filter through a
    second, independent route. Measured here on the raw text, cheaply, before
    any of it costs GPU time.
    """
    def _fn():
        cfg = _load_cfg(cfg_path)
        pool_path = Path(str(cfg.get("data_files")))
        if not pool_path.exists():
            return _WARN, "pool missing; run the pools step first"
        max_len = int(cfg["max_seq_len"])
        from tag.modeling.loader import load_tokenizer
        tok = load_tokenizer(str(cfg["model_path"]))
        recs = json.load(open(pool_path))
        # A sample of 2000 is plenty for a rate and keeps this a few seconds.
        step = max(1, len(recs) // 2000)
        sample = recs[::step]
        n_trunc = 0
        for r in sample:
            # Approximate the real budget: prompt tokens + response tokens + EOS.
            n_p = len(tok(str(r.get("instruction", "")) + str(r.get("input", "")),
                          add_special_tokens=False)["input_ids"])
            n_r = len(tok(str(r.get("output", "")), add_special_tokens=False)["input_ids"])
            # +8 is slack for the chat template's role/marker tokens.
            if n_p + n_r + 1 + 8 > max_len:
                n_trunc += 1
        rate = n_trunc / max(1, len(sample))
        c_trunc = float(((cfg.get("selection") or {}).get("tag") or {}).get("c_trunc", 0.2))
        detail = (
            f"max_seq_len={max_len}: ~{100 * rate:.1f}% of the pool is "
            f"budget-truncated (sampled {len(sample)})"
        )
        if rate > 0.15:
            return _WARN, detail + (
                f" — those responses lose their appended EOS, so the "
                f"completeness check marks them incomplete and they take "
                f"c_trunc={c_trunc} (a {1 / c_trunc:.0f}x gate cut) despite "
                f"being clean. It hits long responses only, compounding the "
                f"Delta^min length drift. Either raise max_seq_len, or report "
                f"this rate alongside the length-bias table."
            )
        return _OK, detail
    return _fn


def make_corpus_consistency_check(cfg_path):
    """The candidate pool and the CLEAN reference pool must share a corpus.

    ``s`` is a quantile of Delta_hat measured on the reference. If the
    reference is drawn from a different corpus than the pool being gated,
    every gate value in the run is mis-scaled — and the only symptom is a
    zero-weight rate that looks wrong, which is precisely the number being
    reported. It is an easy mistake to make, because the two pools are
    generated by two separate commands and ALPACA_RAW_JSON can change
    between them.
    """
    def _fn():
        cfg = _load_cfg(cfg_path)
        pool_path = Path(str(cfg.get("data_files")))
        pool_man = pool_path.parent / "corruption_manifest.json"
        tag = (cfg.get("selection") or {}).get("tag") or {}
        ref = str(tag.get("gate_ref_file") or "").strip()
        if not ref:
            return _OK, "no gate_ref_file configured; nothing to cross-check"
        ref_man = Path(ref).parent / "corruption_manifest.json"
        if not pool_man.exists() or not ref_man.exists():
            return _WARN, (
                "cannot cross-check the pool and clean-reference corpora: "
                f"{'pool' if not pool_man.exists() else 'reference'} manifest "
                f"missing. Regenerate the pools to get the check."
            )
        pm = json.load(open(pool_man))
        rm = json.load(open(ref_man))
        pin = pm.get("inputs")
        rin = rm.get("inputs")
        if not pin or not rin:
            return _WARN, (
                "pool manifests predate corpus recording — regenerate both "
                "pools to enable the check (scripts/gpu_cloud/bootstrap.sh "
                "pools, after deleting the old dirs)"
            )
        p_sha = {i.get("sha256") for i in pin}
        r_sha = {i.get("sha256") for i in rin}
        if p_sha != r_sha:
            return _FAIL, (
                f"the candidate pool and the clean reference come from "
                f"DIFFERENT corpora:\n"
                f"           pool      <- {[i.get('path') for i in pin]}\n"
                f"           reference <- {[i.get('path') for i in rin]}\n"
                f"         s is a quantile of Delta_hat measured on the "
                f"reference, so gating this pool with that calibration "
                f"mis-scales every G. Regenerate BOTH from one corpus:\n"
                f"           export ALPACA_RAW_JSON=<the one you want>\n"
                f"           rm -rf {pool_path.parent} {Path(ref).parent}\n"
                f"           bash scripts/gpu_cloud/bootstrap.sh pools"
            )
        return _OK, f"pool and clean reference share one corpus ({[i.get('path') for i in pin][0]})"
    return _fn


def make_gate_ref_check(cfg_path):
    """The calibration reference is what keeps the gate from being pool-relative."""
    def _fn():
        import torch
        cfg = _load_cfg(cfg_path)
        tag = (cfg.get("selection") or {}).get("tag") or {}
        scale = str(tag.get("gate_scale") or "").strip()
        ref = str(tag.get("gate_ref_file") or "").strip()
        want_null = bool(tag.get("null_correction", True))
        if scale and not want_null:
            return _OK, f"gate_scale pinned explicitly to {scale}"
        if not ref:
            if want_null:
                return _FAIL, (
                    "null_correction is on but gate_ref_file is empty. mu(M) "
                    "is measured on a CLEAN pool and has no in-pool fallback "
                    "— fitting it on a 30%-dirty pool would absorb the signal "
                    "the gate looks for. Run: bash "
                    "scripts/gpu_cloud/bootstrap.sh calibrate7b"
                )
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
        # weights_only=False: the artifact carries the null curve as a plain
        # dict alongside its tensors.
        d = torch.load(ref, map_location="cpu", weights_only=False)
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
        null = d.get("null")
        if want_null:
            if null is None:
                return _FAIL, (
                    f"{Path(ref).name} carries no Eq. 5' null curve — it "
                    f"predates the correction. Refit with no GPU: "
                    f"scripts/sweep_gate_config.py --ref {ref} --span-tokens "
                    f"{tag.get('span_tokens', 16)} --refit-out {ref} "
                    f"(or regenerate: bootstrap.sh calibrate7b)."
                )
            tv = float(tag.get("target_zero_rate", 0.05))
            if abs(float(null["target_zero_rate"]) - tv) > 1e-9:
                return _FAIL, (
                    f"{Path(ref).name} was fit at target_zero_rate="
                    f"{null['target_zero_rate']} but the arm asks for {tv}. mu(M) "
                    f"IS that quantile — refit."
                )
            if int(null["span_tokens"]) != int(tag.get("span_tokens", 16)):
                return _FAIL, (
                    f"{Path(ref).name} fit mu(M) at W={null['span_tokens']} "
                    f"but the arm uses W={tag.get('span_tokens', 16)}. M is a "
                    f"span COUNT — refit at the arm's W."
                )

        pos = float((dh > 0).float().mean().item())
        detail = (
            f"{Path(ref).name}: n={dh.numel()}, raw {100 * pos:.1f}% positive, "
            f"s={d.get('scale'):.6g}"
        )
        if want_null:
            # With the correction on, the clean zero-weight rate is the target by
            # construction — that is the number to report, and the raw one is
            # context for how much correcting was needed.
            detail += (
                f", clean zero-weight rate pinned at {100 * float(null['target_zero_rate']):.1f}%"
                f" over {len(null['bin_edges'])} length bin(s)"
            )
            return _OK, detail
        if pos < 0.8:
            return _WARN, detail + (
                " — under 80% of the CLEAN reference has Delta_hat > 0. With "
                "null_correction off that is expected on long-response pools "
                "(Eq. 5's order-statistic drift); this arm is the ablation, "
                "so the number is the finding, not a fault."
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
    check("pool / clean-reference same corpus", make_corpus_consistency_check(cfg_path))
    if not args.skip_model:
        check("max_seq_len vs pool length", make_seq_len_check(cfg_path))
    check("gate calibration reference", make_gate_ref_check(cfg_path))
    check("previous run caches", make_stale_cache_check(cfg_path))

    summarise_and_exit(strict=args.strict)


if __name__ == "__main__":
    main()
