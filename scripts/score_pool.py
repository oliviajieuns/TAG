#!/usr/bin/env python
"""Forward-only pool diagnostics: which selection signal picks up dirt?

Phase-A experiment of docs/plan_low_quality_multiview.md §4: score a
(corrupted) candidate pool at the base checkpoint — no training — and
measure, per signal, how much corruption each signal's top-K selection
would admit. Ground truth comes from the corruption manifest written by
``scripts/make_corrupted_pool.py``.

Signals compared (each used AS a selection score, higher = selected):
    entropy        mean predictive entropy H_i         (uncertainty view)
    loss           mean response CE L_i
    R              legacy composite reward wL + (1-w)H (old TADS carrier)
    legacy_score   R · (1 + λ·align)                   (old TADS, Eq. 10)
    Q              counterfactual reliability, v3 calibrated sigmoid gate
    Q_rank         v1 rank01 reliability               (ablation arm)
    gate           (Q·c + ε)^γ                          (MVF reliability gate)
    D              learnable difficulty at t=1 = rank01(L)
    mvf_score      full MVF v3 fusion S^1              (new TADS)
    mvf_v2         v2 ablation: raw σ gate (no rezero) + d_floor=0 —
                   the parameterisation whose non-compensation provably
                   reverses (kept as the motivating-counterexample row)
    additive       compensatory fusion (Q·c + D + Ã)/3 — the gated-vs-
                   additive contrast row (plan §2.4)
    ppl            -L(y): base-model fluency filter    (needs --uncond-loss;
                   the "a trivial perplexity filter already solves this"
                   attack, measured instead of assumed)
    ifd            L(y|x)/L(y): IFD (Li et al. 2024)   (needs --uncond-loss;
                   closest published relative of Q)

TAG signals (paper Eqs. 2-6; computed when the config has a tads.tag block,
or with --tag-gate. They need a TOKEN-LEVEL counterfactual forward, which is
one extra pool forward per pool on top of the mean-loss passes above):
    delta_bar      1 - L(y|x)/L(y|x^-)                  (Eq. 3, overall gain)
    delta_min      worst admissible span's gain         (Eq. 5, tail gain)
    delta_hat      min(delta_bar, delta_min)            (Eq. 6 input)
    G              c·(2σ(Δ̂/s) - 1)_+                    (Eq. 6, the gate)
    tag_score      G · R · (1 + λ·align)                (Eq. 1, full TAG)

Comparing delta_bar against delta_min IS the paper's span ablation: if the
tail gain does not beat the overall gain on the localized corruption types
(T5 wrong-answer, T7 fluent-wrong), Eqs. 4-5 are not earning their place.

Metrics per signal:
    dirty@K        corrupted fraction of the top-K selection (K = the
                   configured selection_ratio and any extra --ks)
    ap_dirty       average precision for detecting dirty samples from the
                   signal's REJECTION order (detector = -signal; a good
                   selection score concentrates dirt at the bottom)
    reject@K/type  per-corruption-type fraction NOT selected at ratio K
    cluster_frac   unique-near-dup-cluster fraction of the selection
                   (with --dedup-clusters; duplicate-adjusted diversity)

Usage:
    python scripts/score_pool.py \
        --config configs/experiments/lowq/light_tads_mvf_05b.yaml \
        --manifest $POOLS/composite20/corruption_manifest.json \
        --out $POOLS/composite20/score_report.json
The pool/counterfactual files come from the config's data_files /
tads.mvf.counterfactual_data_files (env-var wired, same as training).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Mirrors the guard in tads/train.py: transformers eager-imports
# `from torchvision.io import VideoReader` via its video model registry,
# which raises on torchvision builds without ffmpeg support — even though
# this script never touches video. Stub the missing attribute BEFORE any
# transformers import so the import resolves to a harmless placeholder.
try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    pass  # torchvision absent entirely is fine — this script never uses it.

import torch  # noqa: E402

from tads.core.reliability import (  # noqa: E402
    completeness_from_dataset,
    compute_pool_loss,
    reliability_from_losses,
)
from tads.core.scorer import (  # noqa: E402
    learnable_difficulty,
    mvf_score,
    pool_reward,
    rank01,
    tads_score,
)
from tads.core.selector import collect_episode  # noqa: E402
from tads.core.trajectory_anchor import TrajectoryAnchor  # noqa: E402
from tads.core.utils import load_config, set_seed, setup_logger  # noqa: E402
from tads.data.alpaca import build_alpaca_dataset  # noqa: E402
from tads.data.corruption import dirty_labels_from_manifest  # noqa: E402
from tads.modeling.loader import load_model, load_tokenizer  # noqa: E402


def average_precision(detector: torch.Tensor, labels: torch.Tensor) -> float:
    """AP for binary ``labels`` (1 = dirty) ranked by ``detector`` desc."""
    order = torch.argsort(detector, descending=True)
    y = labels[order].float()
    n_pos = float(y.sum().item())
    if n_pos == 0:
        return float("nan")
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float(((precision * y).sum() / n_pos).item())


def dirty_at_k(score: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    top = torch.topk(score, k).indices
    return float(labels[top].float().mean().item())


def reject_rate_by_type(
    score: torch.Tensor,
    manifest: Dict,
    k: int,
) -> Dict[str, float]:
    top = set(torch.topk(score, k).indices.tolist())
    by_type: Dict[str, List[int]] = {}
    for idx_str, entry in manifest.get("entries", {}).items():
        by_type.setdefault(entry["type"], []).append(int(idx_str))
    out = {}
    for t, idxs in sorted(by_type.items()):
        rejected = sum(1 for i in idxs if i not in top)
        out[t] = rejected / max(1, len(idxs))
    return out


def cluster_fraction(
    score: torch.Tensor, cluster_ids: Optional[List[int]], k: int,
) -> Optional[float]:
    if cluster_ids is None:
        return None
    top = torch.topk(score, k).indices.tolist()
    seen, distinct = set(), 0
    for i in top:
        cid = cluster_ids[i]
        if cid < 0:
            distinct += 1
        elif cid not in seen:
            seen.add(cid)
            distinct += 1
    return distinct / max(1, len(top))


def _build_gate_cfg(params: Dict, scale: Optional[float]):
    """GateConfig from a tads.tag block — mirrors the training-side builder
    in tads.pipelines.selection so a diagnostic run and a training run
    compute the SAME G from the same YAML."""
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
        undefined_policy=str(params.get("undefined_policy", "pass")),
        scale=scale,
        dispersion_discount=bool(params.get("dispersion_discount", True)),
    )


def _resolve_gate_scale_cli(params: Dict) -> Optional[float]:
    """Explicit gate_scale > gate_ref_file calibration > None (in-pool).

    ``${oc.env:VAR,}`` resolves to the EMPTY STRING when unset, so the
    strip() check is required — `or None` would additionally swallow a
    legitimate 0.0 (which GateConfig rejects anyway, but the distinction
    matters for the error message the user sees).
    """
    from tads.core import gate as gatelib

    scale = params.get("gate_scale")
    if scale is not None and str(scale).strip() != "":
        return float(scale)
    ref_file = params.get("gate_ref_file")
    if ref_file and str(ref_file).strip() != "":
        ref = torch.load(str(ref_file), map_location="cpu", weights_only=True)
        if isinstance(ref, dict):
            if "delta_hat" not in ref:
                raise ValueError(
                    f"gate_ref_file {ref_file} has no 'delta_hat' key (found "
                    f"{sorted(ref.keys())}). The TAG gate calibrates on the "
                    f"Delta_hat ratio, not the MVF raw ΔL — regenerate with "
                    f"scripts/calibrate_reliability.py --mode tag."
                )
            ref = ref["delta_hat"]
        return gatelib.calibrate_gate_scale(
            ref,
            target_pct=float(params.get("calibration_target_pct", 0.10)),
            target_q=float(params.get("calibration_target_q", 0.8)),
        )
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="experiment YAML (pool wiring)")
    p.add_argument("--manifest", required=True, help="corruption_manifest.json")
    p.add_argument("--out", required=True, help="output report JSON path")
    p.add_argument("--ks", default="", help="extra selection ratios, e.g. 0.1,0.5")
    p.add_argument("--dedup-clusters", default=None,
                   help="dedup_clusters.json (defaults to tads.mvf.dedup_clusters_file)")
    p.add_argument("--uncond-loss", default=None,
                   help=".pt from scripts/compute_uncond_loss.py — enables the "
                        "ppl and ifd baseline signal rows")
    p.add_argument("--no-anchor", action="store_true",
                   help="skip the trajectory anchor (drops legacy_score/mvf alignment factor)")
    p.add_argument("--tag-gate", dest="tag_gate", action="store_true", default=None,
                   help="force the TAG signals on (token-level counterfactual "
                        "forward: +1 pool forward per pool). Default: on when "
                        "the config has a tads.tag block.")
    p.add_argument("--no-tag-gate", dest="tag_gate", action="store_false",
                   help="force the TAG signals off")
    p.add_argument("--save-signals", default=None,
                   help="write every per-sample signal vector to this .pt "
                        "(needed for the reliability diagram, the "
                        "predicted-vs-measured Dirty@K figure, and per-view "
                        "attribution — the JSON report is aggregate only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger(str(Path(args.out).parent), name="score_pool")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    with open(args.manifest) as f:
        manifest = json.load(f)
    labels = torch.tensor(dirty_labels_from_manifest(manifest), dtype=torch.long)

    tads_cfg = cfg.get("tads", {}) or {}
    mvf_cfg = tads_cfg.get("mvf", {}) or {}
    tag_cfg = tads_cfg.get("tag", {}) or {}
    lam = float(tads_cfg.get("lam", 1.0))
    gamma = float(mvf_cfg.get("gamma", 1.0))
    eps = float(mvf_cfg.get("eps", 0.01))
    # A TAG arm has no tads.mvf block, so c_trunc must come from whichever
    # score block the config actually carries.
    c_trunc = float(tag_cfg.get("c_trunc", mvf_cfg.get("c_trunc", 0.2)))
    want_tag = bool(tag_cfg) if args.tag_gate is None else bool(args.tag_gate)
    if args.tag_gate and not tag_cfg:
        logger.warning(
            "--tag-gate was requested but the config has no tads.tag block; "
            "the gate will use gate.py defaults and in-pool calibration "
            "(diagnostic-only). Point --config at a TAG arm for reported runs.",
        )

    tokenizer = load_tokenizer(cfg["model_path"])
    model = load_model(
        cfg["model_path"],
        training_mode="full",
        lora_cfg=None,
        use_ddp=False,
        local_rank=0,
        gradient_checkpointing=False,
        attn_implementation=cfg.get("attn_implementation"),
    )
    model.eval()

    cache_dir = str(Path(cfg["data_cache"]) / "score_pool")
    dataset = build_alpaca_dataset(
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_seq_len=int(cfg["max_seq_len"]),
        dataset_name=cfg.get("dataset_name"),
        data_files=cfg.get("data_files"),
        prompt_style=str(cfg.get("prompt_style") or "alpaca_default"),
    )
    n = len(dataset)
    if n != labels.numel():
        raise ValueError(
            f"pool size {n} != manifest n_total {labels.numel()} — config's "
            f"data_files does not match this manifest."
        )

    cf_files = (
        mvf_cfg.get("counterfactual_data_files")
        or tag_cfg.get("counterfactual_data_files")
        or None
    )
    if not cf_files:
        raise ValueError(
            "config has no tads.mvf.counterfactual_data_files nor "
            "tads.tag.counterfactual_data_files — every reliability signal "
            "(Q, gate, mvf_score, delta_*, G, tag_score) needs the "
            "counterfactual pool."
        )
    if not isinstance(cf_files, (list, tuple)):
        # Comma-separated strings express K > 1 through a single env var,
        # matching the training-side plumbing in tads/train.py.
        cf_files = [s.strip() for s in str(cf_files).split(",") if s.strip()]
    cf_datasets = []
    for k_cf, one in enumerate(cf_files, start=1):
        cf_dataset = build_alpaca_dataset(
            tokenizer=tokenizer,
            cache_dir=str(
                Path(cache_dir)
                / ("counterfactual" if k_cf == 1 else f"counterfactual_{k_cf}")
            ),
            max_seq_len=int(cfg["max_seq_len"]),
            dataset_name=None,
            data_files=str(one),
            prompt_style=str(cfg.get("prompt_style") or "alpaca_default"),
        )
        if len(cf_dataset) != n:
            raise ValueError(
                f"counterfactual pool #{k_cf} size {len(cf_dataset)} != pool size {n}"
            )
        cf_datasets.append(cf_dataset)

    cluster_path = args.dedup_clusters or (
        mvf_cfg.get("dedup_clusters_file")
        or tag_cfg.get("dedup_clusters_file")
        or None
    )
    cluster_ids = None
    if cluster_path:
        from tads.core.dedup import load_clusters
        cluster_ids = load_clusters(str(cluster_path))
        if len(cluster_ids) != n:
            raise ValueError(
                f"dedup clusters length {len(cluster_ids)} != pool size {n}"
            )

    # ---- anchor (optional; gives the alignment factor of both scores) ----
    anchor = None
    use_anchor = bool(tads_cfg.get("use_anchor", True)) and not args.no_anchor
    if use_anchor:
        anchor_cfg = cfg.get("anchor", {}) or {}
        anchor = TrajectoryAnchor(
            layer_idx=int(anchor_cfg.get("layer_idx", -1)),
            layer_indices=anchor_cfg.get("layer_indices"),
            max_samples_for_pca=int(anchor_cfg.get("max_samples_for_pca", 2000)),
            pca_batch_size=int(anchor_cfg.get("pca_batch_size", 4)),
            device=str(args.device),
        )
        anchor.update(model=model, dataset=dataset, seed=seed, epoch=1)

    # ---- forward passes: pool (loss/entropy/hidden) + counterfactual ----
    episode = collect_episode(
        model=model,
        dataset=dataset,
        selection_ratio=float(cfg.get("selection_ratio", 0.1)),
        trajectory_anchor=anchor,
        lam=lam,
        use_anchor=use_anchor,
        batch_size=int(cfg.get("episode_batch_size", 1)),
        device=str(args.device),
        seed=seed,
        epoch=1,
        exp_tag="score_pool",
    )
    loss = episode["r_loss"]
    entropy = episode["r_entropy"]
    alignment = episode["alignment"]  # min-max normed or None

    cf_losses = [
        compute_pool_loss(
            model, one_ds,
            batch_size=int(cfg.get("episode_batch_size", 1)),
            device=str(args.device),
            tag=f"counterfactual_{k_cf}" if len(cf_datasets) > 1 else "counterfactual",
        )
        for k_cf, one_ds in enumerate(cf_datasets, start=1)
    ]
    loss_cf = cf_losses[0] if len(cf_losses) == 1 else torch.stack(cf_losses, dim=0)

    # ---- signals ----
    completeness = completeness_from_dataset(
        dataset, eos_token_id=tokenizer.eos_token_id, c_trunc=c_trunc,
    )
    # v3 gate configuration from the config (same resolution as training).
    r_mode = str(mvf_cfg.get("reliability_mode", "sigmoid"))
    r_rezero = bool(mvf_cfg.get("reliability_rezero", True))
    r_scale = mvf_cfg.get("reliability_scale") or None
    if r_scale is not None:
        r_scale = float(r_scale)
    d_floor = float(mvf_cfg.get("d_floor", 0.5))
    q = reliability_from_losses(
        loss, loss_cf, mode=r_mode, scale=r_scale, rezero=r_rezero,
    )
    q_rank = reliability_from_losses(loss, loss_cf, mode="rank")
    q_v2 = reliability_from_losses(
        loss, loss_cf, mode="sigmoid", scale=r_scale, rezero=False,
    )
    gate = torch.pow(q * completeness + eps, gamma)
    d1 = learnable_difficulty(loss, None)
    R, r_weight = pool_reward(loss, entropy)
    legacy = tads_score(R, alignment, lam) if alignment is not None else R
    # rank01 of the min-max alignment == rank01 of the raw alignment
    # (monotone transform), i.e. the MVF pool-CDF normalisation.
    alignment_cdf = rank01(alignment) if alignment is not None else None
    s1 = mvf_score(
        q, completeness, d1, alignment_cdf,
        lam=lam, gamma=gamma, eps=eps, d_floor=d_floor,
    )
    # v2 ablation: raw sigmoid gate + uncompressed D — the parameterisation
    # whose non-compensation provably reverses (theorem counterexample row).
    s_v2 = mvf_score(
        q_v2, completeness, d1, alignment_cdf,
        lam=lam, gamma=gamma, eps=eps, d_floor=0.0,
    )
    # Compensatory contrast: equal-weight additive fusion of the same views.
    additive_views = [q * completeness, d1]
    if alignment_cdf is not None:
        additive_views.append(alignment_cdf)
    s_add = torch.stack(additive_views, dim=0).mean(dim=0)

    signals = {
        "entropy": entropy,
        "loss": loss,
        "R": R,
        "legacy_score": legacy,
        "Q": q,
        "Q_rank": q_rank,
        "gate": gate,
        "D": d1,
        "mvf_score": s1,
        "mvf_v2": s_v2,
        "additive": s_add,
    }

    # ---- TAG signals (paper Eqs. 2-6) ----
    # These need per-token counterfactual losses, which neither
    # compute_rewards nor compute_pool_loss preserves (both reduce to a
    # sequence mean), so this is a separate token-level pass over the true
    # pool and each counterfactual pool.
    tag_components = None
    if want_tag:
        from tads.core import gate as gatelib

        gcfg_params = dict(tag_cfg)
        gcfg_params.setdefault("c_trunc", c_trunc)
        g_scale = _resolve_gate_scale_cli(gcfg_params)
        gcfg = _build_gate_cfg(gcfg_params, g_scale)
        logger.info(
            "TAG gate | W=%d tau=%.3f(%s) min_span=%d tail=%s include_eos=%s "
            "| scale=%s | token-level forwards: %d",
            gcfg.span_tokens, gcfg.tau, gcfg.tau_mode, gcfg.min_span_tokens,
            gcfg.tail_mode, gcfg.include_eos, g_scale, 1 + len(cf_datasets),
        )
        tok_true, n_true = gatelib.compute_pool_token_losses(
            model, dataset,
            batch_size=int(cfg.get("episode_batch_size", 1)),
            device=str(args.device), tag="true",
            eos_token_id=tokenizer.eos_token_id,
            drop_trailing_eos=not gcfg.include_eos,
        )
        tok_cf, n_cf_list = [], []
        for k_cf, one_ds in enumerate(cf_datasets, start=1):
            tc, nc = gatelib.compute_pool_token_losses(
                model, one_ds,
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(args.device),
                tag=f"cf_tokens_{k_cf}" if len(cf_datasets) > 1 else "cf_tokens",
                eos_token_id=tokenizer.eos_token_id,
                drop_trailing_eos=not gcfg.include_eos,
            )
            tok_cf.append(tc)
            n_cf_list.append(nc)
        if gcfg.scale is None:
            probe = gatelib.gate_components(
                tok_true, n_true, tok_cf[0], n_cf_list[0], cfg=gcfg,
            )
            gcfg = _build_gate_cfg(
                gcfg_params, gatelib.resolve_scale(gcfg, probe["delta_hat"]),
            )
        tag_components = gatelib.compute_gate(
            tok_true, n_true, tok_cf, n_cf_list, completeness, cfg=gcfg,
        )
        tag_components["scale_used"] = gcfg.scale
        g_i = tag_components["gate"]
        signals["delta_bar"] = tag_components["delta_bar"]
        signals["delta_min"] = tag_components["delta_min"]
        signals["delta_hat"] = tag_components["delta_hat"]
        signals["G"] = g_i
        signals["tag_score"] = g_i * legacy

    if args.uncond_loss:
        uncond = torch.load(args.uncond_loss, map_location="cpu", weights_only=True)
        if isinstance(uncond, dict):
            uncond = uncond["uncond_loss"]
        uncond = uncond.view(-1).float()
        if uncond.numel() != n:
            raise ValueError(
                f"--uncond-loss length {uncond.numel()} != pool size {n}"
            )
        # PPL filter keeps the most fluent responses → select LOW L(y).
        signals["ppl"] = -uncond
        # IFD (Li et al., NAACL 2024) selects HIGH conditioned/unconditioned
        # loss ratio ("instruction-following difficulty"), EXCLUDING
        # samples with IFD > 1 — the paper treats those as misaligned
        # (instruction makes prediction WORSE) and drops them before
        # top-K. Omitting the exclusion would make the baseline
        # preferentially select exactly our mismatch corruptions and
        # misrepresent it (adversarial review 2026-08). Excluded samples
        # get score 0 (< any admissible ratio), keeping AP well-defined.
        ifd_ratio = loss / uncond.clamp_min(1e-6)
        signals["ifd"] = torch.where(
            ifd_ratio <= 1.0, ifd_ratio, torch.zeros_like(ifd_ratio),
        )

    ratios = [float(cfg.get("selection_ratio", 0.1))]
    for extra in args.ks.split(","):
        if extra.strip():
            ratios.append(float(extra))
    ratios = sorted(set(ratios))

    base_rate = float(labels.float().mean().item())
    report = {
        "config": str(args.config),
        "manifest": str(args.manifest),
        "n": n,
        "dirty_base_rate": base_rate,
        "r_weight": r_weight,
        "use_anchor": use_anchor,
        "ratios": ratios,
        "signals": {},
    }
    for name, sig in signals.items():
        entry = {"ap_dirty_from_rejection": average_precision(-sig, labels)}
        for ratio in ratios:
            k = max(1, int(n * ratio))
            key = f"@{ratio:g}"
            entry[f"dirty{key}"] = dirty_at_k(sig, labels, k)
            entry[f"reject_by_type{key}"] = reject_rate_by_type(sig, manifest, k)
            cf_frac = cluster_fraction(sig, cluster_ids, k)
            if cf_frac is not None:
                entry[f"unique_cluster_frac{key}"] = cf_frac
        report["signals"][name] = entry
        logger.info(
            "signal=%s | AP(dirty)=%.4f | %s",
            name, entry["ap_dirty_from_rejection"],
            " ".join(
                f"dirty@{r:g}={entry[f'dirty@{r:g}']:.3f}" for r in ratios
            ),
        )

    if want_tag and tag_components is not None:
        report["tag"] = {
            "gate_scale": tag_components.get("scale_used"),
            "gate_mean": float(tag_components["gate"].mean().item()),
            "gate_zero_frac": float(
                (tag_components["gate"] == 0).float().mean().item()
            ),
            "n_undefined": int(tag_components["undefined"].sum().item()),
            "n_empty_C": int(tag_components["empty_c"].sum().item()),
            "mean_spans": float(tag_components["n_spans"].float().mean().item()),
            "mean_valid_spans": float(
                tag_components["n_valid_spans"].float().mean().item()
            ),
        }
        # The veto only holds while the admissible set covers the budget —
        # report the headroom so a shortfall cannot go unnoticed in the table.
        n_adm = int((tag_components["gate"] > 0).sum().item())
        report["tag"]["n_admissible"] = n_adm
        report["tag"]["admissible_frac"] = n_adm / max(1, n)
        for ratio in ratios:
            k = max(1, int(n * ratio))
            report["tag"][f"budget_fits@{ratio:g}"] = bool(n_adm >= k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    if args.save_signals:
        sig_path = Path(args.save_signals)
        sig_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signals": {k: v.detach().cpu() for k, v in signals.items()},
            "dirty_labels": labels.detach().cpu(),
            "completeness": completeness.detach().cpu(),
            "n": n,
            "config": str(args.config),
            "manifest": str(args.manifest),
        }
        if tag_components is not None:
            payload["tag_components"] = {
                k: v.detach().cpu()
                for k, v in tag_components.items()
                if torch.is_tensor(v)
            }
        if cluster_ids is not None:
            payload["cluster_ids"] = list(cluster_ids)
        torch.save(payload, sig_path)
        logger.info("Per-sample signals written to %s", sig_path)
    logger.info(
        "Report written to %s | base dirty rate %.3f", out_path, base_rate,
    )


if __name__ == "__main__":
    main()
