"""Episode collection and selection scoring (paper §3.3, Algorithm 1).

Pure deterministic pipeline — no PPO actor, no learned transform, no
z-score, no sigmoid (paper §3.3 final paragraph: "No learned transform,
learned policy, z-score, or sigmoid is applied inside the ranking rule.").
For each candidate sample we compute:

    h̄_l(x_i)              = sequence-mean of layer-l hidden states  (paper Eq. 2)
    L_i                   = mean CE loss over response tokens
    H_i                   = mean predictive entropy over response tokens
    R_i                   = w · L_i + (1-w) · H_i  composite reward (paper Eq. 3)
    w                     = Var(L) / (Var(L) + Var(H) + ε)          (paper Eq. 4)
    align_i               = (1/L) Σ_l ⟨h̄_l(x_i), v_l⟩              (paper §3.3 anchor)
    widetilde-align_i     = min-max-norm(align) ∈ [0, 1]             (paper §3.3 anchor)
    s_i                   = R_i · (1 + λ · widetilde-align_i)        (paper Eq. 10)

Top-B samples by s_i form the training subset for this epoch.
Setting λ=0 (or use_anchor=False) recovers the composite-reward base
ranking exactly (paper §3.3): s_i = R_i.

MVF mode (``mvf`` kwarg; docs/plan_low_quality_multiview.md):
    The low-quality-pool score replaces the uncertainty-carrying R with a
    quality-gated fusion of three genuinely distinct views —

        S_i = (Q_i · c_i + ε)^γ · (D_i^t + ε) · (1 + λ · widetilde-align_i)

    where Q_i is the (static, cached) counterfactual reliability, c_i the
    completeness gate, D_i^t the progress-modulated learnable difficulty,
    and the alignment factor is unchanged. Passing ``mvf=None`` (default)
    keeps the legacy path bit-identical.

Note on "reward" naming: `R`, `r_loss`, `r_entropy`, `r_weight` are kept for
paper / prior-work (DOTS, Active IT) convention compatibility. The TADS
pipeline carries **no RL semantics** — every step is a closed-form
deterministic operation over pool-level statistics.

Pool-level scope (paper Eqs. 3-4 + §3.3 anchor min-max):
    Variance ratio w and min-max bounds for widetilde-align are computed
    across the FULL pool (every candidate sample), not per mini-batch.
    Per-batch variance degenerates to 0 at batch_size=1; we therefore
    accumulate per-sample (L_i, H_i) and raw alignment scores across all
    batches and reduce once after the loop.

Note on forward-pass count (paper §3.3 vs. this implementation):
    Paper §3.3 describes a SINGLE forward pass over D that simultaneously
    provides probe deltas Δh_l(x; θ_{t-1}) for x ∈ D̃_t and candidate mean
    activations h̄_l(x_i; θ_{t-1}) for all i, so the anchor requires no
    separate forward. This implementation runs TWO forwards (probe-only in
    ``TrajectoryAnchor.update`` + full pool here) for engineering
    modularity. The two-forward design is mathematically equivalent —
    same v_l, same alignment, same score, same top-B — but adds ~5-10 %
    wall-clock from the probe forward (~1k samples vs. ~50k pool). A
    future commit can merge them; the algorithm output is unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .dedup import constrained_topk
from .reward import compute_rewards
from .scorer import (
    gated_selection_key,
    learnable_difficulty,
    mvf_score,
    normalize_alignment,
    pool_reward,
    rank01,
    select_top_b,
    tads_score,
    tag_score,
)
from .trajectory_anchor import TrajectoryAnchor
from .utils import cuda_mem_str

logger = logging.getLogger(__name__)


def _flatten_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().view(-1).cpu()


def _unwrap(model):
    """Strip DDP / PEFT wrappers to reach the underlying HF model."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model"):
        m = m.base_model
        if hasattr(m, "model"):
            m = m.model
    return m


@torch.no_grad()
def collect_episode(
    model,
    dataset,
    selection_ratio: float,
    *,
    trajectory_anchor: Optional[TrajectoryAnchor] = None,
    lam: float = 0.0,
    use_anchor: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 42,
    epoch: int = 0,
    exp_tag: Optional[str] = None,
    progress_interval: int = 50,
    empty_cache_interval: int = 10,
    mvf: Optional[Dict[str, Any]] = None,
    tag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one episode over the candidate pool and return selection results.

    ``mvf`` — when None (default) the legacy composite-reward score
    s_i = R_i · (1 + λ·widetilde-align_i) is used, bit-identical to the
    shipped pipeline. When given, the multi-view-fusion score
    S_i = (Q_i·c_i + ε)^γ · (D_i + ε) · (1 + λ·widetilde-align_i)
    replaces it (docs/plan_low_quality_multiview.md §1.4). Expected keys:

        reliability   (N,) Q in [0,1], or None to derive it here from
                      ``loss_cf`` (Q = rank01(loss_cf − loss_orig); done at
                      the epoch that populates the cache)
        loss_cf       (N,) counterfactual pool loss, required when
                      ``reliability`` is None
        completeness  (N,) c in (0,1] — EOS/truncation gate
        loss_prev     (N,) previous-refresh pool loss, or None at t=1
        cluster_ids   list[int] near-duplicate cluster ids (-1 = unique),
                      or None to skip the dedup constraint
        eta, gamma, eps   scalars (plan §1.2 / §1.4)

    ``tag`` — TAG mode (paper Eq. 1), mutually exclusive with ``mvf``. The
    legacy score is kept intact and multiplied by the static reliability
    gate: s_i = G_i · R_i · (1 + λ·widetilde-align_i). Expected keys:

        gate          (N,) G in [0,1] from ``tads.core.gate.compute_gate``,
                      computed at the BASE checkpoint and cached
        cluster_ids   list[int] near-duplicate cluster ids (-1 = unique),
                      or None to skip the dedup constraint

    Unlike MVF, TAG adds no new dynamic machinery — R, the min-max
    alignment, and λ are exactly the legacy ones, so G ≡ 1 reproduces the
    legacy ranking bit-for-bit.

    Memory: ``empty_cache_interval=10`` keeps the CUDA caching allocator
    from fragmenting across the ~3000 episode batches. With Llama-2-7B +
    episode_batch_size=16 the per-batch peak is ~5 GB (logits + entropy
    intermediates); a long run without periodic empty_cache can fragment
    the allocator until a new ~1 GB block can't be placed even though
    gross free memory is high.
    """
    torch.manual_seed(seed + epoch)
    _was_training = model.training
    model.eval()
    # KV cache adds nothing during a feed-forward pass and just leaks memory
    # batch over batch. The loader pins use_cache=False once when
    # gradient_checkpointing is on; this block is a belt-and-braces guard.
    base_model = _unwrap(model)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    all_r_loss: List[torch.Tensor] = []
    all_r_entropy: List[torch.Tensor] = []
    all_alignment_raw: List[torch.Tensor] = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    total_batches = len(loader)
    total_samples = len(dataset)
    t0 = time.time()

    if mvf is not None and tag is not None:
        raise ValueError(
            "collect_episode: 'mvf' and 'tag' are mutually exclusive score "
            "modes — MVF replaces the composite reward, TAG gates it. Pick one.",
        )

    apply_anchor = (
        use_anchor
        and trajectory_anchor is not None
        and trajectory_anchor.is_fitted
    )

    # Pre-cache the per-layer anchor directions on GPU.
    v_cache_gpu: Dict[int, torch.Tensor] = {}
    if apply_anchor:
        if not trajectory_anchor.layer_indices:
            raise RuntimeError(
                "trajectory_anchor.layer_indices is empty — anchor.update() "
                "must run before collect_episode for use_anchor=True.",
            )
        for li in trajectory_anchor.layer_indices:
            v_cache_gpu[li] = trajectory_anchor.v_by_layer[li].to(
                device, dtype=torch.float32, non_blocking=True,
            )

    # NB: named `_log_tag`, not `tag` — `tag` is the TAG-mode parameter.
    _log_tag = f" | tag={exp_tag}" if exp_tag else ""
    logger.info(
        "collect_episode start | epoch=%d | n=%d | bs=%d | batches=%d | "
        "ratio=%.2f | use_anchor=%s | lam=%.3f | mode=%s | %s%s",
        epoch, total_samples, batch_size, total_batches,
        selection_ratio, apply_anchor, lam,
        "mvf" if mvf is not None else ("tag" if tag is not None else "tads"),
        cuda_mem_str(), _log_tag,
    )

    # NCCL heartbeat collective has been REMOVED here. Earlier experiments
    # had rank-0 fire `dist.all_reduce(zeros(1))` every 30s during this loop
    # and have workers do the same inside their file-polling loop. The race:
    # when rank 0 exits the collect loop and writes the ready sentinel,
    # a worker whose 30-second timer is about to elapse can fire one more
    # all_reduce after rank 0 has already left selection.py for SFT —
    # rank 0's next collective is a DDP gradient all_reduce on real tensors,
    # which doesn't match the worker's zeros(1) all_reduce → both hang.
    # NCCL idle protection now relies entirely on the
    # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC env var (set in train.main).
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        decoder_hidden = out.hidden_states[1:]
        B_local = input_ids.size(0)

        if apply_anchor:
            # Paper Eq.6: align_i = (1/L) Σ_l <h̄_l(x_i), v_l>,
            # where h̄_l(x_i) = (1/K_x) Σ_k h_l^(k)(x_i) is the
            # sequence-mean of layer-l hidden states (padding excluded).
            #
            # Cast mask to fp32 BEFORE multiplying with the (bf16/fp16) hidden
            # state — fp16 sum-accumulator over T=512 positions with hidden-
            # state magnitudes ~1.0 can overflow at fp16's 65 504 ceiling.
            # bf16 has the dynamic range to handle it, but doing the multiply
            # + sum in fp32 is uniformly safe across all model dtypes.
            mask_f = attention_mask.to(torch.float32)              # (B, T) fp32
            valid_counts_int = attention_mask.sum(dim=1)           # (B,) int
            if (valid_counts_int == 0).any():
                # All-padding row would silently contribute a zero vector to
                # the alignment min-max — a sign the collate/truncation
                # produced a degenerate batch. Warn but proceed (clamp_min(1)
                # in the divisor below prevents NaN).
                logger.warning(
                    "collect_episode: %d row(s) in this batch have zero valid "
                    "tokens — alignment for those rows will be 0. Likely a "
                    "collate / left-truncation edge case.",
                    int((valid_counts_int == 0).sum().item()),
                )
            valid_counts = valid_counts_int.clamp_min(1).unsqueeze(-1).float()  # (B,1)

            if trajectory_anchor.is_multi_layer:
                batch_align = torch.zeros(B_local, dtype=torch.float32, device=device)
                for li in trajectory_anchor.layer_indices:
                    h_l = decoder_hidden[li].float()               # (B, T, H) fp32
                    masked = h_l * mask_f.unsqueeze(-1)            # (B, T, H) fp32
                    mean_h_l = masked.sum(dim=1) / valid_counts     # (B, H) fp32
                    batch_align += mean_h_l @ v_cache_gpu[li]
                    del mean_h_l, masked, h_l
                # Paper Eq.6 divides by L; we accumulate then divide once.
                batch_align /= float(len(trajectory_anchor.layer_indices))
                all_alignment_raw.append(batch_align.detach().cpu())
                del batch_align
            else:
                # Legacy single-layer mode: project the sequence-mean of the
                # one configured layer onto the single anchor direction.
                li = trajectory_anchor.layer_indices[0]
                v = v_cache_gpu[li]
                h_l = decoder_hidden[li].float()                   # fp32
                masked = h_l * mask_f.unsqueeze(-1)
                mean_h_l = masked.sum(dim=1) / valid_counts        # (B, H) fp32
                all_alignment_raw.append((mean_h_l @ v).detach().cpu())
                del mean_h_l, masked, h_l

        del decoder_hidden

        # Per-sample (L_i, H_i). Per-batch r_weight is degenerate at
        # batch_size=1, so we accumulate and compute w at pool level later.
        r_loss, r_entropy, _ = compute_rewards(out.logits, labels)
        all_r_loss.append(_flatten_cpu_float(r_loss))
        all_r_entropy.append(_flatten_cpu_float(r_entropy))

        del out, input_ids, attention_mask, labels, r_loss, r_entropy

        if (
            torch.cuda.is_available()
            and empty_cache_interval > 0
            and step % empty_cache_interval == 0
        ):
            torch.cuda.empty_cache()

        if step == 1 or step % progress_interval == 0 or step == total_batches:
            elapsed = time.time() - t0
            seen = min(step * batch_size, total_samples)
            pct = 100.0 * seen / max(1, total_samples)
            sec_per_batch = elapsed / max(1, step)
            logger.info(
                "collect_episode | epoch=%d | batch=%d/%d | %d/%d (%.1f%%) "
                "| elapsed=%.1fmin | %.2fs/batch | %s",
                epoch, step, total_batches, seen, total_samples, pct,
                elapsed / 60, sec_per_batch, cuda_mem_str(),
            )

    if not all_r_loss:
        raise RuntimeError("collect_episode produced no samples.")

    all_r_loss_t = torch.cat(all_r_loss, dim=0)
    all_r_entropy_t = torch.cat(all_r_entropy, dim=0)

    # ---- Pool-level composite reward (paper Eqs. 3-4) ----
    R, r_weight_value = pool_reward(all_r_loss_t, all_r_entropy_t)

    # ---- Alignment normalisation (paper §3.3 anchor) ----
    # Legacy path keeps min-max (bit-identical to shipped results). The MVF
    # path uses the pool-CDF (rank01) instead: min-max pins the [0,1]
    # endpoints to the two most extreme samples, so one alignment outlier
    # compresses every other sample's factor — an arbitrary normalisation
    # choice that top-B ranking is NOT invariant to (adversarial review
    # 2026-08). rank01 makes A a pool-CDF like D, with a probabilistic
    # reading and no outlier pinning.
    alignment_raw = None
    if apply_anchor:
        alignment_raw = torch.cat(all_alignment_raw, dim=0).view(-1)
        alignment, _alignment_collapsed = normalize_alignment(alignment_raw)
        if _alignment_collapsed:
            logger.error(
                "TADS alignment COLLAPSED (max-min < 1e-8) | "
                "alignment_raw.std=%.2e. widetilde-align set to 0.5 for "
                "every sample → anchor factor becomes constant (1 + 0.5λ) "
                "and the score reduces to its base ranking for this epoch. "
                "Check anchor PCA gap or multi-layer cancellation.",
                float(alignment_raw.std().item()),
            )
        align_mean = float(alignment.mean().item())
        align_std = float(alignment.std().item())
    else:
        alignment = None
        align_mean = None
        align_std = None
        _alignment_collapsed = False

    # ---- Final score ----
    q = None
    difficulty = None
    gate = None
    ungated_score = None
    if mvf is None:
        # Legacy path (paper Eq. 10): s_i = R_i · (1 + λ · widetilde-align_i).
        # Paper §3.3 final paragraph excludes z-score / R̃ from the ranking
        # rule, so raw R goes into `tads_score` directly.
        if apply_anchor:
            score = tads_score(R, alignment, lam)
            boost = 1.0 + lam * alignment
            logger.info(
                "TADS score | epoch=%d | lam=%.3f | R_mean=%.4f | "
                "align_mean=%.4f | align_std=%.4f | "
                "boost_mean=%.4f | boost_max=%.4f",
                epoch, lam,
                float(R.mean().item()),
                align_mean, align_std,
                float(boost.mean().item()), float(boost.max().item()),
            )
        else:
            # λ = 0 or use_anchor=False — paper §3.3: "Setting λ = 0
            # recovers the composite-reward base ranking exactly."
            score = R
            logger.info(
                "Composite-reward-only score (no anchor) | epoch=%d | R_mean=%.4f",
                epoch, float(R.mean().item()),
            )
        if tag is not None:
            # TAG (paper Eq. 1): s_i = G_i · R_i · (1 + λ·widetilde-align_i).
            # The legacy score computed just above IS the dynamic part; the
            # gate multiplies it. G is static and cached, so it never
            # changes across refreshes — see tads/core/gate.py.
            gate = tag.get("gate")
            if gate is None:
                raise ValueError(
                    "collect_episode(tag=...): 'gate' is required. G is defined "
                    "at the BASE checkpoint (paper §1, Eq. 6) and is computed by "
                    "the pipeline before scoring — see tads/core/gate.py.",
                )
            gate = gate.view(-1).float()
            if gate.numel() != total_samples:
                raise ValueError(
                    f"collect_episode(tag=...): gate length {gate.numel()} != "
                    f"pool size {total_samples} — stale cache or wrong pool?",
                )
            if bool(torch.isnan(gate).any()) or bool((gate < 0).any()) or bool((gate > 1).any()):
                raise ValueError(
                    "collect_episode(tag=...): gate must lie in [0,1] and be "
                    "NaN-free (paper Eq. 6 clamps to that range).",
                )
            ungated_score = score
            score = tag_score(gate, R, alignment if apply_anchor else None, lam)
            n_vetoed = int((gate == 0).sum().item())
            logger.info(
                "TAG score | epoch=%d | lam=%.3f | G_mean=%.4f | G==0: %d/%d "
                "(%.1f%%) | R_mean=%.4f | s_mean=%.4f",
                epoch, lam if apply_anchor else 0.0,
                float(gate.mean().item()), n_vetoed, total_samples,
                100.0 * n_vetoed / max(1, total_samples),
                float(R.mean().item()), float(score.mean().item()),
            )
    else:
        # MVF path (plan §2, v3):
        #   S = (Q·c + ε)^γ · (D' + ε) · (1 + λ_eff · Ã),  Ã = rank01(align_raw)
        from .reliability import reliability_from_losses  # local: avoid cycle

        q = mvf.get("reliability")
        if q is None:
            # Q is DEFINED at the base checkpoint (plan §2.1). Deriving it
            # here at a later epoch — e.g. after a resume that lost
            # reliability_cache.pt — would silently change the view's
            # meaning to "counterfactual fidelity under the current
            # checkpoint". Fail loudly instead of degrading quietly.
            if epoch > 1 and not bool(mvf.get("allow_late_reliability", False)):
                raise RuntimeError(
                    f"collect_episode(mvf=...): no cached reliability at "
                    f"epoch {epoch} (> 1). Q must come from the base "
                    f"checkpoint — restore reliability_cache.pt from the "
                    f"run's output dir, or pass allow_late_reliability=True "
                    f"to explicitly accept a wrong-checkpoint Q.",
                )
            loss_cf = mvf.get("loss_cf")
            if loss_cf is None:
                raise ValueError(
                    "collect_episode(mvf=...): either 'reliability' or "
                    "'loss_cf' must be provided.",
                )
            q = reliability_from_losses(
                all_r_loss_t, loss_cf,
                mode=str(mvf.get("reliability_mode", "sigmoid")),
                scale=mvf.get("reliability_scale"),
                rezero=bool(mvf.get("reliability_rezero", True)),
            )
        q = q.view(-1).float()
        completeness = mvf["completeness"].view(-1).float()
        loss_prev = mvf.get("loss_prev")
        if loss_prev is not None:
            loss_prev = loss_prev.view(-1).float()
        eta = float(mvf.get("eta", 0.5))
        gamma = float(mvf.get("gamma", 1.0))
        eps = float(mvf.get("eps", 0.01))
        d_floor = float(mvf.get("d_floor", 0.5))
        lam_scale = float(mvf.get("lam_scale", 1.0))
        lam_eff = lam * lam_scale
        for name, t in (("reliability", q), ("completeness", completeness)):
            if t.numel() != total_samples:
                raise ValueError(
                    f"collect_episode(mvf=...): {name} length {t.numel()} != "
                    f"pool size {total_samples} — stale cache or wrong pool?",
                )
        difficulty = learnable_difficulty(
            all_r_loss_t, loss_prev, eta,
            selected_prev=mvf.get("selected_prev"),
            progress_mode=str(mvf.get("progress_mode", "split")),
        )
        # Pool-CDF normalisation — but NOT on a collapsed anchor: rank01 of
        # sub-1e-8 numerical noise would fabricate a full [0,1] spread from
        # nothing (the legacy path's normalize_alignment guards this with
        # the 0.5-constant fallback; here we drop the factor entirely).
        if alignment_raw is not None and not _alignment_collapsed:
            alignment_mvf = rank01(alignment_raw)
        else:
            alignment_mvf = None
            if alignment_raw is not None and _alignment_collapsed:
                logger.error(
                    "MVF alignment COLLAPSED (max-min < 1e-8) — dropping the "
                    "Geometry factor for this epoch (S = gate · difficulty).",
                )
        score = mvf_score(
            q, completeness, difficulty, alignment_mvf,
            lam=lam_eff, gamma=gamma, eps=eps, d_floor=d_floor,
        )
        logger.info(
            "MVF score | epoch=%d | lam=%.3f (scale=%.3f) | gamma=%.2f | "
            "eta=%.2f | d_floor=%.2f | Q_mean=%.4f | c_mean=%.4f | "
            "D_mean=%.4f | align_cdf_mean=%s | progress=%s (%s)",
            epoch, lam_eff, lam_scale, gamma, eta, d_floor,
            float(q.mean().item()),
            float(completeness.mean().item()),
            float(difficulty.mean().item()),
            # Diagnostics must report the statistic the score actually uses
            # (the CDF), not the legacy min-max value.
            f"{float(alignment_mvf.mean().item()):.4f}" if alignment_mvf is not None else "n/a",
            "on" if loss_prev is not None else "off (t=1)",
            str(mvf.get("progress_mode", "split")),
        )

    # ---- Top-B selection ----
    k = max(1, int(total_samples * selection_ratio))
    if total_samples == 0:
        raise RuntimeError(
            "collect_episode: total_samples == 0 — empty candidate pool. "
            "Check dataset_subset_size / data path.",
        )
    if mvf is not None:
        cluster_ids = mvf.get("cluster_ids")
    elif tag is not None:
        # The legacy path has never deduplicated (dedup was introduced with
        # MVF), so TAG must thread cluster_ids explicitly or duplicated
        # instructions would be selected repeatedly despite the gate.
        cluster_ids = tag.get("cluster_ids")
    else:
        cluster_ids = None

    ranking = score
    if tag is not None and gate is not None:
        # Vetoed samples all score exactly 0; rank them below every
        # admissible sample and break the zero-block by the ungated score
        # instead of by pool file order (see scorer.gated_selection_key).
        ranking, n_admissible = gated_selection_key(
            score,
            ungated_score if ungated_score is not None else R,
            gate,
        )
        if n_admissible < k:
            logger.warning(
                "TAG: only %d/%d samples pass the reliability gate but the "
                "budget is B=%d — %d slot(s) must be filled with VETOED "
                "samples (ranked by the ungated score). The non-compensation "
                "guarantee does not hold for those slots; report this count, "
                "or lower selection_ratio / raise the gate scale.",
                n_admissible, total_samples, k, k - n_admissible,
            )
        else:
            logger.info(
                "TAG selection | admissible=%d/%d | B=%d — budget fits inside "
                "the gated set.", n_admissible, total_samples, k,
            )

    if cluster_ids is not None:
        selected_indices: List[int] = (
            constrained_topk(ranking, k, cluster_ids).cpu().tolist()
        )
    else:
        selected_indices = select_top_b(ranking, k).cpu().tolist()
    logger.info(
        "Selection topk | k=%d/%d | first5=%s",
        k, total_samples, selected_indices[:5],
    )

    var_loss_val = (
        float(all_r_loss_t.var().item()) if all_r_loss_t.numel() > 1 else 0.0
    )
    var_entropy_val = (
        float(all_r_entropy_t.var().item()) if all_r_entropy_t.numel() > 1 else 0.0
    )

    elapsed = time.time() - t0
    logger.info(
        "Episode done | epoch=%d | selected=%d/%d | "
        "R_loss=%.4f (var=%.6f) | R_entropy=%.4f (var=%.6f) | "
        "r_weight=%.4f | R_mean=%.4f | elapsed=%.1fmin | %s",
        epoch, k, total_samples,
        float(all_r_loss_t.mean().item()), var_loss_val,
        float(all_r_entropy_t.mean().item()), var_entropy_val,
        r_weight_value,
        float(R.mean().item()),
        elapsed / 60, cuda_mem_str(),
    )

    r_loss_mean = float(all_r_loss_t.mean().item())
    r_entropy_mean = float(all_r_entropy_t.mean().item())

    if _was_training:
        model.train()

    return {
        "selected_indices": selected_indices,
        "rewards": R,
        "alignment": alignment,
        # Per-sample vectors — consumed by the MVF pipeline (loss history,
        # reliability cache) and by scripts/score_pool.py diagnostics.
        "r_loss": all_r_loss_t,
        "r_entropy": all_r_entropy_t,
        "reliability": q,
        "difficulty": difficulty,
        "gate": gate,
        "ungated_score": ungated_score,
        "score": score,
        "score_mode": (
            "mvf" if mvf is not None else ("tag" if tag is not None else "tads")
        ),
        "r_loss_mean": r_loss_mean,
        "r_entropy_mean": r_entropy_mean,
        "r_weight": r_weight_value,
        # Compatibility aliases for older Data-Agent analysis scripts.
        "rdiff_mean": r_loss_mean,
        "rconf_mean": r_entropy_mean,
        "r": r_weight_value,
        "var_loss": var_loss_val,
        "var_entropy": var_entropy_val,
        "lam": lam if apply_anchor else 0.0,
        "use_anchor": apply_anchor,
        "align_mean": align_mean,
        "align_std": align_std,
        "alignment_collapsed": _alignment_collapsed,
    }
