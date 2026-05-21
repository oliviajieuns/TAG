"""Episode collection and selection scoring (paper §3, Algorithm 1).

Pure deterministic pipeline — no PPO actor. For each candidate sample we
compute:
    h̄_l(x_i)  = sequence-mean of layer-l hidden states           (paper Eq.6 input)
    L_i        = mean CE loss over response tokens                (== rdiff legacy name)
    H_i        = mean predictive entropy over response tokens     (== rconf legacy name)
    R_i        = w·L_i + (1-w)·H_i  composite reward              (paper Eq.2)
    R̃_i       = (R_i - R̄) / (σ_R + 1e-6)  calibrated utility    (paper Eq.8 inner, pool z-score)
    ã_i       = min-max-norm( Σ_l <h̄_l(x_i), v_l> / L )           (paper Eq.7)
    s_i        = R̃_i · (1 + λ · ã_i)                             (paper Eq.8 outer)

Top-B samples by s_i form the training subset for this epoch.
Setting λ=0 (or use_anchor=False) reduces s_i to R̃_i — the clean-ablation
case where TADS becomes the calibrated-utility baseline.

Note on "reward" naming: `R`, `r_loss`, `r_entropy`, `r_weight` are kept for
paper / prior-work (DOTS, Active IT) convention compatibility. The TADS
pipeline carries **no RL semantics** — every step is a closed-form
deterministic operation over pool-level statistics.

Pool-level scope (paper Eq.3, Eq.7, Eq.8):
    Variance ratio w, min-max bounds, and the z-score moments for R̃ are all
    computed across the FULL pool (every candidate sample), not per
    mini-batch. Per-batch variance degenerates to 0 at batch_size=1; we
    therefore accumulate per-sample (L_i, H_i) and raw alignment scores
    across all batches and reduce once after the loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .reward import compute_rewards
from .scorer import (
    calibrated_utility,
    normalize_alignment,
    pool_reward,
    select_top_b,
    tads_score,
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
) -> Dict[str, Any]:
    """Run one episode over the candidate pool and return selection results.

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

    tag = f" | tag={exp_tag}" if exp_tag else ""
    logger.info(
        "collect_episode start | epoch=%d | n=%d | bs=%d | batches=%d | "
        "ratio=%.2f | use_anchor=%s | lam=%.3f | %s%s",
        epoch, total_samples, batch_size, total_batches,
        selection_ratio, apply_anchor, lam, cuda_mem_str(), tag,
    )

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

    # ---- Pool-level composite reward (paper Eq.2-3) ----
    R, r_weight_value = pool_reward(all_r_loss_t, all_r_entropy_t)

    # ---- Calibrated utility R̃ (paper Eq.8 inner) ----
    R_tilde = calibrated_utility(R)

    # ---- Alignment and final score (paper Eq.7 / Eq.8 outer) ----
    if apply_anchor:
        alignment_raw = torch.cat(all_alignment_raw, dim=0).view(-1)
        alignment, _alignment_collapsed = normalize_alignment(alignment_raw)
        if _alignment_collapsed:
            logger.error(
                "TADS alignment COLLAPSED (max-min < 1e-8) | "
                "alignment_raw.std=%.2e. ã set to 0.5 for every sample → "
                "boost becomes constant and TADS reduces to the calibrated-"
                "utility baseline for this epoch. Check anchor PCA gap "
                "or multi-layer cancellation.",
                float(alignment_raw.std().item()),
            )
        score = tads_score(R_tilde, alignment, lam)
        align_mean = float(alignment.mean().item())
        align_std = float(alignment.std().item())
        boost = 1.0 + lam * alignment
        logger.info(
            "TADS score | epoch=%d | lam=%.3f | "
            "R_mean=%.4f | R̃_mean=%.4f | align_mean=%.4f | align_std=%.4f | "
            "boost_mean=%.4f | boost_max=%.4f",
            epoch, lam,
            float(R.mean().item()), float(R_tilde.mean().item()),
            align_mean, align_std,
            float(boost.mean().item()), float(boost.max().item()),
        )
    else:
        score = R_tilde
        alignment = None
        align_mean = None
        align_std = None
        _alignment_collapsed = False
        logger.info(
            "Calibrated-utility-only score (no anchor) | epoch=%d | "
            "R_mean=%.4f | R̃_mean=%.4f",
            epoch, float(R.mean().item()), float(R_tilde.mean().item()),
        )

    # ---- Top-B selection ----
    k = max(1, int(total_samples * selection_ratio))
    if total_samples == 0:
        raise RuntimeError(
            "collect_episode: total_samples == 0 — empty candidate pool. "
            "Check dataset_subset_size / data path.",
        )
    selected_indices: List[int] = select_top_b(score, k).cpu().tolist()
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
        "calibrated_utility": R_tilde,
        "alignment": alignment,
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
