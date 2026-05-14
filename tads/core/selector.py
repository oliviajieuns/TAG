"""Episode collection and selection scoring (paper §3, Algorithm 1).

For each candidate sample we compute:
    h_i      = last-token hidden state under current θ_t
    R_i      = composite reward (paper Eq. 6)
    a_i      = PPO actor action ∈ [0, 1]
    Align_i  = compute_alignment(h_i) ∈ [0, 1]   (TADS only)

The selection score is
    s_i^(t) = R_i · a_i · (1 + λ · Align_i^(t))
and the top-K samples form the training subset for this epoch.
Setting λ=0 (or use_anchor=False) recovers the Data Agent baseline.

Reward components (paper Eq. 1, 3, 5, 6):
    r_loss_i    = mean CE loss over response tokens          (== rdiff in the
                                                              Data Agent paper)
    r_entropy_i = mean predictive entropy                     (== rconf)
    r_weight    = Var(r_loss) / (Var(r_loss) + Var(r_entropy) + eps)  (== r)
    R_i         = r_weight * r_loss_i + (1 - r_weight) * r_entropy_i

Adaptive weight scope (paper Eq. 5):
    r_weight is computed at the DATASET LEVEL (over all N candidate
    samples), not per-batch. Per-batch variance is degenerate when
    batch_size=1 (forces var=0 -> r_weight=0). We therefore accumulate
    r_loss / r_entropy across all batches and compute r_weight once
    after the loop, then derive composite reward from it.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .agent import PPOAgent
from .reward import compute_rewards
from .trajectory_anchor import TrajectoryAnchor
from .utils import cuda_mem_str

logger = logging.getLogger(__name__)


def _flatten_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().view(-1).cpu()


def _unwrap(model):
    """Strip DDP / PEFT wrappers to reach the underlying HF model.

    Mirrors the unwrap chain in :func:`tads.modeling.loader.get_hidden_size`:
    DDP exposes the inner module via ``.module``, and PEFT's ``PeftModel``
    nests the base HF model under ``.base_model.model``. Either or both can
    be present depending on training_mode and DDP launch.
    """
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
    agent: PPOAgent,
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

    ``exp_tag`` is a free-form string (e.g. ``"qwen2.5-7b/alpaca/tads"``) used
    only for log readability. It does not affect any numerics.

    Memory: ``empty_cache_interval=10`` keeps the CUDA caching allocator from
    fragmenting across the ~3000 episode batches. With Llama-2-7B + episode_
    batch_size=16 the per-batch peak is ~5 GB (logits + entropy intermediates);
    a long run without periodic empty_cache can fragment the allocator until a
    new ~1 GB block can't be placed even though gross free memory is high.
    """
    torch.manual_seed(seed + epoch)
    model.eval()
    agent.ac.eval()
    # KV cache adds nothing during a feed-forward pass and just leaks memory
    # batch over batch on Mistral / Qwen (both default to use_cache=True).
    # NB: DDP and PEFT both wrap the base model; .config lives on the inner
    # HF causal-LM, not on the wrapper.
    base_model = _unwrap(model)
    _orig_use_cache = getattr(base_model.config, "use_cache", None)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    all_states: List[torch.Tensor] = []
    all_actions: List[torch.Tensor] = []
    all_log_probs: List[torch.Tensor] = []
    all_r_loss: List[torch.Tensor] = []
    all_r_entropy: List[torch.Tensor] = []
    all_alignment_raw: List[torch.Tensor] = []  # streaming alignment (per-batch)

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

    # Pre-cache the per-layer anchor directions on GPU (each is (H,) fp32, total
    # ~512 KB for 32 layers × 4096 H). Done once outside the batch loop so we
    # don't pay a CPU→GPU copy per batch per layer.
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

        # hidden_states[0] is the embedding; decoder layers occupy [1:].
        # We always store the LAST decoder layer's last-token hidden as the
        # state for the PPO actor (shape (B, H)). For multi-layer anchor mode
        # we ALSO compute the per-batch NAIT Eq.5 alignment inline and only
        # accumulate the (B,) scalar — instead of stacking (B, L, H) deltas
        # and aggregating after the loop, which used to peak at ~27 GB on
        # CPU for Llama-2-7B (52K samples × 32 layers × 4096 H × 4 bytes).
        decoder_hidden = out.hidden_states[1:]
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        lengths = lengths.to(decoder_hidden[0].device)
        bidx = torch.arange(decoder_hidden[0].size(0), device=decoder_hidden[0].device)

        # Always (B, H) — last layer, last real token. This is what the actor
        # consumes both at action-sampling time AND inside agent.update.
        agent_input = decoder_hidden[-1][bidx, lengths].detach().float().cpu()

        if apply_anchor:
            B_local = input_ids.size(0)
            if trajectory_anchor.is_multi_layer:
                # Σ_l ⟨Δh_l, v_l⟩  (NAIT Eq.5), accumulated on GPU and then
                # moved to CPU as a small (B,) vector.
                batch_align = torch.zeros(B_local, dtype=torch.float32, device=device)
                for li in trajectory_anchor.layer_indices:
                    h_l = decoder_hidden[li]
                    first_h = h_l[:, 0, :]
                    last_h = h_l[bidx, lengths]
                    delta_l = (last_h - first_h).float()  # (B, H) fp32
                    batch_align += delta_l @ v_cache_gpu[li]
                    del delta_l
                all_alignment_raw.append(batch_align.detach().cpu())
                del batch_align
            else:
                # Legacy single-layer mode: dot-product the last-token hidden
                # with the single anchor direction. Matches the pre-refactor
                # `compute_alignment([N, H])` path (NOT the delta).
                v = v_cache_gpu[trajectory_anchor.layer_indices[0]]
                last_token_gpu = decoder_hidden[-1][bidx, lengths].float()
                all_alignment_raw.append((last_token_gpu @ v).detach().cpu())
                del last_token_gpu

        all_states.append(agent_input)
        del decoder_hidden

        # FIX: ignore per-batch r_weight (degenerate at batch_size=1).
        # r_weight is computed once at dataset level after loop.
        r_loss, r_entropy, _ = compute_rewards(out.logits, labels)

        r_loss_cpu = _flatten_cpu_float(r_loss)
        r_entropy_cpu = _flatten_cpu_float(r_entropy)
        # Drop the model output (logits is the heaviest tensor on GPU at this
        # point — (B, T, V) bf16) BEFORE running the actor / appending CPU
        # buffers, so the next batch's forward starts with maximum free space.
        del out

        states_for_agent = agent_input.to(agent.device, non_blocking=True)
        action, log_prob, _ = agent.ac.get_action(states_for_agent)

        # all_states already received agent_input above the compute_rewards
        # block; here we only append the per-batch action / reward outputs.
        all_actions.append(_flatten_cpu_float(action))
        all_log_probs.append(_flatten_cpu_float(log_prob))
        all_r_loss.append(r_loss_cpu)
        all_r_entropy.append(r_entropy_cpu)

        del input_ids, attention_mask, labels, states_for_agent
        del action, log_prob, r_loss, r_entropy, r_loss_cpu, r_entropy_cpu

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

    if not all_states:
        raise RuntimeError("collect_episode produced no states.")

    all_states = torch.cat(all_states, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_r_loss = torch.cat(all_r_loss, dim=0)
    all_r_entropy = torch.cat(all_r_entropy, dim=0)

    # ---- Dataset-level adaptive weight r_weight (paper Eq. 5) ----
    eps = 1e-8
    if all_r_loss.numel() > 1:
        var_loss = all_r_loss.var()
        var_entropy = all_r_entropy.var()
        r_weight = var_loss / (var_loss + var_entropy + eps)
    else:
        r_weight = torch.tensor(0.5)
    r_weight_value = float(r_weight.item())

    # ---- Composite reward per sample (paper Eq. 6) ----
    all_rewards = r_weight * all_r_loss + (1.0 - r_weight) * all_r_entropy

    # ---- Selection score s_i = R_i · a_i · (1 + λ · Align_i) ----
    R = all_rewards.view(-1)
    a = all_actions.view(-1)

    if apply_anchor:
        # Streaming alignment: per-batch raw scores were accumulated inside the
        # episode loop. Concat and apply the same min-max normalisation as
        # `TrajectoryAnchor.compute_alignment` so downstream behaviour is
        # bit-equivalent to the old "stack all states then project" path.
        alignment_raw = torch.cat(all_alignment_raw, dim=0).view(-1)
        a_min, a_max = alignment_raw.min(), alignment_raw.max()
        if (a_max - a_min) > 1e-8:
            alignment = (alignment_raw - a_min) / (a_max - a_min)
        else:
            alignment = torch.full_like(alignment_raw, 0.5)
        boost = 1.0 + lam * alignment
        score = R * a * boost
        align_mean = float(alignment.mean().item())
        align_std = float(alignment.std().item())
        boost_mean = float(boost.mean().item())
        boost_max = float(boost.max().item())
        logger.info(
            "TADS score | epoch=%d | lam=%.3f | "
            "align_mean=%.4f | align_std=%.4f | "
            "boost_mean=%.4f | boost_max=%.4f",
            epoch, lam, align_mean, align_std, boost_mean, boost_max,
        )
    else:
        score = R * a
        align_mean = None
        align_std = None
        logger.info("DataAgent score (no anchor) | epoch=%d", epoch)

    k = max(1, int(total_samples * selection_ratio))
    print(f"[diag-selector] total_samples={total_samples} ratio={selection_ratio} k={k} score.shape={tuple(score.shape)} dtype={score.dtype} has_nan={bool(torch.isnan(score).any())} has_inf={bool(torch.isinf(score).any())}", flush=True)
    selected_indices: List[int] = score.topk(k).indices.cpu().tolist()
    print(f"[diag-selector] selected_indices type={type(selected_indices).__name__} len={len(selected_indices)} first5={selected_indices[:5]}", flush=True)

    var_loss_val = float(all_r_loss.var().item()) if all_r_loss.numel() > 1 else 0.0
    var_entropy_val = float(all_r_entropy.var().item()) if all_r_entropy.numel() > 1 else 0.0

    elapsed = time.time() - t0
    logger.info(
        "Episode done | epoch=%d | selected=%d/%d | "
        "R_loss=%.4f (var=%.6f) | R_entropy=%.4f (var=%.6f) | "
        "r_weight=%.4f | reward_mean=%.4f | elapsed=%.1fmin | %s",
        epoch, k, total_samples,
        all_r_loss.mean().item(), var_loss_val,
        all_r_entropy.mean().item(), var_entropy_val,
        r_weight_value,
        all_rewards.mean().item(),
        elapsed / 60, cuda_mem_str(),
    )

    r_loss_mean = float(all_r_loss.mean().item())
    r_entropy_mean = float(all_r_entropy.mean().item())

    # Restore the model's use_cache so subsequent eval-time generation paths
    # (which DO want a KV cache) get their original behaviour back.
    if hasattr(base_model.config, "use_cache") and _orig_use_cache is not None:
        base_model.config.use_cache = _orig_use_cache

    return {
        "selected_indices": selected_indices,
        "states": all_states,
        "actions": all_actions,
        "log_probs": all_log_probs,
        "rewards": all_rewards,
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
    }
