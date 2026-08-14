"""Episode collection + PPO update + top-K selection for Data Agent.

For one episode (= one selection epoch in the LLM SFT context):

    1. Forward the LLM over every candidate sample (eval mode) to obtain
       per-sample per-token logits + last-layer hidden states.
    2. State  s_i = sequence-mean of last-layer hidden states (padding excluded).
    3. Action a_i, log_prob_i, value_i ← actor.get_action(s_i)
       (action is a Beta(α, β) sample in [0, 1] — stochastic, not the mean).
    4. Per-sample L_i (mean CE loss over response tokens) and H_i (mean
       predictive entropy over response tokens) via tag.core.reward.compute_rewards.
    5. Paper-faithful normalised reward (Eq.5-6, reference repo):
            R_diff = (L_i - L_min) / (L_max - L_min + ε)
            R_conf = H_i / (H_max + ε)
            r      = Var(R_diff) / (Var(R_diff) + Var(R_conf) + ε)
            R_i    = r·R_diff + (1-r)·R_conf
    6. PPO update on (s, a, log_prob, R) — actor improves; next epoch will
       output different a_i for the same candidates.
    7. Selection: top-K indices ranked by a_i (NOT R_i, NOT R_i·a_i).

The reward formula intentionally differs from the legacy score's Eq.2-3 —
that uses raw (unnormalised) L, H and pool-variance w; Data Agent paper
min-max normalises each component first and then takes the same variance
ratio. We follow the paper here even though it diverges from the legacy
score, because this file IS the paper-faithful baseline.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from tag.core.reward import compute_rewards
from tag.core.timing import PhaseTimer
from tag.core.utils import cuda_mem_str

from .agent import PPOAgent

logger = logging.getLogger(__name__)


def _unwrap(model):
    """Strip DDP / PEFT wrappers — same as tag.core.selector._unwrap."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model"):
        m = m.base_model
        if hasattr(m, "model"):
            m = m.model
    return m


def _flatten_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().view(-1).cpu()


def _normalize_min_max(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-paper normalisation: (x - min) / (max - min + ε)."""
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


@torch.no_grad()
def _extract_states(
    model,
    dataset,
    *,
    batch_size: int,
    device: str,
    progress_interval: int = 50,
    empty_cache_interval: int = 10,
) -> Dict[str, torch.Tensor]:
    """Forward the candidate pool once, returning (states, L_i, H_i).

    states = sequence-mean of the last decoder layer's hidden states with the
    attention mask applied (padding excluded). Shape (N, H).
    """
    _was_training = model.training
    model.eval()
    base_model = _unwrap(model)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    total_batches = len(loader)
    total_samples = len(dataset)

    all_states: List[torch.Tensor] = []
    all_r_loss: List[torch.Tensor] = []
    all_r_entropy: List[torch.Tensor] = []

    t0 = time.time()
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

        # Last decoder layer: hidden_states is (embedding, layer_1, ..., layer_L).
        # The final entry is the output of the last decoder block — what the
        # paper calls the "penultimate feature" for the LLM-as-classifier
        # interpretation, and the natural analogue to the image-classifier
        # f^feat_θ(x) the paper defines its state on.
        h_last = out.hidden_states[-1].float()                  # (B, T, H) fp32
        mask_f = attention_mask.to(torch.float32)                # (B, T)
        valid_counts = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        masked = h_last * mask_f.unsqueeze(-1)
        state = masked.sum(dim=1) / valid_counts                 # (B, H) fp32
        all_states.append(state.detach().cpu())
        del h_last, masked, state

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
            logger.info(
                "DataAgent forward | batch=%d/%d | %d/%d (%.1f%%) | "
                "elapsed=%.1fmin | %s",
                step, total_batches, seen, total_samples, pct,
                elapsed / 60, cuda_mem_str(),
            )

    if _was_training:
        model.train()

    return {
        "states": torch.cat(all_states, dim=0),
        "r_loss": torch.cat(all_r_loss, dim=0),
        "r_entropy": torch.cat(all_r_entropy, dim=0),
    }


def collect_episode_and_select(
    model,
    dataset,
    agent: PPOAgent,
    *,
    selection_ratio: float,
    batch_size: int = 1,
    device: str = "cuda",
    epoch: int = 0,
    seed: int = 42,
    exp_tag: Optional[str] = None,
    timer: Optional[PhaseTimer] = None,
) -> Dict[str, Any]:
    """One Data Agent episode: forward → score → PPO update → top-K by a."""
    torch.manual_seed(seed + epoch)
    total_samples = len(dataset)
    k = max(1, int(total_samples * selection_ratio))
    if total_samples == 0:
        raise RuntimeError(
            "DataAgent.collect_episode_and_select: empty candidate pool — "
            "check dataset_subset_size / data path."
        )

    tag = f" | tag={exp_tag}" if exp_tag else ""
    logger.info(
        "DataAgent episode start | epoch=%d | n=%d | bs=%d | ratio=%.3f | "
        "k=%d | %s%s",
        epoch, total_samples, batch_size, selection_ratio, k,
        cuda_mem_str(), tag,
    )

    from contextlib import nullcontext

    def _t(name: str, category: str = "selection"):
        return timer.phase(name, category) if timer is not None else nullcontext()

    # ---- 1. Forward over the candidate pool once ----
    with _t("data_agent.forward_states"):
        extracted = _extract_states(
            model, dataset, batch_size=batch_size, device=device,
        )
    states = extracted["states"]                                 # (N, H)
    L = extracted["r_loss"]                                      # (N,)
    H = extracted["r_entropy"]                                   # (N,)

    # ---- 2. Paper-faithful normalised reward (Eq.5-6 / reference repo) ----
    R_diff = _normalize_min_max(L)                               # (N,) ∈ [0,1]
    H_max = H.max().clamp_min(1e-8)
    R_conf = H / H_max                                           # (N,) ∈ [0,1]
    if R_diff.numel() > 1:
        var_diff = R_diff.var().item()
        var_conf = R_conf.var().item()
    else:
        var_diff = var_conf = 0.0
    eps = 1e-8
    r_weight = var_diff / (var_diff + var_conf + eps)
    rewards = r_weight * R_diff + (1.0 - r_weight) * R_conf      # (N,)

    # ---- 3. Actor scoring (states → a_i, log_prob_i) ----
    with _t("data_agent.actor_scoring"):
        agent.ac.eval()
        states_dev = states.to(device).float()
        actions, log_probs, values = agent.ac.get_action(states_dev)
        actions_cpu = actions.detach().cpu()                     # (N,) ∈ (0,1)

    # ---- 4. PPO update on (s, a, log_prob, R) ----
    with _t("data_agent.ppo_update"):
        try:
            actor_loss, critic_loss = agent.update(
                states=states_dev,
                actions=actions,
                old_log_probs=log_probs,
                rewards=rewards.to(device),
            )
        finally:
            agent.ac.eval()

    # ---- 5. Top-K by action sample (paper: "top-k highest action weights") ----
    # The selection score is a_i alone — NOT R_i and NOT R_i · a_i.
    with _t("data_agent.topk"):
        selected_indices: List[int] = (
            actions_cpu.topk(k).indices.detach().cpu().tolist()
        )

    a_mean = float(actions_cpu.mean().item())
    a_std = float(actions_cpu.std().item()) if actions_cpu.numel() > 1 else 0.0
    R_mean = float(rewards.mean().item())
    logger.info(
        "DataAgent episode done | epoch=%d | selected=%d/%d | "
        "R_mean=%.4f | r_weight=%.4f | "
        "a_mean=%.4f | a_std=%.4f | "
        "actor_loss=%.4f | critic_loss=%.4f | first5=%s",
        epoch, k, total_samples, R_mean, r_weight,
        a_mean, a_std, actor_loss, critic_loss, selected_indices[:5],
    )

    return {
        "selected_indices": selected_indices,
        "actions": actions_cpu,
        "rewards": rewards.detach().cpu(),
        "r_weight": r_weight,
        "r_loss_mean": float(L.mean().item()),
        "r_entropy_mean": float(H.mean().item()),
        "a_mean": a_mean,
        "a_std": a_std,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "state_dim": int(states.size(1)),
    }
