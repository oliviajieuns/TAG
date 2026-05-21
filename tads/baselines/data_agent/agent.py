"""PPO Actor-Critic for the Data Agent baseline (Yang et al., ICML 2026).

Ported from the previous TADS implementation (`tads/core/agent.py` at commit
1889eaf~1 — removed when TADS dropped its PPO actor) and aligned to the
paper / reference repo (Jackbrocp/Data-Agent):

    Actor       :  3-layer MLP → Beta(α, β) with `softplus + 1.0`
    Critic      :  shared trunk, scalar head
    PPO         :  ε_clip=0.2, k_epochs=4, γ=0.99, GAE λ=0.95
    Entropy bonus: 0.0 (paper has no entropy term; configurable)
    Value clip  :  off by default (paper does not value-clip; configurable)

The numerical guards from the old TADS PPO implementation (Beta-sample
clamp before log_prob, advantage normalisation on the full rollout, optimizer-
state device migration on resume) are kept — they are paper-orthogonal
stability fixes that prevent NaN propagation under fp16/bf16 forwards.
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Shared MLP trunk → Beta(α, β) actor + scalar critic.

    The reference implementation (Jackbrocp/Data-Agent: model.py) uses
    ``hidden_dim=128`` and separate actor/critic trunks. We share the
    trunk because the state is a single per-sample feature vector and the
    two heads agree on the same representation; the parameter count
    difference (≈100 K params either way) is irrelevant next to the
    LLM forward cost.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head_alpha = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)
        self.head_value = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor):
        h = self.trunk(states)
        alpha = F.softplus(self.head_alpha(h)).squeeze(-1) + 1.0
        beta = F.softplus(self.head_beta(h)).squeeze(-1) + 1.0
        value = self.head_value(h).squeeze(-1)
        return alpha, beta, value

    @torch.no_grad()
    def get_action(self, states: torch.Tensor):
        alpha, beta, value = self.forward(states)
        dist = Beta(alpha, beta)
        # Clamp before log_prob: when alpha or beta is large the FP-rounded
        # sample can land exactly at 0.0 or 1.0, sending log_prob to -inf
        # and poisoning the stored old_log_prob — the next PPO update then
        # computes ratio = exp(new - old) = exp(+inf) = inf and NaNs cascade.
        action = dist.sample().clamp(min=1e-6, max=1.0 - 1e-6)
        return action, dist.log_prob(action), value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        alpha, beta, value = self.forward(states)
        dist = Beta(alpha, beta)
        actions = actions.clamp(min=1e-6, max=1.0 - 1e-6)
        return dist.log_prob(actions), dist.entropy(), value


class PPOAgent:
    """Clipped-surrogate PPO with group-relative or GAE advantage.

    Defaults match the paper recipe:
        clip_eps=0.2, ppo_epochs=4, gamma=0.99, gae_lam=0.95,
        entropy_coef=0.0 (paper has no entropy term), value_clip=False.

    ``advantage_mode`` controls advantage estimation:
        "group_relative" — A = (R - mean) / std over the full episode pool.
            Recommended for LLM episodes where samples are scored once per
            epoch with no temporal coupling between them.
        "gae"            — standard GAE(γ, λ) over the candidate sequence.
            Available for paper-strict comparison; the temporal semantics
            are weak when consecutive samples don't share state evolution.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        mb_size: int = 1024,
        advantage_mode: str = "group_relative",
        value_clip: bool = False,
        device: str = "cuda",
    ):
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.mb_size = mb_size
        self.advantage_mode = advantage_mode
        self.value_clip = value_clip
        self.device = device
        self.ac = ActorCritic(state_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)
        logger.info(
            "DataAgent PPOAgent | state_dim=%d hidden=%d mb=%d adv=%s "
            "vclip=%s ent=%.3f clip=%.2f k=%d",
            state_dim, hidden_dim, mb_size, advantage_mode, value_clip,
            entropy_coef, clip_eps, ppo_epochs,
        )

    def _compute_group_advantage(self, rewards: torch.Tensor):
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return adv, rewards

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor):
        N = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device)
        values_ext = torch.cat([values, values.new_zeros(1)])
        for t in reversed(range(N)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] - values_ext[t]
            gae = delta + self.gamma * self.gae_lam * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[float, float]:
        self.ac.train()
        states = states.to(self.device).float()
        actions = actions.to(self.device).float()
        old_log_probs = old_log_probs.to(self.device).float()
        rewards = rewards.to(self.device).float()

        with torch.no_grad():
            _, _, old_values = self.ac.forward(states)

        if self.advantage_mode == "group_relative":
            advantages, returns = self._compute_group_advantage(rewards)
        else:
            advantages, returns = self._compute_gae(rewards, old_values)

        N = states.shape[0]
        if N < 2:
            raise RuntimeError(
                f"PPOAgent.update: need at least 2 samples to normalise "
                f"advantages, got N={N}. Increase episode size or "
                f"selection_ratio.",
            )
        mb_size = min(self.mb_size, N)
        actor_losses, critic_losses = [], []

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, mb_size):
                mb = idx[start:start + mb_size]
                if mb.numel() < 2:
                    continue

                log_prob, entropy, value = self.ac.evaluate(states[mb], actions[mb])
                adv_mb = advantages[mb]
                ratio = torch.exp(log_prob - old_log_probs[mb])
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                if self.value_clip:
                    v_clipped = old_values[mb] + (value - old_values[mb]).clamp(
                        -self.clip_eps, self.clip_eps,
                    )
                    critic_loss = torch.max(
                        (value - returns[mb]).pow(2),
                        (v_clipped - returns[mb]).pow(2),
                    ).mean()
                else:
                    critic_loss = F.mse_loss(value, returns[mb])

                total = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return (
            sum(actor_losses) / max(1, len(actor_losses)),
            sum(critic_losses) / max(1, len(critic_losses)),
        )

    def save(self, path: str) -> None:
        torch.save({
            "ac_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.ac.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Optimizer state was loaded onto CPU but the actor-critic is now on
        # GPU. Calling .step() in that state raises "Expected all tensors to
        # be on the same device" — manually migrate every state tensor.
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device, non_blocking=True)
