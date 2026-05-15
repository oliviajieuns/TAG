"""PPO Actor-Critic for data-selection.

Single canonical implementation (formerly ``agent_v2``). Supports two
advantage modes via ``advantage_mode``:

- ``"group_relative"`` (default, recommended): advantages are computed
  across the current pool of candidate samples as a standardised reward,
  which matches the score-based top-K selection (no time axis).
- ``"gae"``: standard Generalised Advantage Estimation. Available for
  ablations; treats the sample list as a sequence so its temporal
  semantics are weak in this setting.

Mini-batch updates and value clipping are both enabled by default.
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
    """Shared MLP trunk → Beta(α, β) actor + scalar critic."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
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
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        alpha, beta, value = self.forward(states)
        dist = Beta(alpha, beta)
        actions = actions.clamp(min=1e-6, max=1.0 - 1e-6)
        return dist.log_prob(actions), dist.entropy(), value


class PPOAgent:
    """Clipped-surrogate PPO with optional group-relative advantage."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        mb_size: int = 1024,
        advantage_mode: str = "group_relative",
        value_clip: bool = True,
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
            "PPOAgent | state_dim=%d mb=%d adv=%s vclip=%s",
            state_dim, mb_size, advantage_mode, value_clip,
        )

    # -------------------------------------------------------- advantage modes
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

    # ----------------------------------------------------------------- update
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[float, float]:
        # collect_episode put the actor-critic in eval(). The current trunk
        # has no dropout / BN so eval-vs-train is a no-op today, but PPO
        # gradient updates conceptually want train mode and any future
        # dropout/BN added to ActorCritic would silently train under eval.
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
        mb_size = min(self.mb_size, N)
        actor_losses, critic_losses = [], []

        # PPO advantage normalisation needs ≥2 samples per minibatch, so
        # episodes with N < 2 produce no gradient updates — and the
        # return-path average over empty lists then reports actor_loss=0,
        # critic_loss=0, which looks indistinguishable from a successful
        # zero-loss step. Refuse rather than fake success.
        if N < 2:
            raise RuntimeError(
                f"PPOAgent.update: need at least 2 samples to normalise "
                f"advantages, got N={N}. Increase episode size or "
                f"selection_ratio.",
            )

        # Ensure last partial minibatch is also processed — but only if it
        # has at least 2 samples (else advantage std collapses). Mark below.
        self.ac.train()
        for _ in range(self.ppo_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, mb_size):
                mb = idx[start:start + mb_size]
                if mb.numel() < 2:
                    # tail minibatch of size 1 — skip silently (the previous
                    # full minibatches already produced a gradient step).
                    continue

                log_prob, entropy, value = self.ac.evaluate(states[mb], actions[mb])
                adv_mb = advantages[mb]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
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

    # ---------------------------------------------------------------- I/O
    def save(self, path: str) -> None:
        torch.save({
            "ac_state_dict": self.ac.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        # map_location="cpu" + weights_only=False: PPO actor-critic is tiny
        # (~1 M params) so the GPU peak isn't a concern here, but we keep
        # the same convention as the main trainer so future PyTorch versions
        # don't trip on the weights_only=True default.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.ac.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # The optimizer was loaded with map_location="cpu" inside the
        # checkpoint, so its Adam momentum / second-moment tensors live on
        # CPU while the actor-critic parameters have just been moved to GPU.
        # Calling optimizer.step() in that state raises
        #   RuntimeError: Expected all tensors to be on the same device
        # on the very first PPO update after resume. Push every state tensor
        # to the model's device explicitly. (torch.optim.Optimizer has no
        # .to() of its own — this is the standard manual pattern.)
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device, non_blocking=True)
