"""PPOAgent 1-step update sanity (no model required)."""
from __future__ import annotations

import torch

from tads.core.agent import PPOAgent


def _build_agent(state_dim: int = 64, device: str = "cpu") -> PPOAgent:
    return PPOAgent(
        state_dim=state_dim,
        hidden_dim=32,
        lr=1e-3,
        ppo_epochs=1,
        mb_size=8,
        advantage_mode="group_relative",
        device=device,
    )


def test_ppo_update_finite_losses():
    agent = _build_agent()
    N, D = 16, 64
    states = torch.randn(N, D)
    with torch.no_grad():
        action, log_prob, _ = agent.ac.get_action(states)
    rewards = torch.randn(N)
    actor_loss, critic_loss = agent.update(
        states=states, actions=action, old_log_probs=log_prob, rewards=rewards,
    )
    assert torch.isfinite(torch.tensor(actor_loss))
    assert torch.isfinite(torch.tensor(critic_loss))


def test_ppo_save_load(tmp_path):
    agent = _build_agent()
    path = tmp_path / "agent.pt"
    agent.save(str(path))
    agent2 = _build_agent()
    agent2.load(str(path))
    sd1 = agent.ac.state_dict()
    sd2 = agent2.ac.state_dict()
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k])
