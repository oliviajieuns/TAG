"""Data Agent (Yang et al., ICML 2026) — PPO-based dynamic data selection.

Faithfully ports the public reference implementation
(https://github.com/Jackbrocp/Data-Agent) to the LLM SFT setting:

  - State: sequence-mean of the model's last hidden layer  (h̄_L(x_i))
  - Actor: 3-layer MLP → Beta(α, β) distribution
  - Action: a_i = Beta(α_i, β_i).sample()   (stochastic; paper does NOT use the
            distribution mean — `dist.sample()` is stored as the per-sample score)
  - Selection rule: top-K candidates by a_i  (paper §3: "samples with the top-k
            highest action weights are selected")
  - Reward:  R_diff = (L_i - L_min) / (L_max - L_min + ε)         (paper Eq.6 numerator-loss)
             R_conf = H_i / H_max
             r      = Var(R_diff) / (Var(R_diff) + Var(R_conf) + ε)   (Eq.5)
             R_i    = r·R_diff + (1-r)·R_conf                         (Eq.6)
  - PPO:     ε_clip=0.2, k_epochs=4, γ=0.99, GAE λ=0.95  (paper Eq.7/Eq.9 + reference repo)

Selection rule note
-------------------
``score = a_i`` (raw Beta sample) — **not** ``R_i · a_i`` or ``R_i + a_i``.
R_i is the PPO training signal only; the actor's a_i alone determines top-K.

Training entrypoint
-------------------
    python -m tads.baselines.data_agent.train \\
        --config configs/experiments/main_7b/llama2/data_agent_10.yaml \\
        --tag DataAgent-PPO
"""
