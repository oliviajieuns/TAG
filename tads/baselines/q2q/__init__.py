"""Q2Q / Cherry_LLM (Li et al., 2024 NAACL) — self-guided IFD-based selection.

Paper: "From Quantity to Quality: Boosting LLM Performance with Self-Guided
Data Selection for Instruction Tuning." arXiv:2308.12032
Official repo: github.com/tianyi-lab/Cherry_LLM

Core idea (IFD = Instruction-Following Difficulty):
    IFD(x, y) = PPL(y | x) / PPL(y)
High-IFD samples (response is hard to generate even given the instruction)
are the most informative for SFT.
"""
