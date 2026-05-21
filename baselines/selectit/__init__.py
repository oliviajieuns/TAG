"""SelectIT (Liu et al., 2024b) — uncertainty-based data selection.

Ported from https://github.com/Blue-Raincoat/SelectIT. Scoring lives in
``score.py`` (token-level Eq.2 + sentence-level Eq.4 over a fixed
rating-prompt template, with a ``_double_softmax`` NaN guard). Training
entrypoint is ``python -m baselines.selectit.train``.
"""
