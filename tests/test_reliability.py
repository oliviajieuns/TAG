"""Unit tests for tag.core.reliability — completeness gate (text + token
level), the calibrated sigmoid Q gate, and cache round-trip. The
counterfactual forward pass itself is exercised end-to-end on GPU runs;
here we test the pure logic around it."""
from __future__ import annotations

import math

import torch

from tag.core.reliability import (
    calibrate_reliability_scale,
    completeness_from_dataset,
    load_reliability_cache,
    reliability_from_losses,
    save_reliability_cache,
)
from tag.data.sft_prompts import text_is_complete

EOS = 2


class _FakeDataset:
    """Minimal stand-in for the tokenised HF dataset (labels only, plus an
    optional per-row raw-text completeness flag)."""

    def __init__(self, label_rows, text_flags=None):
        self.rows = [torch.tensor(r, dtype=torch.long) for r in label_rows]
        self.text_flags = text_flags

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = {"labels": self.rows[i]}
        if self.text_flags is not None:
            row["text_complete"] = self.text_flags[i]
        return row


def test_completeness_eos_gate():
    ds = _FakeDataset([
        [-100, -100, 5, 6, EOS],        # complete: ends with EOS
        [-100, 5, 6, 7, 8],             # truncated: no EOS
        [-100, -100, -100, -100, -100],  # no response tokens at all
        [-100, 5, EOS, -100, -100],     # EOS then padding-masked tail
    ])
    c = completeness_from_dataset(ds, eos_token_id=EOS, c_trunc=0.2)
    assert torch.allclose(c, torch.tensor([1.0, 0.2, 0.2, 1.0]))


def test_completeness_uses_text_flag():
    """The §1.2 bug regression: tokenisation appends EOS to EVERY response,
    so a T3-truncated text still ends with EOS at the token level. Only the
    raw-text flag can catch it — a row with EOS but text_complete=0 must
    receive c_trunc."""
    ds = _FakeDataset(
        [
            [-100, 5, 6, EOS],   # tokens fine, text fine     → 1.0
            [-100, 5, 6, EOS],   # tokens fine, TEXT CUT (T3) → c_trunc
            [-100, 5, 6, 7],     # no EOS (max_seq_len cut)   → c_trunc
        ],
        text_flags=[1, 0, 1],
    )
    c = completeness_from_dataset(ds, eos_token_id=EOS, c_trunc=0.2)
    assert torch.allclose(c, torch.tensor([1.0, 0.2, 0.2]))


def test_text_is_complete_heuristic():
    assert text_is_complete("The answer is 42")            # numeric ending
    assert text_is_complete("It works.")
    assert text_is_complete('He said "done."')             # wrapped ending
    assert text_is_complete("```python\nprint(1)\n```")    # closed fence
    assert not text_is_complete("")
    assert not text_is_complete("and then the")            # mid-sentence cut
    assert not text_is_complete("```python\nprint(1)")     # open fence
    # T3-style: truncate_text strips trailing .!?"')]} — the remainder ends
    # on a bare word and must be flagged.
    assert not text_is_complete("The capital of France is Paris and the")


def test_structured_endings_are_complete():
    """c_i multiplies the score by c_trunc (0.2), so a false positive is a
    five-fold demotion. The punctuation-only heuristic flagged 14.6% of
    alpaca-cleaned incomplete — far above the true truncation rate — and the
    excess was almost all lists, tables and one-word answers. Structured
    content is written without terminal punctuation as a matter of style."""
    assert text_is_complete("Three tips:\n- Sleep well\n- Eat greens\n- Walk")
    assert text_is_complete("Steps:\n1. Preheat the oven\n2. Mix the batter")
    assert text_is_complete("| Name | Age |\n| --- | --- |\n| Ann | 30 |")
    assert text_is_complete("Name: Ann\nAge: 30")
    # A single line that merely CONTAINS a dash is not a list.
    assert not text_is_complete("The plan - which we discussed - is to")


def test_terse_answers_are_complete_but_dangling_words_are_not():
    """A one- or two-word answer is the dataset's style, not a truncation.
    The guard is the trailing word: a cut lands on whatever was there, very
    often a function word; a deliberate terse answer essentially never does."""
    assert text_is_complete("Paris")
    assert text_is_complete("Blue whale")
    assert text_is_complete("Highly recommended")
    assert not text_is_complete("and then the")
    assert not text_is_complete("It was")
    assert not text_is_complete("because of")
    # Past the short-answer window the rule does not apply at all.
    assert not text_is_complete("One two three four five six seven")


def test_truncated_corruption_is_still_caught_after_the_new_rules():
    """The rules added for precision must not cost the recall they exist to
    protect. Run T3 over realistic responses and check the flag rate."""
    import random

    from tag.data.corruption import truncate_text

    rng = random.Random(0)
    originals = [
        "The mitochondria is the powerhouse of the cell. It generates most "
        "of the chemical energy needed to power the cell's biochemical "
        "reactions, which is stored in ATP.",
        "To bake bread you need flour, water, yeast and salt. Mix them "
        "together, knead for ten minutes, let the dough rise, then bake at "
        "220 degrees for half an hour.",
        "Machine learning models learn patterns from data rather than being "
        "explicitly programmed. The quality of the training data therefore "
        "bounds the quality of the model.",
        "Photosynthesis converts light energy into chemical energy. Plants "
        "use sunlight, water and carbon dioxide to produce glucose and "
        "release oxygen as a by-product.",
    ]
    for text in originals:
        assert text_is_complete(text), text
        cut = truncate_text(text, rng)
        assert not text_is_complete(cut), cut


def test_completeness_validates_c_trunc():
    ds = _FakeDataset([[5, EOS]])
    for bad in (0.0, -1.0, 1.5):
        try:
            completeness_from_dataset(ds, EOS, c_trunc=bad)
            assert False
        except ValueError:
            pass


def test_reliability_shape_mismatch_raises():
    try:
        reliability_from_losses(torch.rand(3), torch.rand(4))
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# v3 calibrated sigmoid gate
# ---------------------------------------------------------------------------

def test_calibration_hits_target_quantile():
    """By construction: σ(P10(ΔL_clean)/s) must equal target_q exactly."""
    ref = torch.linspace(0.1, 2.0, steps=1000)  # clean reference ΔL, all > 0
    s = calibrate_reliability_scale(ref, target_pct=0.10, target_q=0.8)
    p10 = float(torch.quantile(ref, 0.10).item())
    assert abs(1.0 / (1.0 + math.exp(-p10 / s)) - 0.8) < 1e-6


def test_sigmoid_gate_zero_anchor():
    """ΔL ≤ 0 (counterfactual predicts as well or better than the true
    instruction) must map to Q = 0 under the re-zeroed gate — the physical
    reference point that rank01 lacks."""
    loss_orig = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss_cf = torch.tensor([1.0, 0.5, 2.0, 4.0])  # ΔL = 0, -0.5, 1.0, 3.0
    q = reliability_from_losses(loss_orig, loss_cf, mode="sigmoid", scale=1.0)
    assert q[0] == 0.0 and q[1] == 0.0
    assert 0.0 < q[2] < q[3] <= 1.0


def test_sigmoid_gate_is_pool_independent():
    """The v1 rank gate's structural defect: gate strength depended on pool
    composition. The calibrated sigmoid must give the SAME Q to the same
    (ΔL, s) regardless of what the rest of the pool looks like."""
    s = 0.7
    # Same sample (ΔL = 2.0) embedded in a clean pool vs an 80%-dirty pool.
    orig_a = torch.zeros(5)
    cf_a = torch.tensor([2.0, 1.9, 2.1, 1.8, 2.2])
    orig_b = torch.zeros(5)
    cf_b = torch.tensor([2.0, 0.01, 0.0, 0.02, 0.01])
    qa = reliability_from_losses(orig_a, cf_a, mode="sigmoid", scale=s)
    qb = reliability_from_losses(orig_b, cf_b, mode="sigmoid", scale=s)
    assert torch.isclose(qa[0], qb[0])
    # rank mode, by contrast, is pool-dependent for the identical sample.
    ra = reliability_from_losses(orig_a, cf_a, mode="rank")
    rb = reliability_from_losses(orig_b, cf_b, mode="rank")
    assert not torch.isclose(ra[0], rb[0])


def test_rank_mode_is_v1_ablation():
    lo = torch.rand(50)
    lc = lo + torch.randn(50)
    from tag.core.scorer import rank01
    q = reliability_from_losses(lo, lc, mode="rank")
    assert torch.allclose(q, rank01(lc - lo))


def test_k_counterfactuals_dispersion_discount():
    """K > 1: identical counterfactuals must reproduce K=1; disagreeing
    counterfactuals must discount Q."""
    lo = torch.zeros(3)
    cf1 = torch.tensor([2.0, 2.0, 2.0])
    q1 = reliability_from_losses(lo, cf1, mode="sigmoid", scale=1.0)
    q_same = reliability_from_losses(
        lo, torch.stack([cf1, cf1]), mode="sigmoid", scale=1.0,
    )
    assert torch.allclose(q1, q_same)
    cf2 = torch.tensor([2.0, 0.0, 2.0])  # disagrees on sample 1
    q_dis = reliability_from_losses(
        lo, torch.stack([cf1, cf2]), mode="sigmoid", scale=1.0,
    )
    assert q_dis[1] < q_same[1]
    assert torch.isclose(q_dis[0], q_same[0])  # agreement → no discount


def test_cache_roundtrip(tmp_path):
    q = torch.rand(10)
    c = torch.ones(10)
    lo = torch.rand(10)
    lc = lo + torch.rand(10)
    save_reliability_cache(
        tmp_path, q=q, completeness=c, loss_orig=lo, loss_cf=lc, epoch=1,
        mode="sigmoid", scale=0.5, rezero=True,
    )
    cache = load_reliability_cache(tmp_path)
    assert cache is not None
    assert torch.allclose(cache["q"], q)
    assert torch.allclose(cache["completeness"], c)
    assert cache["epoch"] == 1
    assert cache["mode"] == "sigmoid"
    assert cache["scale"] == 0.5
    assert cache["rezero"] is True


def test_cache_missing_returns_none(tmp_path):
    assert load_reliability_cache(tmp_path) is None
