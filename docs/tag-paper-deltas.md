# TAG: proposed amendments to the paper's equations and wording

**Status:** ACTIVE. Written 2026-08-13 while implementing Eqs. 1-6 in
`tag/core/gate.py`. Every item below was found by making the equations
executable — either the maths is undefined on a case the data actually
produces, or the code had to make a choice the paper does not specify, or
the claim as written is stronger than what the code can deliver.

**How to read this.** Each item states what the paper says, what the code
does, why they differ, and the suggested replacement text. Items in §A are
**required** — as written the paper is undefined or wrong on inputs that
occur in every real pool. §B are **recommended** and are about whether the
experiment will survive review. §C is wording precision.

Nothing here changes the method's story: the gate is still a calibrated,
zero-anchored counterfactual contrast used as a multiplicative reliability weight. The changes make it
well-defined and honest about its failure modes.

---

## A. Required — the equations are undefined or wrong as written

### A1. Eq. 5 is undefined when \(\mathcal{C}_i\) is empty

**Paper.** \(\Delta^{\min}_i = \min_{m : \mathcal{S}_m \in \mathcal{C}_i}
\Delta_{i,m}\), where \(\mathcal{C}_i\) excludes low-information spans.

**Problem.** A minimum over the empty set is undefined, and
\(\mathcal{C}_i = \emptyset\) is common: any response that is entirely
boilerplate, or shorter than one admissible span, excludes every span. On a
short-answer-heavy pool this is a few percent of samples. Left undefined,
the natural implementations disagree wildly — \(+\infty\) (never gate) and
\(-\infty\) (always gate) are both "natural".

**Code.** `tail_gain(...)` falls back to \(\bar\Delta_i\): the tail test
abstains and the overall gain decides alone. Zeroing would punish short
answers for being short; passing unconditionally would blind the gate.

**Suggested text.** Add to Eq. 5:

> \[\Delta^{\min}_i = \begin{cases}
> \min_{m:\mathcal{S}_m\in\mathcal{C}_i}\Delta_{i,m} & \mathcal{C}_i \neq \emptyset\\
> \bar\Delta_i & \text{otherwise,}
> \end{cases}\]
> so that when no span carries enough instruction-dependent content to
> judge, the tail test abstains and \(\hat\Delta_i = \bar\Delta_i\).

### A2. Eq. 3's two forms are equal only over a common token set

**Paper.** \(\bar\Delta_i = \frac{\sum_k \delta_{i,k}}{\sum_k \ell_k(y_i|x_i^-)}
= 1 - \frac{L(y_i|x_i)}{L(y_i|x_i^-)}\).

**Problem.** The second equality silently assumes the two sums run over the
same \(k\). They do not, in general: the response is truncated to
`max_seq_len - len(prompt_ids)` and \(x_i\) and \(x_i^-\) have different
lengths, so the true and counterfactual copies of the SAME response can end
at different tokens. (A counterfactual instruction longer than the true one
can even trigger the prompt-overflow guard and leave a 1-token response.)
Comparing token \(k\) of one against nothing in the other is not a contrast.

**Code.** `spans_from_token_losses(...)` trims both sides to
\(k \le \min(n_i^{\text{true}}, n_i^{\text{cf}})\) before anything else, and
samples whose common prefix is shorter than `min_common_tokens` are marked
*undefined* (see A3).

**Suggested text.** After Eq. 2, one sentence:

> Both sums range over the common prefix \(k \le n_i =
> \min(n_i^{\text{true}}, n_i^{\text{cf}})\) of the response as tokenised
> under the two instructions; the response token ids themselves are
> identical by construction, since prompt and response are encoded
> separately, so only the length budget can differ.

### A3. No rule for samples with no usable evidence

**Paper.** Silent.

**Problem.** When the common prefix of A2 is too short (a handful of
tokens), \(\bar\Delta_i\) and \(\Delta^{\min}_i\) are computable but
meaningless, and their noise is large enough to cross the zero anchor
routinely. Zeroing a sample because of a tokenisation artifact is the worst
possible failure mode for a method whose entire pitch is "reliability is a
weight whose floor is attainable".

**Code.** `undefined_policy`, default `pass`: \(G_i = c_i\) — no evidence,
no verdict. `neutral` and `zero` exist as ablation arms, and the count is
logged and reported.

**Suggested text.** A footnote to Eq. 6:

> Samples whose common response prefix is shorter than a minimum length
> carry no usable contrast; we set \(G_i = c_i\) for these (no evidence, no
> verdict) and report their count. On our pools this is under 1%.

*(Replace "under 1%" with the measured number — `report["tag"]["n_undefined"]`.)*

### A4. Eq. 1 has no rule for a budget larger than the admissible set

**Paper.** "a zeroed gate cannot be compensated by any amount of difficulty
or alignment evidence".

**Problem.** True of the SCORE, but selection is top-\(B\). If
\(|\{i : G_i > 0\}| < B\) the selector must fill the remaining slots with
zero-weight samples, all of which score exactly 0. The non-compensation claim then silently
fails for those slots — and worse, ties at exactly 0 are broken by
`torch.topk`'s index order, i.e. by the candidate pool's FILE ORDER.

**Code.** `scorer.gated_selection_key` orders every admissible sample above
every zeroed one and breaks each block by a meaningful statistic (the gated
score above, the ungated score below); the shortfall is logged as a warning
and reported per selection ratio as `budget_fits@K`.

**Two further subtleties the code had to settle.**

*Which reject?* Ordering the zeroed block by the ungated reward is actively
perverse: \(\mathcal{R} = wL + (1-w)H\) increases with response loss, and
the corruptions the gate exists to reject are precisely the high-loss ones,
so the backfill would pull in the MOST corrupted rejects first. The code
orders that block by \(\hat\Delta_i\) — take the least unreliable rejects.

*The dedup constraint can force a backfill even when the budget "fits".*
\(|\{G_i>0\}| \ge B\) does not imply non-compensation held: at most one sample per
near-duplicate cluster may be selected, so the admissible set can be
exhausted before \(B\) is reached. The realised count
(`n_zero_weight_selected`) is therefore measured after selection and written to
`metrics.json`, rather than predicted from \(|\{G_i>0\}|\).

**Suggested text.** After Eq. 1:

> When fewer than \(B\) candidates pass the gate — either because the gated
> set is smaller than the budget or because the near-duplicate constraint
> exhausts it — the remaining slots are filled from the zero-weight set in
> increasing order of \(\hat\Delta_i\), i.e. by the least unreliable
> rejects. We report the realised number of zero-weight samples in each selected
> subset; at our selection ratios it is \(N\).

*(Substitute the measured `n_zero_weight_selected` from `metrics.json`. If it is
not zero, the non-compensation claim must be stated as holding for the
remaining slots, not unconditionally.)*

---

### A5. Eq. 5 tests a length-dependent statistic against a length-independent threshold

**This is the item that decides whether the method works at all at 7B.**

**Paper.** \(\hat\Delta_i = \min(\bar\Delta_i, \Delta^{\min}_i)\), and Eq. 6
zeroes exactly when \(\hat\Delta_i \le 0\).

**Problem — measured, not simulated.** \(\Delta^{\min}\) is a minimum over
\(M_i = \lceil n_i/W \rceil\) spans, so its null *location* falls as
responses get longer. Zero is therefore the wrong threshold for it at every
length but one. On Qwen2.5-7B-Instruct over the 51 760-response clean
reference pool at \(W=16\):

| statistic | mean | reading |
|---|---|---|
| \(\bar\Delta\) (Eq. 3) | **+0.108** | healthy — the instruction explains the response |
| \(\Delta^{\min}\) (Eq. 5) | **−0.265** | 60% of *clean* responses have a span the counterfactual "explains better" |
| \(\hat\Delta > 0\) | **39.6%** | the gate would zero **60.4% of clean data** |

\(P_{10}(\hat\Delta_{\text{clean}}) = -0.410\) and even the median is
\(-0.084\), so `calibrate_gate_scale` could not derive \(s\) at all and fell
back to the diagnostic \(s = 1\). Because Eq. 6 zeroes *exactly* at
\(\hat\Delta \le 0\), **no choice of \(s\) can rescue this** — the zero is
decided before \(s\) is consulted. The two readings that matter are that the
overall gain is fine (so the reference pool is not contaminated and the
counterfactuals are genuinely unrelated) and that the damage is confined to
the tail statistic.

**Amendment.** Recentre the tail test on where the null actually sits at that
span count:

\[
\hat\Delta_i \;=\; \min(\bar\Delta_i, \Delta^{\min}_i) \;-\; \mu(M_i)
\tag{5$'$}
\]

with \(\mu(M)\) the \(\alpha\)-quantile of the uncorrected statistic on a
**clean reference pool** restricted to span count \(M\), estimated in
count-balanced bins and projected onto the non-increasing cone (the
mechanism is monotone; a rise is noise). Eq. 6 is untouched: \(\sigma(0)=1/2\)
still makes \(\hat\Delta \le 0 \Rightarrow G = 0\) exactly, so the fusion
stays non-compensatory. Only the origin moves.

**Two properties follow by construction**, and both are printed by the
calibration rather than asserted:

1. the clean zero-weight rate *is* \(\alpha\) — a dial the experimenter sets
   (`selection.tag.target_zero_rate`, 0.05) instead of an emergent 60%; and
2. it is \(\alpha\) in **every** length bin, which is what removes the
   confound that item B2 raises.

**Why this does not launder corruption.** \(\mu\) is fit on clean data only.
A dirty sample is not compared against its own pool's null but against where
*clean* samples of the same length sit, so a genuinely bad span still lands
negative. Fitting \(\mu\) on the candidate pool would absorb the signal, so
the code refuses to: `null_correction: true` with no `gate_ref_file` is a
hard error with no in-pool fallback, unlike \(s\).

**Measured on a controlled reproduction** (`tests/test_gate.py`, 6 000
synthetic responses with identical per-token dependency at every length):
the uncorrected zero-weight rate runs 11% → 55% from the shortest to the longest
length quintile; after Eq. 5\('\) it is 5% in all five. Injecting one
corrupted span per response then zeroes 100% of the corrupted rows at a 5.4%
clean rate.

**Interaction with \(s\).** \(s\) must be calibrated on the *centred*
statistic, since that is what Eq. 6 sees. Centring puts
\(Q_\alpha(\hat\Delta_{\text{clean}}) = 0\) exactly, so
`calibration_target_pct` must be strictly greater than `target_zero_rate` or
\(s\) is derived from a non-positive quantile; the code raises with that
exact message rather than silently falling back. With
\(\alpha = 0.05,\ \text{target\_pct} = 0.10\) the derived gate is genuinely
graded — on the reproduction above, 5% zeroed, 47% in the soft interior,
48% at the ceiling — not the binary mask a broken calibration produces.

**Ablation arm.** `configs/experiments/lowq/tag_nonull_7b.yaml` runs the
literal Eq. 5 with its own reference and gate cache, so the amendment is
justified by a number in the results table rather than by this argument.

**Suggested text.** Replace Eq. 5's consumption in Eq. 6 with Eq. 5\('\) and
add:

> The span minimum is an order statistic over \(M_i = \lceil n_i/W\rceil\)
> spans, so its null distribution depends on response length; comparing it
> against a fixed zero threshold would zero long responses far more often
> than short ones for reasons unrelated to instruction dependency. We
> therefore recentre \(\hat\Delta_i\) on \(\mu(M_i)\), the \(\alpha\)-quantile
> of the uncentred statistic among clean reference responses with the same
> span count. This makes the clean-reference zero-weight rate equal to \(\alpha\)
> uniformly in length by construction; we set \(\alpha = 0.05\) and report
> the realised per-bin rates. Because \(\mu\) is estimated on clean data
> only, it cannot absorb corruption signal.

---

### A6. MEASURED: the span minimum carries no signal, and the `min` destroys what \(\bar\Delta\) has

**This supersedes B2's recommendation and is the largest single change to the
method.**

**Paper.** Eqs. 4-5 partition the response into spans and take the worst one,
justified by: *"the mean alone can be diluted by localized corruption — a
response that is correct for ninety tokens and wrong for five keeps a healthy
\(\bar\Delta\) while \(\Delta^{\min}\) collapses."* Eq. 6 then consumes
\(\hat\Delta = \min(\bar\Delta, \Delta^{\min})\).

**Measured.** Qwen2.5-7B-Instruct, composite-20 pool (59 516 records, 30.4%
dirty), W=16, Eq. 5\('\) null correction on. AP of each statistic as a
detector of one corruption type against clean only
(`scripts/gate_report.py`):

| type | base | AP \(\bar\Delta\) | AP \(\Delta^{\min}\) | AP \(\hat\Delta\) |
|---|---|---|---|---|
| mismatch | 0.059 | **0.8170** (13.8×) | 0.0873 (1.5×) | **0.1697** |
| noisy | 0.059 | **0.4507** (7.6×) | 0.0574 (≈0) | **0.0719** |
| duplicate | 0.158 | 0.2387 | 0.1714 | 0.1808 |
| wrong_answer | 0.059 | 0.0876 | 0.0903 | 0.0779 |
| truncated | 0.059 | 0.0445 | 0.0505 | 0.0614 |

Three readings, in order of importance:

1. **\(\bar\Delta\) (Eq. 3) is a strong detector** — 13.8× the base rate on
   instruction-response mismatch, 7.6× on injected noise. The counterfactual
   contrast works.
2. **\(\Delta^{\min}\) (Eq. 5) carries no signal on ANY corruption type**,
   including the localized ones it was introduced for.
3. **Because \(\hat\Delta\) is a MINIMUM, the noisy tail does not merely fail
   to help — it destroys \(\bar\Delta\)'s signal.** Mismatch AP falls
   \(0.817 \to 0.170\); noisy falls \(0.451 \to 0.072\). Roughly 80% of the
   available signal is thrown away by the \(\min\).

The pool-level consequence: the shipped gate reaches only AP 0.418 against a
0.304 base (1.37×), and its \(G=0\) block is just 43.8% dirty — barely
1.44× enriched.

**Why the span argument failed — MEASURED** (`scripts/span_profile.py`, clean
reference, W=16). The instruction's predictive contribution is concentrated
almost entirely in the first response span. Mean per-token NLL by span index:

| span \(m\) | \(\ell(y\mid u^+)\) | \(\ell(y\mid u^-)\) | gap |
|---|---|---|---|
| 0 | 1.455 | **5.835** | **4.380** |
| 1 | 1.389 | 1.988 | 0.600 |
| 2 | 1.336 | 1.632 | 0.296 |
| ≥12 | 1.075 | 1.129 | **0.054** |

The gap falls 7× from \(m=0\) to \(m=1\) and 80× by \(m\ge12\):
\(\ell_k(y\mid u)\) conditions on the response prefix \(y_{<k}\) as well as on
\(u\), so once the model has read the opening of \(y\) it infers the topic and
predicts the remainder about as well under a wrong instruction as under the
right one.

A MINIMUM therefore never selects span 0 — it has the highest gain by
construction. Among records with more than one span, \(\arg\min_m\) sits at
span 0 about **1%** of the time (the 24.4% raw rate at \(m=0\) is almost
entirely single-span records, where the minimum is forced). **Eq. 5 reads only
the spans that carry no signal**, which is why its \(P_{05}\) is negative at
every \(m\ge1\) (−0.19 to −0.33) for perfectly clean data.

This also explains why \(\bar\Delta\) works: the sum is dominated by span 0
(4.38 nats × 16 tokens ≈ 70 nats, against ~0.05 nats/token over the rest).

**Per-position centring helps a great deal but still loses.** An earlier
draft of this item claimed centring signal-free spans "yields unbiased noise,
and the minimum of noise is still noise". The measurement refutes that: there
IS signal in the later spans, merely buried under the position bias.
Subtracting a per-position \(\mu(m)\) raises mismatch AP from 0.087 to
**0.454**, a 5.2× improvement. It is nonetheless well short of the prefix
statistic (0.935), and it still leaves the clean tail's \(P_{05}\) at −0.324
(from −0.565), because a minimum over \(M\) positions each calibrated at
\(\alpha\) falls below zero with probability \(1-(1-\alpha)^M\) — the part
Eq. 5\('\)'s \(M\)-centring handles.

So this is B2's recorded limit in its strongest form, with the mechanism now
identified: not "the distributions happen to overlap" but "the statistic is
computed where there is nothing to see".

**The corruption the spans were for is not caught by anything.**
`wrong_answer` scores 0.0876 / 0.0903 / 0.0779 — all ≈ the 0.059 base. A
fluent wrong answer *is* well explained by its instruction, so the likelihood
contrast has little to see. This is an honest limitation of the whole
counterfactual approach, not of the span mechanism specifically, and should
be stated as such rather than quietly dropped.

**`truncated` is caught entirely by \(c_i\), not by the contrast.** Its
\(G=0\) rate is 4.6%, statistically identical to clean's 4.7%, while its mean
\(G\) is 0.227 — the suppression is the completeness heuristic
(\(\bar c = 0.30\)) alone. \(\bar\Delta\)'s AP of 0.0445 is *below* the base
rate, i.e. mildly anti-correlated: a truncated response is a prefix of a
correct one and therefore slightly *more* predictable from its instruction.
The paper must not credit Eqs. 2-6 with catching T3.

**A false alarm worth recording.** The pool's mean \(\bar\Delta\) is 0.447
against the reference's 0.108, which looks like a calibration-invalidating
distribution shift. It is not: the two agree to three decimals at every
quantile (\(P_{10}\) 0.159 vs 0.159, \(P_{50}\) 0.379 vs 0.382, \(P_{90}\)
0.883 vs 0.889). The means differ only because \(\bar\Delta\) has a heavy
left tail — the `eps_den` clamp sends \(1 - L^+/L^-\) far negative whenever
the counterfactual sum is tiny — which drags the reference's mean below even
its own 10th percentile. Any diagnostic on this statistic must use medians
and IQRs, not means and SDs; `gate_report.py` now does.

**Code.** `tail_mode: none` makes the tail abstain everywhere, so
\(\hat\Delta = \bar\Delta\). `configs/experiments/lowq/tag_bar_7b.yaml` is
that arm; it differs from `tag_7b` in exactly one field.

**Suggested paper changes.**
1. Report the per-type AP table above. It is the strongest evidence in the
   paper that the contrast works, and it is also the evidence that Eqs. 4-5
   do not.
2. **Replace the minimum with a PREFIX contrast — measured on the real pool.**
   The paper's intuition ("a sequence average can hide a locally unsupported
   segment") is sound; what is wrong is *where* Eq. 5 looks. Every candidate
   summary, scored against the corruption labels
   (`scripts/sweep_gate_statistic.py`, 59 516 records, AP):

   | statistic | all | duplicate | mismatch | noisy | truncated | wrong_ans |
   |---|---|---|---|---|---|---|
   | *(base rate)* | 0.304 | 0.158 | 0.059 | 0.059 | 0.059 | 0.059 |
   | \(\Delta^{\min}\) (Eq. 5) | 0.334 | 0.171 | 0.087 | 0.057 | 0.051 | 0.090 |
   | \(\Delta^{\min}\) pos-centred | 0.465 | 0.202 | 0.454 | 0.166 | 0.049 | 0.089 |
   | \(\bar\Delta\) (Eq. 3) | 0.542 | 0.239 | 0.817 | 0.451 | 0.045 | 0.088 |
   | span 0 only | 0.582 | 0.259 | 0.933 | 0.584 | 0.062 | 0.063 |
   | **prefix 32 tok** | **0.591** | **0.266** | **0.935** | **0.632** | 0.061 | 0.068 |

   The prefix contrast wins on every type the contrast can see, detecting
   instruction-response mismatch at **15.8× the base rate**. (`prefix 16` and
   `span 0 only` agree to four decimals, as they must at \(W=16\) — a
   consistency check on the implementation.) It is also more length-robust
   than Eq. 3, whose denominator grows with the response while the signal
   does not.

   So Eq. 4-5 should be replaced by

   \[\hat\Delta_i = \Bigl[1 - \tfrac{L(y_i^{<n_p} \mid u_i)}{L(y_i^{<n_p} \mid u_i^-)}\Bigr] - \mu(M_i), \qquad n_p = 32,\]

   with the span minimum reported as the measured-and-rejected variant.
   `configs/experiments/lowq/tag_prefix_7b.yaml` is the arm
   (`prefix_tokens: 32`, `tail_mode: none`). Which configuration is "main" is settled by the
   end-to-end numbers (`tag_bar_7b` vs `tag_7b`), not by this table alone,
   but the table already rules out presenting the span minimum as a
   contribution.
3. State the `wrong_answer` limitation explicitly, and attribute T3 detection
   to \(c_i\) rather than to the contrast.

> The span minimum of Eqs. 4-5 was evaluated and did not earn its place: on
> our pool it detected no corruption type above the base rate, and because
> \(\hat\Delta\) is a minimum it suppressed the overall gain's own signal
> (mismatch AP \(0.82 \to 0.17\)). We therefore gate on \(\bar\Delta\) alone
> and report the span variant as an ablation. We attribute this to the clean
> span minimum being an order statistic over \(M\) noisy ratios, whose
> distribution overlaps the corrupted one regardless of where the threshold
> is placed.

---

## B. Recommended — experiment robustness

### B2. \(\Delta^{\min}\) has an order-statistic bias in the response length, and \(W\) is a first-order hyper-parameter

**This is the most important item in this document.**

> **Update.** Item A5 supersedes the framing below in one respect and
> confirms it in another. The *zero-weight-rate* half of this problem is solved by
> the Eq. 5\('\) null correction, which pins the clean rate at \(\alpha\)
> uniformly in length. The *detection* half is not: re-centring moves the
> threshold, it does not create separation where the clean and dirty minima
> genuinely overlap. That is what the "correcting the null per \(M\)" bullet
> below measured, and it still holds — at large \(M\) with a weak, diluted
> corruption, no threshold separates the two. \(W\) therefore remains a
> first-order choice, because it is the only knob that acts on the
> separation itself rather than on the threshold. Read this section as the
> argument for the \(W\) sweep, not as an argument against A5.

**Paper.** Note (e) in the source comments says the token-level bottom-\(\rho\)
variant "was discarded for order-statistic bias". The span-level minimum has
the *same* disease, only weaker: \(\Delta^{\min}_i\) is a minimum over
\(M_i = \lceil n_i / W \rceil\) spans, and \(M_i\) grows with response
length, so the tail statistic drifts downward for long responses *even when
they are perfectly clean*.

**Why it matters.** With a single global \(s\), that drift becomes a
length-dependent zero-weight rate. Simulating clean-only samples with per-span
gains \(\mathcal{N}(0.45, 0.18)\) at \(W=16\):

| response tokens | \(M\) | clean zero-weight rate |
|---|---|---|
| 32 | 2 | 1.2% |
| 64 | 4 | 2.6% |
| 128 | 8 | 5.1% |
| 256 | 16 | 9.5% |
| 512 | 32 | **18.1%** |
| 1024 | 64 | **32.8%** |

A 15-27× swing in the false-zero rate driven purely by length. This is
directly attackable: truncated (T3) corruptions are SHORT, so the gate
would zero long CLEAN responses more often than short DIRTY ones on this
axis, and a reviewer will read the result as a length filter.

**Two fixes that do not work, and why** (both measured, same model):

- *Capping \(M\)* (widening spans for long responses) flattens the clean
  rate to ~4.7% at every length but destroys detection: a fixed-size
  corrupted region diluted inside a 64-token span gives dirty zero-weight rate
  4.8% — statistically identical to clean. The cap trades the confound for
  blindness.
- *Correcting the null per \(M\)* (subtracting the clean conditional median
  of \(\Delta^{\min}\)) flattens the clean rate to 2-9% but collapses
  detection to 3.3% at \(M \ge 16\), because by then the clean minimum and
  the dirty minimum genuinely overlap. No threshold separates them.
  **This measured the correction as a *detection* fix, which it is not.**
  As a *calibration* fix it is necessary and is now shipped as Eq. 5\('\)
  (item A5) — with two changes: the offset is the \(\alpha\)-quantile, not
  the median (the median puts half the clean pool below zero), and \(s\) is
  recalibrated on the centred statistic. What survives from this bullet is
  the honest limit: at large \(M\) with a diluted corruption the statistic
  has little to separate, and only \(W\) changes that.

**What does work.** Per-span noise is an average over \(W\) tokens and so
shrinks like \(1/\sqrt{W}\), while dilution of a fixed-size corrupted region
grows with \(W\). The two effects trade off, and the optimum is not at the
default. Same model, AUC of \(\hat\Delta\) as a clean-vs-dirty
discriminator, 24-token corrupted region:

| response tokens | \(W\)=16 | \(W\)=32 | \(W\)=64 |
|---|---|---|---|
| 64 | 0.976 | **0.991** | 0.944 |
| 128 | 0.949 | **0.980** | 0.904 |
| 256 | 0.912 | **0.962** | 0.845 |
| 512 | 0.837 | **0.928** | 0.768 |
| 1024 | 0.727 | **0.872** | 0.697 |

\(W = 32\) dominates \(W = 16\) at every length in this model, and \(W=64\)
over-dilutes. **These are simulated numbers under assumed per-span gain
statistics — they justify the sweep, they do not replace it.**

**Actions taken in the code.**
- `store_token_losses: true` in the 0.5B TAG arm, so re-deriving \(G\) for a
  new \(W\) / \(\tau\) / \(s\) costs **no forward pass** — the sweep is free.
- `scripts/score_pool.py` reports `tag.length_bias`: per-length-quantile
  mean gate, zero-weight rate, clean zero-weight rate, and the rank correlation between
  \(G\) and response length.

**Suggested paper changes.**
1. Report the \(W\) sweep as a first-order ablation, not an appendix detail.
2. Report the length-bias diagnostic (clean zero-weight rate by length quantile,
   and \(\rho(G, \text{length})\)) — pre-empting the objection is far
   cheaper than answering it in rebuttal.
3. Add one honest sentence to the method or limitations:

> Because \(\Delta^{\min}_i\) is a minimum over \(M_i \propto n_i/W\) spans,
> its null distribution drifts downward with response length; \(W\) trades
> this order-statistic drift against the dilution of a localized corruption
> inside a wider span. We select \(W\) on a clean reference pool by the
> criterion that the false-zero rate be flat across response-length
> quantiles, and report the resulting profile.

### B1. \(\tau\) should threshold the per-token mean, not the span sum

**Paper.** \(\mathcal{C}_i\) excludes spans "whose counterfactual loss
\(\sum_{k\in\mathcal{S}_m}\ell_k(y_i|x_i^-)\) falls below a threshold".

**Problem.** The SUM scales with span length. Every response whose length is
not a multiple of \(W\) ends in a short partial span, whose sum is
mechanically smaller — so an absolute threshold excludes trailing partial
spans regardless of their content. The exclusion rule becomes a length
filter, which is precisely what it is supposed to avoid (its stated purpose
is to skip *boilerplate*, a content property).

**Code.** `tau_mode: per_token` (default) thresholds
\(\frac{1}{|\mathcal{S}_m|}\sum_{k\in\mathcal{S}_m}\ell_k(y_i|x_i^-)\).
`tau_mode: absolute` keeps the literal reading as an ablation arm, and
`min_span_tokens` drops fragments too short to have a stable ratio.

**Suggested text.** Redefine \(\mathcal{C}_i\) in Eq. 5:

> \[\mathcal{C}_i = \Bigl\{\mathcal{S}_m : |\mathcal{S}_m| \ge W_{\min}
> \ \text{and}\ \tfrac{1}{|\mathcal{S}_m|}\!\!\sum_{k\in\mathcal{S}_m}\!\!
> \ell_k(y_i|x_i^-) \ge \tau \Bigr\}\]
> i.e. the threshold applies to the span's mean counterfactual NLL, so that
> the admissibility of a span depends on its content rather than its length.

### B4. \(c_i\) is a five-fold demotion decided by a string heuristic — measure its error rate

**Paper.** \(c_i \in \{1, c_{\text{trunc}}\}\) is described as a completeness
factor, with no statement of how completeness is decided or how often it is
decided wrongly.

**Problem.** The decision is a text heuristic, and the first version of it
(ends with terminal punctuation, a digit, or a closed code fence) flagged
**14.6% of the composite20 pool** incomplete — against a T3 corruption rate
of roughly 4%. The excess was structural: bulleted lists, numbered steps,
markdown tables, `Key: value` blocks and one-word answers routinely end
without a period. Each of those clean samples had its score multiplied by
0.2. A view introduced to catch truncation was demoting more clean data than
there was truncation in the pool.

**Code.** `text_is_complete` now accepts a structured final line (bullet,
numbered item, table row, `Key: value`) and a terse answer of at most three
words that does not end on a dangling function word, before falling back to
the punctuation test. `scripts/audit_completeness.py` scores the heuristic
against the pool manifest — precision, recall on T3, false-positive rate on
the uncorrupted subset, and the ratio of clean demotions to true catches —
with `--ablate` to attribute each rule.

**One caveat to state rather than bury.** The list rule measures as nearly
free of false negatives on this pool partly for a synthetic reason:
`corruption.truncate_text` rebuilds the response with `" ".join(words)`,
collapsing newlines, so a T3-truncated list arrives as a single line and
cannot end on line 2+. Against naturally truncated text the rule would be
weaker. The short-answer rule is a genuine trade: a 30% cut of a ten-word
response lands in the window and escapes.

**Suggested text.** One sentence in the setup and one number in the results:

> Completeness \(c_i\) is decided by a text heuristic that accepts terminal
> punctuation, closed code fences, structured final lines (list items, table
> rows, field labels) and short answers. On our pool it flags \(X\%\) of
> responses, with recall \(R\) on injected truncations and a false-positive
> rate of \(F\) on uncorrupted ones.

*(Substitute the measured \(X, R, F\) from `scripts/audit_completeness.py`.
If \(F\) exceeds the true truncation rate, lower \(c_{\text{trunc}}\)'s
severity or report \(c_i \equiv 1\) as an ablation — do not ship a view whose
false positives outnumber its catches.)*

### B3. State which statistic \(s\) is calibrated on

**Paper.** "the scale \(s\) is calibrated once per backbone on a clean
reference pool" — but not on *which* quantity.

**Problem.** \(\bar\Delta\) and \(\Delta^{\min}\) have systematically
different distributions (the latter is a minimum, so it is lower by
construction). Calibrating on \(\bar\Delta\) and gating on
\(\hat\Delta = \min(\cdot,\cdot)\) would zero a large slice of clean data.
Anyone reimplementing from the paper has a 50% chance of picking the wrong
one.

**Code.** `calibrate_gate_scale` takes \(\hat\Delta\), and
`scripts/calibrate_reliability.py --mode tag` emits exactly that; the loader
rejects an MVF reference file (raw \(\Delta L\) in nats) by key.

**Suggested text.**

> \(s = \mathrm{P}_{10}(\hat\Delta^{\text{clean}}) / \mathrm{logit}(0.8)\),
> the scale at which 90% of a clean reference pool attains
> \(\sigma(\hat\Delta/s) \ge 0.8\), i.e. \(G \ge 0.6\). The calibration
> statistic is \(\hat\Delta\) itself, not \(\bar\Delta\).

---

## C. Wording precision

### C1. "a single forward pass" is 1 + K

**Paper.** "computed in a single forward pass at the base checkpoint and
cached — no per-refresh cost".

**Reality.** The contrast needs per-token NLLs of the same response under
\(x_i\) and under \(x_i^-\): one pool forward each, so \(1+K\) forwards for
\(K\) counterfactual pairings. The *cached, no-per-refresh-cost* half of the
claim is exactly right and is worth keeping — verified: epoch 2 of a TAG run
hits the cache and runs no gate forward at all.

**Suggested text.** "...computed once at the base checkpoint in one forward
pass per pool (the candidate pool and its counterfactual), then cached, so
later refreshes pay nothing."

### C2. The boundedness argument proves less, and more, than it says

**Paper.** "because both dynamic factors in Eq. (1) are bounded, a zeroed
gate cannot be compensated by any amount of difficulty or alignment
evidence".

**Two corrections, in opposite directions.**

1. For an **exactly** zeroed gate, boundedness is not needed at all:
   \(0 \cdot r = 0\) for any finite \(r\). The claim is stronger than its
   stated premise requires.
2. Boundedness *is* what the argument needs for the case the paper does not
   address: a **small but non-zero** gate. \(G_i = \epsilon\) is outranked by
   a clean \(G_j\) only if
   \(\epsilon\,\mathcal{R}_i(1+\lambda a_i) < G_j\mathcal{R}_j(1+\lambda a_j)\),
   which depends on the pool's reward ratio \(\mathcal{R}_i/\mathcal{R}_j\)
   — bounded on any given pool, but not by a constant of the method. The
   anchor factor contributes at most \((1+\lambda)\).

**Suggested text.** Split the claim:

> The gate's floor is exact: \(G_i = 0 \Rightarrow s_i^{(t)} = 0\)
> regardless of the dynamic factors, since \(\sigma(0)=\tfrac12\) makes the
> clamp in Eq. 6 bite exactly at zero gain. For a small but non-zero gate the
> ordering is governed by the ratio
> \(\frac{G_i}{G_j}\cdot\frac{\mathcal{R}_i}{\mathcal{R}_j}\cdot
> \frac{1+\lambda a_i}{1+\lambda a_j}\), whose last factor is bounded by
> \(1+\lambda\); we report the empirical reward ratio so the achievable
> compensation margin is explicit.

*(The MVF revision computed exactly this margin — 61× suppression against
3.96× compensation. The TAG version of that calculation should appear too;
it is the quantitative core of the non-compensation claim.)*

### C3. "counterfactual form of pointwise mutual information"

Only Eq. 2's numerator is a counterfactual PMI:
\(\sum_k \delta_{i,k} = \log p(y|x) - \log p(y|x^-)\). What the gate
actually thresholds is Eq. 3's **ratio**, that PMI divided by the
counterfactual surprisal. The normalisation is the contribution — it is what
makes the statistic scale-free and what distinguishes it from IFD's
\(L(y|x)/L(y)\) — so it should not be elided in the sentence that names the
statistic.

**Suggested text.** "Eq. 2 is the counterfactual pointwise mutual
information; Eq. 3 normalises it by the counterfactual surprisal, giving a
scale-free *relative* PMI."

### C5. With \(K>1\), the dispersion discount sits outside Eq. 6

**Paper.** Eq. 6 defines \(G_i\) purely as a function of \(\hat\Delta_i\),
and claim (a) is that \(G_i = 0 \iff \hat\Delta_i \le 0\).

**Code.** With \(K>1\) counterfactual pairings the shipped default applies
\(G_i = \bigl(1 - 2\,\mathrm{std}_k(G_i^{(k)})\bigr)_+ \cdot
\mathrm{mean}_k G_i^{(k)}\). That second factor appears nowhere in Eq. 6, and
it can zero the gate even when every \(\hat\Delta^{(k)}_i > 0\) — so under
\(K>1\), "\(G=0\)" means either "no instruction dependency" *or* "the
pairings disagree too much to trust". The per-pairing invariant still holds;
the aggregate one does not.

Note also that the reported/cached \(\hat\Delta_i\) is the across-pairing
mean while the gate is the mean of per-pairing gates (deliberately: the
clamp is convex, so gate-of-mean would collapse straddling evidence to an
exact zero). The two are not in the functional relationship Eq. 6 states.

**Suggested text.** If \(K>1\) is reported at all, state the estimator:

> With \(K\) counterfactual pairings we gate each pairing separately and
> average, discounting by the cross-pairing dispersion:
> \(G_i = (1 - 2 s_k)_+ \cdot \mathrm{mean}_k\,
> c_i(2\sigma(\hat\Delta^{(k)}_i/s)-1)_+\). Gating before averaging is
> deliberate — the clamp is convex, so averaging first would zero any sample
> whose evidence straddles zero. Under this extension a zero gate also
> encodes irreducible disagreement between pairings, not only absent
> instruction dependency.

*(If \(K=1\) is what ships in the paper, say so and move this to the
appendix — it is cleaner than defending a second meaning for \(G=0\).)*

### C6. The counterfactual is matched on RESPONSE length, not instruction length

**Paper (Eq. 2).** "\(x_i^-\) a length-matched unrelated instruction sampled
from the pool."

**Code.** `make_counterfactual` deranges within **response**-length buckets;
the substituted instruction's own length is unconstrained. That is the right
choice — matching on response length is what keeps the denominator
\(\sum_k \ell_k(y_i|x_i^-)\) comparable across samples and stops the gate
becoming a length detector — but it is not what the sentence says, and the
difference is exactly why the true and counterfactual prompts have different
lengths (which is what makes A2's trim necessary).

**Suggested text.** "\(x_i^-\) is an unrelated instruction drawn from the
same response-length stratum of the pool."

### C7. Say that \(G\) is a continuous weight — "gate" will be read as binary

**Paper.** \(G_i\) is called a *gate* throughout, and Eq. 6 is presented as
producing a decision the rest of the score cannot override.

**Problem.** "Gate" is doing two jobs at once and only one of them is
accurate. In the ML literature gating is overwhelmingly *continuous*
multiplicative modulation — LSTM/GRU \(\sigma\) gates, MoE gating networks,
GLU, highway networks, squeeze-and-excitation — so the word itself is fine.
But paired with language like *veto* or *reject*, a reader reconstructs a
threshold classifier with a decision boundary, and then reads every
calibration number as a classification error rate. That is the wrong mental
model of the object and it changes what the reader thinks the failure modes
are.

\(G_i\) is continuous in \(\hat\Delta_i\): \((\cdot)_+\) is a ReLU-style
kink, not a step, since \(2\sigma(z)-1 \to 0\) as \(z \to 0^+\). On the 7B
calibration the weight distribution is

| \(G\) | share of pool |
|---|---|
| \(=0\) (at the floor) | 5% |
| \(0 < G < 0.99\) | **47%** |
| \(\ge 0.99\) | 48% |

so roughly half the pool receives a genuinely intermediate weight. Calling
that a veto is simply a wrong description of the measured object.

**What IS distinctive** is that the floor is *attainable*. A plain sigmoid
gate approaches 0 and never reaches it; \(\sigma(0)=\tfrac12\) makes
\(\hat\Delta_i \le 0 \Rightarrow G_i = 0\) exactly. That one property — not
binariness — is what the non-compensation claim rests on, because with
\(G_i > 0\) strictly, a large enough \(\mathcal{R}_i^{(t)}\) always
compensates.

**Code.** All decision-flavoured naming has been removed from the weight
side: `target_zero_rate`, `zero_rate`, `n_zero_weight_selected`. The
*selection key* keeps two-block language (`n_admissible`) because top-\(B\)
selection genuinely is discrete — the distinction being drawn is that the
weight is continuous and the key built from it is not.

**Suggested text.** Replace decision vocabulary in the method section with:

> \(G_i\) is a continuous function of \(\hat\Delta_i\) taking values in
> \([0,1]\). Unlike a conventional sigmoid gate it *attains* its lower
> bound: \(G_i = 0\) exactly when \(\hat\Delta_i \le 0\), i.e. when the
> instruction contributes nothing to explaining the response. This
> attainability — rather than any thresholding — is what makes the fusion
> non-compensatory; elsewhere \(G_i\) expresses a graded degree of
> reliability, and on our pool \(X\%\) of candidates receive a weight
> strictly between the bounds.

Add one clause to the limitations, since the property is two-sided:

> Because the lower bound is attained, a candidate placed at the floor
> cannot be recovered by any amount of dynamic evidence; errors in
> estimating \(\hat\Delta\) near the origin are therefore not recoverable,
> which is why the null calibration of Eq. 5\('\) is required rather than
> optional.

### C4. The EOS position is excluded from the span aggregation

Tokenisation appends EOS to every response unconditionally, so the final
label position is always EOS unless budget truncation dropped it. Its
predictability reflects whether the response terminated, not whether the
instruction explains it — and termination is already \(c_i\)'s job. The code
excludes it (`include_eos: false`). Worth one clause in the setup so the
token counts in the paper match a reimplementation.

---

## D. Evaluation protocol

### D1. Benchmark generation is batched, and the batch size is part of the number

Every generative evaluator decodes prompts in batches of `TAG_EVAL_GEN_BS`
(default 16; `1` restores one prompt per `model.generate` call). This is what
makes the eight-benchmark suite finishable — measured on BBH with a 7B
backbone, **6.1x** on identical work, 20+ hours per arm down to under three.

It is not bit-exact, and the paper must not claim it is.

Greedy decoding is padding-invariant in exact arithmetic, and the
implementation checks that directly: the first predicted token's argmax is
compared batched-vs-alone at the start of every eval, and the max |logit
diff| measured on LLaMA-class and Qwen-class 7B weights was **0**. The mask
is right. What is not exact is the rest of the chain: cuBLAS reduces a
`(16, T, H)` matmul in a different order than a `(1, T, H)` one, logits move
by ~1e-3, and a 500-token chain-of-thought that passes near a tie takes the
other branch. Measured on BBH, ~25% of continuations fork somewhere; most
forks land *after* the answer and cost nothing.

Measured, BBH at `--limit 8` (216 examples), Qwen2.5-7B-Instruct, the same
checkpoint at `TAG_EVAL_GEN_BS` 16 vs 1:

| | macro-avg |
|---|---|
| batch 16 | 0.6481 |
| batch 1 | 0.6620 |

Five of 27 tasks moved by one example each — four up, one down (a 4/1 split
is p=0.19 under a symmetric null, i.e. no evidence of bias). That is a
**correctness-flip rate of 5/216 = 2.31%**, and it propagates to the macro
average as

$$\mathrm{SD} \;=\; \sqrt{p/n}\,/\,\sqrt{27}, \qquad p = 0.0231$$

with `n` the examples per task. The +1.39pp gap in the table above is a
`--limit 8` artifact: at n=8 the predicted SD is **1.04pp** and 1.39pp is
1.3 sigma, entirely expected. At the full n=250 the predicted SD is
**0.19pp**.

So, for the paper:

* Hold `TAG_EVAL_GEN_BS` **fixed across every arm and every seed**. Within a
  fixed batch size the decode is deterministic — the same checkpoint scored
  0.6481 twice. The value is stamped into every benchmark summary JSON as
  `generation_batch_size`; it is part of the run's provenance, like the git
  sha.
* Report the protocol, and the ~0.2pp figure, in the reproducibility note.
  At three seeds this perturbation is inside the seed-to-seed SD and is
  absorbed by the CI `make_table_v2` already computes.
* If a headline gap ever lands *below* ~0.5pp on a generative benchmark,
  re-score that benchmark alone at `TAG_EVAL_GEN_BS=1` before reporting it.
  Do not re-score one arm and not the other.

### D2. Measured on the Table 2 backbone (LLaMA-2-7B, clean Alpaca-GPT4)

Numbers from the calibration and gate precompute for the Table 2 row
(`main_7b/llama2/tag_10`, pool sha `d5d8ccc6…`, n=52 002, dirty 0.0%;
clean reference `pools/clean_ref`, n=51 760 — **a different snapshot from
the training pool, and the paper's provenance has to say so**).

Calibrated `s = 0.031586` (P10 = 0.0438, target_q = 0.80), null curve fit
on 15 length bins at W=16.

**G on the training pool** — the measured share of intermediate weights that
C7 asks for:

| | |
|---|---|
| `G == 0` | 5.0% (2 590) |
| `0 < G < 0.99` | **42.3%** |
| `G >= 0.99` | 52.7% |

Nearly half the pool carries a weight strictly between the floor and 1. "Gate"
must not be described as binary; on clean data it is overwhelmingly a
reweighting, and the exact zero reaches only the configured 5%. The Qwen
rehearsal on the same corpus gave 4.8% / 49.3%, so the shape is not a
backbone accident.

**A5, the per-length-bin clean zero-weight rates: they are NOT uniform.**
The Eq. 5\('\) centring targets 5.0% in every bin and delivers 3.3%–9.1%:

| M | 1 | 2 | 3 | 4 | 5-6 | 7-8 | 9-10 | 11-12 | 13-14 | 15-16 | 17-18 | 19-20 | 21-23 | 24-26 | 27-30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero % | 5.0 | 5.0 | **9.1** | **7.9** | 6.1 | 5.0 | 3.8 | 4.7 | 4.3 | 4.1 | **3.3** | 3.9 | 4.0 | 4.6 | 5.0 |

The PAVA monotone projection ties \(\mu\) to a single value (+0.2005) across
bins 3-26, so short responses overshoot the target and mid-length ones
undershoot. Only M=1 (+0.2586) and M=2 (+0.2072) get their own level — which
is itself the length effect, since a 32-token prefix covers a 1-span response
entirely. With `tail_mode: none` there is no span minimum and the
M-dependence *should* vanish; that it does not, at the short end, is a
residual length bias and belongs in the limitations, not in a claim that the
correction is uniform.

**B4, the completeness heuristic's cost on clean data.** On the clean Table 2
corpus it flags **4 336 of 52 002 records (8.3%)** as incomplete, each taking
`c_trunc = 0.20` — a five-fold demotion. That is the largest single lever the
gate pulls on a clean pool: the hard zero touches 5%, this touches 8.3% and
divides by five. Not all of them are false positives (Alpaca-GPT4 does
contain truncated responses), but a string heuristic demoting one record in
twelve of a clean corpus is the number `scripts/audit_completeness.py` has to
be pointed at before the paper claims the demotion is targeted.

### D3. The selection advantage saturates at epoch 2, and part of it is the baseline getting worse

Measured on the lowq grid (composite20, n=59 516, 30.4% corrupted,
Qwen2.5-7B-Instruct, seed 42), corrupted fraction of the selected subset:

| epoch | legacy | tag_prefix | gap | Jaccard |
|---|---|---|---|---|
| 1 | 66.4% | 20.1% | 46.3pp | 0.191 |
| 2 | 67.2% | 16.8% | **50.4pp** | 0.146 |
| 3 | 67.3% | 17.2% | 50.1pp | 0.146 |

**92% of the final separation is already present after one epoch**, and both
the gap and the subset overlap reach a fixed point at epoch 2. There is no
compounding curve to plot over three epochs, and no case for keeping
per-epoch checkpoints to chase one.

What *does* move across epochs is the baseline, in the wrong direction:
legacy's over-selection of `noisy` goes 4.82x → 7.03x → 7.02x, a 46% increase
after one epoch of training on it. A selector with no reliability term becomes
more attracted to noise as it trains on noise. A fair share of the +4.1pp the
gap gains from epoch 1 to 2 is that, not TAG improving.

**The one corruption type where the baseline wins.** Per-type selection rate
against each type's share of the pool, epoch 3:

| | duplicate | mismatch | noisy | truncated | wrong_answer |
|---|---|---|---|---|---|
| legacy | 0.89x | 3.37x | 7.02x | 2.18x | **0.24x** |
| tag_prefix | 0.37x | 0.61x | 0.69x | 0.88x | 0.66x |

TAG is 2.4x–10.2x better on four of five types and **2.8x worse on
`wrong_answer`**, and the mechanism is legible: legacy's \(R\) is effectively
loss-seeking, so it hoovers up high-loss garbage (noisy, 7x) and avoids
fluent-but-wrong answers precisely because they are fluent and score low loss.
The gate has no such bias and selects them near the base rate. The paper must
not claim TAG dominates every corruption type — it dominates in aggregate
(17.2% vs 67.3% dirty) while losing this one axis to an accident of the
baseline's own bias.

### D4. On the clean pool the two arms land in different attractors — measured on the Table 2 pair

Same diagnostic on the Table 2 pair (LLaMA-2-7B, clean Alpaca-GPT4, seed 42),
where there are no corruption labels and the only readable quantity is
overlap:

| | Jaccard |
|---|---|
| cross-arm, epoch 1 | 0.266 |
| cross-arm, epoch 2 | **0.052** |
| within-arm legacy, ep1→ep2 | 0.221 |
| within-arm legacy, ep2→ep3 | **0.818** |
| within-arm tag, ep1→ep2 | 0.282 |
| within-arm tag, ep2→ep3 | **0.773** |

Two facts and one mechanism:

* **The ep1→ep2 churn is intrinsic, not gate-specific** — legacy re-selects
  just as violently (0.221) as the gated arm (0.282). Epoch 1 is scored by
  the BASE model; epoch 2 by a model after its first SFT epoch, and base
  LLaMA-2 is not instruction-tuned, so that first epoch transforms the loss
  landscape and both arms re-select nearly from scratch.
* **The near-disjointness at epoch 2 (0.052) is persistent structure, not
  noise.** If it were churn it would keep churning; instead legacy's own
  selection then freezes (ep2→ep3 = 0.818). Each arm has settled into its own
  attractor, and the two attractors share 10% of their mass.
* Mechanism: G is static, but it reorders the epoch-1 subset (0.266), the
  first SFT epoch amplifies that into a different model, and \(R^{(t)}\)
  then locks each arm into its own basin. This is "training-adaptive" doing
  real work, and it contrasts with the corrupted pool (D3), where strong
  anchoring signals hold the arms' overlap steady at ~0.15 instead.

Prediction recorded before the fact — and confirmed: the attractor reading
required tag's ep2→ep3 Jaccard to land near 0.8, and it landed at **0.773**.
Both arms follow the same dynamic (violent re-selection when scoring moves
from the base model to the once-trained one, then a frozen selection) and
settle in different places. `n_zero_weight_selected` was 0 in every epoch:
no G=0 record was ever forced into the selection.

Consequence for Table 2's interpretation: from epoch 2 the arms train on
nearly disjoint subsets, so the row-pair gap measures the END-TO-END effect
of G's initial reordering compounded through training — which is the paper's
actual claim — and cannot be decomposed into "same data, better weighting".

### D5. The sealed seed-42 tables (2026-08-18)

Official numbers from `make_table_v2` over sealed runs — 7-bench macro
(mmlu, bbh, svamp, gsm8k, mbpp, tydiqa, xquad; **humaneval excluded**: its
harness died identically in all four runs with "os.fork is unsafe while
filelock is changing descriptor ownership" — a scoring-harness bug, not a
model result; fix pending). Single seed: **no CI, not final** until seeds
1 and 7 land.

**Robustness (lowq: Qwen2.5-7B-Instruct, composite20, 30.4% corrupted):**

| | mmlu | bbh | svamp | gsm8k | mbpp | tydiqa | xquad | macro |
|---|---|---|---|---|---|---|---|---|
| legacy_7b | 74.30 | 63.26 | 6.00 | 20.55 | 71.60 | 46.88 | 46.01 | 46.94 |
| tag_prefix_7b | 74.24 | 63.46 | **31.33** | **33.66** | 70.82 | **52.89** | 45.70 | **53.16** |
| Δ | −0.1 | +0.2 | **+25.3** | **+13.1** | −0.8 | **+6.0** | −0.3 | **+6.22** |

**Table 2 row pair (main_7b: LLaMA-2-7B, clean Alpaca-GPT4, ρ=10%):**

| | mmlu | bbh | svamp | gsm8k | mbpp | tydiqa | xquad | macro |
|---|---|---|---|---|---|---|---|---|
| legacy_10 (TADS) | 46.15 | 36.69 | 38.00 | 14.94 | 36.96 | 61.31 | 49.89 | 40.56 |
| tag_10 | 45.90 | 36.79 | 39.67 | 16.53 | 33.85 | 59.87 | 49.96 | 40.37 |
| Δ | −0.3 | +0.1 | +1.7 | +1.6 | −3.1 | −1.4 | +0.1 | **−0.20** |

No per-bench difference in the clean pair is significant (MBPP's −3.1pp is
8 problems of 257, inside the ~4.2pp paired SE; TyDiQA's −1.4pp is ~1.4σ).
The clean-pool claim is a TIE, and a tie survives the ±0.2pp batching
perturbation (D1) without a batch-1 rescore — that rule exists for claimed
GAPS.

The sentence the two tables license together: *the gate buys +6.2pp macro
when the pool is corrupted and costs nothing measurable when it is clean.*

Run provenance: lowq pair `runs/20260818_071841`, main pair
`runs/20260818_075532` / `20260818_083139`, all sealed, all
`generation_batch_size=16`, seed 42 pinned via manifest (these four predate
the eval-side cfg.json provenance stamp; every later eval run records seed
and git sha itself).

## Checklist before submission

- [ ] A1-A4 folded into the equations (these are correctness fixes)
- [ ] A6 per-type AP table reported; Eqs. 4-5 demoted to an ablation;
      wrong_answer limitation and c_i's role in T3 stated
- [ ] A5 Eq. 5\('\) added; \(\alpha\) stated; per-length-bin clean zero-weight rates
      reported (measured in D2 — they range 3.3-9.1% against a 5.0% target,
      so report the spread, not the target); the `tag_nonull_7b` ablation in
      the results table
- [ ] B1 \(\mathcal{C}_i\) redefined on the per-token mean
- [ ] B2 \(W\) sweep run and reported; length-bias profile reported
- [ ] B3 calibration statistic named as the CENTRED \(\hat\Delta\)
- [ ] B4 completeness heuristic's precision / recall / FP rate measured with
      `scripts/audit_completeness.py` and reported — D2 has the exposure it
      needs to be measured against: 8.3% of a CLEAN corpus demoted 5-fold
- [ ] C1 "single forward pass" -> "one forward pass per pool, then cached"
- [ ] C2 non-compensation split into the exact-zero case and the
      small-gate ratio, with the measured reward ratio
- [ ] C3 Eq. 3 named as a *relative* PMI, not PMI
- [ ] C5 the K>1 estimator stated, or K=1 declared and K>1 moved to the
      appendix
- [ ] C6 "length-matched instruction" -> "same response-length stratum"
- [ ] C7 decision vocabulary ('veto', 'reject') replaced; G described as a
      continuous weight with an ATTAINABLE floor, with the measured share
      of intermediate weights (D2: 42.3% graded, 5.0% at the floor); the
      two-sided cost noted in limitations
- [ ] `n_zero_weight_selected` from metrics.json reported, not `budget_fits@K`
      inferred — the dedup constraint can force a backfill even when the
      gated set is larger than the budget
- [ ] D1 eval batch size stated in the reproducibility note, with the
      measured ~0.2pp perturbation and the fixed-across-arms rule
- [ ] D3's two negative results stated: the advantage saturates at epoch 2,
      and `wrong_answer` is selected 2.8x MORE by TAG than by the baseline
- [ ] seeds 1 and 7 run for both pairs; CIs from make_table_v2 replace the
      single-seed D5 point estimates
- [ ] humaneval scoring harness fork/filelock crash fixed and the column
      restored (all four D5 runs lost it identically)
- [ ] measured numbers substituted for every placeholder above
