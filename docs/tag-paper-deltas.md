# TAG: proposed amendments to the paper's equations and wording

**Status:** ACTIVE. Written 2026-08-13 while implementing Eqs. 1-6 in
`tads/core/gate.py`. Every item below was found by making the equations
executable — either the maths is undefined on a case the data actually
produces, or the code had to make a choice the paper does not specify, or
the claim as written is stronger than what the code can deliver.

**How to read this.** Each item states what the paper says, what the code
does, why they differ, and the suggested replacement text. Items in §A are
**required** — as written the paper is undefined or wrong on inputs that
occur in every real pool. §B are **recommended** and are about whether the
experiment will survive review. §C is wording precision.

Nothing here changes the method's story: the gate is still a calibrated,
zero-anchored counterfactual contrast used as a veto. The changes make it
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
abstains and the overall gain decides alone. Vetoing would punish short
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
routinely. Vetoing a sample because of a tokenisation artifact is the worst
possible failure mode for a method whose entire pitch is "reliability is a
veto".

**Code.** `undefined_policy`, default `pass`: \(G_i = c_i\) — no evidence,
no verdict. `neutral` and `veto` exist as ablation arms, and the count is
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
vetoed samples, all of which score exactly 0. The veto claim then silently
fails for those slots — and worse, ties at exactly 0 are broken by
`torch.topk`'s index order, i.e. by the candidate pool's FILE ORDER.

**Code.** `scorer.gated_selection_key` orders every admissible sample above
every vetoed one and breaks each block by a meaningful statistic (the gated
score above, the ungated score below); the shortfall is logged as a warning
and reported per selection ratio as `budget_fits@K`.

**Suggested text.** After Eq. 1:

> When fewer than \(B\) candidates pass the gate, the remaining slots are
> filled by the ungated score \(\mathcal{R}_i^{(t)}(1+\lambda\,
> \widetilde{\mathrm{align}}_i^{(t)})\) among the vetoed set, and we report
> how often this occurs; at our selection ratios the gated set covers the
> budget in every run.

*(This last clause must be checked against `budget_fits@K` before it is
claimed. If it does not hold, report the count instead.)*

---

## B. Recommended — experiment robustness

### B2. \(\Delta^{\min}\) has an order-statistic bias in the response length, and \(W\) is a first-order hyper-parameter

**This is the most important item in this document.**

**Paper.** Note (e) in the source comments says the token-level bottom-\(\rho\)
variant "was discarded for order-statistic bias". The span-level minimum has
the *same* disease, only weaker: \(\Delta^{\min}_i\) is a minimum over
\(M_i = \lceil n_i / W \rceil\) spans, and \(M_i\) grows with response
length, so the tail statistic drifts downward for long responses *even when
they are perfectly clean*.

**Why it matters.** With a single global \(s\), that drift becomes a
length-dependent veto rate. Simulating clean-only samples with per-span
gains \(\mathcal{N}(0.45, 0.18)\) at \(W=16\):

| response tokens | \(M\) | clean veto rate |
|---|---|---|
| 32 | 2 | 1.2% |
| 64 | 4 | 2.6% |
| 128 | 8 | 5.1% |
| 256 | 16 | 9.5% |
| 512 | 32 | **18.1%** |
| 1024 | 64 | **32.8%** |

A 15-27× swing in the false-veto rate driven purely by length. This is
directly attackable: truncated (T3) corruptions are SHORT, so the gate
would veto long CLEAN responses more often than short DIRTY ones on this
axis, and a reviewer will read the result as a length filter.

**Two fixes that do not work, and why** (both measured, same model):

- *Capping \(M\)* (widening spans for long responses) flattens the clean
  rate to ~4.7% at every length but destroys detection: a fixed-size
  corrupted region diluted inside a 64-token span gives dirty veto rate
  4.8% — statistically identical to clean. The cap trades the confound for
  blindness.
- *Correcting the null per \(M\)* (subtracting the clean conditional median
  of \(\Delta^{\min}\)) flattens the clean rate to 2-9% but collapses
  detection to 3.3% at \(M \ge 16\), because by then the clean minimum and
  the dirty minimum genuinely overlap. No threshold separates them.

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
  mean gate, veto rate, clean veto rate, and the rank correlation between
  \(G\) and response length.

**Suggested paper changes.**
1. Report the \(W\) sweep as a first-order ablation, not an appendix detail.
2. Report the length-bias diagnostic (clean veto rate by length quantile,
   and \(\rho(G, \text{length})\)) — pre-empting the objection is far
   cheaper than answering it in rebuttal.
3. Add one honest sentence to the method or limitations:

> Because \(\Delta^{\min}_i\) is a minimum over \(M_i \propto n_i/W\) spans,
> its null distribution drifts downward with response length; \(W\) trades
> this order-statistic drift against the dilution of a localized corruption
> inside a wider span. We select \(W\) on a clean reference pool by the
> criterion that the false-veto rate be flat across response-length
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

### B3. State which statistic \(s\) is calibrated on

**Paper.** "the scale \(s\) is calibrated once per backbone on a clean
reference pool" — but not on *which* quantity.

**Problem.** \(\bar\Delta\) and \(\Delta^{\min}\) have systematically
different distributions (the latter is a minimum, so it is lower by
construction). Calibrating on \(\bar\Delta\) and gating on
\(\hat\Delta = \min(\cdot,\cdot)\) would veto a large slice of clean data.
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

> A zeroed gate is an exact veto: \(G_i = 0 \Rightarrow s_i^{(t)} = 0\)
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

\(\sum_k \delta_{i,k}\) is a counterfactual PMI estimate. \(\bar\Delta_i\) is
that quantity **normalised by the counterfactual loss**, which is what makes
it scale-free — the normalisation is the contribution, so it should not be
elided in the sentence that names the statistic.

### C4. The EOS position is excluded from the span aggregation

Tokenisation appends EOS to every response unconditionally, so the final
label position is always EOS unless budget truncation dropped it. Its
predictability reflects whether the response terminated, not whether the
instruction explains it — and termination is already \(c_i\)'s job. The code
excludes it (`include_eos: false`). Worth one clause in the setup so the
token counts in the paper match a reimplementation.

---

## Checklist before submission

- [ ] A1-A4 folded into the equations (these are correctness fixes)
- [ ] B1 \(\mathcal{C}_i\) redefined on the per-token mean
- [ ] B2 \(W\) sweep run and reported; length-bias profile reported
- [ ] B3 calibration statistic named as \(\hat\Delta\)
- [ ] C1 "single forward pass" -> "one forward pass per pool, then cached"
- [ ] C2 non-compensation split into the exact-zero case and the
      small-gate ratio, with the measured reward ratio
- [ ] `budget_fits@K` checked before claiming the gated set covers the budget
- [ ] measured numbers substituted for every placeholder above
