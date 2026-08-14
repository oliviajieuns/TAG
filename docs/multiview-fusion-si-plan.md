# TADS → Information Fusion Special Issue: Execution Plan

> **SUPERSEDED (2026-08-11).** The submission direction has pivoted from
> layer-as-view eigengap fusion to signal-level multi-view fusion
> (reliability × learnability × alignment) on low-quality instruction
> pools — see `docs/plan_low_quality_multiview.md`, which is now the
> active plan and is implemented in `tag/` (score_mode: mvf). This file
> is kept for the record; the eigengap-weighted layer fusion idea remains
> viable future work.
>
> [2026-08-11 v2] §7 Elsevier checklist and the day-10–12 go/no-go gate
> have been ported into the active plan (§10 / §9 there) — maintain them
> THERE, not here. This file is now fully read-only history.

**Status:** design doc, nothing in this plan implemented yet.
**Target:** Elsevier *Information Fusion* special issue — "Multi-view Fusion and Learning on Low-quality Data: Foundation Models in Theories, Algorithms and Applications." Submission deadline 2026-08-30.
**Supersedes:** the CIKM 2026 submission target (see caveat in §6 — dual-submission must be resolved with co-authors before starting).

This plan was produced by independently developing four candidate multi-view framings of TADS (layer-as-view, dataset-source-as-view, signal-as-view, multi-backbone-as-view), scoring all four with three independently-lensed judge panels (scope guardian, deadline realist, contribution scientist), and synthesizing. All load-bearing claims below were verified directly against `sections/03_method.tex`, `sections/04_experiments.tex`, `sections/07_appendix.tex` (in the `tads-rev-overleaf` paper repo), and `tag/core/trajectory_anchor.py`, `tag/core/selector.py`, `tag/core/scorer.py`, `tag/core/reward.py`, `baselines/nait/direction.py`, `references.bib` in this repo.

---

## 1. Recommended core framing

Lead with **layer-as-view adaptive fusion**: each of TADS's L transformer decoder layers already produces an independent PCA capability direction `v_l` from its own probe-delta covariance (`03_method.tex` Eqs. anchor-covariance–anchor-eigenvector), and the current score fuses these L views by a flat, unweighted sum with zero quality-awareness (Eq. `raw-align`, confirmed line-for-line identical in `trajectory_anchor.py::compute_alignment`, though that function is dead code — see §2). The paper's own diagnostic already caught this as a problem: boundary layers 1 and 24 are hand-excluded from the stability table for having "near-degenerate local covariance spectra" (`03_method.tex` lines 553–560), yet production configs (`layer_indices: all`) still fuse those same known-bad views with identical weight to every inner layer. The contribution is to close that gap: replace flat fusion with eigengap-weighted fusion, prove a Corollary showing why this specifically fixes the degenerate-view failure mode, and validate it with a missing/degraded-view robustness grid. This draws primarily on Candidate 1 and Candidate 4 (layer-as-view and its cross-backbone variant), which are near-identical in their core mechanism — all three judge panels ranked these two first and second, unanimously, in every ranking supplied, and all three hybrid suggestions independently converged on this exact spine without seeing each other's reasoning.

Two decisions on top of that convergence:

- **(a) Schedule discipline over feature breadth.** Adopt a small, honestly-scoped "must-ship" primary track with a hard go/no-go gate at day 10–12, rather than a flatter, more feature-loaded plan. 19 days with engineering/writing (not compute) as the bottleneck does not support building 4 fusion modes, stochastic dropout, a noise-injection hook, a 2-backbone grid, *and* the elsarticle migration at full quality.
- **(b) One cheap graft, one explicit refusal.** Graft in the difficulty-only/uncertainty-only carrier ablations (from the signal-as-view candidate) as a cheap bonus table — but explicitly refuse that candidate's framing of "L_i, H_i, align_i are three views." Two of those three are statistics of the same softmax from one forward pass, and every judge panel flagged this as the weakest multi-view claim on the table.

The dataset-source-as-view pooling idea and the full cross-backbone fusion stretch are both real, CFP-relevant ideas, but both require new training pipelines / core `selector.py` surgery with genuine confound or engineering risk, and none of the three judge panels' hybrids executed them — they're future work, not this cycle.

---

## 2. What already exists (near-zero-cost reframing)

- **Per-layer PCA "view" extraction**, `v_l` from `Δh_l` covariance — `03_method.tex` Eqs. anchor-delta-mean through anchor-eigenvector (lines 136–168); implemented in `TrajectoryAnchor._pca_top1` (lines 135–164) called from `update()`. This *is* per-view representation extraction already; needs vocabulary, not code.
- **Flat, unweighted fusion of L views** — Eq. `raw-align`, `03_method.tex` lines 179–188 (`\tilde a_i = Σ_l ⟨h̄_l(x_i), v_l⟩`, no weighting, no 1/L). This becomes the "naive/equal-trust fusion" baseline arm for free.
- **Boundary-layer near-degeneracy** — `03_method.tex` lines 553–560, already-written evidence of imbalanced view quality; just needs multi-view framing, not a new experiment.
- **`gap_by_layer` (per-layer eigengap λ1−λ2)** — computed every `update()` call (`trajectory_anchor.py` line 371) and persisted to `state_dict` (line 483). Confirmed via repo-wide grep: written and nowhere else read. This is the ready-made, already-logged fusion-weight signal — genuinely free.
- **Theorem 1 (Davis–Kahan-style local anchor stability)** — `03_method.tex` lines 269–302, proved in `07_appendix.tex` (`Proof of Theorem`, starting line 22). Bound: `‖v_l^(t)−v_l^(t-1)‖ ≤ (2C_{Σ,l}/γ_l)η_{t-1}`. This blows up exactly as `γ_l→0` — i.e., exactly in the boundary-layer regime the paper already flags by hand. This is the springboard for the new Corollary (§4) — no new machinery needed, just one more derivation on top of an already-proved theorem.
- **`layer_indices` config knob** (`"all" | "middle_to_last" | explicit list`) — `_resolve_layer_indices` in `trajectory_anchor.py` (lines 54–78), wired via `configs/base.yaml` line 70. Confirmed via grep across every experiment YAML: only `"all"` has ever actually been run. The plumbing for a missing-view grid already exists; only the experiments don't.
- **The `dstep-spread` table** (`03_method.tex` lines 572–596, sign-flip rates 1.7% TADS vs 11.8% CO) is already a per-view stability diagnostic under a low-quality-view lens — reframe the caption/text, run nothing new.
- **CO (λ=0) row**, already in `tab:main-results` (`04_experiments.tex`), is already exactly a "trajectory-view-missing" condition — reuse as one arm of the missing-view story for free.
- **NAIT baseline also does flat, unweighted layer fusion** — verified in `baselines/nait/direction.py::score_candidates` (`Σ_l ⟨Δh_l, v_l⟩`, no eigengap term anywhere). This sharpens the novelty claim: the equal-weight-fusion gap is a property of the whole hidden-state-geometry method family, not a TADS-specific oversight — worth one sentence in Related Work.
- **Backbone-robustness, dataset-transfer, and efficiency tables** (`tab:backbone-robustness`, `tab:dataset-transfer`, `tab:efficiency` in `04_experiments.tex`) — reusable as-is under the new framing (CFP topics 9, 10), just retitle/recaption.
- **The GenAI disclosure section is already positioned directly before the bibliography** in `main.tex` (right after `\input{sections/06_conclusion}`, right before `\bibliographystyle`) — only the title text needs to change, not the placement.

**One thing to fix, not just reframe** (must happen before any new fusion-weight code): `TrajectoryAnchor.compute_alignment()` (`trajectory_anchor.py` lines 422–465) matches the paper's flat-sum equation exactly — but it is **dead code**, called nowhere outside its own file. The actual production path is the inline loop in `selector.py::collect_episode` (line ~206: `batch_align /= float(len(trajectory_anchor.layer_indices))`), which divides by L. This division is currently numerically inert (a positive scalar rescaling that `normalize_alignment`'s min-max step cancels exactly), so **no existing reported number is wrong**. But four places (`trajectory_anchor.py` docstring line 20, `scorer.py` docstring line 13, `selector.py` comment line 13, `configs/methods/legacy.yaml` header) all describe a "(1/L) average" formula that matches none of the actual live code paths, and `legacy.yaml`'s header additionally describes a z-scored `R̃` pipeline that `scorer.py`'s own docstring says is explicitly *not* the paper-faithful main path. Fix this and unify the two fusion code paths into one function before adding any weight — the moment weights stop being uniform, the "numerically inert" argument breaks, and a reviewer or co-author cross-checking equations against code will otherwise find a real-looking discrepancy.

---

## 3. New experiments, prioritized

1. **Code unification + regression test.** Merge the dead `compute_alignment()` and the live `selector.py` divide-by-L path into one shared `fuse(states, v_by_layer, weights)` function, add a `fusion_mode` config (`flat` default). Assert bit-identical selections to current shipped results on a fixed seed/checkpoint under `flat`. *Why:* precondition — every adaptive-fusion number downstream is unverifiable without this, and it's the fix for the doc/code mismatch above. **Must-have.** ~1.5 days.
2. **Fusion-mode ablation**: `flat` (control) vs `eigengap-raw` (`π_l ∝ γ_l`) vs `eigengap-normalized` (`π_l ∝ γ_l/Σ_k γ_k`) on the flagship setting (LLaMA-2-7B + Alpaca-GPT4, ρ=10%, full 8-task eval, `04_experiments.tex` line 143 setup) plus Qwen2.5-0.5B for a fast second-backbone check. *Why:* the single most direct hit on CFP topic 11 (Adaptive Fusion Strategies). **Must-have.** ~2.5 days.
3. **Missing/degraded-layer-view robustness grid**: cross {flat, best mode from #2} × {all, `middle_to_last` (existing, never run), `boundary_only`=[1,24] (adversarial low-quality-only control), random per-refresh dropout at p∈{0.25, 0.5} — new, small extension to `_resolve_layer_indices`} on Qwen2.5-0.5B (cheap iteration) and confirm on LLaMA-2-7B. *Why:* this is the structural core of the SI — directly answers CFP topics 3 (Incomplete) and 5 (Imbalanced). **Must-have.** ~3 days.
4. **Interpretability figure**: plot `π_l^(t)` under eigengap fusion vs. the flat baseline's implicit equal weight over refresh steps, using `gap_by_layer` (already logged, zero new instrumentation). *Why:* cheap, direct hit on CFP topics 7 and 11 simultaneously — shows the mechanism, not just the outcome. **Must-have** (near-free). ~0.5 day.
5. **Difficulty-only / uncertainty-only carrier ablation** (force `w=1`/`w=0` in `pool_reward`, ~20–30 LOC). Runs alongside the already-shipped CO(λ=0) row to complete a 3-signals × missing-that-one picture at the carrier level. *Why CFP fit:* modest — frame this strictly as a **secondary/bonus robustness table**, never as "a third view." **Nice-to-have**, cut first if behind schedule. ~1 day.
6. **Noisy-view stress test** (Gaussian-noise injection into `Δh_l` at σ∈{0, 0.5×, 1×, 2×} of natural probe-delta std, flat vs. adaptive fusion). Needs a genuinely new noise-injection hook — the highest new-code-risk item on this list. *Why:* graded (not hard-zero) view corruption is the CFP's more literal language ("data with noise... low-quality multi-view data"). **Gated stretch** — cut if behind schedule, gate behind a hard go/no-go at day 10–11: only attempt if #1–4 are done and stable.

**Explicitly not this cycle**: cross-backbone score fusion and cross-source dataset pooling. Both require new training-loop surgery or a from-scratch reliability-weighting algorithm with real confound risk, and neither is necessary to clear the CFP's scope bar. One future-work paragraph in Discussion covers both.

---

## 4. New theory

A genuine Corollary to Theorem 1 — concrete enough to hand to a co-author today, pure algebra on an already-proved result, no GPU time needed.

**Setup.** Generalize the fused score to `ã_i^{(t)} = Σ_l π_l ⟨h̄_l(x_i), v_l^{(t)}⟩` with weights `π_l ≥ 0`. The current paper is the special case `π_l = 1` (unweighted). Define `ε_l := (2C_{Σ,l}/γ_l)η_{t-1}` — the RHS of Theorem 1.

**Step 1 (per-layer score sensitivity, Cauchy–Schwarz).**
`|⟨h̄_l(x_i), v_l^{(t)}⟩ − ⟨h̄_l(x_i), v_l^{(t-1)}⟩| = |⟨h̄_l(x_i), v_l^{(t)}−v_l^{(t-1)}⟩| ≤ ‖h̄_l(x_i)‖·ε_l`

**Step 2 (fused score sensitivity, triangle inequality).**
`|ã_i^{(t)} − ã_i^{(t-1)}| ≤ Σ_l π_l ‖h̄_l(x_i)‖ ε_l = Σ_l π_l ‖h̄_l(x_i)‖ (2C_{Σ,l}/γ_l) η_{t-1}`

**Step 3 (the punchline — flat vs. eigengap-normalized).** Under `π_l = 1` (the current paper), the bound is a **sum of L individual `1/γ_l` terms** — a single near-degenerate layer (`γ_l→0`) can make the whole bound unbounded regardless of every other layer's health. Under eigengap-normalized weighting `π_l = γ_l/Σ_k γ_k`, the `γ_l` **cancels**:
`π_l · (2C_{Σ,l}/γ_l) = 2C_{Σ,l}/Σ_k γ_k`, giving
`|ã_i^{(t)} − ã_i^{(t-1)}| ≤ (2η_{t-1}/Σ_k γ_k) Σ_l ‖h̄_l(x_i)‖ C_{Σ,l}`.
No single layer's spectral degeneracy can dominate this bound anymore — it's controlled by the *total* eigengap mass across all retained views, not the worst one. This is a clean, checkable, non-relabeled result: it formally explains *why* naive equal-weight fusion is fragile to exactly the boundary-layer phenomenon the paper's own diagnostic already found, and *why* eigengap weighting fixes it — not just an empirical trick.

**One-line missing-view corollary** (near-free addition): dropping view `l'` sets `π_{l'}=0`, removing a non-negative term from the Step 2 RHS — so any view-subset's fused-score sensitivity bound is weakly tighter than the full-view bound. This gives a formal complement to the empirical missing-view grid (§3, item 3).

Both go in a new subsection of `03_method.tex` (mirroring `sec:local-anchor-stability`'s structure), with the proof deferred to `07_appendix.tex` next to Theorem 1's existing proof — should take well under a day since it's pure algebra on top of an already-proved result. This is the strongest CFP topic 1 (Theoretical Foundations) hit available and should be written regardless of how the empirical timeline goes — it needs no GPU time.

---

## 5. Paper structure changes

- **Title**: keep "TADS" and "Trajectory-Anchored" (still literally accurate). Add an explicit multi-view-fusion descriptor, e.g. *"TADS: Trajectory-Anchored Dynamic Selection via Quality-Aware Multi-View Fusion of Layer Representations."*
- **Abstract**: rewrite to lead with the layer-as-view fusion contribution; subordinate the instruction-selection framing to second position.
- **Introduction** (661 words → add ~150–250): new opening paragraph establishing "a transformer layer's hidden-state subspace is a view of an instruction," grounded in classical multi-view-learning definitions, *before* the instruction-tuning motivation.
- **Related work** (657 words, already at capacity): add ~8–10 genuine MVL citations (fusion survey, incomplete MVL, imbalanced/adaptive-fusion MVL, trusted/explainable MVL); one sentence each on NAIT's and Data Agent's non-adaptive fusion. Forces cutting ~11–13 of the current 53 refs.
- **Method** (2366 words → add ~400–600 + Corollary): new subsection formalizing layer=view fusion notation, flat fusion as the "equal-trust" special case, `π_l` weighting, the new Corollary (proof in appendix). Update Algorithm 1 to show the generalized weighted sum with flat as default.
- **Experiments** (2819 words → add ~500–800): new tables/figure for fusion-mode ablation, missing-view grid, interpretability plot, and (if kept) the difficulty/uncertainty-only bonus table. Existing tables survive as-is given the flat-mode regression test — retitle `dataset-transfer` toward "cross-domain generalization" and `efficiency` toward "scalable fusion" language.
- **Discussion** (375 words → add ~150–200): explicit future-work paragraph naming cross-backbone and cross-source fusion as not pursued here; explicit limitation that layer-views are co-registered, so CFP topic 4 (Unaligned MVL) is not centrally addressed.
- **Conclusion** (126 words): light touch, one sentence reframing the headline contribution as adaptive multi-view fusion.
- **Appendix** (1248 words → add ~200–300): proof of the new Corollary next to Theorem 1's proof; config detail table for the missing-view grid.
- **Keywords**: replace ACM-style phrases with Elsevier's preferred single words: *Multi-view; Fusion; Foundation models; Instruction-tuning; Robustness; Data-selection; Representation-learning* (5–7 words, avoid "X and Y" phrasing).

---

## 6. Risks and honest caveats

- **The layer-as-view claim is a genuine reframing, not the SI's headline examples** (multi-omics, multi-modal imaging, sensor fusion — heterogeneous *external* sources). Justify it explicitly and early via classical MVL definitions plus CFP topic 2's own license for foundation-model internal representations. Don't assume the reader grants this.
- **Do not force CFP topic 4** (Unaligned MVL) — layer-views are inherently co-registered. State this limitation yourself rather than let a reviewer catch it.
- **If item 5 (§3) is kept, never call `L_i`/`H_i`/`align_i` "three views."** Two of the three are statistics of one softmax from one forward pass — the weakest multi-view claim across every candidate considered.
- **Don't overclaim a strict accuracy win.** The existing lambda-ablation already shows a sharp inverted-U at λ=1.0 (`04_experiments.tex sec:lambda-ablation`) — the current design is already fairly well-tuned on clean data. Frame the eigengap result as *more robust to missing/degraded views*, never *strictly better on clean data*.
- **Fix the code/doc discrepancy before, not after, building on top of it** (§2).
- **Reference budget is a real curation task, not arithmetic**: 53 existing entries, 50 cap, ~8–10 new MVL citations needed → net ~11–13 cuts. Start this in week one alongside the related-work rewrite.
- **elsarticle migration is unverified risk**: tikz figures, algorithm floats, `threeparttable`, `tabularx` all need to survive the class swap; trial-compile on day 1–2, in parallel with code work.
- **Dual-submission policy is a decision, not an engineering problem.** Elsevier's submission declaration implies the CIKM 2026 submission must be withdrawn or not pursued concurrently. Resolve this explicitly with all co-authors before investing the full 19-day effort.
- **Novelty-narrowness objection**: a skeptical guest editor could see "flat sum → eigengap weight" as a small tweak. Preempt this by leading with the proof (§4) as the qualitative claim, backed by the concrete missing-view grid, not theory alone.

---

## 7. Elsevier formatting/process checklist

1. Trial-compile under elsarticle immediately (day 1–2, parallel to code work) — swap ACM boilerplate for elsarticle's double-blind option; verify tikz/algorithm/threeparttable/tabularx survive; choose double-column.
2. Swap bibliography style: `ACM-Reference-Format` → elsarticle-num.
3. Trim references from 53 → target ~40–42 before adding ~8–10 new MVL citations, landing at/under the 50 cap.
4. Retitle the GenAI disclosure section to "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process." Placement needs no change.
5. Draft a Highlights file now: 3–5 bullets, ≤85 characters each.
6. Produce a graphical abstract, 531×1328 px (h×w) — repurpose the existing trajectory-anchor schematic or the new interpretability figure.
7. Write the CRediT author-contribution statement.
8. Write the funding-source statement.
9. Complete the competing-interests declaration via Elsevier's declarations tool at submission time.
10. Write the data-availability statement.
11. Add Acknowledgements as its own section, directly before the reference list.
12. Replace the keyword list with 1–7 single-word Elsevier-preferred terms.
13. Verify the double-blind class option in elsarticle matches the CFP's "single anonymized review" requirement.
14. Submit full editable source — .tex, .bib, all figure files, Highlights file, graphical abstract image. PDF-only submissions are rejected.
15. Re-verify total page count under elsarticle's actual page geometry once the migration is done.

---

**Bottom line**: primary contribution is eigengap-weighted layer-as-view fusion + a real Corollary to Theorem 1 + a missing/degraded-layer-view robustness grid (items 1–4 in §3, non-negotiable). Difficulty/uncertainty-only ablations are a cheap bonus, framed carefully. Noisy-view injection is a gated stretch. Cross-backbone and cross-source fusion are future work, stated in one paragraph, not executed. Fix the fusion-code discrepancy first. Trial-compile elsarticle on day 1. Resolve the CIKM-withdrawal decision this week, not at the deadline.
