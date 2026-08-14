"""TAG reliability gate — the static view of the fused score (paper Eqs. 2-6).

The gate answers one question per candidate, once, before training: *does the
true instruction actually explain this response?* It is a counterfactual
likelihood contrast — the response is held fixed and the INSTRUCTION is
swapped for an unrelated one — aggregated at two granularities and squashed
into an absolute [0, 1] value.

    delta_{i,k}  = l_k(y_i | x_i^-) - l_k(y_i | x_i)                  (Eq. 2)

    Delta_bar_i  = sum_k delta_{i,k} / sum_k l_k(y_i | x_i^-)
                 = 1 - L(y_i | x_i) / L(y_i | x_i^-)                  (Eq. 3)

    Delta_{i,m}  = 1 - sum_{k in S_m} l_k(y_i | x_i)
                     / sum_{k in S_m} l_k(y_i | x_i^-)                (Eq. 4)

    Delta_min_i  = min_{m : S_m in C_i} Delta_{i,m}                   (Eq. 5)

    Delta_hat_i  = min(Delta_bar_i, Delta_min_i) - mu(M_i)            (Eq. 5')

    G_i          = c_i * (2*sigma(Delta_hat_i / s) - 1)_+             (Eq. 6)

Eq. 5' is an amendment the implementation forced (docs item A5). Eq. 5 is a
minimum over ``M = ceil(n/W)`` spans, so its null LOCATION depends on the
response length; testing it against a fixed zero vetoed 60 % of clean 7B
data, almost all of it long. ``mu(M)`` is the ``target_veto``-quantile of the
uncentred statistic on a CLEAN reference pool at span count ``M``, so the
clean veto rate becomes a dial the experimenter sets (5 % by default) and is
uniform in length. See :class:`NullCalibration`.

Why the ratio and not the raw difference (Eq. 3 vs. the v3 ``reliability``
module's ``ΔL``): the raw difference ``L(y|x^-) - L(y|x)`` is in nats and
scales with how hard the response is intrinsically. A long technical answer
can post a large raw gain while remaining barely explained by its
instruction, and a short crisp answer can post a small raw gain while being
fully determined by it. Normalising by the counterfactual loss makes the
statistic a scale-free *fraction of the response's unexplained content that
the instruction accounts for*, which is what the gate is supposed to
measure.

Why spans (Eqs. 4-5): the mean is diluted by localized corruption. A
wrong-answer sample is correct for ninety tokens and wrong for five, so
``Delta_bar`` stays healthy. Partitioning the response into contiguous spans
and taking the WORST one restores sensitivity to exactly the corruption
types that motivated the gate (T5 wrong-answer, T7 fluent-wrong, T2 spliced
noise). The token-level variant of this idea (worst bottom-rho fraction of
individual tokens) was discarded: single-token NLL is dominated by
tokenisation accidents, so its minimum is an order statistic of noise.

Alignment contract (verified against ``tads.data.sft_prompts.tokenize_alpaca``):
prompt and response are tokenised in SEPARATE tokenizer calls and the id
lists concatenated, so the response token ids of record ``i`` are identical
under ``x_i`` and ``x_i^-`` — no BPE merge can cross the boundary. The two
per-token NLL vectors are therefore index-aligned by construction. The one
exception is length budgeting: the response is truncated to
``max_seq_len - len(prompt_ids)`` and the two prompts differ in length, so
the vectors agree only over the first ``min(n_true, n_cf)`` positions.
:func:`spans_from_token_losses` trims both sides to that common prefix
before doing anything else; samples whose common prefix is too short to
judge are marked *undefined* and (by default) pass the gate rather than
being vetoed on a tokenisation artifact.

G is a property of the DATA, not of the checkpoint, so it is computed once
at the base checkpoint and cached (``tag_gate_cache.pt``) — no per-refresh
cost, and no drift in what the view means across refreshes.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import cuda_mem_str

logger = logging.getLogger(__name__)

CACHE_FILENAME = "tag_gate_cache.pt"
CACHE_VERSION = 2

_TAIL_MODES = ("min", "quantile")
_TAU_MODES = ("per_token", "absolute")
_UNDEFINED_POLICIES = ("pass", "neutral", "veto")


# ---------------------------------------------------------------------------
# Length-conditional null calibration (paper Eq. 5', see docs item A5)
# ---------------------------------------------------------------------------

def _pava_nonincreasing(y: Sequence[float], w: Sequence[float]) -> List[float]:
    """Weighted isotonic regression onto the non-increasing cone (PAVA).

    Applied to the null curve because the mechanism is monotone: ``Delta^min``
    is a minimum over ``M`` spans, so a lower quantile of it can only fall as
    ``M`` grows. Any rise in the raw per-bin quantiles is estimation noise,
    and projecting it out costs nothing while stabilising the small-count
    bins at the long-response end.
    """
    # Non-increasing in x == non-decreasing in reversed x.
    vals = [float(v) for v in reversed(list(y))]
    wts = [float(v) for v in reversed(list(w))]
    stack_v: List[float] = []
    stack_w: List[float] = []
    stack_c: List[int] = []
    for v, wt in zip(vals, wts):
        stack_v.append(v)
        stack_w.append(max(wt, 1e-12))
        stack_c.append(1)
        while len(stack_v) > 1 and stack_v[-2] > stack_v[-1]:
            v2, w2, c2 = stack_v.pop(), stack_w.pop(), stack_c.pop()
            v1, w1, c1 = stack_v.pop(), stack_w.pop(), stack_c.pop()
            wn = w1 + w2
            stack_v.append((v1 * w1 + v2 * w2) / wn)
            stack_w.append(wn)
            stack_c.append(c1 + c2)
    out: List[float] = []
    for v, c in zip(stack_v, stack_c):
        out.extend([v] * c)
    return list(reversed(out))


@dataclass(frozen=True)
class NullCalibration:
    """The clean-reference null of ``Delta_hat`` as a function of span count.

    Why this exists. ``Delta^min`` (Eq. 5) is a MINIMUM over ``M = ceil(n/W)``
    spans, so its distribution is an order statistic whose location depends on
    ``M``. Testing it against the fixed threshold ``Delta_hat <= 0`` therefore
    vetoes long responses far more often than short ones for a reason that has
    nothing to do with instruction dependency. Measured on the 7B backbone
    with W=16 over 51 760 CLEAN alpaca responses: only 39.6 % had
    ``Delta_hat > 0``, i.e. the gate vetoed 60 % of data it should have passed
    — while ``Delta_bar`` (Eq. 3, no order statistic) averaged a healthy
    +0.108. The pathology is entirely in the tail statistic's null.

    The fix is to compare ``Delta^min`` against where the null actually sits
    at that ``M`` rather than against zero:

        Delta_hat_i = min(Delta_bar_i, Delta_min_i) - mu(M_i)          (Eq. 5')

    with ``mu(M)`` the ``target_veto``-quantile of the raw statistic on a
    CLEAN reference pool restricted to span count ``M``. Two properties
    follow by construction:

      * the clean veto rate equals ``target_veto`` in EVERY bin, so it is a
        dial the experimenter sets (0.05 by default) rather than an emergent
        60 %; and
      * it is length-uniform, which is what removes the bias.

    ``mu`` is estimated on clean data ONLY. It cannot absorb corruption
    signal: a dirty sample whose worst span sits far below where clean
    samples of the same length sit still lands negative and is still vetoed.
    That is the difference between this and self-calibrating on the candidate
    pool, which would.

    Eq. 6 is untouched: ``sigma(0) = 1/2`` still makes ``Delta_hat <= 0``
    produce ``G == 0`` exactly, so the fusion stays non-compensatory. Only
    the origin of ``Delta_hat`` moves.

    Fields:
        bin_edges: inclusive upper edges on ``M``; the last bin catches
            everything above the second-to-last edge.
        mu: one offset per bin, same length as ``bin_edges``.
        counts: reference samples per bin (diagnostics, and PAVA weights).
        target_veto: the clean-reference veto rate this curve targets.
        span_tokens: the ``W`` the curve was fit at — ``M`` means nothing
            without it, so a curve fit at one ``W`` must never be used at
            another.
    """

    bin_edges: Tuple[int, ...]
    mu: Tuple[float, ...]
    counts: Tuple[int, ...]
    target_veto: float
    span_tokens: int
    n_ref: int
    monotone: bool = True

    def __post_init__(self) -> None:
        if len(self.bin_edges) != len(self.mu) or len(self.mu) != len(self.counts):
            raise ValueError(
                f"NullCalibration: bin_edges/mu/counts must be the same length, got "
                f"{len(self.bin_edges)}/{len(self.mu)}/{len(self.counts)}"
            )
        if not self.bin_edges:
            raise ValueError("NullCalibration: at least one bin is required")
        if list(self.bin_edges) != sorted(set(self.bin_edges)):
            raise ValueError(
                f"NullCalibration: bin_edges must be strictly increasing, got {self.bin_edges}"
            )
        if not (0.0 < self.target_veto < 1.0):
            raise ValueError(
                f"NullCalibration: target_veto must be in (0,1), got {self.target_veto}"
            )
        if self.span_tokens < 1:
            raise ValueError(
                f"NullCalibration: span_tokens must be >= 1, got {self.span_tokens}"
            )

    def lookup(self, n_spans: torch.Tensor) -> torch.Tensor:
        """``mu(M_i)`` for each sample, with the last bin absorbing the tail."""
        edges = torch.as_tensor(self.bin_edges, dtype=torch.long)
        mu = torch.as_tensor(self.mu, dtype=torch.float32)
        # right=False: boundaries[i-1] < v <= boundaries[i], i.e. the edges
        # are inclusive upper bounds. Values past the last edge clamp into it.
        idx = torch.bucketize(n_spans.long(), edges, right=False)
        idx = idx.clamp(max=len(self.mu) - 1)
        return mu[idx]

    def apply(self, delta_hat_raw: torch.Tensor, n_spans: torch.Tensor) -> torch.Tensor:
        return delta_hat_raw.float() - self.lookup(n_spans)

    def digest(self) -> str:
        """Short content hash — the cache compares curves without printing them."""
        payload = "|".join(
            [
                ",".join(str(int(e)) for e in self.bin_edges),
                ",".join(f"{float(v):.8g}" for v in self.mu),
                f"{float(self.target_veto):.8g}",
                str(int(self.span_tokens)),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_edges": list(int(e) for e in self.bin_edges),
            "mu": list(float(v) for v in self.mu),
            "counts": list(int(c) for c in self.counts),
            "target_veto": float(self.target_veto),
            "span_tokens": int(self.span_tokens),
            "n_ref": int(self.n_ref),
            "monotone": bool(self.monotone),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NullCalibration":
        return NullCalibration(
            bin_edges=tuple(int(e) for e in d["bin_edges"]),
            mu=tuple(float(v) for v in d["mu"]),
            counts=tuple(int(c) for c in d["counts"]),
            target_veto=float(d["target_veto"]),
            span_tokens=int(d["span_tokens"]),
            n_ref=int(d["n_ref"]),
            monotone=bool(d.get("monotone", True)),
        )


def fit_null_calibration(
    delta_hat_raw: torch.Tensor,
    n_spans: torch.Tensor,
    *,
    target_veto: float,
    span_tokens: int,
    min_bin_count: int = 400,
    max_bins: int = 24,
    monotone: bool = True,
) -> NullCalibration:
    """Fit ``mu(M)`` on a CLEAN reference pool (see :class:`NullCalibration`).

    Bins are built by walking the distinct ``M`` values in order and closing a
    bin once it holds ``min_bin_count`` samples, so short responses — where
    the population is dense and the curve moves fastest — get fine bins and
    the sparse long tail gets one wide one. A trailing bin below the minimum
    is merged back into its predecessor rather than left to estimate a
    quantile from a handful of points.
    """
    if not (0.0 < target_veto < 1.0):
        raise ValueError(f"fit_null_calibration: target_veto in (0,1), got {target_veto}")
    if min_bin_count < 1:
        raise ValueError(f"fit_null_calibration: min_bin_count >= 1, got {min_bin_count}")
    if max_bins < 1:
        raise ValueError(f"fit_null_calibration: max_bins >= 1, got {max_bins}")
    stat = delta_hat_raw.detach().float().view(-1)
    m = n_spans.detach().long().view(-1)
    if stat.numel() != m.numel():
        raise ValueError(
            f"fit_null_calibration: delta_hat_raw has {stat.numel()} entries but "
            f"n_spans has {m.numel()}"
        )
    finite = torch.isfinite(stat)
    if not bool(finite.all()):
        logger.warning(
            "fit_null_calibration: dropping %d non-finite reference values.",
            int((~finite).sum().item()),
        )
    stat, m = stat[finite], m[finite]
    if stat.numel() == 0:
        raise ValueError("fit_null_calibration: reference contains no finite values")

    uniq = torch.unique(m, sorted=True)
    # Aim for max_bins even when the pool is small enough that min_bin_count
    # alone would allow more.
    per_bin = max(int(min_bin_count), int(math.ceil(stat.numel() / max_bins)))
    edges: List[int] = []
    acc = 0
    for value in uniq.tolist():
        acc += int((m == value).sum().item())
        if acc >= per_bin:
            edges.append(int(value))
            acc = 0
    last = int(uniq.max().item())
    if not edges:
        edges = [last]
    elif edges[-1] != last:
        # The leftover tail is below per_bin: widen the final bin instead of
        # estimating a quantile from it.
        edges[-1] = last

    mus: List[float] = []
    counts: List[int] = []
    lo = -1
    for hi in edges:
        sel = (m > lo) & (m <= hi)
        vals = stat[sel]
        counts.append(int(vals.numel()))
        mus.append(
            float(torch.quantile(vals, float(target_veto)).item())
            if vals.numel()
            else float("nan")
        )
        lo = hi
    # An empty interior bin cannot happen given how edges were built, but a
    # NaN here would silently poison every gate downstream.
    if any(math.isnan(v) for v in mus):
        raise RuntimeError(f"fit_null_calibration: empty bin among edges {edges}")

    raw = list(mus)
    if monotone:
        mus = _pava_nonincreasing(mus, counts)

    cal = NullCalibration(
        bin_edges=tuple(edges),
        mu=tuple(mus),
        counts=tuple(counts),
        target_veto=float(target_veto),
        span_tokens=int(span_tokens),
        n_ref=int(stat.numel()),
        monotone=bool(monotone),
    )
    logger.info(
        "fit_null_calibration | n_ref=%d | W=%d | target_veto=%.3f | %d bin(s)",
        stat.numel(), span_tokens, target_veto, len(edges),
    )
    for i, hi in enumerate(edges):
        lo_i = 1 if i == 0 else edges[i - 1] + 1
        logger.info(
            "  M in [%d, %s] | n=%d | mu_raw=%+.4f | mu=%+.4f",
            lo_i, "inf" if i == len(edges) - 1 else str(hi),
            counts[i], raw[i], mus[i],
        )
    return cal


def null_veto_report(
    cal: NullCalibration,
    delta_hat_raw: torch.Tensor,
    n_spans: torch.Tensor,
) -> List[Dict[str, float]]:
    """Per-bin veto rate after centering — the check that the fix worked.

    The point of the null correction is that the clean veto rate is uniform in
    length. That is a claim about the data, so it gets measured and logged
    rather than asserted.
    """
    centered = cal.apply(delta_hat_raw, n_spans)
    m = n_spans.detach().long().view(-1)
    rows: List[Dict[str, float]] = []
    lo = -1
    for i, hi in enumerate(cal.bin_edges):
        sel = (m > lo) & (m <= hi) if i < len(cal.bin_edges) - 1 else (m > lo)
        vals = centered[sel]
        rows.append(
            {
                "m_lo": float(1 if i == 0 else cal.bin_edges[i - 1] + 1),
                "m_hi": float(hi),
                "n": float(vals.numel()),
                "mu": float(cal.mu[i]),
                "veto_rate": float((vals <= 0).float().mean().item()) if vals.numel() else float("nan"),
            }
        )
        lo = hi
    return rows


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    """Hyper-parameters of the reliability gate (paper Eqs. 4-6).

    span_tokens (W): fixed span width in response tokens (Eq. 4). Sentence
        spans are the paper's other option; fixed windows are used here
        because they need no detokenisation and no sentence splitter, and
        because a corrupted region rarely respects sentence boundaries.
    tau: low-information threshold defining C_i (Eq. 5). A span whose
        COUNTERFACTUAL loss is below tau carries no instruction dependency
        to begin with — boilerplate, closing pleasantries, formatting — and
        must not be allowed to trigger the gate.
    tau_mode: "per_token" (default) compares tau against the span's MEAN
        counterfactual NLL; "absolute" compares against the span's SUM, the
        literal reading of Eq. 5. Per-token is the default because the sum
        scales with span length, so an absolute threshold systematically
        excludes the final partial span of every response and makes the
        exclusion rule a length filter in disguise. See ``docs/tag-paper-
        deltas.md`` item P1.
    min_span_tokens: spans shorter than this are dropped from C_i outright
        (a 1-2 token trailing fragment has a high-variance ratio).
    tail_mode / tail_quantile: "min" is Eq. 5 exactly. "quantile" takes the
        ``tail_quantile``-th quantile of the valid span gains instead, a
        length-bias-robust ablation arm (see docs item P2).
    include_eos: whether the appended EOS position participates in the span
        aggregation. Default False: termination is already covered by c_i,
        and the EOS position's predictability reflects response completeness
        rather than instruction dependency.
    c_trunc: completeness value for a response that does not terminate
        properly (paper Eq. 6, c_i in {1, c_trunc}).
    eps_den: floor on any denominator in Eqs. 3-4.
    min_common_tokens: a sample whose true/counterfactual responses share
        fewer than this many token positions has no usable evidence.
    undefined_policy: what to do with such samples. "neutral" (default)
        assigns G = c_i * undefined_gate_value — the gate value a typical
        clean sample receives, so a sample nobody could judge is neither
        promoted nor punished. "pass" (G = c_i) is the SUPREMUM no evidenced
        sample can reach and exists only as an ablation showing the gate is
        doing the work; "veto" (G = 0) is the opposite ablation.
    undefined_gate_value: the neutral gate value, default 0.6 = 2*0.8 - 1,
        i.e. exactly what the default calibration target (90% of clean data
        at raw sigma >= 0.8) assigns to the clean reference's 10th
        percentile.
    null_correction: whether to recentre Delta_hat on its clean-reference
        null at the same span count (Eq. 5', :class:`NullCalibration`).
        Default True: without it the tail min's order-statistic drift vetoed
        60% of CLEAN 7B data and made the veto rate a function of response
        length. False reproduces the literal Eq. 5 and is the ablation arm
        that shows why the correction is needed.
    target_veto: the clean-reference veto rate the correction targets. This
        is the "how much should the gate reject" dial; it must be strictly
        below ``calibration_target_pct`` (default 0.10), otherwise the
        scale calibration's own quantile lands at or below zero.
    null: the fitted curve itself, carried from the calibration artifact.
        ``null_correction=True`` with ``null=None`` is an error at gate time,
        exactly like an unresolved ``scale``.
    scale (s): sigmoid scale of Eq. 6, calibrated once per backbone on a
        clean reference pool via :func:`calibrate_gate_scale`. None triggers
        an in-pool fallback that is DIAGNOSTIC ONLY.
    dispersion_discount: with K > 1 counterfactuals, multiply the mean gate
        by (1 - 2*std_k)_+ so that unstable evidence is trusted less.
    """

    span_tokens: int = 16
    tau: float = 0.5
    tau_mode: str = "per_token"
    min_span_tokens: int = 4
    tail_mode: str = "min"
    tail_quantile: float = 0.0
    include_eos: bool = False
    c_trunc: float = 0.2
    eps_den: float = 1e-3
    min_common_tokens: int = 8
    undefined_policy: str = "neutral"
    undefined_gate_value: float = 0.6
    null_correction: bool = True
    target_veto: float = 0.05
    null: Optional[NullCalibration] = None
    scale: Optional[float] = None
    dispersion_discount: bool = True

    def __post_init__(self) -> None:
        if self.span_tokens < 1:
            raise ValueError(f"GateConfig: span_tokens must be >= 1, got {self.span_tokens}")
        if self.tau < 0:
            raise ValueError(f"GateConfig: tau must be >= 0, got {self.tau}")
        if self.tau_mode not in _TAU_MODES:
            raise ValueError(f"GateConfig: tau_mode must be one of {_TAU_MODES}, got {self.tau_mode!r}")
        if self.min_span_tokens < 1:
            raise ValueError(f"GateConfig: min_span_tokens must be >= 1, got {self.min_span_tokens}")
        if self.tail_mode not in _TAIL_MODES:
            raise ValueError(f"GateConfig: tail_mode must be one of {_TAIL_MODES}, got {self.tail_mode!r}")
        if not (0.0 <= self.tail_quantile <= 1.0):
            raise ValueError(f"GateConfig: tail_quantile must be in [0,1], got {self.tail_quantile}")
        if not (0.0 < self.c_trunc <= 1.0):
            raise ValueError(f"GateConfig: c_trunc must be in (0,1], got {self.c_trunc}")
        if self.eps_den <= 0:
            raise ValueError(f"GateConfig: eps_den must be > 0, got {self.eps_den}")
        if self.min_common_tokens < 1:
            raise ValueError(
                f"GateConfig: min_common_tokens must be >= 1, got {self.min_common_tokens}"
            )
        if self.undefined_policy not in _UNDEFINED_POLICIES:
            raise ValueError(
                f"GateConfig: undefined_policy must be one of {_UNDEFINED_POLICIES}, "
                f"got {self.undefined_policy!r}"
            )
        if not (0.0 <= self.undefined_gate_value <= 1.0):
            raise ValueError(
                f"GateConfig: undefined_gate_value must be in [0,1], "
                f"got {self.undefined_gate_value}"
            )
        if self.tau_mode == "absolute" and self.tau < 1.0:
            logger.warning(
                "GateConfig: tau_mode='absolute' thresholds the span's SUMMED "
                "counterfactual NLL, but tau=%.3f looks like a per-token value "
                "(a full span of %d tokens sums to roughly %.1f x that). The "
                "absolute arm needs its own tau — otherwise it measures a "
                "~%dx threshold change, not sum-vs-mean.",
                self.tau, self.span_tokens, float(self.span_tokens),
                self.span_tokens,
            )
        if self.scale is not None and self.scale <= 0:
            raise ValueError(f"GateConfig: scale must be > 0 or None, got {self.scale}")
        if not (0.0 < self.target_veto < 1.0):
            raise ValueError(
                f"GateConfig: target_veto must be in (0,1), got {self.target_veto}"
            )
        if self.null is not None and self.null.span_tokens != self.span_tokens:
            # M is a span COUNT, so mu(M) means a different thing at a
            # different W. Reusing the curve across W would look fine and
            # silently shift every gate value.
            raise ValueError(
                f"GateConfig: null calibration was fit at span_tokens="
                f"{self.null.span_tokens} but this config uses "
                f"{self.span_tokens}. Refit it at the new W with "
                f"scripts/calibrate_reliability.py --mode tag (or "
                f"scripts/sweep_gate_config.py to compare W first)."
            )
        if self.null is not None and abs(self.null.target_veto - self.target_veto) > 1e-9:
            raise ValueError(
                f"GateConfig: the null calibration was fit at target_veto="
                f"{self.null.target_veto} but this config sets target_veto="
                f"{self.target_veto}. mu(M) IS that quantile, so it cannot be "
                f"reused at a different target — refit it with "
                f"scripts/sweep_gate_config.py --target-veto {self.target_veto}."
            )

    def identity(self) -> Dict[str, Any]:
        """The subset of fields that changes G — used for cache validation.

        ``scale`` is included: re-calibrating the backbone changes every
        gate value even though no forward pass is needed to redo it.

        The null curve is reduced to a content digest. It changes G exactly
        as much as any other field, so it must participate in the comparison,
        but printing two dozen floats in every cache-mismatch message buries
        the field that actually differs.
        """
        d = asdict(self)
        d["null"] = (
            None
            if self.null is None
            else {
                "digest": self.null.digest(),
                "n_bins": len(self.null.bin_edges),
                "target_veto": float(self.null.target_veto),
                "span_tokens": int(self.null.span_tokens),
                "n_ref": int(self.null.n_ref),
            }
        )
        return d


# ---------------------------------------------------------------------------
# Per-token forward (the only GPU work the gate does)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pool_token_losses(
    model,
    dataset,
    *,
    batch_size: int = 1,
    device: str = "cuda",
    progress_interval: int = 200,
    empty_cache_interval: int = 10,
    tag: str = "",
    eos_token_id: Optional[int] = None,
    drop_trailing_eos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token NLL over response tokens for the whole pool.

    This is :func:`tads.core.reliability.compute_pool_loss` without the final
    collapse to a sequence mean — the span aggregation of Eqs. 4-5 needs the
    vector, not its average.

    Args:
        drop_trailing_eos: drop the final response position when it is the
            appended EOS token (``GateConfig.include_eos = False``). The
            check is on the label id, so a budget-truncated response — whose
            EOS was cut off — keeps all of its tokens.

    Returns:
        ``(token_loss, n_tokens)`` where ``token_loss`` is a padded
        ``(N, T_max)`` float32 CPU tensor (zeros past each row's length) and
        ``n_tokens`` is ``(N,)`` int64. Rows with zero response tokens are
        possible in principle and are reported as length 0.
    """
    if drop_trailing_eos and eos_token_id is None:
        raise ValueError(
            "compute_pool_token_losses: drop_trailing_eos=True requires eos_token_id"
        )
    was_training = model.training
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    rows: List[torch.Tensor] = []
    t0 = time.time()
    total_batches = len(loader)
    n_dropped_eos = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        resp_mask = shift_labels != -100
        B = shift_logits.size(0)
        for i in range(B):
            sel = resp_mask[i]
            if not bool(sel.any()):
                rows.append(torch.zeros(0, dtype=torch.float32))
                continue
            tgt = shift_labels[i][sel]
            ce = F.cross_entropy(
                shift_logits[i][sel].float(), tgt, reduction="none",
            )
            if drop_trailing_eos and int(tgt[-1].item()) == int(eos_token_id):
                ce = ce[:-1]
                n_dropped_eos += 1
            rows.append(ce.detach().cpu().float())
        del out, input_ids, attention_mask, labels, shift_logits, shift_labels, resp_mask
        if (
            torch.cuda.is_available()
            and empty_cache_interval > 0
            and step % empty_cache_interval == 0
        ):
            torch.cuda.empty_cache()
        if step == 1 or step % progress_interval == 0 or step == total_batches:
            logger.info(
                "compute_pool_token_losses%s | batch=%d/%d | elapsed=%.1fmin | %s",
                f" [{tag}]" if tag else "", step, total_batches,
                (time.time() - t0) / 60, cuda_mem_str(),
            )
    if was_training:
        model.train()
    n_tokens = torch.tensor([r.numel() for r in rows], dtype=torch.long)
    t_max = int(n_tokens.max().item()) if n_tokens.numel() else 0
    token_loss = torch.zeros((len(rows), max(t_max, 1)), dtype=torch.float32)
    for i, r in enumerate(rows):
        if r.numel():
            token_loss[i, : r.numel()] = r
    logger.info(
        "compute_pool_token_losses%s | done | n=%d | T_max=%d | dropped_eos=%d | %.1fmin",
        f" [{tag}]" if tag else "", len(rows), t_max, n_dropped_eos,
        (time.time() - t0) / 60,
    )
    return token_loss, n_tokens


# ---------------------------------------------------------------------------
# Span aggregation (Eq. 4 machinery)
# ---------------------------------------------------------------------------

def spans_from_token_losses(
    token_true: torch.Tensor,
    n_true: torch.Tensor,
    token_cf: torch.Tensor,
    n_cf: torch.Tensor,
    *,
    span_tokens: int,
) -> Dict[str, torch.Tensor]:
    """Trim both pools to their common prefix and sum losses within spans.

    The trim is the load-bearing step. Response token IDS are identical
    across the two pools, but the two prompts have different lengths, so
    ``max_seq_len`` budget truncation can leave the two response segments at
    different lengths. Only the first ``min(n_true, n_cf)`` positions are
    genuinely paired; comparing beyond that silently contrasts token ``k`` of
    one response against nothing at all.

    Spans are the fixed windows ``[0,W), [W,2W), ...`` over the trimmed
    prefix, so span ``m`` covers the same token positions in both pools.

    Returns a dict with:
        span_true / span_cf : (N, M_max) float32 span-summed NLLs
        span_len            : (N, M_max) int64 tokens per span (0 = padding)
        n_spans             : (N,) int64
        total_true/total_cf : (N,) float32 sums over the trimmed prefix
        n_common            : (N,) int64 trimmed length
    """
    if span_tokens < 1:
        raise ValueError(f"spans_from_token_losses: span_tokens must be >= 1, got {span_tokens}")
    if token_true.dim() != 2 or token_cf.dim() != 2:
        raise ValueError(
            f"spans_from_token_losses: token losses must be 2-D padded tensors, got "
            f"{tuple(token_true.shape)} and {tuple(token_cf.shape)}"
        )
    n = token_true.size(0)
    if token_cf.size(0) != n or n_true.numel() != n or n_cf.numel() != n:
        raise ValueError(
            f"spans_from_token_losses: pool size mismatch — token_true N={n}, "
            f"token_cf N={token_cf.size(0)}, n_true={n_true.numel()}, n_cf={n_cf.numel()}"
        )
    n_common = torch.minimum(n_true.long(), n_cf.long())
    m_max = max(1, int(math.ceil(max(1, int(n_common.max().item())) / span_tokens)))

    span_true = torch.zeros((n, m_max), dtype=torch.float32)
    span_cf = torch.zeros((n, m_max), dtype=torch.float32)
    span_len = torch.zeros((n, m_max), dtype=torch.long)

    # Vectorised scatter-add over the padded token axis: a token at position
    # k belongs to span k // W, and is dropped when k >= n_common[i].
    t_axis = max(token_true.size(1), token_cf.size(1))
    if token_true.size(1) < t_axis:
        token_true = F.pad(token_true, (0, t_axis - token_true.size(1)))
    if token_cf.size(1) < t_axis:
        token_cf = F.pad(token_cf, (0, t_axis - token_cf.size(1)))
    pos = torch.arange(t_axis, dtype=torch.long).unsqueeze(0).expand(n, t_axis)
    valid = pos < n_common.unsqueeze(1)
    span_idx = torch.clamp(pos // span_tokens, max=m_max - 1)
    src_true = torch.where(valid, token_true[:, :t_axis].float(), torch.zeros(1))
    src_cf = torch.where(valid, token_cf[:, :t_axis].float(), torch.zeros(1))
    ones = torch.where(valid, torch.ones(1), torch.zeros(1))
    span_true.scatter_add_(1, span_idx, src_true)
    span_cf.scatter_add_(1, span_idx, src_cf)
    span_len.scatter_add_(1, span_idx, ones.long())

    n_spans = torch.ceil(n_common.float() / float(span_tokens)).long()
    return {
        "span_true": span_true,
        "span_cf": span_cf,
        "span_len": span_len,
        "n_spans": n_spans,
        "total_true": span_true.sum(dim=1),
        "total_cf": span_cf.sum(dim=1),
        "n_common": n_common,
    }


# ---------------------------------------------------------------------------
# Gains (Eqs. 3-5)
# ---------------------------------------------------------------------------

def overall_gain(
    total_true: torch.Tensor,
    total_cf: torch.Tensor,
    *,
    eps_den: float = 1e-3,
) -> torch.Tensor:
    """Eq. 3: ``Delta_bar = 1 - L(y|x) / L(y|x^-)``, denominator-guarded.

    Because both sums run over the SAME trimmed token set, the sum-ratio of
    Eq. 3 and the mean-ratio are identical — the trim in
    :func:`spans_from_token_losses` is what makes that true.
    """
    den = total_cf.float().clamp(min=eps_den)
    return 1.0 - total_true.float() / den


def span_gains(
    span_true: torch.Tensor,
    span_cf: torch.Tensor,
    span_len: torch.Tensor,
    *,
    eps_den: float = 1e-3,
) -> torch.Tensor:
    """Eq. 4: per-span relative gain. Padding spans return 0 (masked later)."""
    den = span_cf.float().clamp(min=eps_den)
    out = 1.0 - span_true.float() / den
    return torch.where(span_len > 0, out, torch.zeros_like(out))


def valid_span_mask(
    span_cf: torch.Tensor,
    span_len: torch.Tensor,
    *,
    tau: float,
    tau_mode: str = "per_token",
    min_span_tokens: int = 4,
) -> torch.Tensor:
    """C_i of Eq. 5: spans with enough content to judge.

    ``tau_mode="per_token"`` thresholds the span's MEAN counterfactual NLL.
    The literal Eq. 5 reading is the SUM (``"absolute"``), which is length-
    dependent: with W = 16 and tau tuned for full spans, every trailing
    partial span is excluded regardless of its content. See docs item P1.
    """
    if tau_mode not in _TAU_MODES:
        raise ValueError(f"valid_span_mask: tau_mode must be one of {_TAU_MODES}, got {tau_mode!r}")
    occupied = span_len > 0
    long_enough = span_len >= int(min_span_tokens)
    if tau_mode == "per_token":
        denom = span_len.clamp(min=1).float()
        informative = (span_cf.float() / denom) >= float(tau)
    else:
        informative = span_cf.float() >= float(tau)
    return occupied & long_enough & informative


def tail_gain(
    gains: torch.Tensor,
    mask: torch.Tensor,
    fallback: torch.Tensor,
    *,
    mode: str = "min",
    quantile: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 5: the worst admissible span, with an explicit empty-C_i rule.

    When ``C_i`` is empty — every span is boilerplate, or the response is
    shorter than ``min_span_tokens`` — Eq. 5 is undefined. We fall back to
    ``fallback`` (in practice ``Delta_bar``), i.e. the tail test abstains and
    the overall gain decides alone. Vetoing here would punish short answers
    for being short; passing unconditionally would blind the gate. See docs
    item P3.

    Returns ``(tail, used_fallback)``.
    """
    if mode not in _TAIL_MODES:
        raise ValueError(f"tail_gain: mode must be one of {_TAIL_MODES}, got {mode!r}")
    has_valid = mask.any(dim=1)
    if mode == "min":
        big = torch.full_like(gains, float("inf"))
        masked = torch.where(mask, gains, big)
        tail = masked.min(dim=1).values
    else:
        tail = torch.empty(gains.size(0), dtype=torch.float32)
        for i in range(gains.size(0)):
            row = gains[i][mask[i]]
            tail[i] = (
                torch.quantile(row.float(), float(quantile))
                if row.numel()
                else float("inf")
            )
    tail = torch.where(has_valid, tail, fallback.float())
    return tail, ~has_valid


# ---------------------------------------------------------------------------
# The gate itself (Eq. 6)
# ---------------------------------------------------------------------------

def gate_from_delta_hat(
    delta_hat: torch.Tensor,
    completeness: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Eq. 6: ``G = c * (2*sigma(Delta_hat / s) - 1)_+``.

    ``sigma(0) = 1/2`` makes the clamp bite exactly at zero gain, so
    ``Delta_hat <= 0 => G == 0.0`` in floating point, not merely close to it.
    That exactness is what makes the fusion non-compensatory: the product
    ``G * R * (1 + lam*a)`` is identically zero no matter how large the
    dynamic factors get.
    """
    if scale <= 0:
        raise ValueError(f"gate_from_delta_hat: scale must be > 0, got {scale}")
    if delta_hat.shape != completeness.shape:
        raise ValueError(
            f"gate_from_delta_hat: shape mismatch delta_hat={tuple(delta_hat.shape)} "
            f"vs completeness={tuple(completeness.shape)}"
        )
    raw = torch.sigmoid(delta_hat.float() / float(scale))
    return completeness.float() * torch.clamp(2.0 * raw - 1.0, min=0.0, max=1.0)


def calibrate_gate_scale(
    ref_delta_hat: torch.Tensor,
    *,
    target_pct: float = 0.10,
    target_q: float = 0.8,
    null_corrected: bool = False,
) -> float:
    """Calibrate ``s`` on a CLEAN reference pool's ``Delta_hat``.

        s = quantile_{target_pct}(Delta_hat_clean) / logit(target_q)

    Reads as "(1 - target_pct) of clean reference pairs receive raw
    sigma >= target_q", i.e. with the defaults 90% of clean data lands at
    ``G >= 2*0.8 - 1 = 0.6``. The gate is therefore near-inert on clean
    pools by construction, while a pair whose instruction contributes
    nothing (``Delta_hat <= 0``) is pinned at zero regardless of s.

    The calibration statistic MUST be ``Delta_hat``, not ``Delta_bar``: the
    tail min is systematically lower than the mean, so calibrating on the
    mean and gating on the min would silently veto a large slice of clean
    data. Calibrating on ``Delta_hat`` also absorbs the population-level
    length dependence of the min (docs item P2).
    """
    if not (0.0 < target_pct < 1.0):
        raise ValueError(f"calibrate_gate_scale: target_pct in (0,1), got {target_pct}")
    if not (0.5 < target_q < 1.0):
        raise ValueError(f"calibrate_gate_scale: target_q in (0.5,1), got {target_q}")
    ref = ref_delta_hat.detach().float().view(-1)
    ref = ref[torch.isfinite(ref)]
    if ref.numel() == 0:
        raise ValueError("calibrate_gate_scale: reference contains no finite values")
    if ref.numel() < 100:
        logger.warning(
            "calibrate_gate_scale: reference has only %d samples — quantile "
            "estimate will be unstable.", ref.numel(),
        )
    logit = math.log(target_q / (1.0 - target_q))
    q = float(torch.quantile(ref, target_pct).item())
    if q <= 0:
        med = float(ref.median().item())
        if null_corrected:
            # After Eq. 5' centering, quantile_{target_veto} == 0 by
            # construction, so a non-positive quantile at target_pct means
            # target_pct <= target_veto. That is a config error with an exact
            # fix, not something to paper over with the median.
            raise ValueError(
                f"calibrate_gate_scale: P{int(100 * target_pct)}(Delta_hat) = "
                f"{q:.4f} on a NULL-CORRECTED reference. Centering puts the "
                f"target_veto quantile at exactly 0, so calibration_target_pct "
                f"must be strictly greater than target_veto. Raise "
                f"calibration_target_pct or lower tads.tag.target_veto."
            )
        logger.warning(
            "calibrate_gate_scale: P%d(Delta_hat_clean)=%.4f is not positive — the "
            "clean reference looks contaminated, or the counterfactual instructions "
            "are not actually unrelated. Falling back to the median (%.4f).",
            int(100 * target_pct), q, med,
        )
        q = med
    if q <= 0:
        logger.warning(
            "calibrate_gate_scale: median Delta_hat_clean is also non-positive (%.4f). "
            "Using s=1.0 — treat every G from this calibration as diagnostic-only.", q,
        )
        return 1.0
    s = q / logit
    logger.info(
        "calibrate_gate_scale | n_ref=%d | P%d=%.4f | target_q=%.2f | s=%.5f",
        ref.numel(), int(100 * target_pct), q, target_q, s,
    )
    return s


def fit_calibration(
    delta_hat_raw: torch.Tensor,
    n_spans: torch.Tensor,
    *,
    span_tokens: int,
    target_veto: float = 0.05,
    target_pct: float = 0.10,
    target_q: float = 0.8,
    null_correction: bool = True,
    min_bin_count: int = 400,
    max_bins: int = 24,
    monotone: bool = True,
) -> Dict[str, Any]:
    """Both halves of the clean-reference calibration, in the required order.

    ``mu(M)`` first, then ``s`` from the ALREADY-CENTRED statistic. The order
    matters: s is a quantile of whatever Eq. 6 will actually see, so deriving
    it from the uncentred reference and then centring at gate time would
    shift every value by mu while leaving the scale that interprets them put.

    Returns ``{"null", "scale", "delta_hat", "report", "veto_rate",
    "frac_positive"}``. ``delta_hat`` is the centred reference statistic —
    the thing to inspect when asking "what will this gate do".
    """
    if null_correction and not (target_veto < target_pct):
        raise ValueError(
            f"fit_calibration: target_veto ({target_veto}) must be strictly "
            f"below target_pct ({target_pct}) — centring puts the target_veto "
            f"quantile at exactly 0, so s would be derived from a "
            f"non-positive quantile."
        )
    null = (
        fit_null_calibration(
            delta_hat_raw, n_spans,
            target_veto=target_veto, span_tokens=span_tokens,
            min_bin_count=min_bin_count, max_bins=max_bins, monotone=monotone,
        )
        if null_correction
        else None
    )
    stat = null.apply(delta_hat_raw, n_spans) if null is not None else delta_hat_raw.float()
    s = calibrate_gate_scale(
        stat, target_pct=target_pct, target_q=target_q,
        null_corrected=null is not None,
    )
    report = null_veto_report(null, delta_hat_raw, n_spans) if null is not None else []
    return {
        "null": null,
        "scale": s,
        "delta_hat": stat,
        "report": report,
        "veto_rate": float((stat <= 0).float().mean().item()),
        "frac_positive": float((stat > 0).float().mean().item()),
    }


# ---------------------------------------------------------------------------
# End-to-end assembly
# ---------------------------------------------------------------------------

def gate_components(
    token_true: torch.Tensor,
    n_true: torch.Tensor,
    token_cf: torch.Tensor,
    n_cf: torch.Tensor,
    *,
    cfg: GateConfig,
) -> Dict[str, torch.Tensor]:
    """Eqs. 2-5 for ONE counterfactual pairing: gains and their diagnostics.

    Returns ``delta_bar``, ``delta_min``, ``delta_hat`` plus the masks and
    counters the per-view attribution figures need. The gate itself (Eq. 6)
    is applied by :func:`compute_gate`, which also handles K > 1.

    ``delta_hat`` is the value Eq. 6 consumes: ``min(Delta_bar, Delta_min)``
    recentred on the clean-reference null when ``cfg.null`` is present
    (Eq. 5'). ``delta_hat_raw`` is the uncentred quantity, kept because it is
    what a null curve is fit on and what the ablation arm reports.
    """
    sp = spans_from_token_losses(
        token_true, n_true, token_cf, n_cf, span_tokens=cfg.span_tokens,
    )
    d_bar = overall_gain(sp["total_true"], sp["total_cf"], eps_den=cfg.eps_den)
    gains = span_gains(sp["span_true"], sp["span_cf"], sp["span_len"], eps_den=cfg.eps_den)
    mask = valid_span_mask(
        sp["span_cf"], sp["span_len"],
        tau=cfg.tau, tau_mode=cfg.tau_mode, min_span_tokens=cfg.min_span_tokens,
    )
    d_min, empty_c = tail_gain(
        gains, mask, d_bar, mode=cfg.tail_mode, quantile=cfg.tail_quantile,
    )
    d_hat_raw = torch.minimum(d_bar, d_min)
    if cfg.null_correction and cfg.null is None:
        raise ValueError(
            "gate_components: null_correction is on but no null calibration was "
            "supplied. Fit one on a CLEAN reference pool with "
            "scripts/calibrate_reliability.py --mode tag and point "
            "tads.tag.gate_ref_file at it, or set tads.tag.null_correction: "
            "false to run the uncorrected Eq. 5 ablation (which vetoed 60% of "
            "clean 7B data)."
        )
    d_hat = cfg.null.apply(d_hat_raw, sp["n_spans"]) if cfg.null_correction else d_hat_raw

    undefined = sp["n_common"] < int(cfg.min_common_tokens)
    return {
        "delta_bar": d_bar,
        "delta_min": d_min,
        "delta_hat": d_hat,
        "delta_hat_raw": d_hat_raw,
        "span_gains": gains,
        "span_valid": mask,
        "n_spans": sp["n_spans"],
        "n_valid_spans": mask.sum(dim=1),
        "n_common": sp["n_common"],
        "empty_c": empty_c,
        "undefined": undefined,
        "total_true": sp["total_true"],
        "total_cf": sp["total_cf"],
    }


def _apply_undefined_policy(
    gate: torch.Tensor,
    completeness: torch.Tensor,
    undefined: torch.Tensor,
    policy: str,
    neutral_value: float,
) -> torch.Tensor:
    """No evidence, no verdict — see ``GateConfig.undefined_policy``.

    Note why ``pass`` is NOT the default. ``2*sigma(z) - 1 < 1`` for every
    finite z, so ``G = c_i`` is a value no EVIDENCED sample can ever attain:
    handing it to zero-evidence samples would rank them above every sample
    the gate actually examined. Short responses would then be promoted
    precisely because they were too short to judge. ``neutral`` instead
    assigns the gate value a typical clean sample receives.
    """
    if not bool(undefined.any()):
        return gate
    if policy == "pass":
        repl = completeness.float()
    elif policy == "neutral":
        repl = completeness.float() * float(neutral_value)
    else:  # "veto"
        repl = torch.zeros_like(gate)
    return torch.where(undefined, repl, gate)


def compute_gate(
    token_true: torch.Tensor,
    n_true: torch.Tensor,
    token_cf: Sequence[torch.Tensor],
    n_cf: Sequence[torch.Tensor],
    completeness: torch.Tensor,
    *,
    cfg: GateConfig,
) -> Dict[str, torch.Tensor]:
    """Full gate (Eqs. 2-6), including the K > 1 counterfactual extension.

    With K > 1 the gate is computed per counterfactual pairing and averaged,
    then discounted by the cross-pairing dispersion. Gating FIRST and
    averaging second (rather than averaging ``Delta_hat`` and gating once) is
    deliberate and matches ``tads.core.reliability``: the clamp in Eq. 6 is
    convex, so gate-of-mean <= mean-of-gates by Jensen, and evidence that
    straddles zero would collapse to an exact veto under gate-of-mean even
    when some pairings show a clear positive dependency.

    Args:
        token_cf / n_cf: K parallel lists, one entry per counterfactual pool.
        completeness: (N,) c_i in (0, 1].

    Returns a dict with ``gate`` (N,) plus every intermediate the diagnostics
    and the paper's attribution figure need. When K > 1 the reported
    ``delta_*`` are the across-pairing means.
    """
    if len(token_cf) != len(n_cf):
        raise ValueError(
            f"compute_gate: got {len(token_cf)} counterfactual loss tensors but "
            f"{len(n_cf)} length vectors"
        )
    if not token_cf:
        raise ValueError("compute_gate: at least one counterfactual pool is required")
    scale = cfg.scale
    if scale is None:
        raise ValueError(
            "compute_gate: cfg.scale is None — calibrate it on a clean reference pool "
            "with calibrate_gate_scale(), or use resolve_scale() for the explicitly "
            "diagnostic in-pool fallback."
        )
    n = token_true.size(0)
    if completeness.numel() != n:
        raise ValueError(
            f"compute_gate: completeness length {completeness.numel()} != pool size {n}"
        )

    per_k: List[Dict[str, torch.Tensor]] = []
    gates: List[torch.Tensor] = []
    for k, (tc, nc) in enumerate(zip(token_cf, n_cf)):
        comp = gate_components(token_true, n_true, tc, nc, cfg=cfg)
        g = gate_from_delta_hat(comp["delta_hat"], completeness, scale=scale)
        g = _apply_undefined_policy(
            g, completeness, comp["undefined"],
            cfg.undefined_policy, cfg.undefined_gate_value,
        )
        per_k.append(comp)
        gates.append(g)

    gate_stack = torch.stack(gates, dim=0)  # (K, N)
    gate = gate_stack.mean(dim=0)
    dispersion = (
        gate_stack.std(dim=0, unbiased=False)
        if gate_stack.size(0) > 1
        else torch.zeros(n)
    )
    if cfg.dispersion_discount and gate_stack.size(0) > 1:
        gate = gate * torch.clamp(1.0 - 2.0 * dispersion, min=0.0)

    def _mean(key: str) -> torch.Tensor:
        return torch.stack([c[key].float() for c in per_k], dim=0).mean(dim=0)

    undefined_any = torch.stack([c["undefined"] for c in per_k], dim=0).any(dim=0)
    empty_c_any = torch.stack([c["empty_c"] for c in per_k], dim=0).any(dim=0)

    out = {
        "gate": gate,
        "gate_per_cf": gate_stack,
        "gate_dispersion": dispersion,
        "delta_bar": _mean("delta_bar"),
        "delta_min": _mean("delta_min"),
        "delta_hat": _mean("delta_hat"),
        "delta_hat_raw": _mean("delta_hat_raw"),
        "n_spans": per_k[0]["n_spans"],
        "n_valid_spans": _mean("n_valid_spans"),
        "n_common": per_k[0]["n_common"],
        "undefined": undefined_any,
        "empty_c": empty_c_any,
        "completeness": completeness.float(),
    }
    logger.info(
        "compute_gate | n=%d | K=%d | s=%.5f | null=%s | G_mean=%.4f | "
        "G==0: %d (%.1f%%) | undefined=%d | empty_C=%d | delta_bar=%.4f | "
        "delta_min=%.4f | delta_hat=%.4f",
        n, len(token_cf), scale,
        f"W{cfg.null.span_tokens}/v{cfg.null.target_veto:.2f}/{cfg.null.digest()}"
        if (cfg.null_correction and cfg.null is not None) else "off",
        float(gate.mean().item()),
        int((gate == 0).sum().item()), 100.0 * float((gate == 0).float().mean().item()),
        int(undefined_any.sum().item()), int(empty_c_any.sum().item()),
        float(out["delta_bar"].mean().item()), float(out["delta_min"].mean().item()),
        float(out["delta_hat"].mean().item()),
    )
    if cfg.null_correction and cfg.null is not None:
        # The candidate pool is dirtier than the reference, so its veto rate
        # SHOULD exceed target_veto. A rate far below it means the reference
        # does not describe this pool (wrong backbone, wrong W, wrong pool);
        # far above it means the pool is much dirtier than expected. Both are
        # worth seeing before a multi-hour run rather than after.
        veto = float((gate == 0).float().mean().item())
        if veto < 0.5 * cfg.null.target_veto:
            logger.warning(
                "compute_gate: pool veto rate %.1f%% is far BELOW the clean "
                "reference target %.1f%% — the null curve may not describe "
                "this pool (different backbone or pool than it was fit on).",
                100.0 * veto, 100.0 * cfg.null.target_veto,
            )
        elif veto > 0.5:
            logger.warning(
                "compute_gate: pool veto rate %.1f%% — the gate is rejecting "
                "more than half the pool. Expected roughly the dirty fraction "
                "plus %.1f%%; check the calibration before trusting the run.",
                100.0 * veto, 100.0 * cfg.null.target_veto,
            )
    return out


def resolve_scale(
    cfg: GateConfig,
    delta_hat_in_pool: torch.Tensor,
    *,
    target_q: float = 0.8,
) -> float:
    """Return ``cfg.scale``, or a loudly-warned in-pool fallback.

    Self-calibration reintroduces exactly the pool dependence that anchoring
    at ``Delta_hat = 0`` removes — the same statistic gets a different gate
    depending on how dirty its neighbours are. Acceptable for forward-only
    diagnostics; never for a reported selection run.
    """
    if cfg.scale is not None:
        return float(cfg.scale)
    pos = delta_hat_in_pool.detach().float().view(-1)
    pos = pos[torch.isfinite(pos) & (pos > 0)]
    if pos.numel() == 0:
        logger.warning(
            "resolve_scale: no positive Delta_hat mass for the in-pool fallback — "
            "using s=1.0 (diagnostic-only).",
        )
        return 1.0
    s = float(pos.median().item()) / math.log(target_q / (1.0 - target_q))
    logger.warning(
        "resolve_scale: no calibrated scale provided — using IN-POOL fallback "
        "s=%.5f from the median positive Delta_hat. This reintroduces pool "
        "dependence; pass tads.tag.gate_scale or tads.tag.gate_ref_file for "
        "reported runs.", s,
    )
    return max(s, 1e-6)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path_for(output_dir) -> Path:
    return Path(output_dir) / CACHE_FILENAME


def cache_identity(*, model_path, pool_files, n_pool: int) -> Dict[str, Any]:
    """What a gate cache is valid FOR.

    G is a function of (pool, base checkpoint, gate config) and nothing else
    — not the seed, not the arm, not the epoch. That is what makes it safe to
    compute once and share across every arm and seed of an experiment, which
    is the whole point of :mod:`scripts.precompute_gate`.

    It is also what makes a shared cache dangerous if unlabelled: a cache
    from a different pool or a different backbone has the right shape and
    would be reused silently. So the producer records this identity and the
    consumer checks it.
    """
    return {
        "model_path": str(model_path),
        "pool_files": str(pool_files),
        "n_pool": int(n_pool),
    }


def check_cache_identity(cache: Dict[str, Any], want: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable reason the cache does not apply, or None.

    A cache written before identities existed carries no ``identity`` key; it
    is accepted with a warning rather than discarded, since the per-run
    caches that predate sharing were never cross-used.
    """
    got = cache.get("identity")
    if got is None:
        logger.warning(
            "TAG gate cache has no identity block (written before shared "
            "caches existed) — cannot verify it belongs to this pool and "
            "backbone. Accepting it; regenerate to get the check.",
        )
        return None
    diffs = {k: (got.get(k), want.get(k)) for k in want if got.get(k) != want.get(k)}
    if not diffs:
        return None
    return "; ".join(f"{k}: cache={a!r} run={b!r}" for k, (a, b) in diffs.items())


def load_gate_cache(output_dir, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load ``tag_gate_cache.pt``; None when absent, unreadable, or too old.

    Deliberately a SEPARATE file from ``reliability_cache.pt``: the MVF cache
    stores a different statistic under the same key names, and the MVF
    validity check would happily accept a TAG cache (and vice versa).
    """
    p = Path(path) if path is not None else cache_path_for(output_dir)
    if not p.exists():
        return None
    try:
        cache = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # corrupted cache: recompute rather than crash
        logger.warning("Could not load TAG gate cache at %s (%s)", p, e)
        return None
    if int(cache.get("cache_version", -1)) != CACHE_VERSION:
        logger.warning(
            "TAG gate cache at %s has version %s (expected %d); ignoring it.",
            p, cache.get("cache_version"), CACHE_VERSION,
        )
        return None
    for key in ("gate", "completeness", "config"):
        if key not in cache:
            logger.warning("TAG gate cache at %s missing key %r; ignoring", p, key)
            return None
    logger.info(
        "Loaded TAG gate cache from %s | n=%d | computed_at_epoch=%s",
        p, cache["gate"].numel(), cache.get("epoch"),
    )
    return cache


def save_gate_cache(
    output_dir,
    *,
    result: Dict[str, torch.Tensor],
    cfg: GateConfig,
    epoch: int,
    token_true: Optional[torch.Tensor] = None,
    n_true: Optional[torch.Tensor] = None,
    token_cf: Optional[Sequence[torch.Tensor]] = None,
    n_cf: Optional[Sequence[torch.Tensor]] = None,
    store_token_losses: bool = False,
    identity: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> None:
    """Persist G, its diagnostics, and the config that produced it.

    With ``store_token_losses`` the raw per-token NLLs are stored too (fp16,
    roughly ``2*N*T_max`` bytes per pool), which lets a later run re-derive G
    under a different W / tau / s WITHOUT a forward pass — the same
    affordance the MVF cache provides by storing its raw loss vectors.
    """
    p = Path(path) if path is not None else cache_path_for(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "gate": result["gate"].cpu(),
        "completeness": result["completeness"].cpu(),
        "delta_bar": result["delta_bar"].cpu(),
        "delta_min": result["delta_min"].cpu(),
        "delta_hat": result["delta_hat"].cpu(),
        # .get, and None rather than a fallback to delta_hat: with the
        # correction on the two differ by mu, and storing the centred value
        # under the raw key would corrupt any later refit that reads it.
        "delta_hat_raw": (
            result["delta_hat_raw"].cpu() if "delta_hat_raw" in result else None
        ),
        # The full curve, not just the digest identity() carries: a cache
        # re-derived under a new W needs to know what the old one was, and a
        # reader inspecting the cache should not have to find the reference.
        "null": cfg.null.to_dict() if cfg.null is not None else None,
        "n_spans": result["n_spans"].cpu(),
        "n_common": result["n_common"].cpu(),
        "undefined": result["undefined"].cpu(),
        "empty_c": result["empty_c"].cpu(),
        "config": cfg.identity(),
        "epoch": epoch,
    }
    if store_token_losses and token_true is not None and token_cf is not None:
        payload["token_true"] = token_true.to(torch.float16).cpu()
        payload["n_true"] = n_true.cpu() if n_true is not None else None
        payload["token_cf"] = [t.to(torch.float16).cpu() for t in token_cf]
        payload["n_cf"] = [t.cpu() for t in (n_cf or [])]
    tmp = p.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    logger.info(
        "Saved TAG gate cache to %s (n=%d, token_losses=%s)",
        p, result["gate"].numel(), bool(store_token_losses),
    )


# Fields whose value is baked into artifacts the cache CANNOT re-derive:
# ``include_eos`` decides which tokens the forward pass kept, and ``c_trunc``
# is baked into the completeness vector the caller computed from the dataset.
# Changing either invalidates the cache outright — re-deriving would silently
# apply the OLD value while stamping the cache with the NEW identity, so every
# later run would then get a "cache hit" on a wrong gate.
_FORWARD_BOUND_FIELDS = ("include_eos", "c_trunc")


def recompute_gate_from_cache(
    cache: Dict[str, Any],
    cfg: GateConfig,
) -> Optional[Dict[str, torch.Tensor]]:
    """Re-derive G from cached token losses under a new config, no forward.

    Returns None when the cache cannot honour the requested config — either
    it does not carry the raw token losses, or the change touches a field
    that is baked into the cached artifacts (see ``_FORWARD_BOUND_FIELDS``).
    The caller must then re-run the forward pass.
    """
    if "token_true" not in cache or "token_cf" not in cache:
        return None
    cached_cfg = cache.get("config") or {}
    stale = [
        f for f in _FORWARD_BOUND_FIELDS
        if f in cached_cfg and cached_cfg[f] != getattr(cfg, f)
    ]
    if stale:
        logger.warning(
            "TAG gate cache cannot be re-derived: %s changed (%s -> %s). "
            "These are baked into the cached token losses / completeness "
            "vector, so a no-forward re-derivation would apply the OLD value "
            "under the NEW config identity. Recomputing from scratch.",
            ", ".join(stale),
            {f: cached_cfg.get(f) for f in stale},
            {f: getattr(cfg, f) for f in stale},
        )
        return None
    token_true = cache["token_true"].float()
    n_true = cache.get("n_true")
    token_cf = [t.float() for t in cache["token_cf"]]
    n_cf = list(cache.get("n_cf") or [])
    if n_true is None or len(n_cf) != len(token_cf):
        return None
    completeness = cache["completeness"].float()
    logger.info(
        "Recomputing TAG gate from cached token losses under a new config "
        "(no forward pass needed)."
    )
    return compute_gate(
        token_true, n_true, token_cf, n_cf, completeness, cfg=cfg,
    )
