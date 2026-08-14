"""NAIT direction extraction.

Computes per-layer ``Δh = h_last - h_first`` over a seed JSON, fits a
top-1 PCA direction (sign-calibrated), and scores candidate samples by
the same Δh projection onto that direction (paper Eq 5:
``s_y = Σ_l ⟨Δh_l(y), v_l⟩``).

For NAIT, the seed and candidate sides intentionally use the SAME
definition of the contextualization vector (Δh on both) — the directions
were fitted on Δh variance, so projecting anything else (mean-pooled
hidden state, last-token hidden state alone, etc.) onto v_l is not the
inner product the NAIT paper defines and produces a different ranking.

Note: TAG's legacy/tag selection score (tag/core/selector.py) intentionally
departs from this — it extracts v_l from the same Δh PCA
(tag/core/trajectory_anchor.py) but SCORES candidates by sequence-mean h̄_l
projection (paper Eq.6). So the two methods share the anchor-extraction side
and differ on the scoring side by design; that's the algorithmic distinction
between TAG and NAIT.

This module deduplicates the ``extract_delta_from_seed`` definitions in
the original ``train_nait_v2.py`` (which had two identical copies).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from tag.data.sft_prompts import alpaca_input_part

logger = logging.getLogger(__name__)

ALPACA_PROMPT_FULL = (
    "Below is an instruction that describes a task"
    "{input_part}. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


@torch.no_grad()
def extract_delta_from_seed(
    model,
    tokenizer,
    seed_items: List[dict],
    device,
    layers: List[int],
    max_seq_len: int = 2048,
    batch_size: int = 8,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, torch.Tensor]:
    """For each seed item, encode the full Alpaca-style text and return
    ``{layer: stacked Δh tensor (N, H)}``.

    Batched (2026-05-22): the original loop forwarded one seed at a time —
    1500 seeds × ~0.5 s on 7B = ~12 min. With batch_size=8 + right-pad
    collation the forward count drops to ~190 and the seed-extract phase
    becomes ~1.5 min. Numerics are identical to the un-batched version:
    causal LM with attention_mask + right-padding makes non-pad-position
    hidden states pad-invariant, and per-sample first/last-non-pad
    indexing reproduces the same Δh as the single-sample forward.
    """
    model.eval()
    # Build all texts upfront so the tokenizer call is the only python work
    # left inside the loop.
    texts: List[str] = [
        ALPACA_PROMPT_FULL.format(
            input_part=alpaca_input_part(item.get("input", "")),
            instruction=item["instruction"],
            output=item.get("output", ""),
        )
        for item in seed_items
    ]
    n_total = len(texts)

    # Tokenizer state — restore on exit so the shared tokenizer doesn't
    # surprise downstream code paths.
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    orig_pad_token = tokenizer.pad_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    delta_per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    t0 = time.time()
    try:
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            batch_texts = texts[batch_start:batch_end]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            outputs = model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            seq_lens = enc["attention_mask"].sum(dim=1)              # (B,)
            valid = (seq_lens >= 2).cpu()                            # (B,) bool
            B = enc["input_ids"].size(0)
            bidx = torch.arange(B, device=device)
            last_idx = (seq_lens - 1).clamp_min(0)

            for l in layers:
                actual_l = l if l >= 0 else len(hidden_states) + l
                # Slice first, THEN float-cast — avoid materialising a
                # fresh (B, T, H) fp32 tensor just to read two positions.
                first_h = hidden_states[actual_l][:, 0, :].float()        # (B, H)
                last_h = hidden_states[actual_l][bidx, last_idx, :].float()  # (B, H)
                delta = (last_h - first_h).cpu()                          # (B, H)
                if valid.all():
                    delta_per_layer[l].append(delta)
                else:
                    rows = delta[valid]
                    if rows.size(0) > 0:
                        delta_per_layer[l].append(rows)

            del outputs, hidden_states

            if logger and (batch_end // 100) > (batch_start // 100):
                logger.info(
                    "  Seed batched %d / %d (%.1fmin)",
                    batch_end, n_total, (time.time() - t0) / 60,
                )

        return {l: torch.cat(v, dim=0) for l, v in delta_per_layer.items() if v}
    finally:
        tokenizer.padding_side = orig_padding_side
        tokenizer.pad_token = orig_pad_token


def fit_directions(
    delta_per_layer: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Top-1 PCA per layer with sign calibration ``⟨v, μ⟩ > 0``."""
    directions: Dict[int, torch.Tensor] = {}
    for l, delta in delta_per_layer.items():
        N, H = delta.shape
        mu = delta.mean(dim=0, keepdim=True)
        centred = delta - mu
        if N < H:
            gram = centred @ centred.T / N
            _, gram_vecs = torch.linalg.eigh(gram)
            top = gram_vecs[:, -1]
            v = centred.T @ top
            v = v / (v.norm() + 1e-8)
        else:
            cov = centred.T @ centred / N
            _, vecs = torch.linalg.eigh(cov)
            v = vecs[:, -1]
        if torch.dot(v, mu.squeeze(0)) < 0:
            v = -v
        directions[l] = v
    return directions


@torch.no_grad()
def score_candidates(
    model,
    dataset,
    directions: Dict[int, torch.Tensor],
    device,
    batch_size: int = 4,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """Per-sample contextualization score s_y = Σ_l ⟨Δh_l, v_l⟩ (paper Eq 5).

    ``Δh_l(y) = h_l[last-non-pad-token] - h_l[first-token]`` for sample y at
    layer l. ``v_l`` was fitted on this same Δh quantity over the seed set
    in :func:`fit_directions`, so projecting Δh — not mean-pooled hidden
    state — is what the principal direction is defined against.

    Perf history (2026-05-22): four hot-path fixes vs. the original loop
    over a 50K-sample pool with 32 decoder layers:
      1. ``hidden_states[l].float()`` materialised a fresh (B, T, H) fp32
         tensor (~16 MB/layer at B=2,T=512,H=4096) just to slice two token
         positions out of it. We now slice first, then convert to fp32 —
         every batch saves ~512 MB of allocator churn across 32 layers.
      2. ``v_l_gpu = v_l.to(h.device)`` ran *inside* the per-layer loop,
         re-uploading the direction matrix every batch (≈832K transfers
         for a 13K-batch run). We now cache once at function entry.
      3. ``batch_scores += (delta @ v_l_gpu).cpu()`` synced GPU→CPU per
         layer (≈416K syncs). We now accumulate on GPU and copy once per
         batch — a single sync replaces 32.
      4. Pre-resolve negative ``actual_l`` once at entry; the inner loop
         no longer depends on ``len(hidden_states)``.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Cache directions on GPU once. ``v_gpu_by_layer`` keeps fp32 vectors
    # ready for matmul — no re-uploading inside the batch loop.
    v_gpu_by_layer: Dict[int, torch.Tensor] = {
        l: v_l.to(device, dtype=torch.float32, non_blocking=True)
        for l, v_l in directions.items()
    }

    # Layer-index resolution depends only on hidden_states tuple length,
    # which is constant for a given model. We resolve lazily on the first
    # batch and reuse for every subsequent one.
    actual_l_cache: Optional[Dict[int, int]] = None

    scores: List[torch.Tensor] = []
    total_batches = len(loader)
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if actual_l_cache is None:
            actual_l_cache = {
                l: (l if l >= 0 else len(hidden_states) + l)
                for l in v_gpu_by_layer.keys()
            }
        # Index of the last non-pad token per row (clamp_min(1) - 1).
        last_idx = attention_mask.sum(dim=1).clamp_min(1) - 1
        B = input_ids.size(0)
        bidx = torch.arange(B, device=input_ids.device)
        batch_scores = torch.zeros(B, device=device, dtype=torch.float32)

        for l, v_l_gpu in v_gpu_by_layer.items():
            actual_l = actual_l_cache[l]
            h_layer = hidden_states[actual_l]              # (B, T, H), bf16/fp16
            # Slice the two token positions first, THEN cast to fp32 —
            # avoids the (B, T, H) fp32 temp allocation that dominated
            # allocator pressure in the original loop.
            first_h = h_layer[:, 0, :].float()             # (B, H) fp32
            last_h = h_layer[bidx, last_idx, :].float()    # (B, H) fp32
            delta = last_h - first_h                       # (B, H) fp32
            batch_scores += delta @ v_l_gpu                # GPU accumulate

        scores.append(batch_scores.cpu())                  # single sync/batch
        del outputs, hidden_states

        if logger and (step % 100 == 0 or step == total_batches):
            elapsed = time.time() - t0
            logger.info(
                "  Scoring batch %d/%d | elapsed=%.1fmin",
                step, total_batches, elapsed / 60,
            )
    return torch.cat(scores, dim=0)
