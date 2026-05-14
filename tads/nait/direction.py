"""NAIT direction extraction.

Computes per-layer ``Δh = h_last - h_first`` over a seed JSON, fits a
top-1 PCA direction (sign-calibrated), and scores candidate samples by
mean-pooled projection onto that direction.

This module deduplicates the ``extract_delta_from_seed`` definitions in
the original ``train_nait_v2.py`` (which had two identical copies).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..data.sft_prompts import alpaca_input_part

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
    logger: Optional[logging.Logger] = None,
) -> Dict[int, torch.Tensor]:
    """For each seed item, encode the full Alpaca-style text and return
    ``{layer: stacked Δh tensor (N, H)}``.
    """
    model.eval()
    delta_per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    for idx, item in enumerate(seed_items):
        text = ALPACA_PROMPT_FULL.format(
            input_part=alpaca_input_part(item.get("input", "")),
            instruction=item["instruction"],
            output=item.get("output", ""),
        )
        inputs = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        seq_len = inputs.input_ids.shape[1]
        if seq_len < 2:
            del outputs, hidden_states
            continue
        for l in layers:
            actual_l = l if l >= 0 else len(hidden_states) + l
            h = hidden_states[actual_l]
            first_h = h[0, 0].float().cpu()
            last_h = h[0, seq_len - 1].float().cpu()
            delta_per_layer[l].append(last_h - first_h)
        del outputs, hidden_states
        if logger and (idx + 1) % 20 == 0:
            logger.info("  Seed %d/%d", idx + 1, len(seed_items))

    return {l: torch.stack(v) for l, v in delta_per_layer.items() if v}


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
    """Mean-pooled hidden state projected onto each direction; sum across layers."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    scores = []
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
        lengths = attention_mask.sum(dim=1).clamp_min(1)
        batch_scores = torch.zeros(input_ids.size(0))

        for l, v_l in directions.items():
            actual_l = l if l >= 0 else len(hidden_states) + l
            h = hidden_states[actual_l].float()
            mask = attention_mask.unsqueeze(-1).float()
            mean_h = (h * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
            v_l_gpu = v_l.to(h.device)
            batch_scores += (mean_h @ v_l_gpu).cpu()

        scores.append(batch_scores)
        del outputs, hidden_states

        if logger and (step % 100 == 0 or step == total_batches):
            elapsed = time.time() - t0
            logger.info(
                "  Scoring batch %d/%d | elapsed=%.1fmin",
                step, total_batches, elapsed / 60,
            )
    return torch.cat(scores, dim=0)
