"""Shared machinery for the whole-pool forward passes.

Two places in TAG push the entire candidate pool through the model once and
read only a per-token or per-sequence loss out of it:
:func:`tag.core.reliability.compute_pool_loss` (the counterfactual pool) and
:func:`tag.core.gate.compute_pool_token_losses` (Eqs. 2-6). Both were
spending most of their time on work the result does not depend on. The
helpers here remove that work, and every one of them is chosen so that the
numbers coming out are the numbers that came out before:

``crop_padded_batch``
    ``tokenize_alpaca`` right-pads every record to ``max_seq_len``, so a
    60-token record costs a 512-token forward. Under a causal mask a
    trailing padded position cannot influence any earlier position, and it
    carries no label, so dropping the columns that are padding in *every*
    row of the batch is exact.

``length_sorted_batches``
    Which records share a batch is pure parallelism — each row's loss
    depends only on that row — so it only decides how much padding survives
    the crop. Descending order also puts the widest batch first, so an OOM
    surfaces in seconds rather than twenty minutes in.

``split_lm_head`` / ``ce_from_hidden``
    HF's ``forward`` projects EVERY position to the vocabulary. With Qwen's
    151 643-entry vocabulary that is ~5 GB of logits per batch at bs=32,
    of which the ~30 % response positions are all that is ever read.
    Running the head on the gathered response hidden states computes the
    same rows.

The last one is the only helper that depends on model internals: a family
that transforms hidden states between the decoder and the head (Cohere's
``logit_scale``, Gemma-2's soft-capping) would produce different logits.
So callers verify it against the full-logits path on the first batch and
fall back if the two disagree — see :data:`SPLIT_HEAD_TOL`.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Max absolute CE disagreement (nats) tolerated between the split-head path
# and the reference full-logits path. Both consume the same hidden states;
# the only legitimate source of difference is GEMM reduction order at a
# different matrix shape, which lands around 1e-3 in bf16. A model that
# post-processes logits disagrees by order 1.
SPLIT_HEAD_TOL = 0.02

# Rows of the vocabulary projection to hold at once. The fp32 cross-entropy
# input is ``chunk * vocab * 4`` bytes — 1.2 GB at Qwen's vocabulary.
DEFAULT_HEAD_CHUNK = 2048


def split_lm_head(model):
    """Return ``(decoder, lm_head)`` when the model exposes both, else ``(None, None)``."""
    get_dec = getattr(model, "get_decoder", None)
    get_head = getattr(model, "get_output_embeddings", None)
    if not callable(get_dec) or not callable(get_head):
        return None, None
    try:
        decoder = get_dec()
        head = get_head()
    except (AttributeError, NotImplementedError, TypeError):
        return None, None
    if decoder is None or head is None or decoder is model:
        return None, None
    return decoder, head


def dataset_token_lengths(dataset) -> Optional[torch.Tensor]:
    """Unpadded length per record from ``attention_mask``, or None if unreadable.

    Only used to order the batches, so a failure here costs speed and
    nothing else — the caller falls back to sequential order.
    """
    try:
        n = len(dataset)
    except TypeError:
        return None
    lens = torch.empty(n, dtype=torch.long)
    try:
        for i in range(n):
            am = dataset[i]["attention_mask"]
            lens[i] = int(am.sum()) if torch.is_tensor(am) else int(sum(am))
    except (KeyError, TypeError, IndexError, AttributeError):
        return None
    return lens


def length_sorted_batches(
    dataset, batch_size: int,
) -> Optional[Tuple[List[List[int]], List[int]]]:
    """``(batches, emit_order)`` grouping similar-length records together.

    ``emit_order`` is the concatenation of the batches: the k-th row the
    loader yields belongs to dataset record ``emit_order[k]``. Returns None
    when the lengths cannot be read, leaving the caller on sequential order.
    """
    lens = dataset_token_lengths(dataset)
    if lens is None:
        return None
    order = torch.argsort(lens, descending=True, stable=True).tolist()
    batches = [order[a : a + batch_size] for a in range(0, len(order), batch_size)]
    return batches, order


def crop_padded_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Drop trailing columns that are padding in every row. Returns ``(..., T_eff)``.

    Both masks are consulted: a label outside the attention mask would be a
    tokenisation bug, but cropping it away would turn that bug into a
    silently shorter loss vector.
    """
    t = input_ids.size(1)
    pos1 = torch.arange(1, t + 1, device=input_ids.device)
    live = (attention_mask > 0) | (labels != -100)
    # >= 2 positions so at least one shifted target survives.
    t_eff = max(int((live * pos1).max().item()), 2)
    if t_eff >= t:
        return input_ids, attention_mask, labels, t
    return (
        input_ids[:, :t_eff],
        attention_mask[:, :t_eff],
        labels[:, :t_eff],
        t_eff,
    )


def ce_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, chunk: int = DEFAULT_HEAD_CHUNK,
) -> torch.Tensor:
    """Per-position CE over an already-gathered ``(n, V)`` logit block."""
    if targets.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=targets.device)
    parts = []
    for a in range(0, targets.numel(), chunk):
        blk = logits[a : a + chunk].float()
        parts.append(F.cross_entropy(blk, targets[a : a + chunk], reduction="none"))
        del blk
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def ce_from_hidden(
    hidden: torch.Tensor,
    head,
    targets: torch.Tensor,
    chunk: int = DEFAULT_HEAD_CHUNK,
) -> torch.Tensor:
    """Per-position CE, projecting ``(n, H)`` hidden states in chunks."""
    if targets.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=targets.device)
    parts = []
    for a in range(0, targets.numel(), chunk):
        blk = head(hidden[a : a + chunk]).float()
        parts.append(F.cross_entropy(blk, targets[a : a + chunk], reduction="none"))
        del blk
    return torch.cat(parts) if len(parts) > 1 else parts[0]


class ResponseCE:
    """Per-response-position CE for one batch, on the cheapest correct path.

    Prefers the response-only projection and verifies it against the
    full-logits path on the FIRST batch it is asked for. Two things can send
    it back to full logits, both permanently and both loudly: the two paths
    disagreeing beyond :data:`SPLIT_HEAD_TOL`, or the bare decoder call
    raising at all (a model whose ``forward`` does setup the submodule does
    not). Neither is worth losing a cluster run over, so neither is fatal.
    """

    def __init__(self, model, *, split_head: bool = True,
                 chunk: int = DEFAULT_HEAD_CHUNK, tag: str = ""):
        self.model = model
        self.chunk = int(chunk)
        self.where = f" [{tag}]" if tag else ""
        self.decoder, self.head = split_lm_head(model) if split_head else (None, None)
        self.use_split = self.decoder is not None
        self._unverified = self.use_split

    def _full(self, input_ids, attention_mask, sel, targets):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        ce = ce_from_logits(out.logits[:, :-1, :][sel], targets, self.chunk)
        del out
        return ce

    def __call__(self, input_ids, attention_mask, sel, targets) -> torch.Tensor:
        if not self.use_split:
            return self._full(input_ids, attention_mask, sel, targets)
        try:
            hidden = self.decoder(
                input_ids=input_ids, attention_mask=attention_mask,
            ).last_hidden_state
            ce = ce_from_hidden(hidden[:, :-1, :][sel], self.head, targets, self.chunk)
            del hidden
        except Exception as exc:  # noqa: BLE001 — any failure means "use the slow path"
            if not self._unverified:
                raise
            self.use_split = self._unverified = False
            logger.warning(
                "response-only lm_head%s could not run (%s: %s) — falling back "
                "to full logits for the rest of the run.",
                self.where, type(exc).__name__, exc,
            )
            return self._full(input_ids, attention_mask, sel, targets)
        if self._unverified:
            self._unverified = False
            ref = self._full(input_ids, attention_mask, sel, targets)
            diff = float((ce - ref).abs().max().item()) if ref.numel() else 0.0
            if diff > SPLIT_HEAD_TOL:
                self.use_split = False
                logger.warning(
                    "response-only lm_head%s disagrees with the full-logits "
                    "path by %.4g nats (tol %.4g) — this model transforms "
                    "hidden states between the decoder and the head. Falling "
                    "back to full logits for the rest of the run.",
                    self.where, diff, SPLIT_HEAD_TOL,
                )
                return ref
            logger.info(
                "response-only lm_head%s verified against full logits "
                "(max diff %.3g nats)", self.where, diff,
            )
            del ref
        return ce
