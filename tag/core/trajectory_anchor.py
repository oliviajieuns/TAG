"""Trajectory Anchor — multi-layer capability direction v_l (paper §3.3).

For each layer l and sample x, the contextualisation vector is (paper Eq. 1):
    Δh_l(x; θ_t) := h_l^{(K_x)}(x; θ_t) - h_l^{(1)}(x; θ_t).

At refresh step t, over a uniformly-sampled probe subset D̃_t ⊂ D
(|D̃_t| = n_p), we compute (paper Eqs. 7-8):
    mean probe delta   Δh̄_l^{(t)} = (1/|D̃_t|) Σ_{x ∈ D̃_t} Δh_l(x; θ_{t-1})
    centred covariance Σ̂_l^{(t)} = (1/|D̃_t|) Σ_x (Δh_l - Δh̄)(Δh_l - Δh̄)^T
The uncalibrated anchor ṽ_l^{(t)} is the top unit eigenvector of Σ̂_l^{(t)}
(paper Eq. 9).

Sign calibration (paper §3.3, immediately after Eq. 9; also Theorem 1
assumption (3) "Consistent temporal sign calibration"):
    t = 1:  align ṽ_l^{(1)} with Δh̄_l^{(1)}   (spatial: ⟨ṽ, Δh̄⟩ > 0)
    t > 1:  align ṽ_l^{(t)} with v_l^{(t-1)}    (temporal: ⟨ṽ_l^{(t)}, v_l^{(t-1)}⟩ > 0)
The resulting sign-calibrated anchor is denoted v_l^{(t)}.

At scoring time the trajectory-anchored branch (legacy/tag score modes)
projects h̄_l (paper Eq. 2 sequence-mean activation):
    align_i^{(t)} = (1/L) Σ_l ⟨h̄_l(x_i; θ_{t-1}), v_l^{(t)}⟩  (paper §3.3 anchor)
    widetilde-align_i^{(t)} = min-max-norm(align) ∈ [0, 1]    (paper §3.3 anchor)

This v_l extraction is shared between TAG and NAIT (both consume the
same {Δh_l} PCA), but the scoring projection differs by design: TAG
uses h̄_l (sequence-mean, Eq. 2); NAIT (Chen et al., ICLR 2026, Eq. 5)
uses Δh_l on the candidate side too.

Layer selection (``layer_indices`` arg):
    "all"             → every decoder layer (paper default, recommended)
    "middle_to_last"  → layers L//2 .. L-1 (memory-friendlier ablation)
    list[int]         → explicit indices
    None              → falls back to legacy single-layer mode using ``layer_idx``
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


LayerSpec = Union[str, List[int], None]

# Maximum number of past epochs of (v, lambda, stability) history we keep
# in memory. NAIT-faithful runs do 3 epochs so this is effectively unlimited;
# the cap matters only for long ablation sweeps that call .update() many
# times against the same anchor instance.
_MAX_HISTORY = 50


def _resolve_layer_indices(spec: LayerSpec, num_decoder_layers: int) -> List[int]:
    """Translate a layer-spec into a concrete list of decoder-layer indices.

    ``num_decoder_layers`` is the number of transformer blocks (L). Returned
    indices are 0-based among the L decoder layers (NOT among hidden_states
    which has L+1 entries because position 0 is the embedding).
    """
    if spec is None:
        raise ValueError("layer_indices spec is None — caller should fall back to legacy mode")
    if isinstance(spec, str):
        if spec == "all":
            return list(range(num_decoder_layers))
        if spec == "middle_to_last":
            return list(range(num_decoder_layers // 2, num_decoder_layers))
        raise ValueError(f"Unknown layer_indices string: {spec!r}")
    if isinstance(spec, (list, tuple)):
        out = [int(x) for x in spec]
        for i in out:
            if not (0 <= i < num_decoder_layers):
                raise ValueError(
                    f"layer index {i} out of range [0, {num_decoder_layers}); "
                    f"model has {num_decoder_layers} decoder layers."
                )
        return out
    raise TypeError(f"layer_indices must be str | list | None, got {type(spec).__name__}")


class TrajectoryAnchor:
    """Multi-layer capability anchor.

    ``layer_indices`` can be passed at construction time (resolved against
    the model's layer count at the first :meth:`update` call), or left as
    None to use the legacy single-layer behaviour driven by ``layer_idx``.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        layer_indices: LayerSpec = None,
        max_samples_for_pca: int = 1024,
        pca_batch_size: int = 4,
        device: str = "cuda",
    ):
        self.layer_idx = layer_idx
        self.layer_indices_spec: LayerSpec = layer_indices
        self.max_samples_for_pca = max_samples_for_pca
        self.pca_batch_size = pca_batch_size
        self.device = device

        # Resolved indices — filled in lazily on first update() so we can
        # use the model's actual layer count. Empty means "not yet resolved
        # OR legacy single-layer mode".
        self.layer_indices: List[int] = []
        # Direction per layer (decoder-layer index → unit vector ∈ R^H).
        self.v_by_layer: Dict[int, torch.Tensor] = {}
        self.lambda1_by_layer: Dict[int, float] = {}
        self.lambda2_by_layer: Dict[int, float] = {}
        self.gap_by_layer: Dict[int, float] = {}

        # Legacy single-layer state (kept for backward compat).
        self.v: Optional[torch.Tensor] = None  # alias of v_by_layer for single-layer mode
        self.lambda_1: float = 0.0
        self.lambda_2: float = 0.0
        self.gap: float = 0.0

        # History (Theorem 1 verification). Track mean over layers in multi mode.
        self.v_history: List[torch.Tensor] = []
        self.lambda1_history: List[float] = []
        self.lambda2_history: List[float] = []
        self.gap_history: List[float] = []
        self.stability_history: List[float] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.v_by_layer)

    @property
    def is_multi_layer(self) -> bool:
        return self.layer_indices_spec is not None

    # ------------------------------------------------------------------ PCA
    @staticmethod
    def _pca_top1(delta: torch.Tensor) -> Dict[str, Any]:
        """Run top-1 PCA on a centred (N, H) delta matrix.

        Returns dict with ``v`` (unit eigenvector), ``lambda_1``, ``lambda_2``,
        and the un-centred mean ``mu`` used for sign calibration upstream.
        """
        N, H = delta.shape
        mu = delta.mean(dim=0, keepdim=True)
        centred = delta - mu
        if N < H:
            gram = centred @ centred.T / N
            eigvals, eigvecs = torch.linalg.eigh(gram)
            lambda_1 = float(eigvals[-1].item())
            lambda_2 = float(eigvals[-2].item()) if N >= 2 else 0.0
            top_u = eigvecs[:, -1]
            v = centred.T @ top_u
            v = v / (v.norm() + 1e-8)
        else:
            cov = centred.T @ centred / N
            eigvals, eigvecs = torch.linalg.eigh(cov)
            lambda_1 = float(eigvals[-1].item())
            lambda_2 = float(eigvals[-2].item())
            v = eigvecs[:, -1]
        return {
            "v": v,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "mu": mu.squeeze(0),
        }

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(
        self,
        model,
        dataset,
        seed: int = 42,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Re-extract the anchor at the start of epoch ``t``.

        Collects Δh per layer over a probe subset and PCAs each independently.

        Probe seed is offset (+1) from the (seed + epoch*100) used by
        ``_random_indices``. Without the offset both RNGs would draw the
        same permutation, so for ratio≤probe_size/N the random-method
        selection and the anchor probe would overlap perfectly — biasing
        the alignment direction toward the very samples that will then
        be SFT'd, an unintended coupling.
        """
        g = torch.Generator()
        g.manual_seed(seed + epoch * 100 + 1)
        n_total = len(dataset)
        n_use = min(self.max_samples_for_pca, n_total)
        perm = torch.randperm(n_total, generator=g).tolist()
        indices = perm[:n_use]

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=self.pca_batch_size,
            shuffle=False,
            num_workers=0,
        )

        model.eval()
        # Accumulator: layer_idx -> List[Tensor (B, H)].
        per_layer_deltas: Dict[int, List[torch.Tensor]] = {}
        resolved_indices: Optional[List[int]] = None

        # Per-batch progress logging: without these the function is silent
        # for the full forward loop (~3-5 min on 7B + 32 decoder layers +
        # probe=1024 + pca_batch_size=4 → ~256 batches), so any hang inside
        # forward / hidden-state extraction / CPU transfer is indistinguish-
        # able from "still running". Log every 20 batches + at the end.
        import time as _time
        _t0 = _time.time()
        total_batches = len(loader)
        logger.info(
            "TrajectoryAnchor.update: forward loop start | epoch=%d | "
            "probe=%d | bs=%d | batches=%d",
            epoch, n_use, self.pca_batch_size, total_batches,
        )

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # tuple, length L+1

            # Decoder layers occupy hidden_states[1:]. Their count is L.
            num_decoder_layers = len(hidden_states) - 1

            # Resolve layer_indices once we know L.
            if resolved_indices is None:
                if self.is_multi_layer:
                    resolved_indices = _resolve_layer_indices(
                        self.layer_indices_spec, num_decoder_layers,
                    )
                else:
                    # Legacy single-layer mode: translate layer_idx (negative
                    # supported, indexing into the full hidden_states tuple
                    # for backward-compat) to a decoder-layer index.
                    li = self.layer_idx
                    if li < 0:
                        li = len(hidden_states) + li
                    # Convert hidden_states-index → decoder-layer-index by
                    # subtracting 1 (embedding offset). When the legacy
                    # spec asks for the embedding itself (li == 0), clamp.
                    decoder_li = max(0, li - 1)
                    resolved_indices = [decoder_li]
                self.layer_indices = resolved_indices

            lengths = (attention_mask.sum(dim=1).clamp_min(1) - 1).to(input_ids.device)
            bidx = torch.arange(input_ids.size(0), device=input_ids.device)

            # Compute all 32 per-layer (B, H) deltas on GPU first, then do a
            # SINGLE bulk transfer to CPU at the end. The previous code did
            # `.detach().float().cpu()` per layer (32 separate sync points)
            # — the new stack-then-cpu pattern produces bitwise-identical
            # values (same operations, same dtype, same order) but cuts the
            # device→host sync count 32× per batch.
            batch_layer_deltas = []
            for li in resolved_indices:
                # hidden_states[li + 1] is the li-th decoder layer
                h = hidden_states[li + 1]                       # (B, T, H)
                first_h = h[:, 0, :]
                last_h = h[bidx, lengths]
                batch_layer_deltas.append(
                    (last_h - first_h).detach().float(),
                )                                               # GPU (B, H) fp32
            stacked = torch.stack(batch_layer_deltas, dim=0).cpu()  # (L, B, H), 1 sync
            for i, li in enumerate(resolved_indices):
                per_layer_deltas.setdefault(li, []).append(stacked[i])

            del hidden_states, outputs, batch_layer_deltas, stacked

            if step == 1 or step % 20 == 0 or step == total_batches:
                _elapsed = _time.time() - _t0
                _per_b = _elapsed / max(1, step)
                _eta = _per_b * (total_batches - step)
                logger.info(
                    "TrajectoryAnchor.update: forward %d/%d | %.1fmin "
                    "elapsed | %.2fs/batch | ETA %.1fmin",
                    step, total_batches, _elapsed / 60, _per_b, _eta / 60,
                )

        logger.info(
            "TrajectoryAnchor.update: forward loop done in %.1fmin — "
            "starting per-layer PCA (%d layers)",
            (_time.time() - _t0) / 60,
            len(resolved_indices) if resolved_indices else 0,
        )
        _t_pca_start = _time.time()

        # Per-layer PCA + sign calibration.
        # Each per_layer_deltas[li] holds a list of (B, H) fp32 CPU tensors
        # totalling N_probe × H ≈ 540 MB at N_probe=1024, H=4096. Across 32
        # decoder layers ("all" mode) the dict alone occupies ~17 GB CPU
        # before this loop starts. Both the original list AND the cat'd
        # tensor are alive simultaneously during _pca_top1 — peak would
        # briefly double for the layer being processed. Drop each layer's
        # raw list as soon as it's been PCA'd (and the cat'd tensor as soon
        # as PCA returns) so the CPU footprint trends DOWN through the loop
        # instead of staying pinned at ~17 GB. Critical for the anchor computation under DDP
        # where rank 0's PCA peak races with workers' grad buffers.
        new_v_by_layer: Dict[int, torch.Tensor] = {}
        new_l1: Dict[int, float] = {}
        new_l2: Dict[int, float] = {}
        n_used = 0
        _n_layers = len(resolved_indices or [])
        for _li_idx, li in enumerate(resolved_indices or [], start=1):
            _t_layer = _time.time()
            delta_l = torch.cat(per_layer_deltas[li], dim=0)  # (N, H)
            # Release the per-batch chunks the moment we have the cat'd
            # matrix. Without this, every chunk for `li` stays referenced
            # by the dict until the function returns.
            per_layer_deltas[li] = []
            n_used = delta_l.shape[0]
            pca = self._pca_top1(delta_l)
            del delta_l  # ~540 MB freed before next layer's cat allocates.
            v_l = pca["v"]
            # Sign calibration (paper §3.3, immediately after Eq. 9; also
            # Theorem 1 assumption (3) "Consistent temporal sign calibration"):
            #   t = 1 (no previous v_l yet): spatial — align ṽ_l^{(1)} with
            #       Δh̄_l^{(1)} so ⟨v_l^{(1)}, Δh̄_l^{(1)}⟩ > 0.
            #   t > 1: temporal — align ṽ_l^{(t)} with v_l^{(t-1)} so
            #       ⟨v_l^{(t)}, v_l^{(t-1)}⟩ > 0.
            # `self.v_by_layer` still holds the PREVIOUS epoch's calibrated v_l
            # (it gets overwritten by `new_v_by_layer` at the end of this loop).
            # At t = 1 it's empty so we fall back to the spatial rule. We also
            # fall back to spatial when the previous-epoch dict is missing this
            # layer key (rare, but possible if `layer_indices_spec` changed
            # between refresh calls).
            prev_v_l = self.v_by_layer.get(li)
            if prev_v_l is not None:
                # Temporal: align with previous epoch's calibrated v_l^{(t-1)}.
                if torch.dot(v_l, prev_v_l) < 0:
                    v_l = -v_l
            else:
                # Spatial (t = 1 or first call): align with mean Δh̄_l.
                if torch.dot(v_l, pca["mu"]) < 0:
                    v_l = -v_l
            new_v_by_layer[li] = v_l
            new_l1[li] = pca["lambda_1"]
            new_l2[li] = pca["lambda_2"]
            if _li_idx == 1 or _li_idx % 8 == 0 or _li_idx == _n_layers:
                logger.info(
                    "TrajectoryAnchor.update: PCA %d/%d (layer=%d) | "
                    "%.2fs/layer | total PCA %.1fmin",
                    _li_idx, _n_layers, li,
                    _time.time() - _t_layer,
                    (_time.time() - _t_pca_start) / 60,
                )
        # Drop the now-empty dict before downstream code allocates more.
        del per_layer_deltas

        # Stability: mean L2 distance between old and new v per layer.
        if self.is_fitted and set(new_v_by_layer.keys()) == set(self.v_by_layer.keys()):
            diffs = [
                float(torch.norm(new_v_by_layer[k] - self.v_by_layer[k]).item())
                for k in new_v_by_layer
            ]
            stability = sum(diffs) / len(diffs) if diffs else float("nan")
        else:
            stability = float("nan")

        # Commit new state.
        self.v_by_layer = new_v_by_layer
        self.lambda1_by_layer = new_l1
        self.lambda2_by_layer = new_l2
        self.gap_by_layer = {k: new_l1[k] - new_l2[k] for k in new_l1}

        # Aggregate scalars for history / logging.
        self.lambda_1 = sum(new_l1.values()) / len(new_l1) if new_l1 else 0.0
        self.lambda_2 = sum(new_l2.values()) / len(new_l2) if new_l2 else 0.0
        self.gap = self.lambda_1 - self.lambda_2

        # Single-layer alias (`self.v`) — set only when legacy mode.
        if not self.is_multi_layer and resolved_indices:
            self.v = self.v_by_layer[resolved_indices[0]]
        else:
            self.v = None

        # Concatenate v_l → flat vector, used for history bookkeeping only.
        flat_v = torch.cat([new_v_by_layer[k] for k in sorted(new_v_by_layer)])
        self.v_history.append(flat_v.clone())
        self.lambda1_history.append(self.lambda_1)
        self.lambda2_history.append(self.lambda_2)
        self.gap_history.append(self.gap)
        self.stability_history.append(stability)
        # Bound history so long-running configurations (>>3 epochs, ablation
        # sweeps that call .update repeatedly) don't accumulate (32*H,)
        # tensors indefinitely. ~50MB per entry at L=32, H=4096, fp32.
        if len(self.v_history) > _MAX_HISTORY:
            drop = len(self.v_history) - _MAX_HISTORY
            self.v_history = self.v_history[drop:]
            self.lambda1_history = self.lambda1_history[drop:]
            self.lambda2_history = self.lambda2_history[drop:]
            self.gap_history = self.gap_history[drop:]
            self.stability_history = self.stability_history[drop:]

        stats = {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "gap": self.gap,
            "stability": stability,
            "n_samples_used": int(n_used),
            "num_layers": len(self.layer_indices),
        }
        stab_str = f"{stability:.4f}" if stability == stability else "N/A"
        logger.info(
            "TrajectoryAnchor.update | epoch=%d | layers=%d (%s) | "
            "λ1̄=%.4f | λ2̄=%.4f | gap̄=%.4f | stability=%s | n=%d",
            epoch, len(self.layer_indices),
            f"[{self.layer_indices[0]}..{self.layer_indices[-1]}]"
            if len(self.layer_indices) > 4 else str(self.layer_indices),
            self.lambda_1, self.lambda_2, self.gap, stab_str, n_used,
        )
        return stats

    # ------------------------------------------------------------------ alignment
    @torch.no_grad()
    def compute_alignment(self, states: torch.Tensor) -> torch.Tensor:
        """Compute per-sample alignment score, NAIT Eq 5.

        Accepts either:
          - ``[N, H]`` — legacy single-layer mode; dot with the single v.
          - ``[N, num_layers, H]`` — multi-layer mode; sum ⟨states_l, v_l⟩ over l.

        The result is min-max normalised into [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Anchor not yet fitted. Call update() first.")
        states = states.float().cpu()
        if states.ndim == 2:
            # Single-layer mode: expect exactly one layer in v_by_layer.
            if len(self.layer_indices) != 1:
                raise RuntimeError(
                    f"compute_alignment got 2-D states but anchor was fitted "
                    f"on {len(self.layer_indices)} layers; expected 3-D input."
                )
            v = self.v_by_layer[self.layer_indices[0]].float().cpu()
            alignment = states @ v
        elif states.ndim == 3:
            n, num_layers, _ = states.shape
            if num_layers != len(self.layer_indices):
                raise RuntimeError(
                    f"compute_alignment: states has {num_layers} layers but "
                    f"anchor fitted on {len(self.layer_indices)} layers.",
                )
            # Σ_l ⟨states[:, i, :], v_l⟩  (NAIT Eq 5)
            alignment = torch.zeros(n, dtype=torch.float32)
            for i, li in enumerate(self.layer_indices):
                v_l = self.v_by_layer[li].float().cpu()
                alignment += states[:, i, :] @ v_l
        else:
            raise ValueError(
                f"states must be 2-D or 3-D; got shape {tuple(states.shape)}",
            )
        a_min, a_max = alignment.min(), alignment.max()
        if (a_max - a_min) > 1e-8:
            alignment = (alignment - a_min) / (a_max - a_min)
        else:
            alignment = torch.full_like(alignment, 0.5)
        return alignment

    # ------------------------------------------------------------------ bookkeeping
    def get_history_summary(self) -> Dict[str, list]:
        return {
            "num_epochs_tracked": len(self.v_history),
            "lambda1_per_epoch": self.lambda1_history,
            "lambda2_per_epoch": self.lambda2_history,
            "gap_per_epoch": self.gap_history,
            "stability_per_epoch": self.stability_history,
        }

    def state_dict(self) -> Dict:
        return {
            "v_by_layer": {k: v.cpu() for k, v in self.v_by_layer.items()},
            "layer_indices": self.layer_indices,
            "lambda1_by_layer": self.lambda1_by_layer,
            "lambda2_by_layer": self.lambda2_by_layer,
            "gap_by_layer": self.gap_by_layer,
            "v_history": [v.cpu() for v in self.v_history],
            "lambda1_history": self.lambda1_history,
            "lambda2_history": self.lambda2_history,
            "gap_history": self.gap_history,
            "stability_history": self.stability_history,
            "layer_idx": self.layer_idx,
            "layer_indices_spec": self.layer_indices_spec,
            "max_samples_for_pca": self.max_samples_for_pca,
        }

    def load_state_dict(self, state: Dict) -> None:
        # Multi-layer state.
        self.v_by_layer = {
            int(k): v for k, v in (state.get("v_by_layer") or {}).items()
        }
        self.layer_indices = list(state.get("layer_indices") or [])
        self.lambda1_by_layer = state.get("lambda1_by_layer", {})
        self.lambda2_by_layer = state.get("lambda2_by_layer", {})
        self.gap_by_layer = state.get("gap_by_layer", {})
        # History.
        self.v_history = state.get("v_history", [])
        self.lambda1_history = state.get("lambda1_history", [])
        self.lambda2_history = state.get("lambda2_history", [])
        self.gap_history = state.get("gap_history", [])
        self.stability_history = state.get("stability_history", [])
        # Config.
        self.layer_idx = state.get("layer_idx", -1)
        self.layer_indices_spec = state.get("layer_indices_spec", None)
        self.max_samples_for_pca = state.get("max_samples_for_pca", 1024)
        # Legacy single-layer alias.
        if not self.is_multi_layer and self.layer_indices:
            self.v = self.v_by_layer.get(self.layer_indices[0])
        else:
            self.v = None
        # Legacy single-layer state.
        if "v" in state and state["v"] is not None and not self.v_by_layer:
            # Pre-multi-layer checkpoint — translate.
            v = state["v"]
            self.v = v
            decoder_li = max(0, self.layer_idx if self.layer_idx >= 0 else 0)
            self.v_by_layer = {decoder_li: v}
            self.layer_indices = [decoder_li]
