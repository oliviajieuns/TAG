"""Trajectory Anchor — capability direction extracted from the model's own
training trajectory (paper §3.1).

For each layer l and sample x, the contextualization vector is
    Δh_l(x; θ_t) := h_l^last(x; θ_t) - h_l^first(x; θ_t),
where h^first and h^last denote the hidden states at the first and last
token positions. The trajectory anchor at epoch t is the top-1 eigenvector
of the empirical covariance Σ_l^(t) of {Δh_l(x; θ_t)} over a probe subset.
The anchor is sign-calibrated so that ⟨v_l^(t), E[Δh_l]⟩ > 0.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class TrajectoryAnchor:
    """Single-layer PCA-based capability anchor (Phase A).

    Phase B extension: pass a list of layer indices and aggregate
    alignments. This class is intentionally minimal for clarity.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        max_samples_for_pca: int = 2000,
        pca_batch_size: int = 4,
        device: str = "cuda",
    ):
        self.layer_idx = layer_idx
        self.max_samples_for_pca = max_samples_for_pca
        self.pca_batch_size = pca_batch_size
        self.device = device

        # Current state.
        self.v: Optional[torch.Tensor] = None  # (hidden_dim,)
        self.lambda_1: float = 0.0
        self.lambda_2: float = 0.0
        self.gap: float = 0.0  # λ_1 - λ_2

        # History (Theorem 1 verification).
        self.v_history: List[torch.Tensor] = []
        self.lambda1_history: List[float] = []
        self.lambda2_history: List[float] = []
        self.gap_history: List[float] = []
        self.stability_history: List[float] = []

    @property
    def is_fitted(self) -> bool:
        return self.v is not None

    @torch.no_grad()
    def update(
        self,
        model,
        dataset,
        seed: int = 42,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Re-extract the anchor at the start of epoch `t`.

        Samples a random probe subset (deterministic given seed+epoch),
        computes contextualization deltas Δh = h_last - h_first, runs
        PCA on the centered covariance, and updates v.
        """
        g = torch.Generator()
        g.manual_seed(seed + epoch * 100)
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
        delta_list: List[torch.Tensor] = []
        actual_layer = self.layer_idx  # set after first iter

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            # Call the model as a whole — `output_hidden_states=True` is
            # honoured by HF CausalLM, PEFT-wrapped models, and DDP wrappers
            # alike. (The earlier `model.model(...)` form is unsafe under
            # PEFT, where `.model` may resolve to LoraModel rather than the
            # underlying transformer.)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # tuple, length L+1

            actual_layer = (
                self.layer_idx
                if self.layer_idx >= 0
                else len(hidden_states) + self.layer_idx
            )
            h = hidden_states[actual_layer]  # (B, T, H)

            lengths = (attention_mask.sum(dim=1).clamp_min(1) - 1).to(h.device)
            bidx = torch.arange(h.size(0), device=h.device)

            first_h = h[:, 0, :]
            last_h = h[bidx, lengths]

            delta = (last_h - first_h).detach().float().cpu()  # (B, H)
            delta_list.append(delta)
            del h, hidden_states, outputs

        delta = torch.cat(delta_list, dim=0)  # (N, H)
        N, H = delta.shape

        mu = delta.mean(dim=0, keepdim=True)
        centred = delta - mu

        if N < H:
            gram = centred @ centred.T / N
            eigvals, eigvecs = torch.linalg.eigh(gram)
            self.lambda_1 = float(eigvals[-1].item())
            self.lambda_2 = float(eigvals[-2].item()) if N >= 2 else 0.0
            top_u = eigvecs[:, -1]
            v = centred.T @ top_u
            v = v / (v.norm() + 1e-8)
        else:
            cov = centred.T @ centred / N
            eigvals, eigvecs = torch.linalg.eigh(cov)
            self.lambda_1 = float(eigvals[-1].item())
            self.lambda_2 = float(eigvals[-2].item())
            v = eigvecs[:, -1]

        mu_vec = mu.squeeze(0)
        if torch.dot(v, mu_vec) < 0:
            v = -v

        if self.is_fitted:
            stability = float(torch.norm(v - self.v).item())
        else:
            stability = float("nan")

        self.v = v
        self.gap = self.lambda_1 - self.lambda_2
        self.v_history.append(v.clone())
        self.lambda1_history.append(self.lambda_1)
        self.lambda2_history.append(self.lambda_2)
        self.gap_history.append(self.gap)
        self.stability_history.append(stability)

        stats = {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "gap": self.gap,
            "stability": stability,
            "n_samples_used": int(N),
        }
        stab_str = f"{stability:.4f}" if stability == stability else "N/A"
        logger.info(
            "TrajectoryAnchor.update | epoch=%d | layer=%d | "
            "lambda1=%.4f | lambda2=%.4f | gap=%.4f | stability=%s | n=%d",
            epoch, actual_layer,
            self.lambda_1, self.lambda_2, self.gap, stab_str, N,
        )
        return stats

    @torch.no_grad()
    def compute_alignment(self, states: torch.Tensor) -> torch.Tensor:
        """Project states onto v and min-max normalise into [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Anchor not yet fitted. Call update() first.")
        states = states.float().cpu()
        v = self.v.float().cpu()
        alignment = states @ v
        a_min, a_max = alignment.min(), alignment.max()
        if (a_max - a_min) > 1e-8:
            alignment = (alignment - a_min) / (a_max - a_min)
        else:
            alignment = torch.full_like(alignment, 0.5)
        return alignment

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
            "v": self.v.cpu() if self.v is not None else None,
            "v_history": [v.cpu() for v in self.v_history],
            "lambda1_history": self.lambda1_history,
            "lambda2_history": self.lambda2_history,
            "gap_history": self.gap_history,
            "stability_history": self.stability_history,
            "layer_idx": self.layer_idx,
            "max_samples_for_pca": self.max_samples_for_pca,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.v = state.get("v")
        self.v_history = state.get("v_history", [])
        self.lambda1_history = state.get("lambda1_history", [])
        self.lambda2_history = state.get("lambda2_history", [])
        self.gap_history = state.get("gap_history", [])
        self.stability_history = state.get("stability_history", [])
        self.layer_idx = state.get("layer_idx", -1)
        self.max_samples_for_pca = state.get("max_samples_for_pca", 2000)
