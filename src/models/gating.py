"""Confidence-based and learned gating for dual-pathway ensemble."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceEnsemble(nn.Module):
    """Confidence-weighted adaptive fusion of global and local pathways.

    The weight for each pathway is derived from the max predicted probability
    (i.e. confidence) raised to a sharpening power ``tau``:

    .. math::

        w_g = \\frac{c_g^\\tau}{c_g^\\tau + c_l^\\tau}, \\quad
        w_l = 1 - w_g

    where :math:`c_g = \\max(p_g)` and :math:`c_l = \\max(p_l)`.

    Higher ``tau`` amplifies the difference between the two confidences,
    giving more weight to the more confident pathway.

    Args:
        tau: Sharpening exponent (default 2.0).
    """

    def __init__(self, tau: float = 2.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        p_global: torch.Tensor,
        p_local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse two probability distributions with confidence weighting.

        Args:
            p_global: Global pathway probabilities ``(B, C)``.
            p_local: Local pathway probabilities ``(B, C)``.

        Returns:
            Dict with keys:
            - ``p_final``: Fused probabilities ``(B, C)``.
            - ``w_global``: Per-sample global weight ``(B,)``.
            - ``w_local``: Per-sample local weight ``(B,)``.
            - ``c_global``: Global confidence ``(B,)``.
            - ``c_local``: Local confidence ``(B,)``.
        """
        c_global = p_global.max(dim=-1).values  # (B,)
        c_local = p_local.max(dim=-1).values    # (B,)

        c_g_tau = c_global.pow(self.tau)
        c_l_tau = c_local.pow(self.tau)

        denom = c_g_tau + c_l_tau + 1e-8  # avoid division by zero
        w_global = c_g_tau / denom  # (B,)
        w_local = c_l_tau / denom   # (B,)

        # Weighted combination: (B, 1) * (B, C)
        p_final = w_global.unsqueeze(1) * p_global + w_local.unsqueeze(1) * p_local

        return {
            "p_final": p_final,
            "w_global": w_global,
            "w_local": w_local,
            "c_global": c_global,
            "c_local": c_local,
        }


class LearnedGating(nn.Module):
    """MLP-based learned gating that takes concatenated features from both pathways.

    Produces per-sample softmax weights ``[w_global, w_local]`` from the
    concatenation of global and local feature vectors.

    Args:
        feat_dim_global: Feature dimension of the global pathway.
        feat_dim_local: Feature dimension of the local pathway.
        hidden_dim: Hidden layer size.
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        feat_dim_global: int = 1792,
        feat_dim_local: int = 2048,
        hidden_dim: int = 256,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim_global + feat_dim_local, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        feat_global: torch.Tensor,
        feat_local: torch.Tensor,
        p_global: torch.Tensor,
        p_local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute learned fusion weights and combine probabilities.

        Args:
            feat_global: Global features ``(B, D_g)``.
            feat_local: Local features ``(B, D_l)``.
            p_global: Global pathway probabilities ``(B, C)``.
            p_local: Local pathway probabilities ``(B, C)``.

        Returns:
            Dict with ``p_final``, ``w_global``, ``w_local``.
        """
        combined = torch.cat([feat_global, feat_local], dim=1)
        weights = F.softmax(self.mlp(combined), dim=1)  # (B, 2)

        w_global = weights[:, 0]  # (B,)
        w_local = weights[:, 1]   # (B,)

        p_final = w_global.unsqueeze(1) * p_global + w_local.unsqueeze(1) * p_local

        return {
            "p_final": p_final,
            "w_global": w_global,
            "w_local": w_local,
        }

    def fit(
        self,
        feat_global: torch.Tensor,
        feat_local: torch.Tensor,
        p_global: torch.Tensor,
        p_local: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> list[float]:
        """Train the gating MLP on validation set predictions.

        Args:
            feat_global: Global features ``(N, D_g)``.
            feat_local: Local features ``(N, D_l)``.
            p_global: Global probabilities ``(N, C)``.
            p_local: Local probabilities ``(N, C)``.
            labels: Ground-truth labels ``(N,)``.
            lr: Learning rate.
            epochs: Training epochs.

        Returns:
            List of per-epoch NLL losses.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses: list[float] = []

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.forward(feat_global, feat_local, p_global, p_local)
            loss = F.cross_entropy(out["p_final"].log().clamp(min=-100), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.eval()
        return losses
