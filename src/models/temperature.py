"""Post-hoc temperature scaling for classifier calibration."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities ``(N, C)``.
        labels: Ground-truth integer labels ``(N,)``.
        n_bins: Number of confidence bins.

    Returns:
        ECE value in ``[0, 1]``.
    """
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels).float()

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return float(ece.item())


class TemperatureScaler(nn.Module):
    """Learnable temperature scaling for post-hoc calibration.

    Divides logits by a single scalar temperature ``T`` before softmax.
    ``T`` is optimised on a held-out validation set using L-BFGS to
    minimise NLL.

    Args:
        init_temperature: Initial value of ``T``.
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        # Store in log-space to guarantee T > 0 (prevents L-BFGS driving T negative)
        self.log_temperature = nn.Parameter(
            torch.tensor([float(np.log(init_temperature))], dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature via exp of log-space parameter."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature.

        Args:
            logits: Raw logits ``(B, C)``.

        Returns:
            Scaled logits ``(B, C)``.
        """
        return logits / self.temperature

    def get_temperature(self) -> float:
        """Return the current temperature value."""
        return float(self.temperature.item())

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> dict[str, float]:
        """Optimise temperature on validation logits using L-BFGS.

        Args:
            val_logits: Validation logits ``(N, C)``.
            val_labels: Validation labels ``(N,)`` (long).
            lr: Learning rate for L-BFGS.
            max_iter: Maximum optimisation iterations.

        Returns:
            Dict with ``temperature``, ``ece_before``, ``ece_after``,
            ``nll_before``, ``nll_after``.
        """
        # Compute pre-calibration metrics
        with torch.no_grad():
            probs_before = F.softmax(val_logits, dim=1)
            nll_before = F.cross_entropy(val_logits, val_labels).item()
            ece_before = expected_calibration_error(probs_before, val_labels)

        # Optimise temperature
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = F.cross_entropy(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Compute post-calibration metrics
        with torch.no_grad():
            scaled_logits = self.forward(val_logits)
            probs_after = F.softmax(scaled_logits, dim=1)
            nll_after = F.cross_entropy(scaled_logits, val_labels).item()
            ece_after = expected_calibration_error(probs_after, val_labels)

        return {
            "temperature": self.get_temperature(),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "nll_before": nll_before,
            "nll_after": nll_after,
        }
