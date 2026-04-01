"""Focal Loss for multi-class classification with label smoothing support."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional per-class alpha and label smoothing.

    Reduces the contribution of easy examples so that the model focuses on
    hard, misclassified samples — critical for the HAM10000 class imbalance.

    Supports both hard labels ``(B,)`` and soft / mixed targets ``(B, C)``,
    making it compatible with Mixup and CutMix.

    Args:
        gamma: Focusing parameter.  ``gamma=0`` recovers standard CE.
        alpha: Per-class weight tensor of shape ``(C,)``, or ``None``.
            Pass ``"auto"`` to :func:`from_config` to load from
            ``configs/class_weights.json``.
        label_smoothing: Label-smoothing factor in ``[0, 1)``.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        gamma: float = 1.0,
        label_smoothing: float = 0.1,
        weights_path: str | Path = "configs/class_weights.json",
        num_classes: int = 7,
        reduction: str = "mean",
    ) -> "FocalLoss":
        """Construct a FocalLoss loading alpha from a JSON weights file.

        Args:
            gamma: Focusing parameter.
            label_smoothing: Smoothing factor.
            weights_path: Path to ``class_weights.json``.
            num_classes: Number of classes.
            reduction: Reduction mode.

        Returns:
            Configured ``FocalLoss`` instance.
        """
        weights_path = Path(weights_path)
        if weights_path.exists():
            with weights_path.open("r") as fh:
                raw = json.load(fh)
            alpha = torch.tensor(list(raw.values())[:num_classes], dtype=torch.float32)
        else:
            alpha = None

        return cls(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _smooth_targets(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert hard labels to smoothed one-hot targets.

        Args:
            targets: Integer labels ``(B,)``.
            num_classes: Number of classes.

        Returns:
            Soft targets ``(B, C)``.
        """
        one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        smooth = self.label_smoothing / num_classes
        return one_hot * (1.0 - self.label_smoothing) + smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw predictions ``(B, C)``.
            targets: Either hard labels ``(B,)`` with dtype ``long``, or
                soft targets ``(B, C)`` (e.g. from Mixup).

        Returns:
            Scalar loss (unless ``reduction="none"``).
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)   # (B, C)
        probs = log_probs.exp()                     # (B, C)

        # --- Build soft target matrix ---
        if targets.dim() == 1:
            soft_targets = self._smooth_targets(targets, num_classes)
        else:
            # Already soft (Mixup / CutMix) — skip smoothing to avoid double application
            soft_targets = targets

        # --- Focal modulation ---
        # p_t for each class weighted by its target probability
        focal_weight = (1.0 - probs) ** self.gamma           # (B, C)

        # --- Per-class alpha weighting ---
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)              # (C,)
            focal_weight = focal_weight * alpha.unsqueeze(0)  # (B, C)

        # --- Loss ---
        loss = -focal_weight * soft_targets * log_probs       # (B, C)
        loss = loss.sum(dim=1)                                # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
