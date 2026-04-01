"""Dice Loss for binary segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Measures the overlap between predicted and ground-truth masks.
    A smooth factor prevents division by zero when both masks are empty.

    Args:
        smooth: Laplace smoothing constant added to numerator and denominator.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"`` over the batch dimension.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw predictions before sigmoid, shape ``(B, 1, H, W)``.
            targets: Binary ground-truth masks, shape ``(B, 1, H, W)`` or
                ``(B, H, W)``.

        Returns:
            Scalar loss (unless ``reduction="none"``).
        """
        probs = torch.sigmoid(logits)

        # Ensure consistent shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Flatten spatial dims: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
