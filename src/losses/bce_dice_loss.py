"""Combined BCE + Dice loss for binary segmentation."""

import torch
import torch.nn as nn

from src.losses.dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):
    """Weighted combination of ``BCEWithLogitsLoss`` and :class:`DiceLoss`.

    Using logits (not probabilities) for the BCE term ensures numerical
    stability via the log-sum-exp trick inside PyTorch.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
        smooth: Smooth factor forwarded to :class:`DiceLoss`.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    @classmethod
    def from_config(cls, cfg: dict) -> "BCEDiceLoss":
        """Construct from the ``segmentation.loss`` config section.

        Args:
            cfg: Dict with keys ``bce_weight`` and ``dice_weight``.

        Returns:
            Configured ``BCEDiceLoss`` instance.
        """
        return cls(
            bce_weight=cfg.get("bce_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Raw predictions ``(B, 1, H, W)``.
            targets: Binary masks ``(B, 1, H, W)`` or ``(B, H, W)``.

        Returns:
            Weighted scalar loss.
        """
        # Ensure consistent shape for BCE
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
