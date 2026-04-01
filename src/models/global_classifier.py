"""EfficientNet-B4 global pathway classifier using timm."""

from pathlib import Path
from typing import Any, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalClassifier(nn.Module):
    """Global pathway: full dermoscopic image classification.

    Uses a timm EfficientNet-B4 backbone with a custom two-layer head.
    The penultimate 512-dim feature vector is exposed for ensemble gating.

    Args:
        model_name: timm model identifier.
        pretrained: Load ImageNet-pretrained weights.
        num_classes: Number of output classes.
        drop_rate: Dropout rate before the final linear layer.
        hidden_dim: Hidden dimension between pooling and classifier.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 7,
        drop_rate: float = 0.4,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Build backbone (no built-in classifier head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove default head → returns pooled features
            drop_rate=0.0,  # we handle dropout ourselves
        )
        # Feature dim from the backbone pooling layer
        self.feat_dim: int = self.backbone.num_features  # 1792 for B4

        # Custom two-layer classification head
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GlobalClassifier":
        """Build from the ``global_classifier`` config section.

        Args:
            cfg: Dict with keys ``model_name``, ``pretrained``, etc.

        Returns:
            Configured ``GlobalClassifier``.
        """
        return cls(
            model_name=cfg.get("model_name", "efficientnet_b4"),
            pretrained=cfg.get("pretrained", True),
            num_classes=cfg.get("num_classes", 7),
            drop_rate=cfg.get("drop_rate", 0.4),
            hidden_dim=cfg.get("hidden_dim", 512),
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        cfg: dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> "GlobalClassifier":
        """Load trained weights from a checkpoint.

        Args:
            ckpt_path: Path to ``.pt`` file.
            cfg: Config section for architecture reconstruction.
            device: Target device.

        Returns:
            Model in eval mode.
        """
        model = cls.from_config(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    # ------------------------------------------------------------------
    # Freeze / unfreeze for phased training
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1: train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone_partial(self, n_blocks: int = 2) -> None:
        """Unfreeze the last ``n_blocks`` of the backbone (Phase 2).

        For EfficientNet-B4 the blocks live in ``self.backbone.blocks``.

        Args:
            n_blocks: Number of blocks to unfreeze from the end.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then unfreeze last n_blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the entire model (Phase 3)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Return discriminative LR parameter groups.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for the classification head.

        Returns:
            List of param group dicts for an optimizer.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features (before the classification head).

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Feature tensor ``(B, feat_dim)`` — 1792-dim for EfficientNet-B4.
        """
        return self.backbone(x)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw classification logits.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.get_features(x)
        return self.head(features)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Probability tensor ``(B, num_classes)``.
        """
        return F.softmax(self.get_logits(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (compatible with loss functions).

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        return self.get_logits(x)
