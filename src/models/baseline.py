"""Generic baseline classifier wrapping any timm model for fair SOTA comparison."""

from pathlib import Path
from typing import Any, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Architecture-specific block discovery for partial unfreezing
_LAYER_ATTR_MAP: dict[str, str] = {
    "resnet": "layer",          # layer1, layer2, layer3, layer4
    "vgg": "features",          # sequential — split by MaxPool indices
    "densenet": "features",     # denseblock1..4
    "efficientnet": "blocks",   # blocks[0..6]
    "vit": "blocks",            # blocks[0..11]
    "swin": "layers",           # layers[0..3]
}


def _get_block_groups(backbone: nn.Module, model_name: str) -> list[nn.Module]:
    """Identify repeatable block groups in a timm backbone.

    Returns a list of module groups ordered shallow → deep so that
    ``groups[-n:]`` gives the last *n* blocks.
    """
    name_lower = model_name.lower()

    # ResNet-family: layer1..layer4
    if "resnet" in name_lower or "resnext" in name_lower:
        return [
            m for n, m in backbone.named_children() if n.startswith("layer")
        ]

    # VGG: split features by MaxPool layers
    if "vgg" in name_lower:
        groups: list[nn.Sequential] = []
        current: list[nn.Module] = []
        for m in backbone.features:
            current.append(m)
            if isinstance(m, nn.MaxPool2d):
                groups.append(nn.Sequential(*current))
                current = []
        if current:
            groups.append(nn.Sequential(*current))
        return groups

    # DenseNet: denseblock1..4
    if "densenet" in name_lower or "dense" in name_lower:
        return [
            m for n, m in backbone.features.named_children()
            if "denseblock" in n
        ]

    # EfficientNet: blocks[0..N]
    if "efficientnet" in name_lower or "efficientnetv2" in name_lower:
        if hasattr(backbone, "blocks"):
            return list(backbone.blocks)

    # ViT: blocks[0..N]
    if "vit" in name_lower:
        if hasattr(backbone, "blocks"):
            return list(backbone.blocks)

    # Swin: layers[0..N]
    if "swin" in name_lower:
        if hasattr(backbone, "layers"):
            return list(backbone.layers)

    # ConvNeXt: stages[0..3]
    if "convnext" in name_lower:
        if hasattr(backbone, "stages"):
            return list(backbone.stages)

    # Fallback: treat all children as blocks
    return list(backbone.children())


class BaselineClassifier(nn.Module):
    """Generic baseline classifier wrapping any timm model.

    Provides the same interface as ``GlobalClassifier`` / ``LocalClassifier``
    so it works with the ``Trainer`` class unchanged: ``freeze_backbone``,
    ``unfreeze_backbone_partial``, ``unfreeze_all``, ``get_param_groups``.

    Args:
        model_name: timm model identifier.
        pretrained: Load ImageNet pre-trained weights.
        num_classes: Output classes.
        drop_rate: Dropout rate in the head.
        hidden_dim: Hidden dimension in the head.
    """

    # Default input sizes for common architectures
    INPUT_SIZES: dict[str, int] = {
        "resnet50": 224,
        "vgg16_bn": 224,
        "densenet121": 224,
        "densenet201": 224,
        "efficientnet_b4": 380,
        "efficientnetv2_rw_s": 288,
        "tf_efficientnetv2_s": 384,
        "convnext_tiny": 224,
        "vit_base_patch16_224": 224,
        "vit_base_patch14_dinov2.lvd142m": 518,
        "vit_small_patch14_dinov2.lvd142m": 518,
        "vit_large_patch14_dinov2.lvd142m": 518,
        "swin_tiny_patch4_window7_224": 224,
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 7,
        drop_rate: float = 0.3,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
        )
        self.feat_dim: int = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Cache block groups for partial unfreezing
        self._block_groups = _get_block_groups(self.backbone, model_name)

    @classmethod
    def get_input_size(cls, model_name: str) -> int:
        """Return the native input size for a model."""
        return cls.INPUT_SIZES.get(model_name, 224)

    # ------------------------------------------------------------------
    # Freeze / unfreeze (same interface as GlobalClassifier)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone_partial(self, n_blocks: int = 2) -> None:
        """Unfreeze the last ``n_blocks`` groups of the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for group in self._block_groups[-n_blocks:]:
            for param in group.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the entire model."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Discriminative LR parameter groups."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        return self.head(self.get_features(x))

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=1)
