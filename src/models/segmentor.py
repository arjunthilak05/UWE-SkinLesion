"""U-Net lesion segmentation model wrapping segmentation_models_pytorch."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LesionSegmentor(nn.Module):
    """Binary lesion segmentation network.

    Wraps a ``segmentation_models_pytorch.Unet`` with a configurable encoder
    (default: ResNet-34 pre-trained on ImageNet).  The model outputs raw
    logits; call :meth:`predict_mask` for thresholded binary predictions.

    Args:
        encoder_name: Backbone name understood by ``smp`` (e.g. ``"resnet34"``).
        encoder_weights: Pre-trained weight identifier (``"imagenet"`` or ``None``).
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output channels (1 for binary segmentation).
        decoder_channels: Decoder channel sequence forwarded to ``smp.Unet``.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: list[int] | None = None,
    ) -> None:
        super().__init__()

        kwargs: dict[str, Any] = dict(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        if decoder_channels is not None:
            kwargs["decoder_channels"] = decoder_channels

        self.unet: nn.Module = smp.Unet(**kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LesionSegmentor":
        """Build from the ``segmentation`` config section.

        Args:
            cfg: Dict with keys ``encoder``, ``encoder_weights``, etc.

        Returns:
            Configured ``LesionSegmentor``.
        """
        return cls(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=cfg.get("encoder_weights", "imagenet"),
            in_channels=cfg.get("in_channels", 3),
            classes=cfg.get("classes", 1),
            decoder_channels=cfg.get("decoder_channels"),
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        cfg: dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> "LesionSegmentor":
        """Load a trained model from a checkpoint file.

        Args:
            ckpt_path: Path to the ``.pt`` checkpoint.
            cfg: Segmentation config section (needed to rebuild architecture).
            device: Device to map tensors to.

        Returns:
            Model with loaded weights in eval mode.
        """
        model = cls.from_config(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (for warm-up epochs)."""
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.unet.encoder.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logit tensor ``(B, 1, H, W)``.
        """
        return self.unet(x)

    @torch.no_grad()
    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predict a binary mask from an input image tensor.

        Applies sigmoid to the logits and thresholds at ``threshold``.

        Args:
            x: Input tensor ``(B, 3, H, W)`` or ``(3, H, W)``.
            threshold: Probability threshold for binarisation.

        Returns:
            Binary mask tensor ``(B, 1, H, W)`` with values in ``{0, 1}``.
        """
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        mask = (probs >= threshold).float()

        return mask.squeeze(0) if single else mask
