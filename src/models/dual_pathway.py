"""Full dual-pathway classification system: global + segmentation + local + ensemble."""

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gating import ConfidenceEnsemble
from src.models.global_classifier import GlobalClassifier
from src.models.local_classifier import LocalClassifier
from src.models.segmentor import LesionSegmentor
from src.models.temperature import TemperatureScaler
from src.postprocessing import clean_mask, crop_lesion


class DualPathwaySystem(nn.Module):
    """End-to-end dual-pathway skin lesion classification.

    Pipeline per image:

    1. **Global pathway**: full image → EfficientNet-B4 → logits_global
    2. **Segmentation**: full image → U-Net → binary mask → post-process
    3. **Local pathway**: cropped lesion → ResNet-50 → logits_local
    4. **Temperature scaling**: calibrate both logit vectors
    5. **Confidence ensemble**: adaptive fusion → final prediction

    Args:
        global_model: Trained global classifier.
        segmentor: Trained lesion segmentor.
        local_model: Trained local classifier.
        ensemble: Confidence-based gating module.
        temp_global: Temperature scaler for global logits.
        temp_local: Temperature scaler for local logits.
        seg_input_size: Spatial size expected by the segmentor.
        local_input_size: Spatial size expected by the local classifier.
        crop_margin: Bounding-box margin for lesion cropping.
    """

    def __init__(
        self,
        global_model: GlobalClassifier,
        segmentor: LesionSegmentor,
        local_model: LocalClassifier,
        ensemble: ConfidenceEnsemble,
        temp_global: TemperatureScaler,
        temp_local: TemperatureScaler,
        seg_input_size: int = 256,
        local_input_size: int = 224,
        crop_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_model = global_model
        self.segmentor = segmentor
        self.local_model = local_model
        self.ensemble = ensemble
        self.temp_global = temp_global
        self.temp_local = temp_local
        self.seg_input_size = seg_input_size
        self.local_input_size = local_input_size
        self.crop_margin = crop_margin

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_and_crop(
        self,
        images: torch.Tensor,
        normalize_fn: Optional[Any] = None,
    ) -> torch.Tensor:
        """Run segmentation and crop lesion regions for the local pathway.

        Works in tensor space: resizes for the segmentor, predicts mask,
        then crops and resizes each image for the local classifier.

        Args:
            images: Input batch ``(B, 3, H, W)`` (already normalised for global).
            normalize_fn: Optional re-normalisation callable (not used if
                global and local share the same normalisation).

        Returns:
            Cropped and resized images ``(B, 3, local_input_size, local_input_size)``.
        """
        B, C, H, W = images.shape
        device = images.device

        # Resize for segmentor
        seg_input = F.interpolate(
            images, size=(self.seg_input_size, self.seg_input_size),
            mode="bilinear", align_corners=False,
        )
        masks = self.segmentor.predict_mask(seg_input, threshold=0.5)  # (B, 1, seg_h, seg_w)

        # Resize masks back to input resolution
        masks_full = F.interpolate(
            masks, size=(H, W), mode="nearest",
        )  # (B, 1, H, W)

        crops = []
        for i in range(B):
            # Convert to numpy for postprocessing
            mask_np = masks_full[i, 0].cpu().numpy().astype(np.uint8)
            mask_np = clean_mask(mask_np)

            # Get image as numpy (denormalise approximately for cropping)
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            # Map back to [0, 255] approximately
            img_uint8 = ((img_np * 0.15 + 0.6) * 255).clip(0, 255).astype(np.uint8)

            # Crop
            cropped = crop_lesion(
                img_uint8, mask_np,
                target_size=self.local_input_size,
                margin_pct=self.crop_margin,
            )

            # Back to tensor and re-normalise
            crop_t = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
            crops.append(crop_t)

        return torch.stack(crops, dim=0).to(device)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full dual-pathway pipeline.

        Args:
            images: Input batch ``(B, 3, H, W)``.

        Returns:
            Dict with all intermediate and final outputs.
        """
        self.eval()

        # 1. Global pathway
        logits_global = self.global_model(images)
        scaled_global = self.temp_global(logits_global)
        p_global = F.softmax(scaled_global, dim=1)

        # 2. Segment and crop
        crops = self._segment_and_crop(images)

        # 3. Local pathway
        logits_local = self.local_model(crops)
        scaled_local = self.temp_local(logits_local)
        p_local = F.softmax(scaled_local, dim=1)

        # 4. Ensemble
        ens_out = self.ensemble(p_global, p_local)

        return {
            "logits_global": logits_global,
            "logits_local": logits_local,
            "p_global": p_global,
            "p_local": p_local,
            "p_final": ens_out["p_final"],
            "w_global": ens_out["w_global"],
            "w_local": ens_out["w_local"],
            "c_global": ens_out["c_global"],
            "c_local": ens_out["c_local"],
            "crops": crops,
        }

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict class labels with confidence and pathway weights.

        Args:
            images: Input batch ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(class_labels, confidences, w_global, w_local)``.
        """
        out = self.forward(images)
        confidences, class_labels = out["p_final"].max(dim=1)
        return class_labels, confidences, out["w_global"], out["w_local"]

    @torch.no_grad()
    def predict_with_details(self, images: torch.Tensor) -> dict[str, Any]:
        """Predict with full diagnostic details.

        Args:
            images: Input batch ``(B, 3, H, W)``.

        Returns:
            Dict with all tensor outputs converted to numpy, plus
            ``predicted_class``, ``confidence``, and
            ``global_predicted_class``, ``local_predicted_class``.
        """
        out = self.forward(images)

        confidences, class_labels = out["p_final"].max(dim=1)
        _, global_preds = out["p_global"].max(dim=1)
        _, local_preds = out["p_local"].max(dim=1)

        return {
            "predicted_class": class_labels.cpu().numpy(),
            "confidence": confidences.cpu().numpy(),
            "global_predicted_class": global_preds.cpu().numpy(),
            "local_predicted_class": local_preds.cpu().numpy(),
            "p_final": out["p_final"].cpu().numpy(),
            "p_global": out["p_global"].cpu().numpy(),
            "p_local": out["p_local"].cpu().numpy(),
            "w_global": out["w_global"].cpu().numpy(),
            "w_local": out["w_local"].cpu().numpy(),
            "c_global": out["c_global"].cpu().numpy(),
            "c_local": out["c_local"].cpu().numpy(),
            "temp_global": self.temp_global.get_temperature(),
            "temp_local": self.temp_local.get_temperature(),
        }
