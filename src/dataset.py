"""HAM10000 dataset with support for global, local (cropped), and segmentation modes."""

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Canonical label mapping
LABEL_MAP: dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for HAM10000 dermoscopic images.

    Supports three loading modes controlled at __getitem__ time:
    - **classification**: returns the full image resized to ``img_size``.
    - **segmentation**: returns image *and* binary mask (for U-Net training).
    - **local**: returns a cropped lesion region (requires pre-computed masks
      or bounding-box annotations).

    Args:
        csv_path: Path to a split CSV (must contain ``image_id`` and ``dx`` columns).
        images_dir: Directory with ``<image_id>.jpg`` files.
        transform: Albumentations transform pipeline (applied to image & mask).
        img_size: Target spatial size after resize (height == width).
        masks_dir: Optional directory with ``<image_id>_segmentation.png`` masks.
        crop_padding: Fractional padding around the mask bounding box for local crops.
        mode: One of ``"classification"``, ``"segmentation"``, or ``"local"``.
        filter_existing: If ``True``, drop rows whose images are not on disk.
    """

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        transform: Any = None,
        img_size: int = 380,
        masks_dir: Optional[Path] = None,
        crop_padding: float = 0.15,
        mode: str = "classification",
        filter_existing: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.img_size = img_size
        self.crop_padding = crop_padding
        self.mode = mode

        self.df = pd.read_csv(csv_path)
        # Ensure required columns exist
        for col in ("image_id", "dx"):
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # Optionally filter to only images that exist on disk
        if filter_existing:
            before = len(self.df)
            mask = self.df["image_id"].apply(
                lambda iid: (self.images_dir / f"{iid}.jpg").exists()
            )
            self.df = self.df[mask].reset_index(drop=True)
            dropped = before - len(self.df)
            if dropped > 0:
                import warnings
                warnings.warn(
                    f"Filtered {dropped}/{before} missing images from {csv_path.name}"
                )

        self.labels = self.df["dx"].map(LABEL_MAP).values
        self.image_ids = self.df["image_id"].values

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(LABEL_MAP)

    @property
    def lesion_ids(self) -> np.ndarray:
        """Return lesion_id array (for GroupKFold splitting)."""
        return self.df["lesion_id"].values if "lesion_id" in self.df.columns else None

    def get_labels(self) -> np.ndarray:
        """Return integer label array (useful for samplers / stratification)."""
        return self.labels

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_image(self, image_id: str) -> np.ndarray:
        """Load an image as RGB uint8 numpy array."""
        path = self.images_dir / f"{image_id}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, image_id: str) -> Optional[np.ndarray]:
        """Load a binary segmentation mask if available."""
        if self.masks_dir is None:
            return None
        # Try common naming conventions
        for suffix in (
            f"{image_id}_segmentation.png",
            f"{image_id}.png",
        ):
            mask_path = self.masks_dir / suffix
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                return (mask > 127).astype(np.uint8)
        return None

    def _crop_to_lesion(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Crop the image to the mask bounding box with padding.

        Args:
            image: (H, W, 3) RGB image.
            mask: (H, W) binary mask.

        Returns:
            Cropped image region.
        """
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            # Fallback: centre crop
            pad = int(min(h, w) * self.crop_padding)
            return image[pad : h - pad, pad : w - pad]

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        bh = y_max - y_min
        bw = x_max - x_min
        pad_y = int(bh * self.crop_padding)
        pad_x = int(bw * self.crop_padding)

        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)

        return image[y_min:y_max, x_min:x_max]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample dict.

        Keys always present: ``image`` (Tensor), ``label`` (int), ``image_id`` (str).
        If mode == ``"segmentation"``: also ``mask`` (Tensor).
        Metadata keys: ``dx``, ``lesion_id``, ``age``, ``sex``, ``localization``
        (when available in the CSV).
        """
        row = self.df.iloc[idx]
        image_id: str = row["image_id"]
        label: int = int(self.labels[idx])

        image = self._load_image(image_id)
        mask = self._load_mask(image_id)

        # --- local mode: crop around the lesion ---
        if self.mode == "local" and mask is not None:
            image = self._crop_to_lesion(image, mask)

        # --- resize ---
        image = cv2.resize(image, (self.img_size, self.img_size))
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        # --- apply transforms ---
        if self.transform is not None:
            if self.mode == "segmentation" and mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).unsqueeze(0).float()

        # --- build output dict ---
        sample: dict[str, Any] = {
            "image": image,
            "label": label,
            "image_id": image_id,
        }

        if self.mode == "segmentation" and mask is not None:
            sample["mask"] = mask

        # Attach metadata
        metadata: dict[str, Any] = {}
        for col in ("dx", "lesion_id", "dx_type", "age", "sex", "localization"):
            if col in row.index:
                metadata[col] = row[col]
        sample["metadata"] = metadata

        return sample
