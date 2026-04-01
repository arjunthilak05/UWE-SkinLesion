"""Mask post-processing: morphological cleanup, bounding box extraction, lesion cropping."""

from typing import Optional

import cv2
import numpy as np


def clean_mask(
    binary_mask: np.ndarray,
    close_kernel: int = 5,
    open_kernel: int = 3,
) -> np.ndarray:
    """Clean a binary segmentation mask with morphological operations.

    Pipeline:
    1. Morphological closing (fills small holes).
    2. Morphological opening (removes small noise blobs).
    3. Keep only the largest connected component.

    Args:
        binary_mask: 2-D ``uint8`` array with values in ``{0, 1}`` or ``{0, 255}``.
        close_kernel: Kernel size for the closing operation.
        open_kernel: Kernel size for the opening operation.

    Returns:
        Cleaned binary mask (``uint8``, values ``{0, 1}``).
    """
    # Normalise to {0, 255} for OpenCV morphology
    mask = (binary_mask > 0).astype(np.uint8) * 255

    # 1. Closing — fill small holes inside the lesion
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 2. Opening — remove small noise islands
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # 3. Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        # Only background — return zeros
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # Label 0 is background; find the largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    cleaned = (labels == largest_label).astype(np.uint8)

    return cleaned


def extract_bounding_box(
    mask: np.ndarray,
    margin_pct: float = 0.1,
) -> tuple[int, int, int, int]:
    """Get a padded bounding box around non-zero pixels in a binary mask.

    Args:
        mask: 2-D binary mask (``{0, 1}`` or ``{0, 255}``).
        margin_pct: Fractional padding to add on each side relative to the
            bounding-box dimensions.

    Returns:
        ``(x1, y1, x2, y2)`` clipped to image boundaries.  If the mask is
        entirely zero, returns the full image extent ``(0, 0, W, H)``.
    """
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        return (0, 0, w, h)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    bw = x_max - x_min
    bh = y_max - y_min

    margin_x = int(bw * margin_pct)
    margin_y = int(bh * margin_pct)

    x1 = max(0, x_min - margin_x)
    y1 = max(0, y_min - margin_y)
    x2 = min(w, x_max + margin_x + 1)
    y2 = min(h, y_max + margin_y + 1)

    return (x1, y1, x2, y2)


def crop_lesion(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 224,
    margin_pct: float = 0.1,
) -> np.ndarray:
    """Crop the lesion region from an image and resize to a square.

    Args:
        image: RGB image ``(H, W, 3)`` as ``uint8``.
        mask: Binary mask ``(H, W)``.
        target_size: Output spatial size (square).
        margin_pct: Bounding-box margin forwarded to :func:`extract_bounding_box`.

    Returns:
        Cropped and resized RGB image ``(target_size, target_size, 3)``.
    """
    x1, y1, x2, y2 = extract_bounding_box(mask, margin_pct=margin_pct)
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized
