"""Classification and segmentation evaluation metrics.

All public functions accept NumPy arrays or PyTorch tensors and return
plain Python floats (or lists of floats for per-class metrics).
"""

from typing import Union

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


# =========================================================================
# Helpers
# =========================================================================


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a tensor or array to a NumPy array on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================================
# Classification Metrics
# =========================================================================


def balanced_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Mean per-class recall (balanced accuracy).

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_pred: Predicted integer labels ``(N,)``.

    Returns:
        Balanced accuracy in ``[0, 1]``.
    """
    return float(balanced_accuracy_score(_to_numpy(y_true), _to_numpy(y_pred)))


def macro_f1(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Macro-averaged F1 score across all classes.

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_pred: Predicted integer labels ``(N,)``.

    Returns:
        Macro F1 in ``[0, 1]``.
    """
    return float(
        f1_score(
            _to_numpy(y_true),
            _to_numpy(y_pred),
            average="macro",
            zero_division=0,
        )
    )


def per_class_auc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_probs: Union[np.ndarray, torch.Tensor],
    num_classes: int = 7,
) -> list[float]:
    """One-vs-rest AUC for each class.

    If a class has no positive samples in ``y_true`` the AUC for that
    class is returned as ``0.0`` rather than raising an error.

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_probs: Predicted probabilities ``(N, C)``.
        num_classes: Number of classes.

    Returns:
        List of per-class AUC values (length ``num_classes``).
    """
    y_true_np = _to_numpy(y_true)
    y_probs_np = _to_numpy(y_probs)

    aucs: list[float] = []
    for c in range(num_classes):
        binary_true = (y_true_np == c).astype(int)
        # Need at least one positive and one negative sample
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            aucs.append(0.0)
        else:
            try:
                aucs.append(float(roc_auc_score(binary_true, y_probs_np[:, c])))
            except ValueError:
                aucs.append(0.0)
    return aucs


def macro_auc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_probs: Union[np.ndarray, torch.Tensor],
    num_classes: int = 7,
) -> float:
    """Mean of per-class one-vs-rest AUC values.

    Classes with AUC = 0.0 (no positive samples) are excluded from the
    average to avoid pessimistic bias on small batches.

    Args:
        y_true: Ground-truth integer labels ``(N,)``.
        y_probs: Predicted probabilities ``(N, C)``.
        num_classes: Number of classes.

    Returns:
        Macro AUC in ``[0, 1]``.
    """
    class_aucs = per_class_auc(y_true, y_probs, num_classes)
    valid = [a for a in class_aucs if a > 0.0]
    return float(np.mean(valid)) if valid else 0.0


# =========================================================================
# Segmentation Metrics
# =========================================================================


def dice_score(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Sørensen–Dice coefficient for binary segmentation.

    Args:
        pred_mask: Predicted probabilities or binary mask.  If values are
            outside ``{0, 1}`` they are thresholded at ``threshold``.
        gt_mask: Ground-truth binary mask (same spatial shape).
        threshold: Binarisation threshold for ``pred_mask``.
        smooth: Smoothing term to prevent division by zero.

    Returns:
        Dice score in ``[0, 1]``.
    """
    pred = _to_numpy(pred_mask).astype(np.float32).ravel()
    gt = _to_numpy(gt_mask).astype(np.float32).ravel()

    pred = (pred >= threshold).astype(np.float32)
    gt = (gt >= threshold).astype(np.float32)

    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def iou_score(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Intersection over Union (Jaccard index) for binary segmentation.

    Args:
        pred_mask: Predicted probabilities or binary mask.
        gt_mask: Ground-truth binary mask.
        threshold: Binarisation threshold.
        smooth: Smoothing term.

    Returns:
        IoU score in ``[0, 1]``.
    """
    pred = _to_numpy(pred_mask).astype(np.float32).ravel()
    gt = _to_numpy(gt_mask).astype(np.float32).ravel()

    pred = (pred >= threshold).astype(np.float32)
    gt = (gt >= threshold).astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return float((intersection + smooth) / (union + smooth))
