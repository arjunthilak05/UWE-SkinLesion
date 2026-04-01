"""Albumentations pipelines for training, validation, and test-time augmentation."""

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Default ImageNet stats — overridden by config at call sites
_DEFAULT_MEAN = (0.7635, 0.5461, 0.5705)
_DEFAULT_STD = (0.1404, 0.1521, 0.1697)


def get_train_transforms(
    img_size: int = 380,
    mean: tuple[float, ...] = _DEFAULT_MEAN,
    std: tuple[float, ...] = _DEFAULT_STD,
    cfg: dict[str, Any] | None = None,
) -> A.Compose:
    """Build the full training augmentation pipeline.

    Args:
        img_size: Target spatial size (height == width).
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        cfg: Optional augmentation config dict (from ``default.yaml``).

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    c = cfg or {}

    rrc = c.get("random_resized_crop", {})
    scale = tuple(rrc.get("scale", [0.8, 1.0]))
    ratio = tuple(rrc.get("ratio", [0.9, 1.1]))

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=scale,
                ratio=ratio,
                p=1.0,
            ),
            A.HorizontalFlip(p=c.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=c.get("vertical_flip_p", 0.5)),
            A.RandomRotate90(p=c.get("rotate90_p", 0.5)),
            A.Transpose(p=c.get("transpose_p", 0.5)),
            # Geometric distortions (pick one)
            A.OneOf(
                [
                    A.OpticalDistortion(p=c.get("optical_distortion_p", 0.3)),
                    A.GridDistortion(p=c.get("grid_distortion_p", 0.3)),
                    A.ElasticTransform(p=c.get("elastic_transform_p", 0.3)),
                ],
                p=c.get("distortion_oneof_p", 0.3),
            ),
            # Color augmentations (pick one)
            A.OneOf(
                [
                    A.CLAHE(
                        clip_limit=c.get("clahe_clip_limit", 4.0),
                        p=c.get("clahe_p", 0.5),
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=c.get("brightness_limit", 0.2),
                        contrast_limit=c.get("contrast_limit", 0.2),
                        p=c.get("brightness_contrast_p", 0.5),
                    ),
                ],
                p=c.get("color_oneof_p", 0.5),
            ),
            A.HueSaturationValue(
                hue_shift_limit=c.get("hue_shift_limit", 20),
                sat_shift_limit=c.get("sat_shift_limit", 30),
                val_shift_limit=c.get("val_shift_limit", 20),
                p=c.get("hue_sat_p", 0.5),
            ),
            A.CoarseDropout(
                num_holes_range=(c.get("coarse_dropout_min_holes", 4), c.get("coarse_dropout_max_holes", 8)),
                hole_height_range=(c.get("coarse_dropout_min_height", 16), c.get("coarse_dropout_max_height", 32)),
                hole_width_range=(c.get("coarse_dropout_min_width", 16), c.get("coarse_dropout_max_width", 32)),
                p=c.get("coarse_dropout_p", 0.3),
            ),
            A.GaussNoise(
                std_range=(c.get("gauss_noise_std_min", 0.02), c.get("gauss_noise_std_max", 0.1)),
                p=c.get("gauss_noise_p", 0.2),
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    img_size: int = 380,
    mean: tuple[float, ...] = _DEFAULT_MEAN,
    std: tuple[float, ...] = _DEFAULT_STD,
) -> A.Compose:
    """Build the validation / test transform pipeline (deterministic).

    Args:
        img_size: Target spatial size.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(
    img_size: int = 380,
    mean: tuple[float, ...] = _DEFAULT_MEAN,
    std: tuple[float, ...] = _DEFAULT_STD,
) -> list[A.Compose]:
    """Return a list of 10 deterministic TTA transform variants.

    The first variant is the identity (plain resize). The remaining nine
    apply various flips, rotations, and mild color shifts to increase
    prediction diversity at inference time.

    Args:
        img_size: Target spatial size.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        List of 10 ``albumentations.Compose`` pipelines.
    """
    base = [A.Resize(img_size, img_size)]
    tail = [A.Normalize(mean=mean, std=std), ToTensorV2()]

    variants: list[list[A.BasicTransform]] = [
        # 0 — identity
        [],
        # 1 — horizontal flip
        [A.HorizontalFlip(p=1.0)],
        # 2 — vertical flip
        [A.VerticalFlip(p=1.0)],
        # 3 — h + v flip
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
        # 4 — rotate 90
        [A.Rotate(limit=(90, 90), p=1.0, border_mode=0)],
        # 5 — rotate 180
        [A.Rotate(limit=(180, 180), p=1.0, border_mode=0)],
        # 6 — rotate 270
        [A.Rotate(limit=(270, 270), p=1.0, border_mode=0)],
        # 7 — transpose
        [A.Transpose(p=1.0)],
        # 8 — slight brightness boost
        [A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0)],
        # 9 — slight contrast boost
        [A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.1, 0.1), p=1.0)],
    ]

    return [A.Compose(base + v + tail) for v in variants]
