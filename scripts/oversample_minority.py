"""Offline oversampling of minority classes via heavy augmentation.

Generates augmented copies of underrepresented classes and saves them
alongside originals so the dataset class needs no changes.

Usage::

    python scripts/oversample_minority.py \
        --train_csv D:/skin_data/splits/fold0_train.csv \
        --images_dir D:/skin_data/raw/images \
        --target_count 1000 \
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import seed_everything


def get_heavy_augmentation() -> A.Compose:
    """Heavy augmentation pipeline for oversampling."""
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.OneOf([
            A.ElasticTransform(p=0.4),
            A.GridDistortion(p=0.4),
            A.OpticalDistortion(p=0.4),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ], p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5,
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Oversample minority classes.")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="Output CSV path (default: {train_csv stem}_oversampled.csv)")
    parser.add_argument("--target_count", type=int, default=1000,
                        help="Minimum samples per class after oversampling.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    df = pd.read_csv(args.train_csv)
    print(f"[INFO] Loaded {len(df)} samples from {args.train_csv}")

    class_counts = df["dx"].value_counts()
    print("\n[INFO] Original class distribution:")
    for cls, cnt in class_counts.items():
        print(f"  {cls:8s}: {cnt}")

    aug = get_heavy_augmentation()
    new_rows = []
    total_generated = 0

    for cls_name in class_counts.index:
        current = class_counts[cls_name]
        needed = max(0, args.target_count - current)
        if needed == 0:
            print(f"  {cls_name}: {current} >= {args.target_count}, skip")
            continue

        cls_df = df[df["dx"] == cls_name]
        print(f"\n[INFO] Oversampling {cls_name}: {current} → {args.target_count} (+{needed})")

        generated = 0
        while generated < needed:
            for _, row in cls_df.iterrows():
                if generated >= needed:
                    break

                img_path = args.images_dir / f"{row['image_id']}.jpg"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                augmented = aug(image=img)["image"]
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

                aug_id = f"{row['image_id']}_aug_{generated:04d}"
                out_path = args.images_dir / f"{aug_id}.jpg"
                cv2.imwrite(str(out_path), augmented)

                new_row = row.copy()
                new_row["image_id"] = aug_id
                new_rows.append(new_row)
                generated += 1

        total_generated += generated
        print(f"  Generated {generated} augmented images for {cls_name}")

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        combined = pd.concat([df, aug_df], ignore_index=True)
    else:
        combined = df

    output_csv = args.output_csv or args.train_csv.parent / f"{args.train_csv.stem}_oversampled.csv"
    combined.to_csv(output_csv, index=False)

    print(f"\n[INFO] Total generated: {total_generated}")
    print(f"[INFO] Combined dataset: {len(combined)} samples")
    print(f"[INFO] Saved to: {output_csv}")

    print("\n[INFO] New class distribution:")
    for cls, cnt in combined["dx"].value_counts().items():
        print(f"  {cls:8s}: {cnt}")


if __name__ == "__main__":
    main()
