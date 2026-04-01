"""Generate predicted segmentation masks and cropped lesion images for all HAM10000 images.

Usage::

    python scripts/generate_masks.py \
        --checkpoint checkpoints/segmentation/best.pt \
        --config configs/default.yaml
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.metrics import dice_score, iou_score
from src.models.segmentor import LesionSegmentor
from src.postprocessing import clean_mask, crop_lesion, extract_bounding_box
from src.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate masks & crops for all images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to segmentation checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_clean", action="store_true", help="Skip morphological post-processing.")
    return parser.parse_args()


def build_inference_transform(size: int, mean: tuple, std: tuple) -> A.Compose:
    """Deterministic resize + normalise pipeline for inference."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    seg_cfg = cfg["segmentation"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    seed_everything(cfg.get("training", {}).get("seed", 42))
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    # --- Paths ---
    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    gt_masks_dir_str = data_cfg.get("masks_dir")
    gt_masks_dir = Path(gt_masks_dir_str) if gt_masks_dir_str else None

    pred_masks_dir = data_dir / "predicted_masks"
    crops_dir = data_dir / "cropped_lesions"
    pred_masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    model = LesionSegmentor.from_checkpoint(args.checkpoint, seg_cfg, device=device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # --- Transform ---
    norm_cfg = aug_cfg.get("normalize", {})
    mean = tuple(norm_cfg.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm_cfg.get("std", [0.1404, 0.1521, 0.1697]))
    input_size = seg_cfg.get("input_size", 256)
    transform = build_inference_transform(input_size, mean, std)

    crop_size = cfg.get("local_classifier", {}).get("input_size", 224)
    margin_pct = cfg.get("local_classifier", {}).get("crop_padding", 0.1)

    # --- Collect all images ---
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"[INFO] Found {len(image_paths)} images in {images_dir}")

    # --- Inference ---
    bbox_rows: list[dict] = []
    dice_scores: list[float] = []
    mask_areas: list[float] = []

    t0 = time.time()
    for img_path in tqdm(image_paths, desc="Generating masks"):
        image_id = img_path.stem

        # Load original image
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Prepare input tensor
        transformed = transform(image=img_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(device)

        # Predict
        pred_mask_small = model.predict_mask(img_tensor, threshold=args.threshold)
        pred_np = pred_mask_small.squeeze().cpu().numpy().astype(np.uint8)

        # Resize mask back to original resolution
        pred_full = cv2.resize(pred_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Post-processing
        if not args.no_clean:
            pred_full = clean_mask(pred_full)

        # Save predicted mask
        cv2.imwrite(str(pred_masks_dir / f"{image_id}.png"), pred_full * 255)

        # Bounding box
        x1, y1, x2, y2 = extract_bounding_box(pred_full, margin_pct=margin_pct)
        bbox_rows.append({
            "image_id": image_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "mask_area_ratio": float(pred_full.sum()) / (h_orig * w_orig),
        })
        mask_areas.append(bbox_rows[-1]["mask_area_ratio"])

        # Crop lesion
        cropped = crop_lesion(img_rgb, pred_full, target_size=crop_size, margin_pct=margin_pct)
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(crops_dir / f"{image_id}.jpg"), cropped_bgr)

        # Dice against GT if available
        if gt_masks_dir is not None:
            for suffix in (f"{image_id}_segmentation.png", f"{image_id}.png"):
                gt_path = gt_masks_dir / suffix
                if gt_path.exists():
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    gt = (gt > 127).astype(np.uint8)
                    dice_scores.append(dice_score(pred_full, gt))
                    break

    elapsed = time.time() - t0

    # --- Save bounding box CSV ---
    bbox_csv_path = data_dir / "bounding_boxes.csv"
    with bbox_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "x1", "y1", "x2", "y2", "mask_area_ratio"])
        writer.writeheader()
        writer.writerows(bbox_rows)

    # --- Statistics ---
    print("\n" + "=" * 50)
    print("Mask Generation Statistics")
    print("=" * 50)
    print(f"Images processed:      {len(bbox_rows)}")
    print(f"Time elapsed:          {elapsed:.1f}s ({len(bbox_rows) / elapsed:.1f} img/s)")
    print(f"Predicted masks dir:   {pred_masks_dir}")
    print(f"Cropped lesions dir:   {crops_dir}")
    print(f"Bounding boxes CSV:    {bbox_csv_path}")
    print(f"Mean mask area ratio:  {np.mean(mask_areas):.4f}")
    print(f"Median mask area:      {np.median(mask_areas):.4f}")
    print(f"Min / Max area:        {np.min(mask_areas):.4f} / {np.max(mask_areas):.4f}")

    if dice_scores:
        print(f"\nGT Dice (mean):        {np.mean(dice_scores):.4f}")
        print(f"GT Dice (median):      {np.median(dice_scores):.4f}")
        print(f"GT Dice (min/max):     {np.min(dice_scores):.4f} / {np.max(dice_scores):.4f}")
    else:
        print("\nNo GT masks found — Dice evaluation skipped.")

    print("=" * 50)
    print("[DONE]")


if __name__ == "__main__":
    main()
