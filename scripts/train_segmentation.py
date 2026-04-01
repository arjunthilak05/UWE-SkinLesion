"""Train the U-Net lesion segmentation model (Stage 2).

Usage::

    python scripts/train_segmentation.py --config configs/default.yaml
    python scripts/train_segmentation.py --config configs/default.yaml --smoke_test 100
"""

import argparse
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HAM10000Dataset
from src.losses import BCEDiceLoss
from src.metrics import dice_score, iou_score
from src.models.segmentor import LesionSegmentor
from src.utils import (
    AverageMeter,
    EarlyStopping,
    get_device,
    load_config,
    save_checkpoint,
    seed_everything,
    setup_wandb,
)

ISIC_MASKS_INSTRUCTIONS = """
================================================================
  Ground-truth segmentation masks not found.

  To train the U-Net you need ISIC 2018 Task 1 GT masks.

  Download options:
  1. ISIC Archive:
     https://challenge.isic-archive.com/data/#2018
     -> Download "Task 1: Training GT"

  2. Kaggle (alternative mirror):
     Search for "ISIC 2018 Task 1" on kaggle.com/datasets

  Using synthetic masks for this run (centre ellipse).
================================================================
"""


# =========================================================================
# Transforms
# =========================================================================


def get_seg_train_transforms(size: int, mean: tuple, std: tuple) -> A.Compose:
    """Augmentation pipeline for segmentation training (image + mask)."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(p=0.3),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=4.0, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                ],
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_seg_val_transforms(size: int, mean: tuple, std: tuple) -> A.Compose:
    """Deterministic pipeline for segmentation validation."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


# =========================================================================
# Synthetic mask fallback
# =========================================================================


def _generate_synthetic_masks(images_dir: Path, masks_dir: Path, csv_path: Path) -> None:
    """Create centre-ellipse synthetic masks when GT masks are unavailable.

    This enables the pipeline to run end-to-end as a smoke test even
    without real segmentation annotations.

    Args:
        images_dir: Directory containing source JPEG images.
        masks_dir: Target directory for synthetic mask PNGs.
        csv_path: Split CSV to read image IDs from.
    """
    import cv2
    import pandas as pd

    masks_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    for image_id in df["image_id"].values:
        mask_path = masks_dir / f"{image_id}_segmentation.png"
        if mask_path.exists():
            continue

        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        ax, ay = int(w * 0.35), int(h * 0.35)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
        cv2.imwrite(str(mask_path), mask)


# =========================================================================
# Training loop
# =========================================================================


def train_one_epoch(
    model: LesionSegmentor,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run a single training epoch.

    Args:
        model: Segmentation model.
        loader: Training DataLoader.
        criterion: Combined BCE + Dice loss.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number (for display).

    Returns:
        Dict with ``loss``, ``dice``, ``iou``.
    """
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train E{epoch:02d}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics (detached)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            bs = images.size(0)
            loss_meter.update(loss.item(), bs)
            dice_meter.update(
                dice_score(probs.cpu(), masks.cpu()), bs,
            )
            iou_meter.update(
                iou_score(probs.cpu(), masks.cpu()), bs,
            )
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", dice=f"{dice_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "dice": dice_meter.avg, "iou": iou_meter.avg}


@torch.no_grad()
def validate(
    model: LesionSegmentor,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run a single validation epoch.

    Args:
        model: Segmentation model.
        loader: Validation DataLoader.
        criterion: Combined BCE + Dice loss.
        device: Compute device.
        epoch: Current epoch number.

    Returns:
        Dict with ``loss``, ``dice``, ``iou``.
    """
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Val   E{epoch:02d}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        dice_meter.update(dice_score(probs.cpu(), masks.cpu()), bs)
        iou_meter.update(iou_score(probs.cpu(), masks.cpu()), bs)

    return {"loss": loss_meter.avg, "dice": dice_meter.avg, "iou": iou_meter.avg}


def log_sample_predictions(
    model: LesionSegmentor,
    dataset: HAM10000Dataset,
    device: torch.device,
    wandb_run: object,
    epoch: int,
    n_samples: int = 4,
) -> None:
    """Log a grid of sample predictions to W&B.

    Args:
        model: Segmentation model (eval mode).
        dataset: Validation dataset.
        device: Compute device.
        wandb_run: Active wandb run (or ``None``).
        epoch: Current epoch.
        n_samples: Number of samples to log.
    """
    if wandb_run is None:
        return

    import wandb

    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    images_list = []

    for idx in indices:
        sample = dataset[int(idx)]
        img_t = sample["image"].unsqueeze(0).to(device)
        pred = model.predict_mask(img_t, threshold=0.5).squeeze().cpu().numpy()

        gt = sample.get("mask")
        if gt is not None:
            if isinstance(gt, torch.Tensor):
                gt = gt.squeeze().numpy()
        else:
            gt = np.zeros_like(pred)

        masks_dict = {
            "prediction": {"mask_data": (pred * 255).astype(np.uint8)},
            "ground_truth": {"mask_data": (gt * 255).astype(np.uint8)},
        }
        # De-normalise image for display
        img_np = sample["image"].permute(1, 2, 0).numpy()
        img_np = (img_np * 0.15 + 0.6).clip(0, 1)  # approximate de-norm

        images_list.append(
            wandb.Image(img_np, masks=masks_dict, caption=sample["image_id"]),
        )

    wandb_run.log({f"predictions/epoch_{epoch}": images_list}, step=epoch)


# =========================================================================
# Main
# =========================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--fold", type=int, default=0, help="Fold index to use.")
    parser.add_argument("--smoke_test", type=int, default=0, help="Limit dataset to N samples (0 = full).")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    """Entry point for segmentation training."""
    args = parse_args()
    cfg = load_config(args.config)
    seg_cfg = cfg["segmentation"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    seed = train_cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    # --- Paths ---
    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    train_csv = splits_dir / f"fold{args.fold}_train.csv"
    val_csv = splits_dir / f"fold{args.fold}_val.csv"

    if not train_csv.exists():
        print(f"[ERROR] Split CSV not found: {train_csv}")
        print("       Run: python scripts/prepare_splits.py")
        sys.exit(1)

    # --- Masks ---
    masks_dir_str = data_cfg.get("masks_dir")
    if masks_dir_str:
        masks_dir = Path(masks_dir_str)
    else:
        masks_dir = data_dir / "masks"

    has_gt_masks = masks_dir.exists() and any(masks_dir.glob("*.png"))
    if not has_gt_masks:
        print(ISIC_MASKS_INSTRUCTIONS)
        print("[INFO] Generating synthetic centre-ellipse masks ...")
        _generate_synthetic_masks(images_dir, masks_dir, train_csv)
        _generate_synthetic_masks(images_dir, masks_dir, val_csv)
        print(f"[INFO] Synthetic masks written to {masks_dir}")

    # --- Transforms ---
    norm_cfg = aug_cfg.get("normalize", {})
    mean = tuple(norm_cfg.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm_cfg.get("std", [0.1404, 0.1521, 0.1697]))
    input_size = seg_cfg.get("input_size", 256)

    train_tf = get_seg_train_transforms(input_size, mean, std)
    val_tf = get_seg_val_transforms(input_size, mean, std)

    # --- Datasets ---
    train_ds = HAM10000Dataset(
        csv_path=train_csv,
        images_dir=images_dir,
        transform=train_tf,
        img_size=input_size,
        masks_dir=masks_dir,
        mode="segmentation",
    )
    val_ds = HAM10000Dataset(
        csv_path=val_csv,
        images_dir=images_dir,
        transform=val_tf,
        img_size=input_size,
        masks_dir=masks_dir,
        mode="segmentation",
    )

    # Smoke test — subsample
    if args.smoke_test > 0:
        n = min(args.smoke_test, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = Subset(val_ds, list(range(min(n // 4 or 1, len(val_ds)))))
        print(f"[SMOKE TEST] train={len(train_ds)}, val={len(val_ds)}")

    batch_size = seg_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # --- Model ---
    model = LesionSegmentor.from_config(seg_cfg).to(device)
    print(f"[INFO] Model: U-Net encoder={seg_cfg.get('encoder', 'resnet34')}")

    # --- Loss ---
    criterion = BCEDiceLoss.from_config(seg_cfg.get("loss", {}))

    # --- Optimizer ---
    lr = seg_cfg.get("lr", 1e-4)
    weight_decay = seg_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # --- Early stopping ---
    es_cfg = seg_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode=es_cfg.get("mode", "max"),
    )

    # --- Encoder freeze schedule ---
    warmup_epochs = seg_cfg.get("warmup_epochs", 5)
    model.freeze_encoder()
    print(f"[INFO] Encoder frozen for first {warmup_epochs} epoch(s)")

    # --- W&B ---
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(cfg, run_name=f"seg_fold{args.fold}")

    # --- Training ---
    epochs = seg_cfg.get("epochs", 50)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "segmentation"
    best_dice = 0.0

    print(f"[INFO] Training for {epochs} epochs, batch_size={batch_size}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after warmup
        if epoch == warmup_epochs + 1:
            model.unfreeze_encoder()
            print(f"[INFO] Encoder unfrozen at epoch {epoch}")

        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        elapsed = time.time() - t0

        # Scheduler step
        scheduler.step(val_metrics["dice"])

        # Log
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} | "
            f"lr={lr_current:.2e} | {elapsed:.1f}s"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/dice": train_metrics["dice"],
                    "train/iou": train_metrics["iou"],
                    "val/loss": val_metrics["loss"],
                    "val/dice": val_metrics["dice"],
                    "val/iou": val_metrics["iou"],
                    "lr": lr_current,
                    "epoch": epoch,
                },
                step=epoch,
            )

        # Sample predictions every 5 epochs
        if epoch % 5 == 0 and wandb_run is not None:
            base_val_ds = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
            log_sample_predictions(model, base_val_ds, device, wandb_run, epoch)

        # Checkpoint
        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "config": seg_cfg,
            },
            filepath=ckpt_dir / f"epoch_{epoch:03d}.pt",
            is_best=is_best,
        )

        # Early stopping
        if early_stopper(val_metrics["dice"]):
            print(f"[INFO] Early stopping at epoch {epoch} (best dice={best_dice:.4f})")
            break

    print("=" * 60)
    print(f"[DONE] Best validation Dice: {best_dice:.4f}")
    print(f"[DONE] Best checkpoint: {ckpt_dir / 'best.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
