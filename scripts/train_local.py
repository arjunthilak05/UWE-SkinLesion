"""Train the ResNet-50 local pathway classifier on cropped lesion images (Stage 4).

Expects pre-generated cropped lesion images in ``data/HAM10000/cropped_lesions/``.
If that directory is missing, run ``scripts/generate_masks.py`` first.

Usage::

    python scripts/train_local.py --config configs/default.yaml
    python scripts/train_local.py --config configs/default.yaml --smoke_test 200 --phase_epochs 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HAM10000Dataset
from src.losses import FocalLoss
from src.models.local_classifier import LocalClassifier
from src.trainer import Trainer
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import get_device, load_config, seed_everything, setup_wandb

GENERATE_MASKS_MSG = """
╔══════════════════════════════════════════════════════════════════╗
║  Cropped lesion images not found!                               ║
║                                                                 ║
║  The local classifier trains on cropped lesion regions.         ║
║  Generate them first by running:                                ║
║                                                                 ║
║    python scripts/generate_masks.py \\                           ║
║        --checkpoint checkpoints/segmentation/best.pt \\          ║
║        --config configs/default.yaml                            ║
║                                                                 ║
║  This will produce:                                             ║
║    data/HAM10000/predicted_masks/  — binary segmentation masks  ║
║    data/HAM10000/cropped_lesions/  — cropped + resized JPEGs    ║
║    data/HAM10000/bounding_boxes.csv                             ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet-50 local classifier on cropped lesions.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--fold", type=int, default=0, help="Fold index.")
    parser.add_argument("--smoke_test", type=int, default=0, help="Limit dataset to N samples (0 = full).")
    parser.add_argument("--phase_epochs", type=int, default=0, help="Override epochs per phase (for testing).")
    parser.add_argument("--train_csv", type=Path, default=None,
                        help="Override train CSV (e.g. oversampled).")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler with inverse-frequency weights.

    Args:
        labels: Integer label array for the training set.

    Returns:
        Configured ``WeightedRandomSampler``.
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    cls_cfg = cfg["local_classifier"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    seed = train_cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")

    # --- Override phase epochs for smoke test ---
    if args.phase_epochs > 0:
        cls_cfg["phase1_epochs"] = args.phase_epochs
        cls_cfg["phase2_epochs"] = args.phase_epochs
        cls_cfg["epochs"] = args.phase_epochs * 3
        print(f"[SMOKE] Overriding: {args.phase_epochs} epochs per phase, {args.phase_epochs * 3} total")

    # --- Paths ---
    data_dir = Path(data_cfg["data_dir"])
    crops_dir = data_dir / "cropped_lesions"

    # Check that cropped lesions exist
    if not crops_dir.exists() or not any(crops_dir.glob("*.jpg")):
        print(GENERATE_MASKS_MSG)
        # Auto-generate from segmentation checkpoint if available
        seg_ckpt = Path("checkpoints/segmentation/best.pt")
        if seg_ckpt.exists():
            print("[INFO] Found segmentation checkpoint — auto-generating crops ...")
            import subprocess
            result = subprocess.run(
                [
                    sys.executable, "scripts/generate_masks.py",
                    "--checkpoint", str(seg_ckpt),
                    "--config", str(args.config),
                    "--device", args.device,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[ERROR] generate_masks.py failed:\n{result.stderr}")
                sys.exit(1)
            print("[INFO] Crops generated successfully.")
        else:
            print("[ERROR] No segmentation checkpoint found. Train segmentation first.")
            sys.exit(1)

    images_dir = crops_dir  # local classifier reads from cropped_lesions/
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    train_csv = args.train_csv if args.train_csv else splits_dir / f"fold{args.fold}_train.csv"
    val_csv = splits_dir / f"fold{args.fold}_val.csv"

    if not train_csv.exists():
        print(f"[ERROR] Split CSV not found: {train_csv}")
        print("       Run: python scripts/prepare_splits.py")
        sys.exit(1)

    # --- Transforms ---
    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))
    img_size = cls_cfg.get("input_size", 224)

    train_tf = get_train_transforms(img_size, mean, std, cfg=aug_cfg)
    val_tf = get_val_transforms(img_size, mean, std)

    # --- Datasets ---
    train_ds = HAM10000Dataset(
        csv_path=train_csv,
        images_dir=images_dir,
        transform=train_tf,
        img_size=img_size,
        mode="classification",
        filter_existing=True,
    )
    val_ds = HAM10000Dataset(
        csv_path=val_csv,
        images_dir=images_dir,
        transform=val_tf,
        img_size=img_size,
        mode="classification",
        filter_existing=True,
    )

    # --- Smoke test subset ---
    if args.smoke_test > 0:
        n = min(args.smoke_test, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = Subset(val_ds, list(range(min(n // 4 or 1, len(val_ds)))))
        print(f"[SMOKE TEST] train={len(train_ds)}, val={len(val_ds)}")

    # --- WeightedRandomSampler ---
    if isinstance(train_ds, Subset):
        train_labels = np.array([train_ds.dataset.labels[i] for i in train_ds.indices])
    else:
        train_labels = train_ds.get_labels()

    sampler = build_weighted_sampler(train_labels)

    batch_size = cls_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
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
    model = LocalClassifier.from_config(cls_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {cls_cfg.get('model_name', 'resnet50')}")
    print(f"[INFO] Parameters: {total:,} total, {trainable:,} trainable")
    print(f"[INFO] Feature dim: {model.feat_dim}")
    print(f"[INFO] Input: cropped lesions at {img_size}x{img_size} from {images_dir}")

    # --- Loss ---
    loss_cfg = cls_cfg.get("loss", {})
    criterion = FocalLoss(
        gamma=loss_cfg.get("gamma", 2.0),
        alpha=None,  # Sampler handles class balance; no double weighting
        label_smoothing=cls_cfg.get("label_smoothing", 0.1),
    )
    print(f"[INFO] Loss: FocalLoss(gamma={loss_cfg.get('gamma', 1.0)}, smoothing={cls_cfg.get('label_smoothing', 0.1)})")

    # --- W&B ---
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(cfg, run_name=f"local_fold{args.fold}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=cfg,
        config_key="local_classifier",
        wandb_run=wandb_run,
        device=device,
    )

    # --- Train ---
    history = trainer.fit()

    # --- Save final metrics ---
    results_dir = Path(train_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"local_fold{args.fold}_history.json"

    serializable_history = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(serializable_history, fh, indent=2)
    print(f"[INFO] Training history saved to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
