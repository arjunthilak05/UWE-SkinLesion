"""Train a single baseline model for SOTA comparison.

Uses the exact same data splits, augmentation, loss, and evaluation protocol
as the dual-pathway classifiers for a fair comparison.

Usage::

    python scripts/train_baselines.py --model_name resnet50 --fold 0
    python scripts/train_baselines.py --model_name vit_base_patch16_224 --fold 0 --epochs 30
    python scripts/train_baselines.py --model_name densenet121 --smoke_test 200 --phase_epochs 2
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
from src.models.baseline import BaselineClassifier
from src.trainer import Trainer
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import get_device, load_config, seed_everything, setup_wandb

SUPPORTED_MODELS = [
    "resnet50",
    "vgg16_bn",
    "densenet121",
    "densenet201",
    "efficientnet_b4",
    "tf_efficientnetv2_s",
    "convnext_tiny",
    "vit_base_patch16_224",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2.lvd142m",
    "swin_tiny_patch4_window7_224",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a baseline model for SOTA comparison.")
    parser.add_argument(
        "--model_name", type=str, required=True, choices=SUPPORTED_MODELS,
        help="timm model name.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0, help="Override total epochs (0 = use config).")
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--phase_epochs", type=int, default=0, help="Override epochs per phase (for testing).")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size (0 = auto).")
    parser.add_argument("--train_csv", type=Path, default=None,
                        help="Override train CSV (e.g. oversampled version).")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler."""
    counts = np.bincount(labels)
    weights = 1.0 / np.maximum(counts, 1).astype(float)
    sample_w = weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).double(),
        num_samples=len(labels),
        replacement=True,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})

    seed = train_cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(args.device)

    model_name = args.model_name
    img_size = BaselineClassifier.get_input_size(model_name)
    config_key = f"baseline_{model_name}"

    # Build a synthetic config section for this baseline
    # so the Trainer reads phase/lr/epoch settings from it
    baseline_cfg = {
        "model_name": model_name,
        "pretrained": True,
        "num_classes": 7,
        "drop_rate": 0.3,
        "hidden_dim": 512,
        "input_size": img_size,
        "batch_size": args.batch_size if args.batch_size > 0 else (16 if img_size >= 380 else 32),
        "gradient_clip": 1.0,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "loss": {"gamma": 2.0},
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "epochs": 60,
        "n_unfreeze_blocks": 2,
        "phase1_lr": 1e-3,
        "phase2_backbone_lr": 1e-5,
        "phase2_head_lr": 1e-4,
        "scheduler_params": {"T_0": 10, "T_mult": 2, "eta_min": 1e-7},
        "early_stopping": {"patience": 15, "mode": "max"},
    }

    # Apply CLI overrides
    if args.epochs > 0:
        baseline_cfg["epochs"] = args.epochs
    if args.phase_epochs > 0:
        baseline_cfg["phase1_epochs"] = args.phase_epochs
        baseline_cfg["phase2_epochs"] = args.phase_epochs
        baseline_cfg["epochs"] = args.phase_epochs * 3

    # Inject into config so Trainer can find it
    cfg[config_key] = baseline_cfg

    print(f"[INFO] Model:      {model_name}")
    print(f"[INFO] Input size: {img_size}x{img_size}")
    print(f"[INFO] Device:     {device}")
    if args.phase_epochs > 0:
        print(f"[SMOKE] {args.phase_epochs} epochs/phase, {baseline_cfg['epochs']} total")

    # --- Paths ---
    data_dir = Path(data_cfg["data_dir"])
    images_dir = data_dir / data_cfg["images_subdir"]
    splits_dir = Path(data_cfg.get("splits_dir", "data/splits"))
    train_csv = args.train_csv if args.train_csv else splits_dir / f"fold{args.fold}_train.csv"
    val_csv = splits_dir / f"fold{args.fold}_val.csv"

    if not train_csv.exists():
        print(f"[ERROR] Split CSV not found: {train_csv}")
        sys.exit(1)

    # --- Transforms (identical pipeline, different img_size) ---
    norm = aug_cfg.get("normalize", {})
    mean = tuple(norm.get("mean", [0.7635, 0.5461, 0.5705]))
    std = tuple(norm.get("std", [0.1404, 0.1521, 0.1697]))

    train_tf = get_train_transforms(img_size, mean, std, cfg=aug_cfg)
    val_tf = get_val_transforms(img_size, mean, std)

    # --- Datasets ---
    train_ds = HAM10000Dataset(
        csv_path=train_csv, images_dir=images_dir,
        transform=train_tf, img_size=img_size, mode="classification",
    )
    val_ds = HAM10000Dataset(
        csv_path=val_csv, images_dir=images_dir,
        transform=val_tf, img_size=img_size, mode="classification",
    )

    if args.smoke_test > 0:
        n = min(args.smoke_test, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = Subset(val_ds, list(range(min(n // 4 or 1, len(val_ds)))))
        print(f"[SMOKE TEST] train={len(train_ds)}, val={len(val_ds)}")

    # --- Sampler ---
    if isinstance(train_ds, Subset):
        labels = np.array([train_ds.dataset.labels[i] for i in train_ds.indices])
    else:
        labels = train_ds.get_labels()
    sampler = build_weighted_sampler(labels)

    batch_size = baseline_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=data_cfg.get("pin_memory", True),
    )

    # --- Model ---
    model = BaselineClassifier(
        model_name=model_name,
        pretrained=baseline_cfg["pretrained"],
        num_classes=baseline_cfg["num_classes"],
        drop_rate=baseline_cfg["drop_rate"],
        hidden_dim=baseline_cfg["hidden_dim"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params:,}")
    print(f"[INFO] Feature dim: {model.feat_dim}")

    # --- Loss (identical to main classifiers) ---
    criterion = FocalLoss.from_config(
        gamma=baseline_cfg["loss"]["gamma"],
        label_smoothing=baseline_cfg["label_smoothing"],
    )

    # --- W&B ---
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(cfg, run_name=f"baseline_{model_name}_fold{args.fold}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=cfg,
        config_key=config_key,
        wandb_run=wandb_run,
        device=device,
    )

    # Override checkpoint dir to baselines/
    trainer.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "baselines" / model_name

    # --- Train ---
    history = trainer.fit()

    # --- Save ---
    results_dir = Path(train_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    hist_path = results_dir / f"baseline_{model_name}_fold{args.fold}_history.json"
    with hist_path.open("w", encoding="utf-8") as fh:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, fh, indent=2)
    print(f"[INFO] History saved to {hist_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
