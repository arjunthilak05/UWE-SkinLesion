"""Reusable multi-phase classification trainer with mixed-precision, gradient clipping, and W&B logging."""

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import balanced_accuracy, macro_auc, macro_f1, per_class_auc
from src.utils import AverageMeter, EarlyStopping, save_checkpoint


class Trainer:
    """Multi-phase classification trainer.

    Implements a three-phase progressive unfreezing schedule:

    - **Phase 1** (warm-up): backbone frozen, only the head trains at a high LR.
    - **Phase 2** (partial unfreeze): last ``n_unfreeze_blocks`` of backbone unfrozen
      with discriminative LR (backbone < head).
    - **Phase 3** (full fine-tune): entire model unfrozen with cosine annealing
      warm-restarts.

    Args:
        model: Classification model with ``freeze_backbone``, ``unfreeze_backbone_partial``,
            ``unfreeze_all``, and ``get_param_groups`` methods.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        config: Full config dict (needs a classifier section and ``training`` keys).
        config_key: Key for the classifier config section (e.g.
            ``"global_classifier"`` or ``"local_classifier"``).
        wandb_run: Active W&B run or ``None``.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: dict[str, Any],
        config_key: str = "global_classifier",
        wandb_run: Any = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.config_key = config_key
        self.wandb_run = wandb_run
        self.device = device or torch.device("cpu")

        cls_cfg = config.get(config_key, config)
        train_cfg = config.get("training", {})

        self.gradient_clip = cls_cfg.get("gradient_clip", 1.0)
        self.num_classes = cls_cfg.get("num_classes", 7)

        # Mixed precision — only for CUDA
        self.use_amp = train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Phase boundaries (epoch indices, 0-based)
        self.phase1_epochs = cls_cfg.get("phase1_epochs", 5)
        self.phase2_epochs = cls_cfg.get("phase2_epochs", 10)
        # Phase 3 runs until total epochs
        self.total_epochs = cls_cfg.get("epochs", 60)
        self.n_unfreeze_blocks = cls_cfg.get("n_unfreeze_blocks", 2)

        # LR settings per phase
        self.phase1_lr = cls_cfg.get("phase1_lr", 1e-3)
        self.phase2_backbone_lr = cls_cfg.get("phase2_backbone_lr", 1e-5)
        self.phase2_head_lr = cls_cfg.get("phase2_head_lr", 1e-4)
        self.phase3_lr = cls_cfg.get("lr", 3e-4)
        self.weight_decay = cls_cfg.get("weight_decay", 1e-5)

        # Scheduler params for Phase 3
        sched_params = cls_cfg.get("scheduler_params", {})
        self.T_0 = sched_params.get("T_0", 10)
        self.T_mult = sched_params.get("T_mult", 2)
        self.eta_min = sched_params.get("eta_min", 1e-7)

        # Early stopping
        es_cfg = cls_cfg.get("early_stopping", {})
        self.early_stopper = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            mode=es_cfg.get("mode", "max"),
        )

        # Checkpoint dir
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / config_key

        # Mixup / CutMix
        mixup_alpha = cls_cfg.get("mixup_alpha", 0.0)
        cutmix_alpha = cls_cfg.get("cutmix_alpha", 0.0)
        self.mixup_fn = None
        if mixup_alpha > 0 or cutmix_alpha > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=0.0,  # FocalLoss handles smoothing
                num_classes=self.num_classes,
            )
            print(f"[INFO] Mixup(alpha={mixup_alpha}) + CutMix(alpha={cutmix_alpha}) enabled")

        # Will be set by _enter_phase
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.current_phase: int = 0

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _enter_phase(self, phase: int) -> None:
        """Configure model freezing, optimizer, and scheduler for a training phase.

        Args:
            phase: 1, 2, or 3.
        """
        self.current_phase = phase

        if phase == 1:
            self.model.freeze_backbone()
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.phase1_lr,
                weight_decay=self.weight_decay,
            )
            self.scheduler = None
            print(f"[Phase 1] Backbone frozen — head only, lr={self.phase1_lr}")

        elif phase == 2:
            self.model.unfreeze_backbone_partial(n_blocks=self.n_unfreeze_blocks)
            param_groups = self.model.get_param_groups(
                backbone_lr=self.phase2_backbone_lr,
                head_lr=self.phase2_head_lr,
            )
            # Filter out frozen params from backbone group
            param_groups[0]["params"] = [
                p for p in param_groups[0]["params"] if p.requires_grad
            ]
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.weight_decay,
            )
            self.scheduler = None
            print(
                f"[Phase 2] Last {self.n_unfreeze_blocks} blocks unfrozen — "
                f"backbone lr={self.phase2_backbone_lr}, head lr={self.phase2_head_lr}"
            )

        elif phase == 3:
            self.model.unfreeze_all()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.phase3_lr,
                weight_decay=self.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
            )
            print(
                f"[Phase 3] Fully unfrozen — cosine warm restarts, "
                f"lr={self.phase3_lr}, T_0={self.T_0}"
            )

    def _get_phase(self, epoch: int) -> int:
        """Determine which phase an epoch belongs to (0-based epoch).

        Args:
            epoch: 0-based epoch index.

        Returns:
            Phase number (1, 2, or 3).
        """
        if epoch < self.phase1_epochs:
            return 1
        if epoch < self.phase1_epochs + self.phase2_epochs:
            return 2
        return 3

    # ------------------------------------------------------------------
    # Training & validation
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train for a single epoch.

        Args:
            epoch: 0-based epoch index (for display).

        Returns:
            Dict with ``loss``.
        """
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch + 1:02d} P{self.current_phase}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Apply Mixup / CutMix
            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            bs = images.size(0)
            loss_meter.update(loss.item(), bs)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation and compute all metrics.

        Args:
            epoch: 0-based epoch index (for display).

        Returns:
            Dict with ``loss``, ``balanced_acc``, ``macro_f1``, ``macro_auc``,
            ``per_class_auc``.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        pbar = tqdm(self.val_loader, desc=f"Val   E{epoch + 1:02d}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            bs = images.size(0)
            loss_meter.update(loss.item(), bs)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.concatenate(all_probs, axis=0)

        bal_acc = balanced_accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        m_auc = macro_auc(y_true, y_probs, self.num_classes)
        pc_auc = per_class_auc(y_true, y_probs, self.num_classes)

        return {
            "loss": loss_meter.avg,
            "balanced_acc": bal_acc,
            "macro_f1": mf1,
            "macro_auc": m_auc,
            "per_class_auc": pc_auc,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list]:
        """Run the full multi-phase training loop.

        Returns:
            History dict with per-epoch metrics for train and val.
        """
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_balanced_acc": [],
            "val_macro_f1": [],
            "val_macro_auc": [],
        }

        best_bal_acc = 0.0
        current_phase = 0

        self.model.to(self.device)

        print(f"[INFO] Training for {self.total_epochs} epochs across 3 phases")
        print(f"  Phase 1: epochs 1–{self.phase1_epochs} (frozen backbone)")
        print(f"  Phase 2: epochs {self.phase1_epochs + 1}–{self.phase1_epochs + self.phase2_epochs} (partial unfreeze)")
        print(f"  Phase 3: epochs {self.phase1_epochs + self.phase2_epochs + 1}–{self.total_epochs} (full fine-tune)")
        print("=" * 70)

        for epoch in range(self.total_epochs):
            phase = self._get_phase(epoch)
            if phase != current_phase:
                self._enter_phase(phase)
                current_phase = phase

            t0 = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0

            # Current LR
            lr_current = self.optimizer.param_groups[-1]["lr"]  # head lr

            # Store history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_balanced_acc"].append(val_metrics["balanced_acc"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_macro_auc"].append(val_metrics["macro_auc"])

            # Log
            print(
                f"E{epoch + 1:03d} P{phase} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"bal_acc={val_metrics['balanced_acc']:.4f} "
                f"f1={val_metrics['macro_f1']:.4f} "
                f"auc={val_metrics['macro_auc']:.4f} | "
                f"lr={lr_current:.2e} | {elapsed:.1f}s"
            )

            if self.wandb_run is not None:
                log_dict = {
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "val/balanced_acc": val_metrics["balanced_acc"],
                    "val/macro_f1": val_metrics["macro_f1"],
                    "val/macro_auc": val_metrics["macro_auc"],
                    "lr": lr_current,
                    "phase": phase,
                    "epoch": epoch + 1,
                }
                # Per-class AUC
                class_names = self.config.get("data", {}).get(
                    "class_names", [f"class_{i}" for i in range(self.num_classes)]
                )
                for i, name in enumerate(class_names):
                    log_dict[f"val/auc_{name}"] = val_metrics["per_class_auc"][i]
                self.wandb_run.log(log_dict, step=epoch + 1)

            # Checkpoint
            is_best = val_metrics["balanced_acc"] > best_bal_acc
            if is_best:
                best_bal_acc = val_metrics["balanced_acc"]

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "phase": phase,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_balanced_acc": val_metrics["balanced_acc"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_macro_auc": val_metrics["macro_auc"],
                    "val_loss": val_metrics["loss"],
                    "config": self.config.get(self.config_key, {}),
                },
                filepath=self.ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                is_best=is_best,
            )

            # Early stopping
            if self.early_stopper(val_metrics["balanced_acc"]):
                print(
                    f"[INFO] Early stopping at epoch {epoch + 1} "
                    f"(best balanced_acc={best_bal_acc:.4f})"
                )
                break

        print("=" * 70)
        print(f"[DONE] Best val balanced accuracy: {best_bal_acc:.4f}")
        print(f"[DONE] Best checkpoint: {self.ckpt_dir / 'best.pt'}")

        return history
