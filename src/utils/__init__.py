"""Utility subpackage: seeding, config I/O, logging, checkpointing, and training helpers."""

import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from src.utils.config import load_config, load_configs, save_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


# =========================================================================
# Device
# =========================================================================


def get_device(preference: str = "auto") -> torch.device:
    """Return the best available device.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.

    Returns:
        A ``torch.device``.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# =========================================================================
# Weights & Biases
# =========================================================================


def setup_wandb(
    config: dict[str, Any],
    run_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> Any:
    """Initialise a Weights & Biases run.

    Args:
        config: Full config dict (logged as hyperparameters).
        run_name: Display name for the run.
        project: W&B project name (falls back to config).
        entity: W&B entity / team (falls back to config).

    Returns:
        The ``wandb.Run`` object, or ``None`` if wandb is not installed.
    """
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed — skipping experiment tracking.")
        return None

    train_cfg = config.get("training", {})
    return wandb.init(
        project=project or train_cfg.get("wandb_project", "skin-lesion"),
        entity=entity or train_cfg.get("wandb_entity"),
        name=run_name,
        config=config,
    )


# =========================================================================
# Early Stopping
# =========================================================================


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        mode: ``"min"`` or ``"max"`` — whether lower or higher is better.
        min_delta: Minimum change to count as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, value: float) -> bool:
        """Update state and return ``True`` if training should stop.

        Args:
            value: Current metric value.

        Returns:
            ``True`` when patience is exhausted.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# =========================================================================
# Average Meter
# =========================================================================


class AverageMeter:
    """Running average tracker for a single scalar metric.

    Usage::

        meter = AverageMeter()
        for loss in losses:
            meter.update(loss, n=batch_size)
        print(meter.avg)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all counters."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new observation.

        Args:
            val: Metric value (scalar).
            n: Number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


# =========================================================================
# Checkpointing
# =========================================================================


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint.

    Args:
        state: Dict with ``model_state_dict``, ``optimizer_state_dict``,
               ``epoch``, ``metrics``, etc.
        filepath: Destination path (e.g. ``checkpoints/seg/last.pt``).
        is_best: If ``True``, also copy to ``best.pt`` alongside ``filepath``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)

    if is_best:
        best_path = filepath.parent / "best.pt"
        shutil.copy2(filepath, best_path)


def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load a training checkpoint into a model (and optionally an optimizer).

    Args:
        filepath: Path to the ``.pt`` file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        The full checkpoint dict (for extracting epoch, metrics, etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    ckpt = torch.load(filepath, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt


__all__ = [
    "seed_everything",
    "load_config",
    "load_configs",
    "save_config",
    "get_logger",
    "get_device",
    "setup_wandb",
    "EarlyStopping",
    "AverageMeter",
    "save_checkpoint",
    "load_checkpoint",
]
