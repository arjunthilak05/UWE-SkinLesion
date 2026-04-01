"""Utilities for reproducible experiment seeding."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for full reproducibility.

    Seeds Python's built-in ``random`` module, NumPy, PyTorch (CPU and all
    CUDA devices), and sets the ``PYTHONHASHSEED`` environment variable.
    Also configures cuDNN to operate deterministically.

    Args:
        seed: Integer seed value. Defaults to 42.

    Example:
        >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
