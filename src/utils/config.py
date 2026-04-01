"""YAML configuration loading and merging utilities."""

from pathlib import Path
from typing import Any, Union

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts, with values in *override* winning.

    Args:
        base: The base dictionary.
        override: Dictionary whose values take precedence.

    Returns:
        A new dict that is the deep merge of *base* and *override*.
    """
    result: dict = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a single YAML configuration file.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        A dictionary containing the parsed YAML contents.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.

    Example:
        >>> cfg = load_config("configs/data.yaml")
        >>> cfg["dataset"]["num_classes"]
        7
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg if cfg is not None else {}


def load_configs(*paths: Union[str, Path]) -> dict[str, Any]:
    """Load and deep-merge multiple YAML configuration files.

    Configs are merged left-to-right; later files override earlier ones.

    Args:
        *paths: One or more paths to ``.yaml`` / ``.yml`` files.

    Returns:
        A single merged dictionary.

    Example:
        >>> cfg = load_configs("configs/paths.yaml", "configs/data.yaml")
    """
    merged: dict[str, Any] = {}
    for path in paths:
        cfg = load_config(path)
        merged = _deep_merge(merged, cfg)
    return merged


def save_config(cfg: dict[str, Any], path: Union[str, Path]) -> None:
    """Serialize a configuration dictionary to a YAML file.

    Args:
        cfg: Dictionary to serialize.
        path: Destination file path. Parent directories are created if they
            do not exist.

    Example:
        >>> save_config({"lr": 1e-4}, "checkpoints/run_cfg.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
