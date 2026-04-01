"""Logging utilities for the skin-lesion project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a named logger.

    The returned logger always writes to *stdout* via a
    :class:`logging.StreamHandler`. If *log_file* is provided an additional
    :class:`logging.FileHandler` (append mode, UTF-8) is attached.

    Each handler uses the format::

        YYYY-MM-DD HH:MM:SS | LEVEL    | name | message

    If a logger with the same *name* already exists and has handlers attached,
    it is returned as-is to avoid duplicate output when the function is called
    multiple times with the same name.

    Args:
        name: Logger name (usually ``__name__`` of the calling module).
        log_file: Optional path to a log file. Parent directories are created
            automatically if they do not exist.
        level: Logging level, e.g. ``logging.DEBUG`` or ``logging.INFO``.
            Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__, log_file=Path("logs/train.log"))
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
