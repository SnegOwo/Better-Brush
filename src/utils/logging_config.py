"""Centralised logging configuration."""

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Set up a tidy root logger for the application."""
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called twice
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
