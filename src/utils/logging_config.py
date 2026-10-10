"""Centralised logging configuration and stdio bootstrap for frozen apps."""

import logging
import os
import sys
from pathlib import Path


def log_dir() -> Path:
    """Directory where ``better-brush.log`` lives (user-accessible)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parents[2]
    return base / "logs"


def log_file_path() -> Path:
    """Full path of the on-disk run log."""
    return log_dir() / "better-brush.log"


def add_file_handler(
    level: int = logging.DEBUG,
    max_mb: int = 5,
) -> None:
    """Attach a rotating file handler to the root logger.

    Everything the app logs is mirrored to ``logs/better-brush.log`` so
    users (and support) can inspect what happened even when the app runs
    windowed with no console.
    """
    try:
        import logging.handlers

        log_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file_path(),
            maxBytes=max_mb * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.addHandler(handler)
    except Exception as exc:  # noqa: BLE001 - logging must never crash startup
        logging.getLogger(__name__).warning("Could not attach file log handler: %s", exc)


def init_stdio() -> None:
    """Make sys.stdout / sys.stderr usable in frozen windowed builds.

    PyInstaller's windowed (no-console) bootloader sets both to None, which
    makes any ``sys.stdout.write`` / logging emit raise
    ``AttributeError: 'NoneType' object has no attribute 'write'`` at the first
    log line (e.g. during SAM 2 propagation). We repoint them at a log file
    next to the executable (or devnull as a last resort).
    """
    if not getattr(sys, "frozen", False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log_file = Path(sys.executable).resolve().parent / "better-brush.log"
        sink = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    except Exception:  # noqa: BLE001
        sink = open(os.devnull, "w")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = sink
    if sys.stderr is None:
        sys.stderr = sink


def configure_logging(
    level: int = logging.INFO,
    fmt: str | None = None,
    stream: object | None = None,
) -> None:
    """Set up a tidy root logger for the application."""
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    init_stdio()
    raw = stream or sys.stdout or sys.stderr
    if raw is None:
        raw = open(os.devnull, "w")  # noqa: SIM115
    handler = logging.StreamHandler(raw)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called twice
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
