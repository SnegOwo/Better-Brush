"""Global application configuration and paths."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Better Brush"
APP_VERSION = "0.1.0"

# Default directories (relative to repo root)
CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
EXPORTS_DIR = Path(__file__).resolve().parents[2] / "exports"
