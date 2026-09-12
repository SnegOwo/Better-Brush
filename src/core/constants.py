"""Global application configuration and paths."""

from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Better Brush"
APP_VERSION = "0.1.0"
APP_STAGE = "beta"

# Writable storage root. Source tree: repo root. Frozen bundle: the folder
# that holds the executable (so checkpoints/exports always land in a
# predictable, user-owned location).
if getattr(sys, "frozen", False):
    _ROOT = Path(sys.executable).resolve().parent
else:
    _ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS_DIR = _ROOT / "checkpoints"
EXPORTS_DIR = _ROOT / "exports"
