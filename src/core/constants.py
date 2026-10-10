"""Global application configuration and paths."""

from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Better Brush"
APP_VERSION = "0.3.0"
APP_STAGE = "beta"

# Writable storage root. Source tree: repo root. Frozen bundle: the folder
# that holds the executable (so checkpoints/exports always land in a
# predictable, user-owned location).
if getattr(sys, "frozen", False):
    _ROOT = Path(sys.executable).resolve().parent
else:
    _ROOT = Path(__file__).resolve().parents[2]

# PyInstaller 6 onedir bundles readonly data under <exe>/_internal. Prefer
# the bundled copy of an object (e.g. the face-landmarker model) so frozen
# users never re-download it, and fall back to the writable _ROOT copy.
_MEIPASS = Path(getattr(sys, "_MEIPASS", ""))
if _MEIPASS.is_dir() and (_MEIPASS / "checkpoints").is_dir():
    CHECKPOINTS_DIR = _MEIPASS / "checkpoints"
else:
    CHECKPOINTS_DIR = _ROOT / "checkpoints"
EXPORTS_DIR = _ROOT / "exports"
