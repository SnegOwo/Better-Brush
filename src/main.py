"""Entry point for Better Brush."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` imports work when launched as `python src/main.py` from the
# repo root (Windows uses backslashes, so a naive string split is unsafe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.app import run

if __name__ == "__main__":
    raise SystemExit(run())
