"""Entry point for Better Brush."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` imports work when launched as `python src/main.py` from the
# repo root (Windows uses backslashes, so a naive string split is unsafe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run before anything imports torch/SAM 2: in the frozen windowed build
# sys.stdout/sys.stderr are None and the first log write would crash.
from src.utils.logging_config import init_stdio

init_stdio()

from src.core.app import run

if __name__ == "__main__":
    raise SystemExit(run())
