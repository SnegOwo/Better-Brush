"""Headless Better Brush engine for external hosts (After Effects panel, CI).

Importing this package must never import torch, PySide6 or SAM 2. Heavy
dependencies are imported lazily inside the functions that need them — the same
rule :mod:`src.export.render_queue` follows. That keeps the HTTP surface and the
job registry importable, and testable, on a machine with no GPU.
"""

from __future__ import annotations

ENGINE_NAME = "Better Brush engine"
ENGINE_VERSION = "0.1.0"

#: Bumped whenever the HTTP JSON contract changes in a breaking way. The After
#: Effects panel checks this on connect and refuses politely on a mismatch.
API_VERSION = 1

__all__ = ["API_VERSION", "ENGINE_NAME", "ENGINE_VERSION"]
