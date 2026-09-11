"""Shared asset path helpers (work both from source and frozen bundles)."""

from __future__ import annotations

import sys
from pathlib import Path


def _bundle_root() -> Path:
    """Root that owns the assets/ folder at runtime.

    - Source tree : repo root (two levels above this file)
    - PyInstaller : the extraction dir (sys._MEIPASS, e.g. ".../_internal")
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


ASSETS_DIR = _bundle_root() / "assets"


def asset_path(*parts: str) -> Path:
    return ASSETS_DIR.joinpath(*parts)


def logo_path() -> Path:
    return asset_path("icons", "LogoBetterBrush.png")


def ico_path() -> Path:
    return asset_path("icons", "BetterBrush.ico")


def app_icon() -> str:
    """String path usable by QIcon, preferring the Windows ICO when present."""
    ico = ico_path()
    if ico.is_file():
        return str(ico)
    logo = logo_path()
    return str(logo) if logo.is_file() else ""