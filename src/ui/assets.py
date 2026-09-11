"""Shared asset path helpers."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = _ROOT / "assets"


def asset_path(*parts: str) -> Path:
    return ASSETS_DIR.joinpath(*parts)


def logo_path() -> Path:
    return asset_path("icons", "LogoBetterBrush.png")


def app_icon() -> str:
    """String path usable by QIcon, preferring the logo when present."""
    logo = logo_path()
    return str(logo) if logo.is_file() else ""