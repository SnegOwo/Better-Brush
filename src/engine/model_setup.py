"""Deterministic model selection for the headless engine.

Everything here defers the heavy imports (``torch`` lives behind
``src.hardware.detector``) so ``src.engine.server`` can be imported and tested
without a GPU. Selection is by *total* VRAM via :class:`SAM2Profiler`, never by
live free VRAM — that was the root cause of the "sometimes no detection"
whiplash in the GUI (free VRAM drifts with whatever else is resident).
"""

from __future__ import annotations

import logging
from typing import Any

from src.engine.schemas import ERR_UNSUPPORTED, RequestError

logger = logging.getLogger(__name__)

#: Conservative tiers per requested quality. ``best`` and ``balanced`` both cap
#: at Base+ because that is the largest SAM 2 checkpoint a stock 8 GB laptop
#: GPU propagates comfortably; a fine-tuned ("Lugnica SAM") model is preferred
#: when installed.
_QUALITY_TO_MODEL = {
    "fast": "small",
    "balanced": "base_plus",
    "best": "base_plus",
}

_MODEL_TO_ENUM = {
    "tiny": "TINY",
    "small": "SMALL",
    "base_plus": "BASE_PLUS",
    "large": "LARGE",
    "custom": "CUSTOM",
}


def custom_model_available() -> bool:
    """True when a fine-tuned checkpoint + sidecar is installed."""
    try:
        from src.models.checkpoint_manager import CheckpointManager

        return CheckpointManager().custom_model().checkpoint.is_file()
    except Exception:  # noqa: BLE001 - missing sidecar/checkpoint is normal
        return False


def custom_model_label() -> str | None:
    try:
        from src.models.checkpoint_manager import CheckpointManager

        return CheckpointManager().custom_model().label
    except Exception:  # noqa: BLE001
        return None


def resolve_model_name(quality: str, override: str | None = None) -> str:
    """Map ``(quality, explicit model)`` to a concrete model name."""
    if override:
        return override
    if quality in ("balanced", "best") and custom_model_available():
        return "custom"
    return _QUALITY_TO_MODEL.get(quality, "base_plus")


def _model_enum(name: str):
    from src.hardware.profiler import ModelSize

    attr = _MODEL_TO_ENUM.get(name)
    if attr is None:
        raise RequestError(f"unknown model '{name}'")
    return getattr(ModelSize, attr)


def build_recommendation(
    quality: str = "balanced",
    preset: str = "anime",
    model: str | None = None,
):
    """Return a :class:`SAM2Recommendation` for this machine + request.

    Raises :class:`RequestError` (kind ``unsupported``) when SAM 2 is absent, so
    the panel can tell the user to run the desktop setup instead of exploding.
    """
    try:
        from src.hardware.detector import HardwareDetector
        from src.hardware.profiler import ContentProfile, SAM2Profiler
    except Exception as exc:
        raise RequestError(
            f"Better Brush runtime unavailable: {exc}",
            kind=ERR_UNSUPPORTED,
        ) from exc

    content = {
        "anime": ContentProfile.ANIME,
        "human": ContentProfile.HUMAN,
        "object": ContentProfile.OBJECT,
    }.get(preset, ContentProfile.ANIME)

    model_name = resolve_model_name(quality, model)
    force = _model_enum(model_name)

    profile = HardwareDetector().detect()
    recommendation = SAM2Profiler().recommend(profile, content, force_model=force)

    from src.models.sam2_manager import SAM2Manager

    manager = SAM2Manager(recommendation)
    if not manager.is_available:
        raise RequestError(
            "SAM 2 is not installed for this Python environment. "
            "Run Better Brush's setup once, then retry.",
            kind=ERR_UNSUPPORTED,
        )
    return recommendation


def describe() -> dict[str, Any]:
    """Cheap capability snapshot for ``GET /health`` (no model load)."""
    custom = custom_model_available()
    info: dict[str, Any] = {
        "custom_model": custom,
        "custom_label": custom_model_label() if custom else None,
    }
    try:
        from src.hardware.detector import HardwareDetector

        profile = HardwareDetector().detect()
        info["device"] = profile.recommended_device
        info["backend"] = profile.best_backend.value
        vram = max((g.total_vram_mb for g in profile.gpus), default=0)
        info["total_vram_mb"] = vram
    except Exception as exc:  # noqa: BLE001
        info["device"] = "unavailable"
        info["error"] = str(exc)
    return info
