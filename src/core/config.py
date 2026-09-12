"""Application configuration (settings persistence could be added here)."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.hardware.profiler import ContentProfile, SAM2Recommendation


@dataclass
class AppConfig:
    """Runtime configuration container."""

    recommendation: SAM2Recommendation | None = None
    content_profile: ContentProfile = ContentProfile.GENERAL
    recent_videos: list[str] = field(default_factory=list)
    window_size: tuple[int, int] = (1400, 900)
