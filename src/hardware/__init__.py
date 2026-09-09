"""Hardware detection & recommendation package."""

from src.hardware.detector import Backend, GPUInfo, HardwareDetector, HardwareProfile
from src.hardware.profiler import (
    ContentProfile,
    ModelSize,
    SAM2Profiler,
    SAM2Recommendation,
)

__all__ = [
    "Backend",
    "GPUInfo",
    "HardwareDetector",
    "HardwareProfile",
    "ContentProfile",
    "ModelSize",
    "SAM2Profiler",
    "SAM2Recommendation",
]
