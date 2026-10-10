"""Hardware detection & recommendation package."""

from src.hardware.detector import Backend, GPUInfo, HardwareDetector, HardwareProfile
from src.hardware.monitor import (
    GpuMonitor,
    GpuSampler,
    GpuTelemetry,
    ModelSafetyAssessment,
    ModelSafetyLevel,
    ThermalPressure,
    assess_model_safety,
    estimate_render_time,
)
from src.hardware.profiler import (
    ContentProfile,
    ModelSize,
    SAM2Profiler,
    SAM2Recommendation,
)

__all__ = [
    "Backend",
    "ContentProfile",
    "GPUInfo",
    "GpuMonitor",
    "GpuSampler",
    "GpuTelemetry",
    "HardwareDetector",
    "HardwareProfile",
    "ModelSafetyAssessment",
    "ModelSafetyLevel",
    "ModelSize",
    "SAM2Profiler",
    "SAM2Recommendation",
    "ThermalPressure",
    "assess_model_safety",
    "estimate_render_time",
]
