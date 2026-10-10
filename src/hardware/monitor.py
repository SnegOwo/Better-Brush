"""Optional live GPU telemetry, model-fit checks, and render estimates."""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Final, Self

from src.hardware.profiler import ModelSize

_BYTES_PER_MIB: Final[int] = 1024 * 1024
_BACKEND_ERRORS: Final[tuple[type[Exception], ...]] = (Exception,)


class TelemetryStatus(str, Enum):
    """Availability state for a telemetry sample."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ThermalPressure(str, Enum):
    """Thermal pressure level."""

    UNAVAILABLE = "unavailable"
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return a severity rank where higher is worse."""
        return {
            ThermalPressure.UNAVAILABLE: -1,
            ThermalPressure.NONE: 0,
            ThermalPressure.LOW: 1,
            ThermalPressure.MEDIUM: 2,
            ThermalPressure.HIGH: 3,
            ThermalPressure.CRITICAL: 4,
        }[self]

    @classmethod
    def from_value(cls, value: object) -> ThermalPressure:
        """Coerce a vendor or numeric pressure value."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNAVAILABLE
        if isinstance(value, str):
            names = {
                "unknown": cls.UNAVAILABLE,
                "unavailable": cls.UNAVAILABLE,
                "not available": cls.UNAVAILABLE,
                "normal": cls.NONE,
                "none": cls.NONE,
                "low": cls.LOW,
                "medium": cls.MEDIUM,
                "moderate": cls.MEDIUM,
                "high": cls.HIGH,
                "critical": cls.CRITICAL,
            }
            if value.strip().lower() in names:
                return names[value.strip().lower()]
        number = _finite(value)
        if number is None:
            return cls.UNAVAILABLE
        if number <= 0:
            return cls.NONE
        if number <= 1:
            return cls.LOW
        if number <= 2:
            return cls.MEDIUM
        if number <= 3:
            return cls.HIGH
        return cls.CRITICAL

    @classmethod
    def from_temperature(cls, temperature_c: float | None) -> ThermalPressure:
        """Infer pressure from a Celsius reading."""
        number = _finite(temperature_c)
        if number is None:
            return cls.UNAVAILABLE
        if number >= 95:
            return cls.CRITICAL
        if number >= 85:
            return cls.HIGH
        if number >= 75:
            return cls.MEDIUM
        if number >= 65:
            return cls.LOW
        return cls.NONE


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _positive_int(value: object, name: str) -> int:
    number = _finite(value)
    if number is None or number <= 0 or int(number) != number:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _non_negative_int(value: object, name: str) -> int:
    number = _finite(value)
    if number is None or int(number) != number:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _bytes_to_mib(value: object) -> float | None:
    number = _finite(value)
    return None if number is None else number / _BYTES_PER_MIB


def _milliwatts_to_watts(value: object) -> float | None:
    number = _finite(value)
    return None if number is None else number / 1000.0


def _attribute(value: object, *names: str) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        try:
            return getattr(value, name)
        except AttributeError:
            continue
    return None


def _telemetry_mapping(
    value: Mapping[str, object],
    device_index: int,
    timestamp: float,
) -> GpuTelemetry:
    aliases = {
        "gpu_utilization": "utilization_percent",
        "gpu_utilization_percent": "utilization_percent",
        "utilization": "utilization_percent",
        "vram_used_mb": "used_vram_mb",
        "memory_used_mb": "used_vram_mb",
        "vram_total_mb": "total_vram_mb",
        "memory_total_mb": "total_vram_mb",
        "vram_free_mb": "free_vram_mb",
        "available_vram_mb": "free_vram_mb",
        "temperature": "temperature_c",
        "power": "power_w",
        "pressure": "thermal_pressure",
    }
    values: dict[str, object] = {"device_index": device_index, "timestamp": timestamp}
    for source, target in aliases.items():
        if target not in values and source in value:
            values[target] = value[source]
    byte_aliases = {
        "used_vram_bytes": "used_vram_mb",
        "memory_used_bytes": "used_vram_mb",
        "total_vram_bytes": "total_vram_mb",
        "memory_total_bytes": "total_vram_mb",
        "free_vram_bytes": "free_vram_mb",
        "available_vram_bytes": "free_vram_mb",
    }
    for source, target in byte_aliases.items():
        if target not in values and source in value:
            values[target] = _bytes_to_mib(value[source])
    for name in (
        "utilization_percent",
        "used_vram_mb",
        "total_vram_mb",
        "free_vram_mb",
        "temperature_c",
        "power_w",
        "thermal_pressure",
        "device_name",
        "status",
        "error",
    ):
        if name in value:
            values[name] = value[name]
    return GpuTelemetry(**values)


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    """Immutable live GPU metrics using MiB, Celsius, and seconds."""

    device_index: int = 0
    timestamp: float = field(default_factory=time.time)
    utilization_percent: float | None = None
    used_vram_mb: float | None = None
    total_vram_mb: float | None = None
    free_vram_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    thermal_pressure: ThermalPressure = ThermalPressure.UNAVAILABLE
    device_name: str | None = None
    status: TelemetryStatus = TelemetryStatus.AVAILABLE
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, int) or self.device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        object.__setattr__(self, "device_index", self.device_index)
        timestamp_value = _finite(self.timestamp)
        object.__setattr__(self, "timestamp", time.time() if timestamp_value is None else timestamp_value)
        object.__setattr__(self, "utilization_percent", self._percent(self.utilization_percent))
        object.__setattr__(self, "used_vram_mb", _finite(self.used_vram_mb))
        object.__setattr__(self, "total_vram_mb", _finite(self.total_vram_mb))
        object.__setattr__(self, "free_vram_mb", _finite(self.free_vram_mb))
        object.__setattr__(self, "temperature_c", _finite(self.temperature_c))
        object.__setattr__(self, "power_w", _finite(self.power_w))
        pressure = self.thermal_pressure if isinstance(self.thermal_pressure, ThermalPressure) else ThermalPressure.from_value(self.thermal_pressure)
        object.__setattr__(self, "thermal_pressure", pressure)
        status = self.status if isinstance(self.status, TelemetryStatus) else self._status(self.status)
        object.__setattr__(self, "status", status)
        if self.device_name is not None:
            object.__setattr__(self, "device_name", str(self.device_name))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        self._derive_memory()

    @staticmethod
    def _percent(value: object) -> float | None:
        number = _finite(value)
        return None if number is None or number > 100 else number

    @staticmethod
    def _status(value: object) -> TelemetryStatus:
        try:
            return TelemetryStatus(str(value).lower())
        except ValueError:
            return TelemetryStatus.PARTIAL

    def _derive_memory(self) -> None:
        total = self.total_vram_mb
        used = self.used_vram_mb
        free = self.free_vram_mb
        if total is not None and used is not None and used > total:
            used = None
        if total is not None and free is not None and free > total:
            free = None
        if free is None and total is not None and used is not None:
            free = max(0.0, total - used)
        if used is None and total is not None and free is not None:
            used = max(0.0, total - free)
        if total is None and used is not None and free is not None:
            total = used + free
        object.__setattr__(self, "used_vram_mb", used)
        object.__setattr__(self, "total_vram_mb", total)
        object.__setattr__(self, "free_vram_mb", free)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        device_index: int = 0,
        timestamp: object = None,
    ) -> GpuTelemetry:
        """Build a sample from common telemetry field names."""
        return _telemetry_mapping(value, device_index, _timestamp(timestamp))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        device_index: int = 0,
        timestamp: object = None,
    ) -> GpuTelemetry:
        """Alias for :meth:`from_mapping`."""
        return cls.from_mapping(value, device_index=device_index, timestamp=timestamp)

    @classmethod
    def from_bytes(
        cls,
        *,
        device_index: int = 0,
        timestamp: object = None,
        utilization_percent: object = None,
        used_vram_bytes: object = None,
        total_vram_bytes: object = None,
        free_vram_bytes: object = None,
        temperature_c: object = None,
        power_mw: object = None,
        thermal_pressure: object = None,
        device_name: str | None = None,
        status: TelemetryStatus | str = TelemetryStatus.AVAILABLE,
        error: str | None = None,
    ) -> GpuTelemetry:
        """Build a sample from NVML byte and milliwatt units."""
        return cls(
            device_index=device_index,
            timestamp=_timestamp(timestamp),
            utilization_percent=utilization_percent,
            used_vram_mb=_bytes_to_mib(used_vram_bytes),
            total_vram_mb=_bytes_to_mib(total_vram_bytes),
            free_vram_mb=_bytes_to_mib(free_vram_bytes),
            temperature_c=temperature_c,
            power_w=_milliwatts_to_watts(power_mw),
            thermal_pressure=thermal_pressure or ThermalPressure.UNAVAILABLE,
            device_name=device_name,
            status=status,
            error=error,
        )

    @classmethod
    def unavailable(
        cls,
        error: str = "telemetry unavailable",
        *,
        device_index: int = 0,
        timestamp: object = None,
    ) -> GpuTelemetry:
        """Create a stable unavailable sample."""
        return cls(
            device_index=device_index,
            timestamp=_timestamp(timestamp),
            status=TelemetryStatus.UNAVAILABLE,
            error=error,
        )

    @property
    def available(self) -> bool:
        """Whether at least one live metric is available."""
        return self.status is not TelemetryStatus.UNAVAILABLE and self.has_data

    @property
    def is_available(self) -> bool:
        """Alias for :attr:`available`."""
        return self.available

    @property
    def partial(self) -> bool:
        """Whether some, but not all, metrics are available."""
        return self.status is TelemetryStatus.PARTIAL and self.has_data

    @property
    def has_data(self) -> bool:
        """Whether any live metric is present."""
        return any(
            value is not None
            for value in (
                self.utilization_percent,
                self.used_vram_mb,
                self.total_vram_mb,
                self.free_vram_mb,
                self.temperature_c,
                self.power_w,
            )
        ) or self.thermal_pressure is not ThermalPressure.UNAVAILABLE

    @property
    def used_vram_bytes(self) -> int | None:
        """Return used VRAM in bytes."""
        return None if self.used_vram_mb is None else int(self.used_vram_mb * _BYTES_PER_MIB)

    @property
    def total_vram_bytes(self) -> int | None:
        """Return total VRAM in bytes."""
        return None if self.total_vram_mb is None else int(self.total_vram_mb * _BYTES_PER_MIB)

    @property
    def free_vram_bytes(self) -> int | None:
        """Return free VRAM in bytes."""
        return None if self.free_vram_mb is None else int(self.free_vram_mb * _BYTES_PER_MIB)

    @property
    def utilization(self) -> float | None:
        """Alias for GPU utilization."""
        return self.utilization_percent

    @property
    def memory_used_mb(self) -> float | None:
        """Alias for used VRAM in MiB."""
        return self.used_vram_mb

    @property
    def memory_total_mb(self) -> float | None:
        """Alias for total VRAM in MiB."""
        return self.total_vram_mb

    @property
    def memory_free_mb(self) -> float | None:
        """Alias for free VRAM in MiB."""
        return self.free_vram_mb

    @property
    def vram_used_mb(self) -> float | None:
        """Alias for used VRAM in MiB."""
        return self.used_vram_mb

    @property
    def vram_total_mb(self) -> float | None:
        """Alias for total VRAM in MiB."""
        return self.total_vram_mb

    @property
    def vram_free_mb(self) -> float | None:
        """Alias for free VRAM in MiB."""
        return self.free_vram_mb

    @property
    def available_vram_mb(self) -> float | None:
        """Alias for free VRAM in MiB."""
        return self.free_vram_mb

    @property
    def gpu_utilization_percent(self) -> float | None:
        """Alias for GPU utilization."""
        return self.utilization_percent

    @property
    def gpu_utilization(self) -> float | None:
        """Alias for GPU utilization."""
        return self.utilization_percent

    @property
    def timestamp_datetime(self) -> datetime:
        """Return the sample time as UTC."""
        return datetime.fromtimestamp(self.timestamp, tz=UTC)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "timestamp": self.timestamp,
            "utilization_percent": self.utilization_percent,
            "used_vram_mb": self.used_vram_mb,
            "total_vram_mb": self.total_vram_mb,
            "free_vram_mb": self.free_vram_mb,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "thermal_pressure": self.thermal_pressure.value,
            "status": self.status.value,
            "available": self.available,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, object]:
        """Alias for :meth:`as_dict`."""
        return self.as_dict()

    def assess_model_safety(
        self,
        model: object,
        *,
        policy: ModelSafetyPolicy | None = None,
        required_vram_mb: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> ModelSafetyAssessment:
        """Assess a model against this sample."""
        return assess_model_safety(
            self,
            model,
            policy=policy,
            required_vram_mb=required_vram_mb,
            width=width,
            height=height,
        )


def _timestamp(value: object) -> float:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return normalized.timestamp()
    timestamp = _finite(value)
    return time.time() if timestamp is None else timestamp


def _model(value: object) -> ModelSize | None:
    if isinstance(value, ModelSize):
        return value
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    compact = normalized.replace("_", "")
    aliases = {
        "tiny": ModelSize.TINY,
        "sam2tiny": ModelSize.TINY,
        "sam21hieratiny": ModelSize.TINY,
        "sam2_1_hiera_tiny": ModelSize.TINY,
        "small": ModelSize.SMALL,
        "sam2small": ModelSize.SMALL,
        "sam21hierasmall": ModelSize.SMALL,
        "sam2_1_hiera_small": ModelSize.SMALL,
        "base": ModelSize.BASE_PLUS,
        "baseplus": ModelSize.BASE_PLUS,
        "base_plus": ModelSize.BASE_PLUS,
        "sam2baseplus": ModelSize.BASE_PLUS,
        "sam21hierabaseplus": ModelSize.BASE_PLUS,
        "sam2_1_hiera_base_plus": ModelSize.BASE_PLUS,
        "large": ModelSize.LARGE,
        "sam2large": ModelSize.LARGE,
        "sam21hieralarge": ModelSize.LARGE,
        "sam2_1_hiera_large": ModelSize.LARGE,
        "sam3": ModelSize.SAM3_1,
        "sam3_1": ModelSize.SAM3_1,
        "sam3_1_multiplex": ModelSize.SAM3_1,
        "sam31": ModelSize.SAM3_1,
        "efficienttam": ModelSize.EFFICIENTTAM_S,
        "efficienttam_s": ModelSize.EFFICIENTTAM_S,
        "efficienttam_ti": ModelSize.EFFICIENTTAM_TI,
        "efficienttam_s_512x512": ModelSize.EFFICIENTTAM_S,
        "efficienttam_ti_512x512": ModelSize.EFFICIENTTAM_TI,
    }
    return aliases.get(normalized, aliases.get(compact))


def coerce_model_size(value: object) -> ModelSize:
    """Return a supported model tier or raise ``ValueError``."""
    model = _model(value)
    if model is None:
        raise ValueError(f"Unknown model: {value!r}")
    return model


class RenderFormat(str, Enum):
    """Output formats used by render estimates."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    EXR = "exr"
    TIFF = "tiff"
    MP4 = "mp4"
    H264 = "h264"
    PRORES = "prores"
    QTRLE = "qtrle"
    AVI = "avi"
    MOV = "mov"
    RGB = "rgb"
    RGBA = "rgba"
    RAW = "raw"


def coerce_render_format(value: object) -> RenderFormat:
    """Return a supported output format or raise ``ValueError``."""
    if isinstance(value, RenderFormat):
        return value
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise TypeError(f"Unknown render format: {value!r}")
    normalized = raw.strip().lower().replace("_", "-")
    if normalized in {"jpg", "image/jpeg", "image/jpg"}:
        return RenderFormat.JPEG
    if normalized in {"tiff", "image/tiff"}:
        return RenderFormat.TIFF
    if normalized in {"h264", "h.264", "avc", "libx264"}:
        return RenderFormat.H264
    if normalized in {"prores-4444", "prores_ks", "prores-ks"}:
        return RenderFormat.PRORES
    try:
        return RenderFormat(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown render format: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ModelSafetyPolicy:
    """Thresholds for conservative model-fit decisions."""

    minimum_headroom_fraction: float = 0.35
    minimum_headroom_mb: float = 1024.0
    high_utilization_percent: float = 90.0
    high_temperature_c: float = 85.0
    critical_temperature_c: float = 95.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_headroom_fraction,
            self.minimum_headroom_mb,
            self.high_utilization_percent,
            self.high_temperature_c,
            self.critical_temperature_c,
        )
        if any(_finite(value) is None for value in values):
            raise ValueError("policy values must be finite non-negative numbers")
        if self.minimum_headroom_fraction > 1 or self.high_utilization_percent > 100:
            raise ValueError("policy percentages must be between zero and 100")
        if self.critical_temperature_c < self.high_temperature_c:
            raise ValueError("critical temperature cannot be below high temperature")

    def required_headroom_mb(
        self,
        free_vram_mb: float | None,
        total_vram_mb: float | None = None,
    ) -> float:
        """Return the required reserve for the current memory state."""
        basis = free_vram_mb if free_vram_mb is not None else total_vram_mb
        if basis is None:
            return self.minimum_headroom_mb
        return max(self.minimum_headroom_mb, basis * self.minimum_headroom_fraction)


class ModelSafetyLevel(str, Enum):
    """Model-fit result."""

    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelSafetyAssessment:
    """Explainable result of checking a model against live VRAM."""

    model: str
    level: ModelSafetyLevel
    safe: bool | None
    required_vram_mb: float | None
    free_vram_mb: float | None
    total_vram_mb: float | None
    headroom_mb: float | None
    required_headroom_mb: float | None
    headroom_fraction: float | None
    utilization_percent: float | None
    temperature_c: float | None
    thermal_pressure: ThermalPressure
    reason: str

    @property
    def is_safe(self) -> bool | None:
        """Return the safety result."""
        return self.safe

    @property
    def can_run(self) -> bool:
        """Whether the model may start safely."""
        return self.safe is True

    @property
    def headroom(self) -> float | None:
        """Alias for raw VRAM headroom."""
        return self.headroom_mb

    @property
    def free_vram(self) -> float | None:
        """Alias for free VRAM."""
        return self.free_vram_mb

    @property
    def should_reduce_quality(self) -> bool:
        """Whether a smaller model or lower resolution is advisable."""
        return self.level in {ModelSafetyLevel.CAUTION, ModelSafetyLevel.UNSAFE}


def _memory_values(value: object) -> tuple[float | None, float | None, float | None]:
    if isinstance(value, GpuTelemetry):
        return value.free_vram_mb, value.total_vram_mb, value.used_vram_mb
    free = _attribute(value, "free_vram_mb", "available_vram_mb", "vram_free_mb")
    total = _attribute(value, "total_vram_mb", "vram_total_mb")
    used = _attribute(value, "used_vram_mb", "vram_used_mb")
    return _finite(free), _finite(total), _finite(used)


def _required_vram(model: ModelSize | None, width: int | None, height: int | None, override: float | None) -> float | None:
    if override is not None:
        number = _finite(override)
        if number is None:
            raise ValueError("required_vram_mb must be finite and non-negative")
        return number
    if model is None:
        return None
    requirements = {
        ModelSize.SAM3_1: 15000.0,
        ModelSize.TINY: 2800.0,
        ModelSize.SMALL: 5000.0,
        ModelSize.BASE_PLUS: 8500.0,
        ModelSize.LARGE: 13500.0,
        ModelSize.CUSTOM: 8500.0,
        ModelSize.EFFICIENTTAM_S: 1800.0,
        ModelSize.EFFICIENTTAM_TI: 1100.0,
    }
    required = requirements[model]
    if width is None and height is None:
        return required
    if width is None or height is None:
        raise ValueError("width and height must be supplied together")
    return required * _positive_int(width, "width") * _positive_int(height, "height") / (1920 * 1080)


def assess_model_safety(
    telemetry: object | None,
    model: object | None = None,
    *,
    model_size: object | None = None,
    policy: ModelSafetyPolicy | None = None,
    required_vram_mb: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ModelSafetyAssessment:
    """Assess whether a model fits current memory and thermal headroom."""
    if model is None:
        model = model_size
    elif model_size is not None:
        raise ValueError("pass only one of model and model_size")
    if model is None:
        raise ValueError("model or model_size is required")
    parsed = _model(model)
    label = parsed.value if parsed is not None else str(getattr(model, "value", model))
    active_policy = policy or ModelSafetyPolicy()
    required = _required_vram(parsed, width, height, required_vram_mb)
    free, total, used = _memory_values(telemetry) if telemetry is not None else (None, None, None)
    if free is None and total is not None and used is not None:
        free = max(0.0, total - used)
    headroom = free - required if free is not None and required is not None else None
    headroom_fraction = headroom / total if headroom is not None and total not in (None, 0) else None
    reserve = active_policy.required_headroom_mb(free, total)
    status = _attribute(telemetry, "status") if telemetry is not None else None
    unavailable = telemetry is None or str(status).lower() == TelemetryStatus.UNAVAILABLE.value
    if required is None:
        level = ModelSafetyLevel.UNKNOWN
        reason = "Unknown model tier."
    elif unavailable or free is None:
        level = ModelSafetyLevel.UNKNOWN
        reason = "Live free VRAM is unavailable; model fit cannot be determined."
    elif free < required:
        level = ModelSafetyLevel.UNSAFE
        reason = f"Requires about {required:.0f} MiB but only {free:.0f} MiB is free."
    elif free < required + reserve:
        level = ModelSafetyLevel.CAUTION
        reason = f"Needs about {required:.0f} MiB; reserve is {reserve:.0f} MiB."
    else:
        level = ModelSafetyLevel.SAFE
        reason = f"Fits with about {headroom:.0f} MiB of headroom after reserve."
    pressure = _attribute(telemetry, "thermal_pressure", "pressure") if telemetry is not None else None
    thermal = ThermalPressure.from_value(pressure) if pressure is not None else ThermalPressure.from_temperature(_attribute(telemetry, "temperature_c") if telemetry is not None else None)
    temperature = _finite(_attribute(telemetry, "temperature_c", "temperature") if telemetry is not None else None)
    utilization = _finite(_attribute(telemetry, "utilization_percent", "utilization") if telemetry is not None else None)
    if thermal is ThermalPressure.CRITICAL or (temperature is not None and temperature >= active_policy.critical_temperature_c):
        if level is not ModelSafetyLevel.UNSAFE:
            level = ModelSafetyLevel.UNSAFE
            reason += " Critical thermal pressure detected."
    elif (
        level is ModelSafetyLevel.SAFE
        and (
            thermal in {ThermalPressure.HIGH, ThermalPressure.MEDIUM}
            or (temperature is not None and temperature >= active_policy.high_temperature_c)
            or (utilization is not None and utilization >= active_policy.high_utilization_percent)
        )
    ):
        level = ModelSafetyLevel.CAUTION
        reason += " Current utilization or thermal state is elevated."
    return ModelSafetyAssessment(
        model=label,
        level=level,
        safe=None if level is ModelSafetyLevel.UNKNOWN else level is ModelSafetyLevel.SAFE,
        required_vram_mb=required,
        free_vram_mb=free,
        total_vram_mb=total,
        headroom_mb=headroom,
        required_headroom_mb=reserve,
        headroom_fraction=headroom_fraction,
        utilization_percent=utilization,
        temperature_c=temperature,
        thermal_pressure=thermal,
        reason=reason,
    )


def largest_safe_model(
    telemetry: object | None,
    *,
    policy: ModelSafetyPolicy | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ModelSize | None:
    """Return the largest model currently considered safe."""
    for model in (
        ModelSize.SAM3_1,
        ModelSize.LARGE,
        ModelSize.BASE_PLUS,
        ModelSize.SMALL,
        ModelSize.TINY,
        ModelSize.EFFICIENTTAM_S,
        ModelSize.EFFICIENTTAM_TI,
    ):
        if assess_model_safety(telemetry, model, policy=policy, width=width, height=height).can_run:
            return model
    return None


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """Coefficients for frame-based render estimates."""

    reference_width: int = 1920
    reference_height: int = 1080
    reference_fps: float = 24.0
    resolution_exponent: float = 1.0
    fps_exponent: float = 0.0
    startup_seconds: float = 0.0
    shutdown_seconds: float = 0.0
    base_seconds_per_frame: Mapping[ModelSize, float] = field(
        default_factory=lambda: {
        ModelSize.SAM3_1: 0.45,
        ModelSize.TINY: 0.06,
        ModelSize.SMALL: 0.10,
        ModelSize.BASE_PLUS: 0.18,
        ModelSize.LARGE: 0.30,
        ModelSize.CUSTOM: 0.18,
        ModelSize.EFFICIENTTAM_S: 0.03,
        ModelSize.EFFICIENTTAM_TI: 0.02,
        }
    )
    format_multipliers: Mapping[RenderFormat, float] = field(
        default_factory=lambda: {
            RenderFormat.PNG: 1.35,
            RenderFormat.JPEG: 0.75,
            RenderFormat.WEBP: 0.95,
            RenderFormat.EXR: 2.20,
            RenderFormat.TIFF: 1.40,
            RenderFormat.MP4: 0.85,
            RenderFormat.H264: 0.90,
            RenderFormat.PRORES: 1.80,
            RenderFormat.QTRLE: 1.35,
            RenderFormat.AVI: 1.00,
            RenderFormat.MOV: 0.95,
            RenderFormat.RGB: 0.80,
            RenderFormat.RGBA: 1.00,
            RenderFormat.RAW: 1.10,
        }
    )

    def __post_init__(self) -> None:
        _positive_int(self.reference_width, "reference_width")
        _positive_int(self.reference_height, "reference_height")
        if _finite(self.reference_fps) is None or self.reference_fps <= 0:
            raise ValueError("reference_fps must be positive")
        for value in (self.resolution_exponent, self.fps_exponent, self.startup_seconds, self.shutdown_seconds):
            if _finite(value) is None:
                raise ValueError("render coefficients must be finite")
        for value in self.base_seconds_per_frame.values():
            if _finite(value) is None or value <= 0:
                raise ValueError("model frame costs must be positive")
        for value in self.format_multipliers.values():
            if _finite(value) is None or value <= 0:
                raise ValueError("format multipliers must be positive")


@dataclass(frozen=True, slots=True)
class RenderEstimate:
    """Deterministic render and output duration estimate."""

    frame_count: int
    fps: float
    width: int
    height: int
    model: ModelSize
    format: RenderFormat
    seconds_per_frame: float
    render_seconds: float
    output_duration_seconds: float
    estimated_render_fps: float
    resolution_scale: float

    @property
    def resolution(self) -> tuple[int, int]:
        """Return target width and height."""
        return self.width, self.height

    @property
    def total_seconds(self) -> float:
        """Alias for render duration."""
        return self.render_seconds

    @property
    def duration_seconds(self) -> float:
        """Alias for output duration."""
        return self.output_duration_seconds

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "model": self.model.value,
            "format": self.format.value,
            "seconds_per_frame": self.seconds_per_frame,
            "render_seconds": self.render_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "estimated_render_fps": self.estimated_render_fps,
            "resolution_scale": self.resolution_scale,
        }

    def to_dict(self) -> dict[str, object]:
        """Alias for :meth:`as_dict`."""
        return self.as_dict()


def _estimate(
    frame_count: object,
    fps: object,
    width: object,
    height: object,
    model: object,
    output_format: object,
    settings: RenderSettings,
    startup_seconds: float | None,
    shutdown_seconds: float | None,
) -> RenderEstimate:
    count = _non_negative_int(frame_count, "frame_count")
    if count <= 0:
        raise ValueError("frame_count must be positive")
    frame_rate = _finite(fps)
    if frame_rate is None or frame_rate <= 0:
        raise ValueError("fps must be positive")
    target_width = _positive_int(width, "width")
    target_height = _positive_int(height, "height")
    parsed_model = coerce_model_size(model)
    parsed_format = coerce_render_format(output_format)
    scale = (target_width * target_height / (settings.reference_width * settings.reference_height)) ** settings.resolution_exponent
    per_frame = float(settings.base_seconds_per_frame[parsed_model]) * scale * (frame_rate / settings.reference_fps) ** settings.fps_exponent * float(settings.format_multipliers[parsed_format])
    fixed_start = settings.startup_seconds if startup_seconds is None else startup_seconds
    fixed_stop = settings.shutdown_seconds if shutdown_seconds is None else shutdown_seconds
    render_seconds = count * per_frame + fixed_start + fixed_stop
    output_duration = count / frame_rate
    return RenderEstimate(count, frame_rate, target_width, target_height, parsed_model, parsed_format, per_frame, render_seconds, output_duration, count / render_seconds if render_seconds > 0 else 0.0, scale)


def estimate_render_time(
    frame_count: int,
    fps: float,
    width: int = 1920,
    height: int = 1080,
    model: object = ModelSize.TINY,
    format: object = RenderFormat.PNG,
    *,
    output_format: object | None = None,
    settings: RenderSettings | None = None,
    resolution: tuple[int, int] | list[int] | str | None = None,
    startup_seconds: float | None = None,
    shutdown_seconds: float | None = None,
) -> RenderEstimate:
    """Estimate render time from frame count, resolution, model, and format."""
    if output_format is not None:
        if format is not RenderFormat.PNG:
            raise ValueError("pass only one of format and output_format")
        format = output_format
    if resolution is not None:
        if width != 1920 or height != 1080:
            raise ValueError("resolution cannot be combined with explicit dimensions")
        width, height = _resolution(resolution)
    return _estimate(frame_count, fps, width, height, model, format, settings or RenderSettings(), startup_seconds, shutdown_seconds)


def _resolution(value: tuple[int, int] | list[int] | str) -> tuple[int, int]:
    parts = value.lower().replace("×", "x").split("x") if isinstance(value, str) else list(value)
    if len(parts) != 2:
        raise ValueError("resolution must contain width and height")
    return _positive_int(parts[0], "width"), _positive_int(parts[1], "height")


def estimate_render_time_for_resolution(
    frame_count: int,
    fps: float,
    resolution: tuple[int, int] | list[int] | str,
    model: object,
    format: object,
    **kwargs: Any,
) -> RenderEstimate:
    """Estimate render time using a resolution tuple or string."""
    width, height = _resolution(resolution)
    return estimate_render_time(frame_count, fps, width, height, model, format, **kwargs)


class RenderTimeEstimator:
    """Reusable wrapper around :func:`estimate_render_time`."""

    def __init__(self, settings: RenderSettings | None = None) -> None:
        self.settings = settings or RenderSettings()

    def estimate(
        self,
        frame_count: int,
        fps: float,
        width: int = 1920,
        height: int = 1080,
        model: object = ModelSize.TINY,
        format: object = RenderFormat.PNG,
        *,
        resolution: tuple[int, int] | list[int] | str | None = None,
        startup_seconds: float | None = None,
        shutdown_seconds: float | None = None,
    ) -> RenderEstimate:
        """Return a configured estimate."""
        if resolution is not None:
            width, height = _resolution(resolution)
        return _estimate(frame_count, fps, width, height, model, format, self.settings, startup_seconds, shutdown_seconds)

    def __call__(self, *args: Any, **kwargs: Any) -> RenderEstimate:
        """Delegate to :meth:`estimate`."""
        return self.estimate(*args, **kwargs)


class GpuSampler:
    """Reusable optional-NVML sampler with explicit cleanup."""

    def __init__(
        self,
        device_index: int = 0,
        *,
        nvml_module: Any | None = None,
        nvml: Any | None = None,
        nvml_factory: Callable[[], Any] | None = None,
        telemetry_provider: Callable[..., GpuTelemetry | Mapping[str, object]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if sum(value is not None for value in (nvml_module, nvml, nvml_factory)) > 1:
            raise ValueError("pass only one NVML source")
        if device_index < 0 or int(device_index) != device_index:
            raise ValueError("device_index must be a non-negative integer")
        self.device_index = int(device_index)
        self._injected_nvml = nvml_module or nvml
        self._nvml_factory = nvml_factory
        self._provider = telemetry_provider
        self._clock = clock
        self._nvml = self._injected_nvml
        self._handle: Any | None = None
        self._initialized = False
        self._attempted = False
        self._owns_session = False
        self._closed = False
        self._lock = threading.RLock()
        self._last_error: str | None = None

    @property
    def initialized(self) -> bool:
        """Whether a backend handle is ready."""
        with self._lock:
            return self._initialized

    @property
    def available(self) -> bool:
        """Whether the sampler can be queried."""
        with self._lock:
            return not self._closed and (self._provider is not None or self._initialized)

    @property
    def last_error(self) -> str | None:
        """Return the most recent backend error."""
        with self._lock:
            return self._last_error

    def start(self) -> bool:
        """Initialize the backend and return whether it is ready."""
        with self._lock:
            if self._closed:
                return False
            if self._provider is not None:
                return True
            return self._ensure_ready()

    def sample(self) -> GpuTelemetry:
        """Collect one sample without raising backend errors."""
        with self._lock:
            timestamp = _timestamp(self._clock())
            if self._closed:
                return GpuTelemetry.unavailable("sampler is closed", device_index=self.device_index, timestamp=timestamp)
            if self._provider is not None:
                try:
                    value = self._provider()
                except TypeError:
                    try:
                        value = self._provider(GpuTelemetry.unavailable(timestamp=timestamp))
                    except _BACKEND_ERRORS as exc:
                        return self._provider_error(exc, timestamp)
                except _BACKEND_ERRORS as exc:
                    return self._provider_error(exc, timestamp)
                if isinstance(value, GpuTelemetry):
                    return value
                if isinstance(value, Mapping):
                    return GpuTelemetry.from_mapping(value, device_index=self.device_index, timestamp=timestamp)
                return self._provider_error(TypeError("provider must return telemetry or a mapping"), timestamp)
            if not self._ensure_ready():
                return GpuTelemetry.unavailable(self._last_error or "NVML unavailable", device_index=self.device_index, timestamp=timestamp)
            return self._sample_nvml(timestamp)

    def read(self) -> GpuTelemetry:
        """Alias for :meth:`sample`."""
        return self.sample()

    def get_telemetry(self) -> GpuTelemetry:
        """Alias for :meth:`sample`."""
        return self.sample()

    def assess_model_safety(self, model: object, **kwargs: Any) -> ModelSafetyAssessment:
        """Sample and assess a model in one call."""
        return assess_model_safety(self.sample(), model, **kwargs)

    def close(self) -> None:
        """Release the backend handle and session."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._initialized and self._owns_session and self._nvml is not None:
                shutdown = getattr(self._nvml, "nvmlShutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except _BACKEND_ERRORS:
                        self._last_error = "NVML shutdown failed"
            self._handle = None
            self._initialized = False
            self._owns_session = False

    def stop(self) -> None:
        """Alias for :meth:`close`."""
        self.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def _provider_error(self, exc: Exception, timestamp: float) -> GpuTelemetry:
        self._last_error = f"injected telemetry failed: {exc}"
        return GpuTelemetry.unavailable(self._last_error, device_index=self.device_index, timestamp=timestamp)

    def _ensure_ready(self) -> bool:
        if self._initialized:
            return True
        if self._closed or self._attempted:
            return False
        self._attempted = True
        if self._nvml is None:
            if self._nvml_factory is not None:
                try:
                    self._nvml = self._nvml_factory()
                except _BACKEND_ERRORS as exc:
                    self._last_error = f"NVML factory failed: {exc}"
                    return False
            else:
                try:
                    self._nvml = importlib.import_module("pynvml")
                except ImportError:
                    try:
                        self._nvml = importlib.import_module("nvidia_ml_py")
                    except ImportError as exc:
                        self._last_error = f"NVML unavailable: {exc}"
                        return False
        init = getattr(self._nvml, "nvmlInit", None)
        handle_factory = getattr(self._nvml, "nvmlDeviceGetHandleByIndex", None)
        if not callable(init) or not callable(handle_factory):
            self._last_error = "NVML module does not provide init and handle lookup"
            return False
        try:
            init()
        except _BACKEND_ERRORS as exc:
            if "already initialized" in str(exc).lower():
                self._owns_session = False
            else:
                self._last_error = f"NVML initialization failed: {exc}"
                return False
        else:
            self._owns_session = True
        try:
            self._handle = handle_factory(self.device_index)
        except _BACKEND_ERRORS as exc:
            self._last_error = f"NVML device lookup failed: {exc}"
            self._shutdown_failed_start()
            return False
        if self._handle is None:
            self._last_error = "NVML returned an empty device handle"
            self._shutdown_failed_start()
            return False
        self._initialized = True
        self._last_error = None
        return True

    def _shutdown_failed_start(self) -> None:
        if self._owns_session and self._nvml is not None:
            shutdown = getattr(self._nvml, "nvmlShutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except _BACKEND_ERRORS:
                    pass
        self._handle = None
        self._initialized = False
        self._owns_session = False

    def _sample_nvml(self, timestamp: float) -> GpuTelemetry:
        errors: list[str] = []
        for optional_name in (
            "nvmlDeviceGetUtilizationRates",
            "nvmlDeviceGetMemoryInfo",
            "nvmlDeviceGetTemperature",
            "nvmlDeviceGetPowerUsage",
            "nvmlDeviceGetThermalPressure",
        ):
            if not callable(getattr(self._nvml, optional_name, None)):
                errors.append(f"{optional_name} unavailable")
        name = self._read(self._nvml, "nvmlDeviceGetName", self._handle, errors, decode=True)
        rates = self._read(self._nvml, "nvmlDeviceGetUtilizationRates", self._handle, errors)
        utilization = _finite(_attribute(rates, "gpu", "utilization"))
        if utilization is not None and utilization > 100:
            utilization = None
        memory = self._read(self._nvml, "nvmlDeviceGetMemoryInfo", self._handle, errors)
        used = _bytes_to_mib(_attribute(memory, "used", "used_vram_bytes"))
        total = _bytes_to_mib(_attribute(memory, "total", "total_vram_bytes"))
        free = _bytes_to_mib(_attribute(memory, "free", "free_vram_bytes"))
        sensor = getattr(self._nvml, "NVML_TEMPERATURE_GPU", 0)
        temperature = self._read(self._nvml, "nvmlDeviceGetTemperature", self._handle, errors, extra=sensor)
        power_raw = self._read(self._nvml, "nvmlDeviceGetPowerUsage", self._handle, errors)
        pressure_raw = self._read(self._nvml, "nvmlDeviceGetThermalPressure", self._handle, errors)
        pressure = ThermalPressure.from_value(pressure_raw) if pressure_raw is not None else ThermalPressure.from_temperature(temperature)
        values = (utilization, used, total, free, temperature, _milliwatts_to_watts(power_raw), pressure)
        has_data = any(value is not None for value in values[:6]) or pressure is not ThermalPressure.UNAVAILABLE
        status = TelemetryStatus.UNAVAILABLE if not has_data else TelemetryStatus.PARTIAL if errors else TelemetryStatus.AVAILABLE
        self._last_error = "; ".join(errors) if errors else None
        return GpuTelemetry(device_index=self.device_index, timestamp=timestamp, utilization_percent=utilization, used_vram_mb=used, total_vram_mb=total, free_vram_mb=free, temperature_c=temperature, power_w=_milliwatts_to_watts(power_raw), thermal_pressure=pressure, device_name=str(name) if name is not None else None, status=status, error=self._last_error)

    @staticmethod
    def _read(module: Any, name: str, handle: Any, errors: list[str], extra: Any = None, decode: bool = False) -> object | None:
        function = getattr(module, name, None)
        if not callable(function):
            return None
        try:
            value = function(handle) if extra is None else function(handle, extra)
            if decode and isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value
        except _BACKEND_ERRORS as exc:
            errors.append(f"{name} failed: {exc}")
            return None


class GpuMonitor:
    """Stateful facade around :class:`GpuSampler`."""

    def __init__(self, device_index: int = 0, **sampler_options: Any) -> None:
        self.sampler = GpuSampler(device_index, **sampler_options)
        self._latest: GpuTelemetry | None = None

    @property
    def latest(self) -> GpuTelemetry | None:
        """Return the latest sample."""
        return self._latest

    @property
    def available(self) -> bool:
        """Whether the latest sample has live data."""
        return self._latest is not None and self._latest.available

    def sample(self) -> GpuTelemetry:
        """Sample and retain telemetry."""
        self._latest = self.sampler.sample()
        return self._latest

    def read(self) -> GpuTelemetry:
        """Alias for :meth:`sample`."""
        return self.sample()

    def assess_model_safety(self, model: object, **kwargs: Any) -> ModelSafetyAssessment:
        """Sample and assess a model."""
        return assess_model_safety(self.sample(), model, **kwargs)

    def close(self) -> None:
        """Close the sampler."""
        self.sampler.close()

    def stop(self) -> None:
        """Alias for :meth:`close`."""
        self.close()

    def __enter__(self) -> Self:
        self.sampler.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


GPUSampler = GpuSampler
GPUTelemetry = GpuTelemetry
GpuTelemetrySnapshot = GpuTelemetry
Telemetry = GpuTelemetry
GPUMonitor = GpuMonitor
NvidiaGpuMonitor = GpuMonitor
NvidiaGpuSampler = GpuSampler

__all__ = [
    "GpuMonitor",
    "GpuSampler",
    "GpuTelemetry",
    "GpuTelemetrySnapshot",
    "ModelSafetyAssessment",
    "ModelSafetyLevel",
    "ModelSafetyPolicy",
    "ModelSize",
    "RenderEstimate",
    "RenderFormat",
    "RenderSettings",
    "RenderTimeEstimator",
    "TelemetryStatus",
    "ThermalPressure",
    "assess_model_safety",
    "coerce_model_size",
    "coerce_render_format",
    "estimate_render_time",
    "estimate_render_time_for_resolution",
    "largest_safe_model",
]
