"""Hardware-independent tests for optional GPU monitoring helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.hardware.monitor import (
    GpuSampler,
    GpuTelemetry,
    ModelSafetyLevel,
    ModelSize,
    RenderFormat,
    ThermalPressure,
    assess_model_safety,
    coerce_model_size,
    estimate_render_time,
)


class _FakeNVML:
    """Small NVML-compatible test double."""

    def __init__(self) -> None:
        self.init_calls = 0
        self.handle_calls = 0
        self.shutdown_calls = 0
        self.handle = object()

    def nvmlInit(self) -> None:
        """Record initialization."""

        self.init_calls += 1

    def nvmlDeviceGetHandleByIndex(self, _index: int) -> object:
        """Return a stable device handle."""

        self.handle_calls += 1
        return self.handle

    def nvmlShutdown(self) -> None:
        """Record shutdown."""

        self.shutdown_calls += 1

    def nvmlDeviceGetName(self, _handle: object) -> bytes:
        """Return a test GPU name."""

        return b"Fake NVIDIA GPU"

    def nvmlDeviceGetUtilizationRates(self, _handle: object) -> SimpleNamespace:
        """Return utilization metrics."""

        return SimpleNamespace(gpu=37, memory=12)

    def nvmlDeviceGetMemoryInfo(self, _handle: object) -> SimpleNamespace:
        """Return memory metrics in bytes."""

        return SimpleNamespace(
            used=2 * 1024 * 1024 * 1024,
            total=8 * 1024 * 1024 * 1024,
            free=6 * 1024 * 1024 * 1024,
        )

    def nvmlDeviceGetTemperature(self, _handle: object, _sensor: int) -> int:
        """Return a safe temperature."""

        return 61

    def nvmlDeviceGetPowerUsage(self, _handle: object) -> int:
        """Return power in milliwatts."""

        return 125000

    def nvmlDeviceGetThermalPressure(self, _handle: object) -> str:
        """Return a direct pressure value."""

        return "low"


def test_fake_nvml_returns_live_metrics_and_releases_handle() -> None:
    """The sampler reads all metrics and owns exactly one lifecycle."""

    fake = _FakeNVML()
    sampler = GpuSampler(nvml_module=fake, clock=lambda: 123.0)

    first = sampler.sample()
    second = sampler.sample()

    assert first.available
    assert first.timestamp == 123.0
    assert first.utilization_percent == 37
    assert first.used_vram_mb == 2048
    assert first.total_vram_mb == 8192
    assert first.free_vram_mb == 6144
    assert first.temperature_c == 61
    assert first.power_w == 125
    assert first.thermal_pressure is ThermalPressure.LOW
    assert second.available
    assert fake.init_calls == 1
    assert fake.handle_calls == 1

    sampler.close()
    sampler.close()
    assert fake.shutdown_calls == 1


def test_missing_nvml_is_a_graceful_unavailable_snapshot() -> None:
    """A sampler without a usable backend never raises to its caller."""

    sampler = GpuSampler(nvml_module=SimpleNamespace(nvmlInit=lambda: None))
    snapshot = sampler.sample()

    assert not snapshot.available
    assert snapshot.status.value == "unavailable"
    assert snapshot.utilization_percent is None
    assert snapshot.total_vram_mb is None
    assert snapshot.error


def test_injected_provider_can_return_a_zero_argument_snapshot() -> None:
    """Tests and alternate backends can inject zero-argument telemetry."""

    expected = GpuTelemetry(
        timestamp=10.0,
        free_vram_mb=4000,
        total_vram_mb=8000,
        used_vram_mb=4000,
    )
    sampler = GpuSampler(telemetry_provider=lambda: expected)

    assert sampler.sample() is expected


def test_partial_nvml_queries_preserve_memory_metrics() -> None:
    """Unsupported optional metrics do not hide available memory data."""

    fake = _FakeNVML()
    fake.nvmlDeviceGetTemperature = None
    fake.nvmlDeviceGetPowerUsage = None
    fake.nvmlDeviceGetThermalPressure = None
    sampler = GpuSampler(nvml_module=fake, clock=lambda: 20.0)

    snapshot = sampler.sample()

    assert snapshot.partial
    assert snapshot.free_vram_mb == 6144
    assert snapshot.temperature_c is None
    assert snapshot.power_w is None
    assert snapshot.thermal_pressure is ThermalPressure.UNAVAILABLE
    sampler.close()


def test_model_safety_requires_fit_and_reserve() -> None:
    """Safety uses current free VRAM and a configurable reserve."""

    telemetry = GpuTelemetry(
        free_vram_mb=6_500,
        total_vram_mb=10_000,
        used_vram_mb=3_500,
    )
    caution = assess_model_safety(telemetry, ModelSize.SMALL)
    unsafe = assess_model_safety(telemetry, ModelSize.LARGE)
    safe = assess_model_safety(
        GpuTelemetry(
            free_vram_mb=12_000,
            total_vram_mb=12_288,
            used_vram_mb=288,
        ),
        ModelSize.SMALL,
    )

    assert caution.level is ModelSafetyLevel.CAUTION
    assert caution.headroom_mb == 1_500
    assert caution.required_headroom_mb == pytest.approx(2_275)
    assert unsafe.level is ModelSafetyLevel.UNSAFE
    assert safe.level is ModelSafetyLevel.SAFE
    assert safe.can_run


def test_model_safety_downgrades_under_thermal_pressure() -> None:
    """A fitting model is not reported safe during critical pressure."""

    telemetry = GpuTelemetry(
        free_vram_mb=12_000,
        total_vram_mb=12_288,
        thermal_pressure=ThermalPressure.CRITICAL,
    )
    assessment = assess_model_safety(telemetry, ModelSize.SMALL)

    assert assessment.level is ModelSafetyLevel.UNSAFE
    assert not assessment.can_run


def test_model_safety_unknown_without_live_memory() -> None:
    """Missing telemetry produces an explicit unknown result."""

    assessment = assess_model_safety(None, ModelSize.TINY)

    assert assessment.level is ModelSafetyLevel.UNKNOWN
    assert assessment.safe is None
    assert assessment.free_vram_mb is None


def test_sam3_1_model_name_and_vram_policy() -> None:
    assert coerce_model_size("sam3.1_multiplex") is ModelSize.SAM3_1
    telemetry = GpuTelemetry(
        free_vram_mb=24_000,
        total_vram_mb=24_000,
        used_vram_mb=0,
    )
    assessment = assess_model_safety(telemetry, ModelSize.SAM3_1)
    assert assessment.required_vram_mb == 15_000
    assert assessment.level is ModelSafetyLevel.SAFE


def test_render_estimate_uses_frames_resolution_model_and_format() -> None:
    """The pure estimator responds to all requested planning inputs."""

    baseline = estimate_render_time(100, 30, 1920, 1080, "tiny", "png")
    larger_model = estimate_render_time(100, 30, 1920, 1080, "large", "png")
    larger_resolution = estimate_render_time(100, 30, 3840, 2160, "tiny", "png")
    cheaper_format = estimate_render_time(100, 30, 1920, 1080, "tiny", "jpeg")

    assert baseline.output_duration_seconds == pytest.approx(100 / 30)
    assert baseline.render_seconds > baseline.output_duration_seconds
    assert larger_model.render_seconds > baseline.render_seconds
    assert larger_resolution.render_seconds > baseline.render_seconds
    assert cheaper_format.render_seconds < baseline.render_seconds
    assert baseline.format is RenderFormat.PNG
    assert baseline.model is ModelSize.TINY


def test_render_estimate_rejects_invalid_inputs() -> None:
    """Invalid planning inputs fail clearly before any hardware access."""

    with pytest.raises(ValueError):
        estimate_render_time(0, 0, 1920, 1080, "tiny", "png")
    with pytest.raises(ValueError):
        estimate_render_time(10, 24, 0, 1080, "tiny", "png")
    with pytest.raises(ValueError):
        estimate_render_time(10, 24, 1920, 1080, "unknown", "png")
