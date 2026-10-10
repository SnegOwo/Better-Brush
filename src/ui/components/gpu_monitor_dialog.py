"""Live GPU performance and safety monitor dialog."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.hardware.monitor import (
    GpuMonitor,
    ModelSafetyLevel,
    ThermalPressure,
    assess_model_safety,
    estimate_render_time,
)


class GpuMonitorDialog(QDialog):
    """Display live GPU metrics, render estimates, and thermal warnings."""

    def __init__(
        self,
        parent: QWidget | None = None,
        model: object = "sam2.1_hiera_tiny",
        frame_count: int = 0,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPU Performance Monitor")
        self.setModal(False)
        self.resize(620, 500)
        self._model = model
        self._frame_count = max(0, int(frame_count))
        self._fps = max(0.001, float(fps))
        self._width = max(1, int(width))
        self._height = max(1, int(height))
        self._monitor = GpuMonitor()
        self._build_ui()
        self._sample()
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._sample)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)
        title = QLabel("GPU Performance Monitor")
        title.setStyleSheet("font-size: 17px; font-weight: 600; color: #d4d4d8;")
        root.addWidget(title)
        intro = QLabel(
            "Live utilization, VRAM, temperature, and a conservative model-fit warning. "
            "Sampling is optional and never blocks the editor."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9aa4; font-size: 11px;")
        root.addWidget(intro)

        self._device = QLabel("GPU: waiting for telemetry")
        self._device.setStyleSheet("color: #d4d4d8; font-size: 12px;")
        root.addWidget(self._device)

        grid = QGridLayout()
        self._values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate(
            (
                ("utilization", "GPU utilization"),
                ("vram", "VRAM"),
                ("temperature", "Temperature"),
                ("power", "Power"),
                ("pressure", "Thermal pressure"),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            value = QLabel("—")
            value.setStyleSheet("color: #a0a0a8; font-size: 11px;")
            self._values[key] = value
            grid.addWidget(value, row, 1)
        root.addLayout(grid)

        self._warning = QLabel()
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #dfa345; font-size: 11px;")
        root.addWidget(self._warning)

        self._estimate_title = QLabel("Render estimate")
        self._estimate_title.setStyleSheet("color: #d4d4d8; font-size: 13px; font-weight: 600;")
        root.addWidget(self._estimate_title)
        self._estimate = QLabel("No frame count available")
        self._estimate.setWordWrap(True)
        self._estimate.setStyleSheet("color: #a0a0a8; font-size: 11px;")
        root.addWidget(self._estimate)
        self._estimate_bar = QProgressBar()
        self._estimate_bar.setRange(0, 100)
        self._estimate_bar.setValue(0)
        self._estimate_bar.setTextVisible(False)
        root.addWidget(self._estimate_bar)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #6b6b74; font-size: 10px;")
        root.addWidget(self._status)
        root.addStretch()

        actions = QHBoxLayout()
        actions.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._sample)
        actions.addWidget(refresh)
        close = QPushButton("Close")
        close.setProperty("cssClass", "accent")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        root.addLayout(actions)

    def _sample(self) -> None:
        telemetry = self._monitor.sample()
        self._device.setText(
            f"GPU: {telemetry.device_name or 'unknown device'} · {telemetry.status.value}"
        )
        self._values["utilization"].setText(
            "—" if telemetry.utilization_percent is None else f"{telemetry.utilization_percent:.0f}%"
        )
        if telemetry.total_vram_mb is None or telemetry.used_vram_mb is None:
            self._values["vram"].setText("—")
        else:
            self._values["vram"].setText(
                f"{telemetry.used_vram_mb:,.0f} / {telemetry.total_vram_mb:,.0f} MB"
            )
        self._values["temperature"].setText(
            "—" if telemetry.temperature_c is None else f"{telemetry.temperature_c:.0f} °C"
        )
        self._values["power"].setText(
            "—" if telemetry.power_w is None else f"{telemetry.power_w:.1f} W"
        )
        pressure = telemetry.thermal_pressure
        pressure_label = (
            "unavailable"
            if pressure is ThermalPressure.UNAVAILABLE
            else pressure.value
        )
        self._values["pressure"].setText(pressure_label)
        self._status.setText(telemetry.error or "Telemetry updates every two seconds.")

        assessment = assess_model_safety(
            telemetry,
            self._model,
            width=self._width,
            height=self._height,
        )
        if assessment.level is ModelSafetyLevel.UNKNOWN:
            self._warning.setText(f"Model safety unknown. {assessment.reason}")
        elif assessment.level is ModelSafetyLevel.SAFE:
            self._warning.setText(f"Model fit: safe. {assessment.reason}")
        else:
            self._warning.setText(f"Model fit: {assessment.level.value}. {assessment.reason}")

        if self._frame_count > 0:
            try:
                estimate = estimate_render_time(
                    self._frame_count,
                    self._fps,
                    self._width,
                    self._height,
                    self._model,
                    "prores",
                )
                self._estimate.setText(
                    f"{self._frame_count:,} frames · about {self._format_seconds(estimate.render_seconds)} "
                    f"· estimated {estimate.estimated_render_fps:.1f} fps"
                )
                self._estimate_bar.setValue(min(100, int(min(100.0, assessment.headroom_fraction * 100.0)) if assessment.headroom_fraction is not None else 0))
            except ValueError as exc:
                self._estimate.setText(str(exc))
        else:
            self._estimate.setText("Open a video to estimate render time")

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, remainder = divmod(int(seconds), 60)
        return f"{minutes}m {remainder}s"

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._monitor.close()
        super().closeEvent(event)


__all__ = ["GpuMonitorDialog"]
