"""Tests for the diagnostics console and model-capacity helper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from src.hardware.detector import Backend, GPUInfo, HardwareProfile
from src.hardware.profiler import ModelSize, SAM2Profiler
from src.ui.components.diagnostics_console import DiagnosticsConsole

_APP = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def test_console_submits_commands_and_renders_output() -> None:
    _app()
    console = DiagnosticsConsole()
    commands: list[str] = []
    console.command_submitted.connect(commands.append)

    console._input.setText("gpu")
    console._submit()

    assert commands == ["gpu"]
    assert "> gpu" in console._output.toPlainText()
    assert console._btn_run.text() == "Run"
    console.close()


def _profile(vram_mb: int) -> HardwareProfile:
    return HardwareProfile(
        cpu_name="Test CPU",
        cpu_count_physical=8,
        cpu_count_logical=16,
        ram_total_mb=32768,
        ram_available_mb=16384,
        gpus=[
            GPUInfo(
                index=0,
                name="Test GPU",
                total_vram_mb=vram_mb,
                available_vram_mb=vram_mb,
                backend=Backend.CUDA,
            )
        ],
    )


def test_largest_model_uses_capacity_not_recommendation() -> None:
    profiler = SAM2Profiler()
    assert profiler.largest_model_for_profile(_profile(16000)) == ModelSize.SAM3_1
    assert profiler.largest_model_for_profile(_profile(5000)) == ModelSize.SMALL
    assert profiler.largest_model_for_profile(_profile(1000)) is None


def test_largest_model_without_gpu_is_unknown() -> None:
    profile = HardwareProfile(
        cpu_name="Test CPU",
        cpu_count_physical=2,
        cpu_count_logical=4,
        ram_total_mb=8192,
        ram_available_mb=4096,
    )
    assert SAM2Profiler().largest_model_for_profile(profile) is None


if __name__ == "__main__":
    _app()
    test_console_submits_commands_and_renders_output()
    test_largest_model_uses_capacity_not_recommendation()
    test_largest_model_without_gpu_is_unknown()
