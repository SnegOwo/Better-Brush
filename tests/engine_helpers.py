"""Shared helpers for the headless-engine tests."""

from __future__ import annotations

from src.engine.schemas import JobResult


class SyncStarter:
    """Run the job body inline so assertions stay deterministic."""

    def __init__(self, target, name: str) -> None:
        self._target = target

    def start(self) -> None:
        self._target()

    def join(self, timeout: float | None = None) -> None:
        return None


def make_result(**overrides) -> JobResult:
    base = {
        "output_path": "C:/tmp/alpha.mov",
        "output_kind": "mov_alpha",
        "frame_in": 0,
        "frame_out": 9,
        "frame_count": 10,
        "fps": 24.0,
        "width": 320,
        "height": 240,
        "model": "sam2.1_hiera_base_plus",
    }
    base.update(overrides)
    return JobResult(**base)
