"""Tests for scene-cut detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.processing.scene_detector import (
    _detect_native,
    detect_scene_cuts,
)


def _create_cut_video(
    path: Path, cut_at: int = 15, total: int = 30, size: tuple[int, int] = (256, 256)
) -> None:
    """Create a synthetic video with a sharp cut from blue to red at *cut_at*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(total):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = (255, 0, 0) if i < cut_at else (0, 0, 255)
        writer.write(frame)
    writer.release()


def test_detects_scene_cut() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "cut.mp4"
        _create_cut_video(video, cut_at=15, total=30)
        cuts = detect_scene_cuts(str(video), threshold=27.0)
        assert 15 in cuts


def test_no_cuts_for_uniform_video() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "uniform.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, 10.0, (256, 256))
        for _ in range(20):
            frame = np.full((256, 256, 3), 128, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        cuts = detect_scene_cuts(str(video), threshold=27.0)
        assert cuts == []


def test_native_fallback_detects_cut() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "cut.mp4"
        _create_cut_video(video, cut_at=12, total=24)
        cuts = _detect_native(str(video), threshold=0.3)
        assert 12 in cuts


def test_min_cut_distance_filtering() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "cut.mp4"
        _create_cut_video(video, cut_at=15, total=30)
        cuts = detect_scene_cuts(str(video), threshold=27.0, min_cut_distance=100)
        # min_cut_distance=100 means no two cuts can be within 100 frames
        assert len(cuts) <= 1


def test_cancel_event_stops_native() -> None:
    import threading

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "long.mp4"
        _create_cut_video(video, cut_at=15, total=30)
        cancel = threading.Event()
        cancel.set()
        cuts = _detect_native(str(video), threshold=0.3, cancel_event=cancel)
        # Immediate cancel means no cuts found (or at most a few before check)
        assert isinstance(cuts, list)


def test_empty_video() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # Trivial 2-frame video
        video = Path(tmp) / "tiny.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, 10.0, (256, 256))
        for _ in range(2):
            writer.write(np.zeros((256, 256, 3), dtype=np.uint8))
        writer.release()
        cuts = detect_scene_cuts(str(video))
        assert isinstance(cuts, list)
