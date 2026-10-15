"""Tests for the anime-specific QC metrics."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.sam2_manager import PointPrompt
from src.processing.anime_qc import (
    AnimeQC,
    analyze_anime_matte,
    background_line_ratio,
    boiling_score,
    contour_change,
    line_support,
    suggest_keyframes,
)


def _white(height: int = 80, width: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _square(shape=(40, 40), y=10, x=10, size=20) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + size, x : x + size] = True
    return mask


def test_contour_change_identical_is_zero() -> None:
    mask = _square()
    assert contour_change(mask, mask) == 0.0


def test_contour_change_detects_a_wobble() -> None:
    a = _square()
    b = a.copy()
    b[10, 10] = False
    b[29, 29] = False
    assert contour_change(a, b) > 0.0


def test_boiling_score_zero_on_stable_masks() -> None:
    frame = _white()
    frames = {i: frame.copy() for i in range(3)}
    masks = {i: _square(shape=(80, 100), y=20, x=20, size=30) for i in range(3)}
    score, held, pairs = boiling_score(masks, frames)
    assert score == 0.0
    assert held == 2
    assert pairs == 2


def test_boiling_score_flags_a_wobble_on_a_held_cel() -> None:
    frame = _white()
    frames = {i: frame.copy() for i in range(3)}
    m0 = _square(shape=(80, 100), y=20, x=20, size=30)
    m1 = m0.copy()
    m1[20:30, 20:30] = False  # carve a chunk out of the held drawing
    masks = {0: m0, 1: m1, 2: m0.copy()}
    score, held, _ = boiling_score(masks, frames)
    assert held == 2
    assert score > 0.0


def test_line_support_high_on_an_aligned_contour() -> None:
    frame = _white()
    cv2.rectangle(frame, (20, 10), (70, 60), (0, 0, 0), 1)
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:61, 20:71] = True
    assert line_support(frame, mask, radius=3) > 0.5


def test_background_line_ratio_detects_stray_line_art() -> None:
    frame = _white(120, 140)
    cv2.rectangle(frame, (10, 10), (40, 40), (0, 0, 0), 1)
    cv2.rectangle(frame, (70, 60), (110, 100), (0, 0, 0), 1)
    mask = np.zeros((120, 140), dtype=bool)
    mask[10:41, 10:41] = True
    mask[60:101, 70:111] = True
    ratio = background_line_ratio(frame, mask, [PointPrompt(10, 25, 1)])
    assert ratio > 0.0


def test_suggest_keyframes_flags_boiling_frames() -> None:
    frame = _white()
    frames = {i: frame.copy() for i in range(3)}
    m0 = _square(shape=(80, 100), y=20, x=20, size=30)
    m1 = m0.copy()
    m1[20:30, 20:30] = False
    masks = {0: m0, 1: m1, 2: m0.copy()}
    flagged = suggest_keyframes(masks, frames, threshold=0.05)
    assert 0 in flagged
    assert 1 in flagged


def test_analyze_reports_and_warns_on_background_line_art() -> None:
    frame = _white(120, 140)
    cv2.rectangle(frame, (10, 10), (40, 40), (0, 0, 0), 1)
    cv2.rectangle(frame, (70, 60), (110, 100), (0, 0, 0), 1)
    mask = np.zeros((120, 140), dtype=bool)
    mask[10:41, 10:41] = True
    mask[60:101, 70:111] = True
    report = analyze_anime_matte(frame, mask, points=[PointPrompt(10, 25, 1)])
    assert isinstance(report, AnimeQC)
    assert report.background_line_ratio > 0.0
    assert report.message
    assert 0.0 <= report.score <= 1.0


def test_analyze_boiling_on_a_held_cel() -> None:
    frame = _white()
    m0 = _square(shape=(80, 100), y=20, x=20, size=30)
    m1 = m0.copy()
    m1[20:30, 20:30] = False
    report = analyze_anime_matte(frame, m1, previous_mask=m0, previous_frame=frame)
    assert report.held_pairs == 1
    assert report.boiling > 0.0
    assert "held cel" in report.message


def test_analyze_handles_empty_mask() -> None:
    frame = _white()
    report = analyze_anime_matte(frame, np.zeros((80, 100), dtype=bool))
    assert isinstance(report, AnimeQC)
    assert report.boiling == 0.0
