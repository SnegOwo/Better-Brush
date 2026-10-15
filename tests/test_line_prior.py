"""Tests for the anime line-art prior."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.sam2_manager import PointPrompt
from src.processing.line_prior import (
    background_line_penalty,
    boundary_band,
    boundary_line_support,
    extract_lines,
    snap_to_anime_lines,
    snap_to_lines,
)


def _white_frame(height: int = 80, width: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_extract_lines_detects_a_dark_outline() -> None:
    frame = _white_frame()
    cv2.rectangle(frame, (20, 10), (70, 60), (0, 0, 0), 1)
    lines = extract_lines(frame)
    assert lines.dtype == bool
    assert lines.any()
    # A pixel on the left edge of the outline must be marked.
    assert bool(lines[30, 19] or lines[30, 20] or lines[30, 21])


def test_extract_lines_on_flat_frame_is_empty() -> None:
    assert not extract_lines(np.full((20, 20, 3), 200, dtype=np.uint8)).any()


def test_boundary_band_is_a_ring() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    band = boundary_band(mask, width=1)
    assert bool(band[10, 20]) or bool(band[9, 20]) or bool(band[11, 20])
    assert not band[20, 20]


def test_boundary_support_high_when_aligned() -> None:
    frame = _white_frame()
    cv2.rectangle(frame, (20, 10), (70, 60), (0, 0, 0), 1)
    lines = extract_lines(frame)
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:61, 20:71] = True
    support = boundary_line_support(mask, lines, radius=3)
    assert support > 0.5


def test_boundary_support_zero_without_lines() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    assert boundary_line_support(mask, np.zeros((40, 40), dtype=bool)) == 0.0
    assert boundary_line_support(np.zeros((40, 40), dtype=bool), np.ones((40, 40), dtype=bool)) == 0.0


def test_background_line_penalty_flags_unprompted_line_art() -> None:
    frame = _white_frame(120, 140)
    cv2.rectangle(frame, (10, 10), (40, 40), (0, 0, 0), 1)  # subject
    cv2.rectangle(frame, (70, 60), (110, 100), (0, 0, 0), 1)  # background
    lines = extract_lines(frame)

    mask = np.zeros((120, 140), dtype=bool)
    mask[10:41, 10:41] = True
    mask[60:101, 70:111] = True

    subject_only = [PointPrompt(10, 25, 1)]
    penalty = background_line_penalty(mask, lines, subject_only)
    # The background outline occupies a small but real share of the mask.
    assert 0.0 < penalty < 0.9

    both = [PointPrompt(10, 25, 1), PointPrompt(70, 80, 1)]
    assert background_line_penalty(mask, lines, both) == 0.0


def test_background_line_penalty_is_zero_without_points() -> None:
    frame = _white_frame(60, 60)
    cv2.rectangle(frame, (10, 10), (50, 50), (0, 0, 0), 1)
    lines = extract_lines(frame)
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:51, 10:51] = True
    assert background_line_penalty(mask, lines, []) == 0.0


def test_snap_grows_onto_a_nearby_line() -> None:
    frame = _white_frame(60, 60)
    frame[5:55, 50] = 0  # a vertical line at x=50
    lines = extract_lines(frame)
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True  # right edge at x=49, one pixel shy of the line
    snapped = snap_to_lines(mask, lines, radius=2)
    assert not mask[30, 50]
    assert snapped[30, 50]


def test_snap_no_lines_is_a_noop() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    out = snap_to_lines(mask, np.zeros((40, 40), dtype=bool), radius=2)
    np.testing.assert_array_equal(out, mask)


def test_snap_trims_a_stray_beyond_the_line() -> None:
    frame = _white_frame(70, 70)
    cv2.rectangle(frame, (20, 10), (40, 50), (0, 0, 0), 1)
    lines = extract_lines(frame)
    mask = np.zeros((70, 70), dtype=bool)
    mask[11:50, 21:41] = True  # rides the outline
    mask[25:35, 41:47] = True  # a bump straying 6 px past the line
    snapped = snap_to_lines(mask, lines, radius=2)
    assert mask[30, 46]
    assert not snapped[30, 46]
    assert snapped[30, 30]  # interior preserved


def test_snap_without_shrink_only_grows() -> None:
    frame = _white_frame(70, 70)
    cv2.rectangle(frame, (20, 10), (40, 50), (0, 0, 0), 1)
    lines = extract_lines(frame)
    mask = np.zeros((70, 70), dtype=bool)
    mask[11:50, 21:41] = True
    mask[25:35, 41:47] = True
    snapped = snap_to_lines(mask, lines, radius=2, allow_shrink=False)
    assert not np.any(mask & ~snapped)


def test_snap_to_anime_lines_wrapper() -> None:
    frame = _white_frame(60, 60)
    frame[5:55, 50] = 0
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    snapped = snap_to_anime_lines(mask, frame, radius=2)
    assert snapped[30, 50]
