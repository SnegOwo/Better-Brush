"""Focused tests for the matte quality-control inspector."""

from __future__ import annotations

import numpy as np

from src.processing.matte_qc import (
    MatteQCReport,
    analyze_matte,
    compare_matte_reports,
    render_mask_overlay,
)


def _rectangle_mask(
    shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Create a simple rectangular foreground matte."""
    mask = np.zeros(shape, dtype=bool)
    mask[12:52, 12:52] = True
    return mask


def test_empty_matte_is_safe_and_warned() -> None:
    """An empty matte returns boolean evidence without an exception."""
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    report = analyze_matte(frame, np.zeros((32, 40), dtype=bool))

    assert isinstance(report, MatteQCReport)
    assert report.empty
    assert report.overall_warning
    assert report.mask.dtype == bool
    assert all(issue.mask.dtype == bool for issue in report.issues.values())


def test_mismatched_matte_is_safe() -> None:
    """A current matte with the wrong shape is reported safely."""
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    report = analyze_matte(frame, np.zeros((20, 20), dtype=bool))

    assert report.valid is False
    assert report.overall_warning
    assert report.mask.shape == frame.shape[:2]
    assert report.mask.dtype == bool


def test_holes_are_counted_and_located() -> None:
    """An enclosed false-background component produces a hole issue."""
    frame = np.full((64, 64, 3), 32, dtype=np.uint8)
    mask = _rectangle_mask()
    mask[28:36, 28:36] = False

    report = analyze_matte(frame, mask)

    issue = report.issue("holes")
    assert issue.count == 64
    assert issue.component_count == 1
    assert issue.bboxes == [(28, 28, 36, 36)]
    assert issue.mask.dtype == bool
    assert report.hole_score > 0.0
    assert report.overall_warning


def test_edge_flicker_uses_previous_matte() -> None:
    """A changed boundary is reported when a previous matte is supplied."""
    frame = np.full((64, 64, 3), 32, dtype=np.uint8)
    current = _rectangle_mask()
    previous = current.copy()
    previous[12, 12:52] = False

    report = analyze_matte(frame, current, previous_mask=previous)

    issue = report.issue("edge_flicker")
    assert issue.count > 0
    assert issue.score > 0.0
    assert issue.mask.dtype == bool
    assert report.edge_flicker_score > 0.0
    assert report.has_previous_mask


def test_comparison_reports_issue_improvement() -> None:
    """Removing a hole lowers the corresponding severity score."""
    frame = np.full((64, 64, 3), 32, dtype=np.uint8)
    before_mask = _rectangle_mask()
    before_mask[28:36, 28:36] = False
    after_mask = _rectangle_mask()
    before = analyze_matte(frame, before_mask)
    after = analyze_matte(frame, after_mask)

    comparison = compare_matte_reports(before, after)

    assert comparison.score_deltas["holes"] < 0.0
    assert comparison.count_deltas["holes"] < 0
    assert comparison.improved
    assert not comparison.worsened


def test_overlay_is_contiguous_rgb_for_qt() -> None:
    """The overlay renderer returns a display-ready RGB array."""
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 8:24] = True

    overlay = render_mask_overlay(frame, mask)

    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8
    assert overlay.flags.c_contiguous
    assert np.any(overlay != frame)
