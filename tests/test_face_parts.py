"""Tests for face landmark detection and point generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.face_parts import (
    _HAAR_EYES,
    _HAAR_MOUTH,
    LEFT_EYE,
    MOUTH_INNER_LIP,
    MOUTH_OUTER_LIP,
    RIGHT_EYE,
    _centroid,
    _detect_face_by_skin,
    _detect_haar,
    _validate_haar_rect,
    _validate_landmarks,
    detect_landmarks,
    face_part_points,
)


def _synthetic_landmarks() -> dict[int, tuple[float, float]]:
    """Return a synthetic landmarks dict centered at (128, 128) in a 256x256 frame."""
    landmarks = {}
    for idx in MOUTH_OUTER_LIP:
        angle = (idx % 12) / 12.0 * 2 * np.pi
        landmarks[idx] = (128 + 15 * np.cos(angle), 128 + 10 * np.sin(angle))
    for idx in MOUTH_INNER_LIP:
        angle = (idx % 12) / 12.0 * 2 * np.pi
        landmarks[idx] = (128 + 8 * np.cos(angle), 128 + 5 * np.sin(angle))
    for idx in LEFT_EYE:
        angle = (idx % 7) / 7.0 * 2 * np.pi
        landmarks[idx] = (100 + 5 * np.cos(angle), 100 + 3 * np.sin(angle))
    for idx in RIGHT_EYE:
        angle = (idx % 7) / 7.0 * 2 * np.pi
        landmarks[idx] = (160 + 5 * np.cos(angle), 100 + 3 * np.sin(angle))
    return landmarks


def test_face_part_points_mouth() -> None:
    lm = _synthetic_landmarks()
    points = face_part_points(lm, "mouth", (256, 256))
    # One include point at the lip centre, no negatives.
    assert all(label == 1 for _, _, label in points)
    assert len(points) == 1
    x, y, label = points[0]
    assert label == 1
    assert 0 <= x < 256
    assert 0 <= y < 256


def test_face_part_points_eyes() -> None:
    lm = _synthetic_landmarks()
    points = face_part_points(lm, "eyes", (256, 256))
    # One include point per eye, no negatives.
    assert len(points) == 2
    assert all(label == 1 for _, _, label in points)
    for x, y, _ in points:
        assert 0 <= x < 256
        assert 0 <= y < 256


def test_face_part_points_empty_landmarks() -> None:
    points = face_part_points({}, "mouth", (256, 256))
    assert points == []


def test_face_part_points_invalid_part() -> None:
    lm = _synthetic_landmarks()
    points = face_part_points(lm, "forehead", (256, 256))
    assert points == []


def test_centroid() -> None:
    lm = {0: (10, 10), 1: (20, 20), 2: (30, 30)}
    c = _centroid(lm, [0, 1, 2])
    assert c == pytest.approx((20.0, 20.0))


def test_centroid_empty() -> None:
    assert _centroid({}, []) is None
    assert _centroid({0: (1, 2)}, []) is None


def test_detect_haar_no_face() -> None:
    """Haar should return None for a blank image."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = _detect_haar(frame)
    assert result is None


def test_detect_landmarks_no_face() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = detect_landmarks(frame)
    # May be None (no face detected)
    assert result is None or isinstance(result, dict)


def test_mouth_points_no_negatives() -> None:
    """SAM 2 selects an object from points — never spam red negatives."""
    lm = _synthetic_landmarks()
    points = face_part_points(lm, "mouth", (256, 256))
    excludes = [p for p in points if p[2] == 0]
    assert len(excludes) == 0


def test_haar_tables_cover_include_indices() -> None:
    """Regression: every index the UI needs must exist in the Haar fallback.

    Earlier the eye include indices (226…341) were missing from the Haar
    table, so the fallback produced an empty point list → blank masks.
    """
    for idx in LEFT_EYE + RIGHT_EYE:
        assert idx in _HAAR_EYES, f"missing Haar eye position for landmark {idx}"
    for idx in MOUTH_OUTER_LIP + MOUTH_INNER_LIP:
        assert idx in _HAAR_MOUTH, f"missing Haar mouth position for landmark {idx}"


def test_haar_fallback_produces_full_points() -> None:
    """A Haar-style landmark result must yield include points for eyes+mouth."""
    h, w = 256, 256
    # Simulate a face box detected at (48, 40) sized 160x180.
    fx, fy, fw, fh = 48, 40, 160, 180
    haar_lms = {
        idx: (fx + rx * fw, fy + ry * fh)
        for idx, (rx, ry) in {**_HAAR_EYES, **_HAAR_MOUTH}.items()
    }
    for part, expected in (("eyes", 2), ("mouth", 1)):
        points = face_part_points(haar_lms, part, (h, w))
        includes = [p for p in points if p[2] == 1]
        assert len(includes) == expected, (
            f"{part} should produce {expected} include point(s), got {len(includes)}"
        )
        assert all(label == 1 for _, _, label in points), (
            f"{part} must not emit negative points"
        )


def test_validate_haar_rect_rejects_hand_like_detections() -> None:
    """Hands (wide rect at the bottom of the frame) must be rejected.

    Regression for: "auto eyes/mouth always selects a random low-left
    point, often on the hand".
    """
    h, w = 480, 854
    # A hand-like rectangle low-left: wide (aspect > 1.4) and low (y > 65% h).
    assert not _validate_haar_rect(40, 360, 220, 90, w, h)
    # A realistic face box is accepted.
    assert _validate_haar_rect(320, 60, 150, 170, w, h)
    # Too-small detection (noise) rejected.
    assert not _validate_haar_rect(10, 10, 20, 26, w, h)


def test_validate_landmarks_rejects_bottom_clustered_points() -> None:
    """Landmarks clustered in the lower part of the frame are not a face."""
    h, w = 256, 256
    lm = {i: (60.0, 220.0) for i in range(10)}  # clumped near bottom
    assert not _validate_landmarks(lm, w, h)
    # A spread, centred face passes.
    lm = {i: (80.0 + (i % 5) * 20, 80.0 + (i // 5) * 20) for i in range(25)}
    assert _validate_landmarks(lm, w, h)


def test_detect_face_by_skin_finds_anime_face() -> None:
    """Skin-colour fallback must locate a face on a flat anime-style frame."""
    h, w = 256, 256
    frame = np.full((h, w, 3), 120, dtype=np.uint8)  # grey background
    # A skin-toned face region in the upper-middle of the frame.
    fx, fy, fw, fh = 80, 50, 96, 110
    face_colour = (225, 175, 155)  # RGB anime skin tone
    frame[fy : fy + fh, fx : fx + fw] = face_colour
    result = _detect_face_by_skin(frame)
    assert result is not None, "skin fallback found nothing"
    # Mouth landmarks should land inside the skin region.
    mouth_pts = [result[i] for i in MOUTH_OUTER_LIP if i in result]
    assert mouth_pts
    mx = sum(p[0] for p in mouth_pts) / len(mouth_pts)
    my = sum(p[1] for p in mouth_pts) / len(mouth_pts)
    assert fx <= mx <= fx + fw
    assert fy <= my <= fy + fh
