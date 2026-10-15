"""Auto-subject scoring: the 'characters only' gate must be predictable."""

from __future__ import annotations

import numpy as np

from src.engine import auto_subject as A


def _blob(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_grid_points_are_interior_and_counted():
    points = A.grid_points(200, 100, steps=4)
    assert len(points) == 16
    for x, y in points:
        assert 0 < x < 199
        assert 0 < y < 99


def test_iou_bounds():
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    assert A.iou(a, a) == 1.0
    assert A.iou(a, np.zeros_like(a)) == 0.0
    b = np.zeros((10, 10), dtype=bool)
    b[2:6, 2:6] = True  # identical
    assert A.iou(a, b) == 1.0


def test_make_candidate_rejects_degenerate_sizes():
    frame = (100, 100)
    tiny = np.zeros(frame, dtype=bool)
    tiny[0, 0] = True  # 1 px -> below MIN_AREA_RATIO
    assert A.make_candidate(tiny, 0.9) is None

    huge = np.ones(frame, dtype=bool)  # 100% -> above MAX_AREA_RATIO
    assert A.make_candidate(huge, 0.9) is None


def test_centred_character_beats_edge_background():
    frame = (100, 100)
    char = _blob(100, 100, 50, 50, 25)
    background = np.zeros(frame, dtype=bool)
    background[:90, :40] = True  # a big corner/edge region

    good = A.make_candidate(char, 0.9)
    bad = A.make_candidate(background, 0.9)
    assert good is not None
    assert bad is not None
    assert good.score > bad.score
    assert good.score >= A.MIN_CONFIDENCE


def test_face_inside_is_rewarded():
    char = _blob(100, 100, 50, 50, 25)
    with_face = A.make_candidate(char, 0.5, face_center=(50.0, 50.0))
    without = A.make_candidate(char, 0.5, face_center=(2.0, 2.0))
    assert with_face is not None
    assert without is not None
    assert with_face.score > without.score


def test_evaluate_picks_single_subject():
    mask = _blob(120, 120, 60, 60, 28)
    cand = A.make_candidate(mask, 0.92)
    assert cand is not None
    selection = A.evaluate([cand])
    assert selection.found
    assert selection.mask is not None
    assert selection.box is not None
    assert selection.confidence >= A.MIN_CONFIDENCE


def test_evaluate_flags_multiple_subjects():
    left = _blob(140, 140, 40, 70, 22)
    right = _blob(140, 140, 100, 70, 22)
    a = A.make_candidate(left, 0.9)
    b = A.make_candidate(right, 0.89)
    assert a is not None
    assert b is not None
    selection = A.evaluate([a, b])
    assert selection.multiple
    assert not selection.found


def test_evaluate_refuses_when_below_confidence():
    mask = _blob(100, 100, 50, 50, 8)  # small, so the blended score is low
    cand = A.make_candidate(mask, 0.01)
    assert cand is not None
    selection = A.evaluate([cand], min_confidence=0.95)
    assert not selection.found
    assert not selection.multiple


class _FakeResult:
    def __init__(self, mask: np.ndarray, score: float) -> None:
        self.mask = mask
        self.score = score


class _FakeManager:
    """Returns the character blob for seeds inside it, else the background."""

    def __init__(self, character: np.ndarray) -> None:
        self.character = character

    def predict_frame_multi(self, frame, prompt_sets, multimask_output=True):
        h, w = frame.shape[:2]
        out = []
        for points in prompt_sets:
            p = points[0]
            if self.character[p.y, p.x]:
                out.append([_FakeResult(self.character.copy(), 0.95)])
            else:
                background = np.zeros((h, w), dtype=bool)
                background[: int(h * 0.95), : int(w * 0.95)] = True
                out.append([_FakeResult(background, 0.95)])
        return out


def test_find_subject_returns_one_character():
    character = _blob(120, 120, 60, 60, 26)
    manager = _FakeManager(character)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    selection = A.find_subject(manager, frame, steps=4)
    assert selection.found
    assert selection.confidence > 0


def test_find_subject_handles_tiny_frame():
    manager = _FakeManager(_blob(8, 8, 4, 4, 2))
    selection = A.find_subject(manager, np.zeros((8, 8, 3), dtype=np.uint8))
    assert not selection.found
    assert not selection.multiple
