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


def test_evaluate_prefers_the_whole_character_over_its_face():
    """SAM 2 returns a face inside a head inside a character for one seed.

    The face is the cleaner mask (solid, away from the frame edge), so it wins
    on score — and the job rotoscopes a face. The whole subject is what was
    asked for, so it has to be promoted.
    """
    character = _blob(200, 200, 100, 110, 70)
    face = _blob(200, 200, 100, 60, 18)
    whole = A.make_candidate(character, 0.90)
    part = A.make_candidate(face, 0.95, face_center=(100.0, 60.0))
    assert whole is not None
    assert part is not None
    assert part.score > whole.score, "the part is expected to score higher"

    selection = A.evaluate([part, whole])

    assert selection.found, selection.reasons
    np.testing.assert_array_equal(selection.mask, character)


def test_evaluate_does_not_call_a_part_a_second_subject():
    character = _blob(200, 200, 100, 110, 70)
    face = _blob(200, 200, 100, 60, 18)
    whole = A.make_candidate(character, 0.90)
    part = A.make_candidate(face, 0.95, face_center=(100.0, 60.0))
    assert whole is not None
    assert part is not None

    selection = A.evaluate([whole, part])

    assert not selection.multiple, selection.reasons


def test_evaluate_refuses_when_below_confidence():
    mask = _blob(100, 100, 50, 50, 8)  # small, so the blended score is low
    cand = A.make_candidate(mask, 0.01)
    assert cand is not None
    selection = A.evaluate([cand], min_confidence=0.95)
    assert not selection.found
    assert not selection.multiple


def test_the_frame_background_loses_to_the_face_bearing_character():
    """Measured on a real clip: SAM 2 rates the set far above the subject.

    The background mask scored 0.99 against the character's 0.06, and it is
    the better mask on every signal except one — it never contains the face.
    Anchoring on the face is what makes the character win.
    """
    character = _blob(200, 200, 100, 110, 70)
    background = ~character
    face = _blob(200, 200, 100, 60, 18)
    face_ctr = (100.0, 60.0)

    whole = A.make_candidate(character, 0.06, face_center=face_ctr)
    part = A.make_candidate(face, 0.66, face_center=face_ctr)
    bg = A.make_candidate(background, 0.99, face_center=face_ctr)
    assert whole is not None
    assert part is not None
    assert bg is not None
    assert bg.sam_score > whole.sam_score, "the set is the more confident mask"

    # The set does not overlap the character, so it also reads as a second
    # subject and the whole job would refuse.
    assert A.evaluate([bg, part, whole]).multiple

    anchored = A._anchor_on_face([bg, part, whole])

    assert all(cand is not bg for cand in anchored), "the set must not survive"
    selection = A.evaluate(anchored)
    assert selection.found, selection.reasons
    np.testing.assert_array_equal(selection.mask, character)


def test_face_anchor_stands_down_when_no_mask_holds_the_face():
    """A wrong face hit must not throw away every candidate."""
    character = _blob(120, 120, 60, 60, 30)

    only = A.make_candidate(character, 0.9, face_center=(3.0, 3.0))
    assert only is not None
    assert not only.face_inside

    kept = A._anchor_on_face([only])
    assert len(kept) == 1
    assert kept[0] is only


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
