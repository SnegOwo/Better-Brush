"""Color spill cleaner: profile plumbing, math, and canvas wiring.

The headline case is a subject shot against a saturated background (blue sky)
that leaves a fringe of that color along its silhouette.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.core.profiles import ShotProfile, ShotProfileRecord
from src.core.project import ProjectData, ProjectManager
from src.processing.despill import (
    apply_shot_transform,
    despill_alpha,
    despill_frame,
    edge_band,
    neutralize_cast,
    spill_match,
)

_APP: QApplication | None = None

BLUE = (30, 60, 230)
RED = (200, 40, 40)
VIOLET = (120, 90, 210)


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _fringe_scene(size: int = 60, radius: int = 18, fringe: float = 3.0):
    """A red disc on blue sky whose matte edge carries a blue fringe.

    Returns:
        ``(frame, mask, fringe_pixels)`` - the fringe is the ring just inside
        the matte boundary, which is exactly where sky spill lands.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    dist2 = (yy - size // 2) ** 2 + (xx - size // 2) ** 2
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = BLUE
    disc = dist2 <= radius**2
    frame[disc] = RED
    ring = disc & (dist2 >= (radius - fringe) ** 2)
    frame[ring] = BLUE
    return frame, disc.copy(), ring


def _blue_chroma(frame: np.ndarray, where: np.ndarray) -> float:
    """Mean blue excess (B minus the R/G average) over *where*."""
    values = frame.astype(np.float32)
    chroma = values[:, :, 2] - (values[:, :, 0] + values[:, :, 1]) / 2.0
    return float(chroma[where].mean())


# ------------------------------------------------------------------ #
# Profile model                                                      #
# ------------------------------------------------------------------ #
def test_spill_defaults_are_off() -> None:
    profile = ShotProfile()
    assert profile.spill_mode == "off"
    assert profile.spill_color is None
    assert profile.spill_tolerance == 30
    assert profile.spill_radius == 2
    assert profile.spill_region is None
    assert profile.spill_active is False


def test_spill_needs_a_color_to_be_active() -> None:
    assert not ShotProfile(spill_mode="despill").spill_active
    assert ShotProfile(spill_mode="despill", spill_color=BLUE).spill_active


def test_spill_values_are_clamped() -> None:
    profile = ShotProfile(spill_tolerance=999, spill_radius=-4, spill_mode="nope")
    assert profile.spill_tolerance == 100
    assert profile.spill_radius == 1
    assert profile.spill_mode == "off"
    bad = ShotProfile(spill_tolerance="x", spill_radius=None)
    assert bad.spill_tolerance == 30
    assert bad.spill_radius == 2


def test_spill_color_accepts_tuple_list_and_hex() -> None:
    assert ShotProfile(spill_color=BLUE).spill_color == BLUE
    assert ShotProfile(spill_color=[1, 2, 3]).spill_color == (1, 2, 3)
    assert ShotProfile(spill_color="#010203").spill_color == (1, 2, 3)
    assert ShotProfile(spill_color="010203").spill_color == (1, 2, 3)
    assert ShotProfile(spill_color=(1, 2)).spill_color is None
    assert ShotProfile(spill_color=(1, 2, 999)).spill_color is None
    assert ShotProfile(spill_color="nonsense").spill_color is None


def test_spill_survives_a_copy() -> None:
    """``copy()`` round-trips through to_dict, so a dropped field would be lost."""
    profile = ShotProfile(spill_color=BLUE, spill_mode="despill", spill_tolerance=45)
    clone = profile.copy()
    assert clone.spill_color == BLUE
    assert clone.spill_mode == "despill"
    assert clone.spill_tolerance == 45
    assert clone == profile


def test_spill_is_json_serializable() -> None:
    profile = ShotProfile(spill_color=BLUE, spill_mode="alpha_key", spill_radius=4)
    encoded = json.dumps(profile.to_dict())
    assert ShotProfile.from_dict(json.loads(encoded)) == profile


def test_legacy_project_without_spill_keys_loads() -> None:
    profile = ShotProfile.from_dict({"name": "Old", "mask_inset": 2})
    assert profile.spill_mode == "off"
    assert profile.spill_color is None


def test_spill_region_accepts_a_box() -> None:
    assert ShotProfile(spill_region=(10, 20, 30, 40)).spill_region == (10, 20, 30, 40)
    assert ShotProfile(spill_region=[1, 2, 3, 4]).spill_region == (1, 2, 3, 4)
    # A degenerate box would silently do nothing, so it is dropped instead.
    assert ShotProfile(spill_region=(0, 0, 0, 10)).spill_region is None
    assert ShotProfile(spill_region=(0, 0, 10, 0)).spill_region is None
    assert ShotProfile(spill_region=(0, 0, 1, 1)).spill_region == (0, 0, 1, 1)
    assert ShotProfile(spill_region=(1, 2, 3)).spill_region is None
    assert ShotProfile(spill_region="nope").spill_region is None


def test_region_mode_and_cast_fields_validate() -> None:
    assert ShotProfile().spill_region_mode == "clean"
    assert ShotProfile(spill_region_mode="PROTECT").spill_region_mode == "protect"
    assert ShotProfile(spill_region_mode="nonsense").spill_region_mode == "clean"
    assert ShotProfile().cast_neutralize is False
    assert ShotProfile().cast_strength == 100
    assert ShotProfile(cast_strength=999).cast_strength == 100
    assert ShotProfile(cast_strength=-5).cast_strength == 0
    assert ShotProfile(cast_strength="x").cast_strength == 100
    assert ShotProfile(cast_neutralize=1).cast_neutralize is True


def test_spill_region_survives_a_copy_and_json() -> None:
    profile = ShotProfile(
        spill_color=BLUE,
        spill_mode="despill",
        spill_region=(4, 5, 6, 7),
        spill_region_mode="protect",
        cast_neutralize=True,
        cast_strength=42,
    )
    assert profile.copy() == profile
    assert ShotProfile.from_dict(json.loads(json.dumps(profile.to_dict()))) == profile


# ------------------------------------------------------------------ #
# .bbproj round trip                                                 #
# ------------------------------------------------------------------ #
def test_spill_survives_project_round_trip() -> None:
    spill = ShotProfile(
        spill_color=BLUE,
        spill_mode="despill",
        spill_tolerance=55,
        spill_radius=6,
        spill_region=(8, 12, 200, 90),
        spill_region_mode="protect",
        cast_neutralize=True,
        cast_strength=70,
    )
    data = ProjectData(
        default_profile=spill,
        shot_profiles=[
            ShotProfileRecord(shot_id=1, start_frame=0, end_frame=9, profile=spill)
        ],
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "spill.bbproj"
        ProjectManager.save(path, data)
        loaded = ProjectManager.load(path)
        for profile in (loaded.default_profile, loaded.shot_profiles[0].profile):
            assert profile.spill_color == BLUE
            assert profile.spill_mode == "despill"
            assert profile.spill_tolerance == 55
            assert profile.spill_radius == 6
            assert profile.spill_region == (8, 12, 200, 90)
            assert profile.spill_region_mode == "protect"
            assert profile.cast_neutralize is True
            assert profile.cast_strength == 70


# ------------------------------------------------------------------ #
# Edge band                                                          #
# ------------------------------------------------------------------ #
def test_edge_band_surrounds_the_boundary() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    band = edge_band(mask, 2)
    assert band.any()
    # Only the ring, never the interior or the far exterior.
    assert not band[20, 20]
    assert not band[2, 2]
    assert band[10, 20] and band[29, 20]


def test_edge_band_handles_empty_masks() -> None:
    assert not edge_band(np.zeros((10, 10), dtype=bool), 3).any()
    assert not edge_band(None, 3).any()


# ------------------------------------------------------------------ #
# Despill (frame colors)                                             #
# ------------------------------------------------------------------ #
def test_despill_suppresses_blue_fringe() -> None:
    frame, mask, fringe = _fringe_scene()
    assert _blue_chroma(frame, fringe) > 150, "fixture must actually have spill"

    cleaned = despill_frame(frame, mask, BLUE, tolerance=30, radius=3)

    assert _blue_chroma(cleaned, fringe) < 40, "fringe must lose its blue"
    assert _blue_chroma(cleaned, fringe) < _blue_chroma(frame, fringe) / 3


def test_despill_leaves_the_subject_interior_alone() -> None:
    frame, mask, fringe = _fringe_scene()
    cleaned = despill_frame(frame, mask, BLUE, tolerance=30, radius=3)
    core = mask & ~fringe
    delta = np.abs(cleaned[core].astype(float) - frame[core].astype(float)).mean()
    assert delta == 0.0, "a red subject must not be turned gray"


def test_despill_does_not_touch_distant_background() -> None:
    frame, mask, _ = _fringe_scene()
    cleaned = despill_frame(frame, mask, BLUE, tolerance=30, radius=3)
    far = np.zeros(frame.shape[:2], dtype=bool)
    far[:4, :4] = True
    assert np.array_equal(cleaned[far], frame[far])


def test_despill_does_not_modify_its_input() -> None:
    frame, mask, _ = _fringe_scene()
    original = frame.copy()
    despill_frame(frame, mask, BLUE, tolerance=30, radius=3)
    assert np.array_equal(frame, original)


def test_despill_ignores_an_unrelated_color() -> None:
    frame, mask, _ = _fringe_scene()
    cleaned = despill_frame(frame, mask, (60, 200, 40), tolerance=30, radius=3)
    assert np.array_equal(cleaned, frame), "green spill must not touch a blue fringe"


def test_despill_tolerance_widens_the_window() -> None:
    """A slightly-off target needs a wider window to be caught."""
    frame, mask, fringe = _fringe_scene()
    # Aim a little off the true blue hue.
    off_target = (60, 40, 210)
    tight = despill_frame(frame, mask, off_target, tolerance=5, radius=3)
    wide = despill_frame(frame, mask, off_target, tolerance=60, radius=3)
    assert _blue_chroma(wide, fringe) < _blue_chroma(tight, fringe)


def test_despill_without_a_mask_is_a_noop() -> None:
    frame, _, _ = _fringe_scene()
    assert np.array_equal(despill_frame(frame, None, BLUE, 30, 3), frame)


# ------------------------------------------------------------------ #
# Alpha key (matte coverage)                                         #
# ------------------------------------------------------------------ #
def test_alpha_key_removes_the_fringe_from_the_matte() -> None:
    frame, mask, fringe = _fringe_scene()
    keyed = despill_alpha(mask, frame, BLUE, tolerance=30, radius=3)
    removed = mask & ~keyed
    assert removed.any(), "the fringe must drop out of the matte"
    # Alpha key only ever removes coverage, never adds it.
    assert not (keyed & ~mask).any()
    # A sliver of the ring can land a hair outside the elliptical band, so
    # assert the bulk goes rather than demanding a perfect pixel count.
    assert (fringe & ~keyed).sum() / fringe.sum() > 0.95
    # The solid interior is never touched.
    assert keyed[mask & ~fringe].all(), "the subject interior must be kept"


def test_alpha_key_ignores_an_unrelated_color() -> None:
    frame, mask, _ = _fringe_scene()
    assert np.array_equal(despill_alpha(mask, frame, (60, 200, 40), 30, 3), mask)


def test_alpha_key_does_not_modify_its_input() -> None:
    frame, mask, _ = _fringe_scene()
    original = mask.copy()
    despill_alpha(mask, frame, BLUE, 30, 3)
    assert np.array_equal(mask, original)


def test_gray_pixels_do_not_count_as_spill() -> None:
    """A near-gray fringe has no hue evidence, so nothing should be keyed."""
    frame = np.full((30, 30, 3), 128, dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    assert np.array_equal(
        despill_alpha(mask, frame, BLUE, tolerance=100, radius=3), mask
    )


# ------------------------------------------------------------------ #
# Match scoring                                                      #
# ------------------------------------------------------------------------
def test_spill_match_scores_the_target_highest() -> None:
    frame, _, fringe = _fringe_scene()
    score = spill_match(frame, BLUE, tolerance=30)
    assert score[fringe].mean() > 0.9
    red_core = np.zeros(frame.shape[:2], dtype=bool)
    red_core[28:32, 28:32] = True
    assert score[red_core].mean() < 0.2


def test_spill_match_handles_hue_wraparound() -> None:
    """Red sits at the 0/179 boundary in OpenCV hue; it must still match red."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:] = (230, 30, 30)
    score = spill_match(frame, (230, 30, 30), tolerance=20)
    assert score.mean() > 0.8


# ------------------------------------------------------------------ #
# Region-limited cleaning (garbage matte)                            #
# ------------------------------------------------------------------ #
def _split_scene(size: int = 120):
    """Blue sky on the right half, a blue eye on skin on the left.

    This is the case a region exists for: the same hue has to be removed in one
    place and preserved in the other.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = (200, 170, 140)  # skin
    sky = xx >= size // 2
    frame[sky] = BLUE
    eye = (yy - size // 2) ** 2 + (xx - size // 4) ** 2 <= 8**2
    frame[eye] = (40, 90, 220)  # a blue eye
    return frame, eye, sky


def test_region_cleans_sky_and_spares_the_eye() -> None:
    frame, eye, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    region = (60, 0, 60, 120)

    cleaned = despill_frame(frame, mask, BLUE, tolerance=100, radius=10, region=region)

    assert _blue_chroma(cleaned, sky) < 10, "the boxed sky must be neutralized"
    assert np.array_equal(cleaned[eye], frame[eye]), "the blue eye must survive"


def test_region_leaves_everything_outside_the_box_alone() -> None:
    frame, _, _ = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    region = (60, 0, 60, 120)

    cleaned = despill_frame(frame, mask, BLUE, tolerance=100, radius=10, region=region)

    left = np.zeros(frame.shape[:2], dtype=bool)
    left[:, :60] = True
    assert np.array_equal(cleaned[left], frame[left])


def test_region_limits_alpha_key_too() -> None:
    frame, eye, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    region = (60, 0, 60, 120)

    keyed = despill_alpha(mask, frame, BLUE, tolerance=100, radius=10, region=region)

    assert not keyed[sky].any(), "boxed sky must leave the matte"
    assert keyed[eye].all(), "the eye must stay in the matte"
    left = np.zeros(frame.shape[:2], dtype=bool)
    left[:, :60] = True
    assert keyed[left].all(), "nothing outside the box may be keyed"


def test_a_region_beats_the_band_for_a_wide_area() -> None:
    """With a full-frame matte there is no boundary, so the band cannot help."""
    frame, _, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)

    banded = despill_frame(frame, mask, BLUE, tolerance=100, radius=10)
    boxed = despill_frame(
        frame, mask, BLUE, tolerance=100, radius=10, region=(60, 0, 60, 120)
    )

    assert _blue_chroma(banded, sky) > 100, "the band has nothing to work on here"
    assert _blue_chroma(boxed, sky) < 10


def test_protect_box_spares_the_subject() -> None:
    """The blue-hair-under-a-blue-sky case, which is why protect exists."""
    frame, _, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    eye_box = (10, 40, 45, 45)

    cleaned = despill_frame(
        frame,
        mask,
        BLUE,
        tolerance=100,
        radius=10,
        region=eye_box,
        region_mode="protect",
    )

    inside = np.zeros(frame.shape[:2], dtype=bool)
    inside[40:85, 10:55] = True
    assert np.array_equal(cleaned[inside], frame[inside]), "the box must be shielded"
    assert _blue_chroma(cleaned, sky) < 10, "everything else still gets cleaned"


def test_protect_box_limits_alpha_key_too() -> None:
    frame, eye, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    eye_box = (10, 40, 45, 45)

    keyed = despill_alpha(
        mask,
        frame,
        BLUE,
        tolerance=100,
        radius=10,
        region=eye_box,
        region_mode="protect",
    )

    assert keyed[eye].all(), "the protected eye stays in the matte"
    assert not keyed[sky].any(), "the rest of frame is still keyed"


def test_a_clean_box_and_a_protect_box_are_opposites() -> None:
    frame, _, _ = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    box = (10, 40, 45, 45)
    as_clean = despill_frame(frame, mask, BLUE, 100, 10, box, "clean")
    as_protect = despill_frame(frame, mask, BLUE, 100, 10, box, "protect")
    inside = np.zeros(frame.shape[:2], dtype=bool)
    inside[40:85, 10:55] = True
    assert not np.array_equal(as_clean[inside], frame[inside])
    assert np.array_equal(as_protect[inside], frame[inside])


def test_region_is_clipped_to_the_frame() -> None:
    frame, _, _ = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    # A box hanging off the edge must not raise or spill outside.
    cleaned = despill_frame(
        frame, mask, BLUE, tolerance=100, radius=10, region=(-50, -50, 500, 500)
    )
    assert cleaned.shape == frame.shape


def test_an_empty_region_is_a_noop() -> None:
    frame, _, _ = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    assert np.array_equal(
        despill_frame(frame, mask, BLUE, 100, 10, region=(500, 500, 10, 10)), frame
    )


# ------------------------------------------------------------------ #
# Match strength                                                     #
# ------------------------------------------------------------------ #
def test_a_wide_tolerance_catches_an_off_target_hue() -> None:
    """The plateau is what makes a high tolerance actually do something."""
    frame = np.empty((30, 30, 3), dtype=np.uint8)
    frame[:] = BLUE
    near_miss = (70, 45, 205)  # same family, clearly off hue
    assert float(spill_match(frame, near_miss, tolerance=10).mean()) < 0.1
    assert float(spill_match(frame, near_miss, tolerance=30).mean()) > 0.7
    assert float(spill_match(frame, near_miss, tolerance=100).mean()) > 0.9


def test_alpha_key_clears_the_fringe_at_every_tolerance() -> None:
    """Regression: the old linear falloff only keyed the exact center line."""
    frame, mask, fringe = _fringe_scene()
    for tolerance in (20, 40, 70, 100):
        keyed = despill_alpha(mask, frame, BLUE, tolerance=tolerance, radius=3)
        assert (fringe & ~keyed).sum() / fringe.sum() > 0.9, (
            f"tolerance {tolerance} left most of the fringe behind"
        )


# ------------------------------------------------------------------ #
# Color cast neutralization                                           #
# ------------------------------------------------------------------ #
def _tinted_scene(size: int = 120):
    """A violet-lit shot: violet sky behind a head-and-shoulders subject.

    The subject deliberately occupies well under half the frame so that the
    area outside the matte is genuinely mostly background, which is what the
    cast reference relies on.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = VIOLET  # the sky fills the frame
    head = ((yy - 52) ** 2) / (22**2) + ((xx - 60) ** 2) / (18**2) <= 1.0
    shoulders = (yy >= 76) & (np.abs(xx - 60) <= 34)
    body = head | shoulders
    frame[body] = (150, 120, 90)  # skin
    hair = ((yy - 40) ** 2) / (20**2) + ((xx - 60) ** 2) / (17**2) <= 1.0
    frame[hair] = (170, 140, 110)  # blonde, but lit violet
    mask = body.copy()
    sky = ~body
    return frame, hair, sky, mask


def test_neutralize_cast_removes_the_violet_cast() -> None:
    frame, hair, sky, mask = _tinted_scene()
    # A box over sky only: that is the reference the panel asks the user for.
    fixed = neutralize_cast(frame, mask, (0, 0, 40, 40), "clean", 100)
    assert _blue_chroma(fixed, sky) < 20
    assert _blue_chroma(fixed, sky) < _blue_chroma(frame, sky) / 2
    # The subject is the thing that was wrong, and it improves too.
    assert _blue_chroma(fixed, hair) < _blue_chroma(frame, hair)


def test_neutralize_cast_preserves_brightness_and_shading() -> None:
    frame, hair, _, mask = _tinted_scene()
    fixed = neutralize_cast(frame, mask, (0, 0, 40, 40), "clean", 100)
    before = float(frame[hair].mean())
    after = float(fixed[hair].mean())
    assert abs(after - before) / before < 0.35, "a cast fix must not change exposure"


def test_cast_strength_interpolates() -> None:
    frame, _, sky, mask = _tinted_scene()
    none = _blue_chroma(neutralize_cast(frame, mask, strength=0), sky)
    half = _blue_chroma(neutralize_cast(frame, mask, strength=50), sky)
    full = _blue_chroma(neutralize_cast(frame, mask, strength=100), sky)
    assert none == _blue_chroma(frame, sky)
    assert full < half < none


def test_neutralize_cast_does_not_modify_its_input() -> None:
    frame, _, _, mask = _tinted_scene()
    original = frame.copy()
    neutralize_cast(frame, mask, strength=100)
    assert np.array_equal(frame, original)


def test_neutralize_cast_uses_the_background_by_default() -> None:
    frame, _, sky, mask = _tinted_scene()
    # No box: the area outside the matte is the reference.
    fixed = neutralize_cast(frame, mask, strength=100)
    assert _blue_chroma(fixed, sky) < 20


def test_cast_needs_a_representative_reference() -> None:
    """A box covering everything is useless, because "neutral" then means
    "the average of everything".

    Grey-world treats whatever it is handed as the reference, so a box around
    the sky fixes the cast while a box around the whole frame - or a matte that
    covers nearly everything - normalises to the scene average and does almost
    nothing.  This is why the panel asks for a small Area box.
    """
    frame, _, sky, mask = _tinted_scene()
    good = neutralize_cast(frame, mask, (0, 0, 40, 40), "clean", 100)
    whole_box = neutralize_cast(frame, mask, (0, 0, 120, 120), "clean", 100)
    full_matte = neutralize_cast(frame, np.ones_like(mask), strength=100)
    assert _blue_chroma(good, sky) < _blue_chroma(whole_box, sky)
    assert _blue_chroma(good, sky) < _blue_chroma(full_matte, sky)


def test_neutralize_cast_survives_a_black_reference() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert np.array_equal(neutralize_cast(frame, strength=100), frame)


# ------------------------------------------------------------------ #
# Combined per-shot transform                                        #
# ------------------------------------------------------------------ #
def test_apply_shot_transform_runs_inset_and_despill() -> None:
    frame, mask, fringe = _fringe_scene()
    out_frame, out_mask = apply_shot_transform(
        frame,
        mask,
        spill_mode="despill",
        spill_color=BLUE,
        spill_tolerance=30,
        spill_radius=3,
    )
    assert _blue_chroma(out_frame, fringe) < 40
    assert np.array_equal(out_mask, mask), "despill leaves the matte alone"


def test_apply_shot_transform_applies_inset_before_spill() -> None:
    frame, mask, _ = _fringe_scene()
    _, out_mask = apply_shot_transform(
        frame,
        mask,
        mask_inset=2,
        spill_mode="alpha_key",
        spill_color=BLUE,
        spill_tolerance=30,
        spill_radius=3,
    )
    assert out_mask.sum() < mask.sum()


def test_apply_shot_transform_is_a_noop_when_off() -> None:
    frame, mask, _ = _fringe_scene()
    out_frame, out_mask = apply_shot_transform(frame, mask, spill_mode="off")
    assert out_frame is frame
    assert np.array_equal(out_mask, mask)


def test_apply_shot_transform_without_a_frame_only_keys_when_possible() -> None:
    _, mask, _ = _fringe_scene()
    out_frame, out_mask = apply_shot_transform(
        None, mask, spill_mode="despill", spill_color=BLUE
    )
    assert out_frame is None
    assert np.array_equal(out_mask, mask)


# ------------------------------------------------------------------ #
# Canvas wiring                                                      #
# ------------------------------------------------------------------ #
def test_eyedropper_samples_the_canvas_pixel() -> None:
    from src.ui.video_canvas import VideoCanvas

    _app()
    canvas = VideoCanvas()
    try:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[5:16, 5:16] = (10, 20, 250)
        canvas.set_frame(frame)
        # The whole 5x5 neighbourhood is the same colour, so the median is it.
        assert canvas.pixel_at(10, 10) == (10, 20, 250)
        assert canvas.pixel_at(999, 999) is None

        picked: list[tuple[int, int, int]] = []
        canvas.color_picked.connect(lambda *rgb: picked.append(rgb))
        canvas.begin_color_pick()
        assert canvas.is_picking_color()

        # Route a click straight through the image-coordinate mapper.
        canvas._widget_to_image = lambda wx, wy: (10, 10)
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent

        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(1.0, 1.0),
            QPointF(1.0, 1.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mousePressEvent(event)
        assert picked == [(10, 20, 250)]
        assert not canvas.is_picking_color(), "the pick ends after a sample"
    finally:
        canvas.deleteLater()


def test_eyedropper_ignores_a_stray_noisy_pixel() -> None:
    """The patch median is the point: one artifact must not steer the hue."""
    from src.ui.video_canvas import VideoCanvas

    _app()
    canvas = VideoCanvas()
    try:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[5:16, 5:16] = (10, 20, 250)
        frame[10, 10] = (255, 255, 255)  # a single compression artifact
        canvas.set_frame(frame)
        assert canvas.pixel_at(10, 10) == (10, 20, 250)
    finally:
        canvas.deleteLater()


def test_escape_cancels_the_eyedropper() -> None:
    from src.ui.video_canvas import VideoCanvas

    _app()
    canvas = VideoCanvas()
    try:
        canvas.set_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        cancelled: list[bool] = []
        canvas.color_pick_cancelled.connect(lambda: cancelled.append(True))
        canvas.begin_color_pick()
        from PySide6.QtGui import QKeyEvent

        canvas.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert cancelled == [True]
        assert not canvas.is_picking_color()
    finally:
        canvas.deleteLater()


def test_apply_shot_transform_passes_the_region_through() -> None:
    frame, eye, sky = _split_scene()
    mask = np.ones(frame.shape[:2], dtype=bool)
    region = (60, 0, 60, 120)

    out_frame, _ = apply_shot_transform(
        frame,
        mask,
        spill_mode="despill",
        spill_color=BLUE,
        spill_tolerance=100,
        spill_region=region,
    )
    assert _blue_chroma(out_frame, sky) < 10
    assert np.array_equal(out_frame[eye], frame[eye])

    _, keyed = apply_shot_transform(
        frame,
        mask,
        spill_mode="alpha_key",
        spill_color=BLUE,
        spill_tolerance=100,
        spill_region=region,
    )
    assert not keyed[sky].any()
    assert keyed[eye].all()


def _Buf(frame):
    """A frame-buffer stand-in that always returns the same frame."""
    return type(
        "Buf",
        (),
        {
            "get": staticmethod(lambda i: frame),
            "close": staticmethod(lambda: None),
        },
    )()


def MainWindow_():
    from src.ui.main_window import MainWindow

    return MainWindow()


# ------------------------------------------------------------------ #
# Shot Profile dialog: colour controls moved off the sidebar          #
# ------------------------------------------------------------------ #
def _panel(win, frame=None):
    """Return a live SpillPanel loaded with the current shot's settings."""
    from src.ui.components.spill_panel import SpillPanel

    panel = SpillPanel()
    panel.set_profile(win._active_profile())
    if frame is not None:
        panel.set_frame(frame)
    return panel


def test_sidebar_no_longer_carries_the_colour_controls() -> None:
    """The controls were moved into the Shot Profile dialog to shorten it."""
    _app()
    win = MainWindow_()
    try:
        for gone in (
            "_spill_mode_combo",
            "_btn_spill_pick",
            "_btn_spill_area",
            "_btn_spill_protect",
            "_chk_cast",
            "_cast_slider",
            "_lbl_spill_area",
        ):
            assert not hasattr(win, gone), f"{gone} is still in the sidebar"
    finally:
        win.close()


def test_panel_round_trips_every_colour_setting() -> None:
    _app()
    win = MainWindow_()
    try:
        panel = _panel(win)
        panel._mode.setCurrentIndex(panel._mode.findData("despill"))
        panel._on_canvas_color(10, 20, 250)
        panel._tolerance.slider.setValue(72)
        panel._radius.slider.setValue(5)
        panel._softness.slider.setValue(3)
        panel._cast.setChecked(True)
        panel._cast_strength.slider.setValue(60)
        panel._begin_area("protect")
        panel._on_canvas_region(30, 40, 100, 60)

        got = panel.profile()
        assert got.spill_mode == "despill"
        assert got.spill_color == (10, 20, 250)
        assert got.spill_tolerance == 72
        assert got.spill_radius == 5
        assert got.edge_softness == 3
        assert got.cast_neutralize is True
        assert got.cast_strength == 60
        assert got.spill_region == (30, 40, 100, 60)
        assert got.spill_region_mode == "protect"
    finally:
        win.close()


def test_panel_keeps_the_previous_shot_settings() -> None:
    """set_profile must load without firing change handlers."""
    _app()
    win = MainWindow_()
    try:
        from src.core.profiles import ShotProfile

        stored = ShotProfile(
            spill_color=(1, 2, 3),
            spill_mode="alpha_key",
            spill_tolerance=41,
            spill_region=(1, 2, 3, 4),
            spill_region_mode="protect",
            cast_neutralize=True,
            cast_strength=17,
            edge_softness=4,
        )
        seen: list[bool] = []
        panel = _panel(win)
        panel.changed.connect(lambda: seen.append(True))
        panel.set_profile(stored)
        assert seen == [], "loading must not look like a user edit"
        assert panel.profile() == stored
    finally:
        win.close()


def test_panel_persists_through_the_main_window_bridge() -> None:
    _app()
    win = MainWindow_()
    try:
        panel = _panel(win)
        panel._mode.setCurrentIndex(panel._mode.findData("despill"))
        panel._on_canvas_color(30, 60, 230)
        panel._begin_area("protect")
        panel._on_canvas_region(2, 18, 56, 46)

        win._on_spill_panel_changed(panel.profile())

        stored = win._active_profile()
        assert stored.spill_color == (30, 60, 230)
        assert stored.spill_region == (2, 18, 56, 46)
        assert stored.spill_region_mode == "protect"
        assert win._shot_profiles.default.spill_color == (30, 60, 230)
    finally:
        win.close()


def test_dialog_embeds_the_panel_with_a_live_frame() -> None:
    from src.core.profiles import shot_ranges
    from src.ui.components.shot_profile_dialog import ShotProfileDialog

    _app()
    win = MainWindow_()
    try:
        frame = np.full((40, 40, 3), (30, 60, 230), dtype=np.uint8)
        dialog = ShotProfileDialog(
            "Object",
            shot_ranges(10, []),
            1,
            win._active_profile(),
            parent=win,
            frame=frame,
        )
        try:
            assert hasattr(dialog, "_spill")
            assert dialog._spill._canvas._frame is not None, (
                "the eyedropper needs a frame to sample from"
            )
        finally:
            dialog.close()
    finally:
        win.close()


def test_main_window_protect_box_spares_the_subject() -> None:
    _app()
    win = MainWindow_()
    try:
        frame, eye, sky = _split_scene(size=120)
        win._frame_buffer = _Buf(frame)
        win._masks = {0: np.ones(frame.shape[:2], dtype=bool)}
        win._current_frame_idx = 0

        panel = _panel(win, frame)
        panel._mode.setCurrentIndex(panel._mode.findData("despill"))
        panel._on_canvas_color(*BLUE)
        panel._tolerance.slider.setValue(100)
        panel._begin_area("protect")
        panel._on_canvas_region(10, 40, 45, 45)
        win._on_spill_panel_changed(panel.profile())

        win._show_current_frame()
        shown = win._canvas._frame
        assert np.array_equal(shown[eye], frame[eye]), "protected eye must survive"
        assert _blue_chroma(shown, sky) < 10, "the sky still gets cleaned"
    finally:
        win.close()


def test_main_window_preview_does_not_touch_cached_masks() -> None:
    _app()
    win = MainWindow_()
    try:
        frame, mask, _ = _fringe_scene(size=40, radius=12)
        win._frame_buffer = _Buf(frame)
        win._masks = {0: mask}
        win._current_frame_idx = 0
        original = mask.copy()

        panel = _panel(win, frame)
        panel._mode.setCurrentIndex(panel._mode.findData("despill"))
        panel._on_canvas_color(*BLUE)
        panel._radius.slider.setValue(3)
        win._on_spill_panel_changed(panel.profile())
        win._show_current_frame()

        assert np.array_equal(win._masks[0], original), (
            "spill is a view transform; the cached matte must stay raw"
        )
        assert not np.array_equal(win._canvas._frame, frame)
    finally:
        win.close()


def test_cast_fix_is_independent_of_the_cleaner() -> None:
    _app()
    win = MainWindow_()
    try:
        frame, _, sky, mask = _tinted_scene()
        win._frame_buffer = _Buf(frame)
        win._masks = {0: mask}
        win._current_frame_idx = 0

        panel = _panel(win, frame)
        # Cleaner stays Off, but the cast fix still applies.
        panel._cast.setChecked(True)
        win._on_spill_panel_changed(panel.profile())

        stored = win._active_profile()
        assert stored.cast_neutralize is True
        assert stored.spill_mode == "off"

        win._show_current_frame()
        assert _blue_chroma(win._canvas._frame, sky) < 20
    finally:
        win.close()


if __name__ == "__main__":
    _app()
    import inspect

    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
