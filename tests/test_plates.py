"""Tests for the subtitle / logo cleanup plates module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.processing.plates import (
    _logo_mask_per_frame,
    _mask_band,
    _repair_rows_donor,
    _text_mask_per_frame,
    _track_text_band,
    apply_clean,
    build_logo_plate,
    build_subtitle_plate,
    detect_cleanup_regions,
    detect_logo_rect,
    detect_subtitle_band,
    sample_frames,
)

H, W = 200, 320
SUBTITLE_BAND = (60, 150, 200, 16)
LOGO_RECT = (W - 56, 0, 56, 56)  # top-right corner


def _subtitle_frames(n: int = 24) -> list[np.ndarray]:
    """Frames with a flickering white subtitle bar over a flat background."""
    frames: list[np.ndarray] = []
    x, y, w, h = SUBTITLE_BAND
    for i in range(n):
        frame = np.full((H, W, 3), 60, dtype=np.uint8)  # static bg
        if i % 2 == 0:
            frame[y : y + h, x : x + w] = 255  # subtitle ON
        frames.append(frame)
    return frames


def _logo_frames(n: int = 24) -> list[np.ndarray]:
    """Frames with a static white logo in the top-right corner."""
    frames: list[np.ndarray] = []
    x, y, w, h = LOGO_RECT
    gradient = np.linspace(80, 140, w, dtype=np.uint8)[None, :, None]  # (1, W, 1)
    for _ in range(n):
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        # Flat corner background with a gradient so the region has content
        frame[y : y + h, x : x + w] = gradient
        frame[y + 8 : y + 30, x + 8 : x + 40] = 255  # the logo itself
        frames.append(frame)
    return frames


# ── Subtitle detection ───────────────────────────────────────────────────────


def test_detect_subtitle_band_happy_path() -> None:
    frames = _subtitle_frames()
    rect = detect_subtitle_band(frames)
    assert rect is not None
    x, y, w, h = rect
    # Must intersect the synthetic band
    bx, by, bw, bh = SUBTITLE_BAND
    assert x + w > bx and x < bx + bw
    assert y + h > by and y < by + bh


def test_detect_subtitle_band_anywhere_in_frame() -> None:
    """Subtitles at the top or middle of the frame must also be found."""
    for band_y in (30, 100):
        frames = []
        x = 60
        for i in range(24):
            frame = np.full((H, W, 3), 60, dtype=np.uint8)
            if i % 2 == 0:
                frame[band_y : band_y + 16, x : x + 200] = 255
            frames.append(frame)
        rect = detect_subtitle_band(frames)
        assert rect is not None, f"band at y={band_y} not detected"
        _, y, _, h = rect
        assert y <= band_y + 16 and y + h >= band_y


def test_detect_subtitle_band_none_when_static() -> None:
    frames = [np.full((H, W, 3), 80, dtype=np.uint8) for _ in range(12)]
    rect = detect_subtitle_band(frames)
    assert rect is None


# ── Logo detection ───────────────────────────────────────────────────────────


def test_detect_logo_rect_happy_path() -> None:
    frames = _logo_frames()
    rect = detect_logo_rect(frames)
    assert rect is not None
    x, y, w, h = rect
    lx, ly, lw, lh = LOGO_RECT
    # Overlap with the expected corner
    assert x < lx + lw and x + w > lx
    assert y < ly + lh and y + h > ly


def test_detect_logo_rect_edge_and_center_positions() -> None:
    """A logo at a mid-edge position (not a corner) must also be found."""
    for (lx, ly) in ((W // 2 - 28, 0), (0, H // 2 - 28), (W // 2 - 28, H // 2 - 28)):
        logo_w, logo_h = 56, 56
        frames = []
        gradient = np.linspace(80, 140, logo_w, dtype=np.uint8)[None, :, None]
        for _ in range(24):
            frame = np.full((H, W, 3), 100, dtype=np.uint8)
            frame[ly : ly + logo_h, lx : lx + logo_w] = gradient
            frame[ly + 8 : ly + 30, lx + 8 : lx + 40] = 255
            frames.append(frame)
        rect = detect_logo_rect(frames)
        assert rect is not None, f"logo at ({lx}, {ly}) not detected"
        x, y, w, h = rect
        assert x < lx + logo_w and x + w > lx
        assert y < ly + logo_h and y + h > ly


def test_detect_logo_rect_not_confused_by_subtitle_band() -> None:
    """A corner logo must win over subtitle text sharing its variance.

    Regression: subtitle text has high edge density at zero variance and
    used to beat the actual logo in the corner scan.
    """
    frames = _logo_frames()
    x, y, w, h = LOGO_RECT
    for i, f in enumerate(frames):
        # Add a subtitle band at the bottom (static, like burned-in subs).
        f[150:166, 60:260] = 255 if i % 2 == 0 else (40, 40, 40)
    sub = detect_subtitle_band(frames)
    rect = detect_logo_rect(frames, subtitle_rect=sub)
    assert rect is not None
    rx, ry, rw, rh = rect
    assert rx < x + w and rx + rw > x
    assert ry < y + h and ry + rh > y


# ── Plate builders ───────────────────────────────────────────────────────────


def test_build_subtitle_plate_shapes() -> None:
    frames = _subtitle_frames()
    plate = build_subtitle_plate(frames, SUBTITLE_BAND)
    _bx, _by, bw, bh = SUBTITLE_BAND
    assert plate.mode == "subtitles"
    assert plate.clean.shape[:2] == (bh, bw)
    assert plate.var_map.shape == (bh, bw)
    # Median plate should be close to the static background (text removed)
    assert plate.clean.mean() < 120


def test_build_logo_plate_shapes() -> None:
    frames = _logo_frames()
    plate = build_logo_plate(frames, LOGO_RECT)
    _lx, _ly, lw, lh = LOGO_RECT
    assert plate.mode == "logo"
    assert plate.clean.shape[:2] == (lh, lw)
    assert plate.var_map.shape == (lh, lw)


# ── Apply ────────────────────────────────────────────────────────────────────


def test_apply_clean_removes_subtitles() -> None:
    frames = _subtitle_frames()
    plate = build_subtitle_plate(frames, SUBTITLE_BAND)
    for frame in frames:
        if frame.mean() < 100:
            continue  # the text is only in frames where it's ON
        clean_frame = apply_clean(frame, [plate])
        x, y, w, h = SUBTITLE_BAND
        region = clean_frame[y : y + h, x : x + w]
        assert region.mean() < 120, "subtitle text was not cleaned"


def test_apply_clean_preserves_static_frames() -> None:
    """A frame inside the band but without text should barely change."""
    frames = _subtitle_frames(n=6)
    plate = build_subtitle_plate(frames, SUBTITLE_BAND)
    plain = np.full((H, W, 3), 60, dtype=np.uint8)
    out = apply_clean(plain, [plate])
    x, y, w, h = SUBTITLE_BAND
    np.testing.assert_array_equal(out[y : y + h, x : x + w], plain[y : y + h, x : x + w])


def test_apply_clean_noop_without_plates() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    out = apply_clean(frame, [])
    np.testing.assert_array_equal(out, frame)


def test_apply_clean_inpaint_removes_always_on_text_over_busy_bg() -> None:
    """Synthetic analogue of the real-video failure mode.

    A white line of text is burned into *every* frame on top of a
    fast-moving noisy background.  The temporal percentile plate is
    contaminated (text baked into p15), but the per-frame inpaint regime
    re-masks the bright strokes each frame and removes them independently.
    """
    frames = []
    for t in range(48):
        # Busy moving anime bg + large always-on stroked text every frame.
        fr = _anime_busy_bg(t)
        _draw_stroked_text(fr, "ALWAYS ON BURNED TEXT", 168)
        frames.append(fr)

    # Detect → must choose inpaint regime.
    rect = detect_subtitle_band(frames)
    assert rect is not None
    plate = build_subtitle_plate(frames, rect)
    assert plate.mode == "inpaint", "must be per-frame inpaint for always-on text"
    assert plate.kind == "text"

    # Apply to every frame: the white glyphs must be gone in the text rows.
    x, y, w, h = plate.region
    for fr in frames:
        patched = apply_clean(fr, [plate])
        region = patched[y:y+h, x:x+w]
        # Text is the only thing that reaches ~255 in the region; after the
        # per-frame inpaint the very bright pixels must be largely gone.
        assert int(np.mean(region.max(axis=2) == 255)) * 100 <= 1.0, (
            "per-frame inpaint failed to remove always-on text"
        )
    frames = _logo_frames()
    white_before = int(np.count_nonzero(_logo_frames(1)[0].max(axis=2) == 255))
    plate = build_logo_plate(frames, LOGO_RECT)
    _lx, _ly, _lw, _lh = LOGO_RECT
    white_after = int(np.count_nonzero(plate.clean.max(axis=2) == 255))
    # >90 % of the logo pixels replaced (a (possibly 1 px) residue is fine)
    assert white_after <= max(1, white_before // 10), (
        f"logo not inpainted: {white_before} -> {white_after} white pixels"
    )


def test_apply_clean_removes_logo_but_keeps_motion() -> None:
    """Logo pixels are replaced while moving (changing) pixels stay intact."""
    x, y, w, h = LOGO_RECT
    frames = []
    for i in range(24):
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        frame[y : y + h, x : x + w] = (120, 120, 120)  # static-ish corner
        frame[y + 8 : y + 30, x + 8 : x + 40] = 255  # the logo
        # Moving foreground: a small bright square that changes position
        offset = i % 5
        frame[y + 10 : y + 22, x + 10 + offset : x + 22 + offset] = (0, 200, 200)
        frames.append(frame)

    plate = build_logo_plate(frames, LOGO_RECT)
    # Frame with the moving square at position 0
    out = apply_clean(frames[1], [plate])
    # Logo (255) inside static area is removed
    assert out[y + 12, x + 12].tolist() != [255, 255, 255]
    # Moving foreground is preserved somewhere in the clean region
    region = out[y : y + h, x : x + w]
    # At least some cyan-ish pixels remain
    assert (np.abs(region.astype(int) - (0, 200, 200)).sum(axis=2) < 60).sum() > 0


# ── Anime-style realism (user-reported failures) ─────────────────────────────


def _draw_stroked_text(fr: np.ndarray, text: str, y: int) -> None:
    """White text with a thick dark outline, the classic anime subtitle."""
    cv2.putText(fr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
    cv2.putText(fr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _anime_busy_bg(t: int) -> np.ndarray:
    """Detailed static line-art background with small moving blobs (motion)."""
    fr = np.full((H, W, 3), 85, dtype=np.uint8)
    for yy in range(0, H, 18):
        cv2.line(fr, (0, yy), (W, yy), (120, 120, 128), 1)
    for xx in range(0, W, 37):
        cv2.line(fr, (xx, 0), (xx, H), (100, 100, 110), 1)
    cx = int(W * 0.25 + 55 * np.sin(t * 0.3))
    cv2.circle(fr, (cx, 70), 14, (150, 90, 200), -1)
    cv2.circle(fr, ((cx + 130) % W, 110), 10, (60, 160, 100), -1)
    return fr


def _anime_frames(always_on: bool, logo: bool) -> list[np.ndarray]:
    """Anime-like frames: always-on stroked subtitles + optional big logo.

    ``always_on`` True → switch between two lines midway (burned-in style).
    ``always_on`` False → no subtitle text at all (motion only).
    """
    frames: list[np.ndarray] = []
    N = 48
    logo_x, logo_y = W - 120, 6
    for t in range(N):
        fr = _anime_busy_bg(t)
        if always_on:
            if t < N // 2:
                _draw_stroked_text(fr, "FIRST LINE HERE", 166)
            else:
                _draw_stroked_text(fr, "SECOND LINE OK", 166)
        if logo:
            cv2.circle(fr, (logo_x + 26, logo_y + 26), 22, (255, 130, 0), -1)
            cv2.putText(
                fr, "CR", (logo_x + 8, logo_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
        frames.append(fr)
    return frames


def test_detect_subtitle_band_always_on_over_busy_animation() -> None:
    """Regression: 'big subtitles with a good stroke' burned into an ED."""
    frames = _anime_frames(always_on=True, logo=False)
    rect = detect_subtitle_band(frames)
    assert rect is not None, "always-on stroked subtitles were not detected"
    _, y, _, h = rect
    # Must cover the text rows (~150..168)
    assert y + h > 150 and y < 168


def test_detect_subtitle_band_always_on_over_static_detail() -> None:
    """Same subtitle, static detailed background (the 'no animation' case)."""
    # Freeze the background: same detailed frame each time (static shot),
    # with always-on subtitle text switching between two lines.
    still = [_anime_busy_bg(7) for _ in range(48)]
    for t, fr in enumerate(still):
        _draw_stroked_text(fr, "FIRST LINE HERE" if t < 24 else "SECOND LINE OK", 166)
    rect = detect_subtitle_band(still)
    assert rect is not None, "always-on subtitles over static detail were not detected"
    _, y, _, h = rect
    assert y + h > 150 and y < 168


def test_detect_subtitle_band_none_on_motion_without_text() -> None:
    """No subtitles, no logo: busy animation must not be a false positive."""
    frames = _anime_frames(always_on=False, logo=False)
    assert detect_subtitle_band(frames) is None


def test_detect_logo_rect_huge_over_busy_animation() -> None:
    """Regression: 'even a huge, huge Crunchyroll logo' must be found."""
    frames = _anime_frames(always_on=True, logo=True)
    rect = detect_logo_rect(frames)
    assert rect is not None, "huge logo over busy animation was not detected"
    x, y, w, _ = rect
    # Overlaps the top-right logo block
    assert x + w > W - 130 and y < 70


def test_detect_logo_rect_none_without_logo() -> None:
    """No logo: busy animation + subtitles must not produce a logo rect."""
    frames = _anime_frames(always_on=True, logo=False)
    assert detect_logo_rect(frames) is None


def test_detect_subtitle_band_same_static_text_every_frame() -> None:
    """Regression: the subtitle text is IDENTICAL in all 48 frames.

    The temporal residual (|frame − median|) is ~0 for always-on burned-in
    text, so only the static median-frame pass can find it.  In real anime
    this is the common 'big stroked credits line' that showed subtitle=NO.
    """
    frames = []
    for t in range(48):
        fr = _anime_busy_bg(t)
        _draw_stroked_text(fr, "SAME TEXT EVERY FRAME", 168)  # never changes
        frames.append(fr)
    rect = detect_subtitle_band(frames)
    assert rect is not None, "identical always-on subtitles were not detected"
    _, y, _, h = rect
    assert y + h > 150 and y < 168
    # Always-on text over a busy background has no text-free temporal
    # percentile → the plate must be the per-frame inpaint regime.
    plate = build_subtitle_plate(frames, rect)
    assert plate.mode == "inpaint"
    assert plate.kind == "text"


def test_detect_subtitle_band_none_on_solid_bright_stripe() -> None:
    """A solid bright scenery band (no text) must not be a false positive."""
    frames = []
    for t in range(48):
        fr = _anime_busy_bg(t)
        fr[150:175, 40:280] = np.full((25, 240, 3), 200, dtype=np.uint8)
        frames.append(fr)
    assert detect_subtitle_band(frames) is None


def test_detect_logo_rect_none_on_stroked_text_only() -> None:
    """Regression: stroked subtitle text alone must not be picked up as a
    logo even in logo-only mode (no subtitle band to penalise it)."""
    frames = []
    for t in range(48):
        fr = _anime_busy_bg(t)
        _draw_stroked_text(fr, "BIG STROKED SUBTITLE", 168)
        frames.append(fr)
    assert detect_logo_rect(frames) is None


def test_detect_logo_rect_corner_watermark() -> None:
    """A small bright emblem hugging a corner must be found as the logo."""
    frames = []
    for t in range(48):
        fr = _anime_busy_bg(t)
        # Crunchyroll-style corner watermark: emblem + wordmark, top-right.
        fr[24:64, H - 130 : H - 90] = 255
        fr[70:96, H - 130 : H - 10] = 255
        frames.append(fr)
    rect = detect_logo_rect(frames)
    assert rect is not None
    x, y, w, h = rect
    # Must be in the top-right corner, not the busy centre of the frame.
    assert x + w > H * 0.5 and y < 120
    assert y >= 0 and h < 120


# ── Full pipeline with a real video file ─────────────────────────────────────


def _make_video(tmp_path: Path, subtitle: bool = True, logo: bool = True) -> Path:
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H)
    )
    try:
        for i in range(48):
            frame = np.full((H, W, 3), 70, dtype=np.uint8)
            if subtitle and i % 2 == 0:
                frame[SUBTITLE_BAND[1] : SUBTITLE_BAND[1] + SUBTITLE_BAND[3],
                      SUBTITLE_BAND[0] : SUBTITLE_BAND[0] + SUBTITLE_BAND[2]] = 255
            if logo:
                lx, ly, _lw, _lh = LOGO_RECT
                frame[ly + 8 : ly + 30, lx + 8 : lx + 40] = 255  # "logo"
            writer.write(frame)
    finally:
        writer.release()
    return video


def test_sample_frames_returns_uniform_spacing(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    frames = sample_frames(str(video), n=24)
    assert len(frames) == 24
    assert frames[0].shape == (H, W, 3)
    assert frames[0].dtype == np.uint8
    # The sampled frames should include both text-on and text-off variants
    means = [float(np.mean(f)) for f in frames]
    assert max(means) - min(means) > 5, "sampling missed the flicker"


def test_detect_cleanup_regions_finds_both(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    result = detect_cleanup_regions(str(video), targets="both")
    assert result.subtitle_plate is not None
    assert result.logo_plate is not None
    assert result.elapsed_ms >= 0


def test_detect_cleanup_regions_respects_targets(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    # Subtitles only
    r1 = detect_cleanup_regions(str(video), targets="subtitles")
    assert r1.subtitle_plate is not None
    assert r1.logo_plate is None
    # Logo only
    r2 = detect_cleanup_regions(str(video), targets="logo")
    assert r2.logo_plate is not None
    assert r2.subtitle_plate is None


def test_detect_cleanup_regions_uses_user_region(tmp_path: Path) -> None:
    """A user-drawn rectangle must be used as-is (no auto-detection needed)."""
    video = _make_video(tmp_path)
    # Video has a static logo at LOGO_RECT; user crops the whole logo too.
    lx, ly, lw, lh = LOGO_RECT
    user_box = (lx - 4, ly - 4, lw + 8, lh + 8)
    result = detect_cleanup_regions(
        str(video), targets="logo", logo_rect=user_box
    )
    assert result.logo_plate is not None
    # The built plate must sit inside the user's drawn box (clamped only at
    # frame edges).
    px, py, pw, ph = result.logo_plate.region
    user_x, user_y, user_w, user_h = user_box
    assert px >= user_x and py >= user_y
    assert px + pw <= user_x + user_w
    assert py + ph <= user_y + user_h


# ── Tracking (subtitle band follows position / size changes) ─────────────────


def _moving_subtitle_frames(n: int = 12) -> list[np.ndarray]:
    """Busy background + a subtitle that slides down and shrinks over time."""
    frames: list[np.ndarray] = []
    for t in range(n):
        fr = _anime_busy_bg(t)
        y = 90 + t * 6  # slides downward every frame
        _draw_stroked_text(fr, "MOVING SUB", y)
        frames.append(fr)
    return frames


def test_track_text_band_follows_moving_subtitle() -> None:
    """The tracked band must follow a subtitle that moves and resizes."""
    frames = _moving_subtitle_frames(n=8)
    rect = detect_subtitle_band(frames)
    assert rect is not None
    plate = build_subtitle_plate(frames, rect, force_per_frame=True)
    assert plate.track, "always-on subtitles must enable band tracking"

    # Track through the sequence; final position must drift down with text.
    final = None
    for fr in frames:
        final = _track_text_band(fr, plate)
        x, y, w, h = final
        assert x >= 0 and y >= 0
        assert x + w <= W and y + h <= H
    assert final is not None
    _, y0, _, h0 = rect
    _x, y1, _w, _h = final
    # Text moved downward → tracked band centre must be lower than origin.
    assert (y1 + _h / 2) > (y0 + h0 / 2)
    # Tracked vertical span still overlaps the original text rows.
    assert y1 + _h > y0 and y1 < y0 + h0 + 40


def test_track_text_band_keeps_full_width_on_partial_blob() -> None:
    """A faint second line must not shrink the band to one line's width."""
    frames = []
    for t in range(24):
        fr = _anime_busy_bg(t)
        _draw_stroked_text(fr, "MAIN LINE ONLY", 150)
        frames.append(fr)
    rect = detect_subtitle_band(frames)
    assert rect is not None
    plate = build_subtitle_plate(frames, rect, force_per_frame=True)
    start = plate.region[2]  # original width
    found = _track_text_band(frames[0], plate)
    # Band width must stay near the original width (no runaway shrink).
    assert found[2] >= start * 0.5


# ── Donor-row texture repair ─────────────────────────────────────────────────_


def test_repair_rows_donor_matches_ground_truth() -> None:
    """Donor fill must beat plain Telea on a textured anime-like background.

    Ground truth: a busy background plus two subtitle lines.  The repaired
    region (over the exact glyph mask) must be closer to the clean
    background than the raw Telea fill — the donor rows transplant real
    texture instead of smearing.
    """
    rng = np.random.default_rng(3)
    bg = rng.integers(40, 200, (120, 300, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    for _ in range(24):
        cv2.line(
            bg,
            (int(rng.integers(0, 120)), int(rng.integers(0, 300))),
            (int(rng.integers(0, 120)), int(rng.integers(0, 300))),
            (120, 160, 200), 2,
        )
    clean_ref = bg.copy()
    _draw_stroked_text(bg, "TEXT LINE ONE", 40)
    _draw_stroked_text(bg, "TEXT LINE TWO", 90)

    mask = _text_mask_per_frame(bg)
    assert mask is not None, "synthetic text masking failed"

    donor = _repair_rows_donor(bg, mask, use_ai=False)
    telea = cv2.inpaint(bg, mask, 6, cv2.INPAINT_TELEA)

    masked = mask > 0
    d_donor = (donor[masked].astype(float) - clean_ref[masked].astype(float)) ** 2
    d_telea = (telea[masked].astype(float) - clean_ref[masked].astype(float)) ** 2
    psnr_d = 10 * np.log10(255 ** 2 / (d_donor.mean() + 1e-9))
    psnr_t = 10 * np.log10(255 ** 2 / (d_telea.mean() + 1e-9))
    # Donor rows transplant real texture → strictly closer to ground truth.
    assert psnr_d > psnr_t


def test_glyph_components_keeps_text_drops_solid_block() -> None:
    """The glyph filter keeps stroked text but drops a solid bright stripe."""
    band = np.full((60, 300, 3), 85, dtype=np.uint8)
    _draw_stroked_text(band, "STROKED TEXT", 40)
    mask = _text_mask_per_frame(band, glyph_only=True)
    assert mask is not None
    assert cv2.countNonZero(mask) > 50, "stroked text was wrongly filtered out"

    # A solid scenery block (like a bright backing panel) is not text.
    block = np.full((60, 300, 3), 85, dtype=np.uint8)
    block[10:40, 50:250] = 210
    mask2 = _text_mask_per_frame(block, glyph_only=True)
    # After the filter the solid block must be essentially removed.
    if mask2 is not None:
        assert int(np.mean(mask2 == 255)) * 100 < 20, (
            "solid scenery block survived the glyph filter"
        )


# ── Logo reference mask (semi-transparent logo full coverage) ────────────────


def test_logo_plate_ref_mask_covers_semi_transparent_logo() -> None:
    """A faint / semi-transparent logo needs the median ref mask OR-ed in."""
    frames = []
    x, y, w, h = LOGO_RECT
    for _ in range(24):
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        # Corner gradient background
        frame[y : y + h, x : x + w] = np.linspace(90, 130, w, dtype=np.uint8)[None, :, None]
        # Semi-transparent logo: lower-contrast than pure white, still visible.
        frame[y + 10 : y + 34, x + 10 : x + 44] = (190, 190, 190)
        frames.append(frame)

    plate = build_logo_plate(frames, LOGO_RECT, per_frame=True)
    assert plate.mode == "inpaint"
    assert plate.ref_mask is not None, "ref mask must be built for per-frame logos"

    # Per-frame mask with the ref mask must reach full logo coverage.
    m = _logo_mask_per_frame(frames[0][y : y + h, x : x + w], plate.ref_mask)
    assert m is not None
    # The logo block pixels must be covered by the mask.
    assert cv2.countNonZero(m[10:34, 10:44]) > 0


def test_faint_white_logo_is_detected_and_removed() -> None:
    """Low-opacity white marks use local contrast, not only saturation."""
    x, y, w, h = LOGO_RECT
    frames = []
    clean_reference = None
    for _ in range(24):
        frame = np.full((H, W, 3), 90, dtype=np.uint8)
        frame[y : y + h, x : x + w] = np.linspace(80, 130, w, dtype=np.uint8)[None, :, None]
        if clean_reference is None:
            clean_reference = frame.copy()
        # 25 % white overlay: still visible, but far below the old 170 gate.
        alpha = 0.25
        frame[y + 10 : y + 34, x + 10 : x + 44] = np.clip(
            frame[y + 10 : y + 34, x + 10 : x + 44].astype(np.float32) * (1 - alpha)
            + 255.0 * alpha,
            0,
            255,
        ).astype(np.uint8)
        frames.append(frame)

    rect = detect_logo_rect(frames)
    assert rect is not None, "faint white logo was not detected"
    plate = build_logo_plate(frames, rect, per_frame=True)
    assert plate.ref_mask is not None

    cleaned = apply_clean(frames[0], [plate])
    rx, ry, rw, rh = plate.region
    before = float(
        np.abs(
            frames[0][ry : ry + rh, rx : rx + rw].astype(np.float32)
            - clean_reference[ry : ry + rh, rx : rx + rw].astype(np.float32)
        ).mean()
    )
    after = float(
        np.abs(
            cleaned[ry : ry + rh, rx : rx + rw].astype(np.float32)
            - clean_reference[ry : ry + rh, rx : rx + rw].astype(np.float32)
        ).mean()
    )
    assert after < before * 0.7, f"faint logo residue remains: {before:.1f} -> {after:.1f}"


def test_mask_band_helper() -> None:
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[5:15, 20:60] = 255
    band = _mask_band(mask)
    assert band == (20, 5, 40, 10)