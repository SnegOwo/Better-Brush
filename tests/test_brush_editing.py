"""Offscreen GUI tests: brush editing, preview modes, toolbar wiring.

No GPU / SAM 2 weights are required — only Qt's event machinery.

Run with:
    python -m pytest tests/test_brush_editing.py -v
or standalone:
    python -m tests.test_brush_editing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.video_canvas import VideoCanvas

_APP = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _canvas_with_frame() -> VideoCanvas:
    canvas = VideoCanvas()
    canvas.resize(720, 480)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[:, :] = [10, 120, 200]
    canvas.set_frame(frame)
    return canvas


def _img_center(canvas: VideoCanvas, ix: int, iy: int) -> QPoint:
    """Widget coords for an image pixel (using the current mapping)."""
    ox, oy = canvas._offset.x(), canvas._offset.y()
    return QPoint(int(ox + ix * canvas._scale), int(oy + iy * canvas._scale))


def test_brush_paints_and_emits() -> None:
    _app()
    canvas = _canvas_with_frame()
    canvas.set_edit_mode("brush")
    canvas.set_brush_size(12)

    events: list[str] = []
    canvas.mask_edited.connect(lambda: events.append("edited"))

    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        _img_center(canvas, 80, 50),
    )
    mask = canvas.get_mask()
    assert mask is not None, "brush stroke did not create a mask"
    assert mask.any(), "brush stroke did not paint anything"
    assert events == ["edited"], "mask_edited not emitted on mouse release"

    painted = int(mask.sum())
    canvas.set_brush_erase(True)
    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        _img_center(canvas, 80, 50),
    )
    assert canvas.get_mask().sum() < painted, "erase did not remove pixels"


def test_brush_stroke_on_main_window_stores_mask() -> None:
    _app()
    win = MainWindow()
    try:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[:, :] = [10, 120, 200]
        win._canvas.setFixedSize(720, 480)
        win._canvas.set_frame(frame)
        win._canvas.set_edit_mode("brush")
        win._canvas.set_brush_size(10)
        QTest.mouseClick(
            win._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _img_center(win._canvas, 80, 50),
        )
        assert 0 in win._masks, "brush edit was not stored for the current frame"
        assert win._masks[0].any()
    finally:
        win.close()


def test_points_mode_does_not_touch_mask() -> None:
    _app()
    canvas = _canvas_with_frame()
    canvas.set_edit_mode("points")

    points: list[tuple[int, int, int]] = []
    canvas.point_added.connect(lambda x, y, lbl: points.append((x, y, lbl)))

    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        _img_center(canvas, 40, 30),
    )
    assert points, "point prompt was not emitted in points mode"
    assert canvas.get_mask() is None, "points mode must not touch the mask"


def test_preview_mode_fallback() -> None:
    _app()
    canvas = VideoCanvas()
    canvas.set_preview_mode("cutout")
    assert canvas._preview_mode == "cutout"
    canvas.set_preview_mode("frame")
    assert canvas._preview_mode == "frame"
    canvas.set_preview_mode("bogus")
    assert canvas._preview_mode == "mask"


def test_main_window_toolbar_wiring() -> None:
    _app()
    win = MainWindow()
    try:
        for attr in (
            "_btn_play",
            "_btn_stop",
            "_combo_mode",
            "_combo_model",
            "_combo_preview",
            "_btn_brush",
            "_btn_points",
            "_spin_brush",
            "_chk_erase",
            "_btn_object_color",
            "_chk_post_process",
            "_smooth_slider",
        ):
            assert hasattr(win, attr), f"missing toolbar widget: {attr}"

        assert win._combo_model.itemData(0) is None  # index 0 == Auto
        assert win._config.recommendation is not None, "recommendation missing"

        # Content-mode combo matches the enum order.
        from src.hardware.profiler import ContentProfile

        assert win._combo_mode.count() == len(list(ContentProfile))
        assert win._smooth_slider.minimum() == 1
        assert win._smooth_slider.maximum() == 10
        assert win._chk_post_process.isChecked() == win._post_process_enabled
        assert win._smooth_slider.isEnabled() == win._post_process_enabled
    finally:
        win.close()


def test_tool_shortcuts_sync_sidebar_controls() -> None:
    _app()
    win = MainWindow()
    try:
        QTest.keyClick(win._canvas, Qt.Key.Key_B)
        assert win._canvas.get_edit_mode() == "brush"
        assert win._btn_brush.isChecked()
        assert not win._btn_points.isChecked()
        assert not win._brush_size_widget.isHidden()

        QTest.keyClick(win._canvas, Qt.Key.Key_BracketRight)
        assert win._spin_brush.value() == 22
        QTest.keyClick(win._canvas, Qt.Key.Key_BracketLeft)
        assert win._spin_brush.value() == 20
    finally:
        win.close()


def test_playback_button_reflects_play_state() -> None:
    _app()
    win = MainWindow()
    try:
        win._video_reader = type("Stub", (), {"fps": 30.0, "total_frames": 10})()
        win._toggle_play()
        assert win._btn_play.isChecked()
        assert win._btn_play.text() == "Pause"
        win._pause_playback()
        assert not win._btn_play.isChecked()
        assert win._btn_play.text() == "Play"
    finally:
        win.close()


def test_set_frame_clamps_to_video_end() -> None:
    _app()
    win = MainWindow()
    try:
        win._video_reader = type("Stub", (), {"total_frames": 10})()
        win._set_frame(99)
        assert win._current_frame_idx == 9
    finally:
        win.close()


def test_segment_bounds() -> None:
    _app()
    win = MainWindow()
    try:
        win._current_frame_idx = 4
        win._keyframes = {0, 3, 8}
        win._video_reader = type("Stub", (), {"total_frames": 12})()
        assert win._segment_bounds(1) == (4, 7), (
            "forward must stop before next keyframe"
        )
        assert win._segment_bounds(-1) == (4, 4), (
            "backward must stop after prev keyframe"
        )

        win._current_frame_idx = 11
        assert win._segment_bounds(1) == (11, 11), "forward at the end stays put"
        win._current_frame_idx = 0
        assert win._segment_bounds(-1) == (0, 0), "backward at the start stays put"
    finally:
        win.close()


def test_brush_size_widget_follows_tool() -> None:
    _app()
    win = MainWindow()
    try:
        assert win._brush_size_widget.isHidden(), (
            "brush size hidden by default (points tool active)"
        )
        win._set_tool("brush")
        assert not win._brush_size_widget.isHidden(), (
            "brush size visible in brush mode"
        )
        win._set_tool("points")
        assert win._brush_size_widget.isHidden(), (
            "brush size hidden again when switching back to points"
        )
    finally:
        win.close()


def test_segment_bounds_stop_at_scene_cuts() -> None:
    _app()
    win = MainWindow()
    try:
        win._current_frame_idx = 4
        win._keyframes = {0, 3}
        win._video_reader = type("Stub", (), {"total_frames": 30})()
        win._cuts_stop_propagation = True
        win._scene_cuts = [0, 12, 25]

        # Forward from 4: next cut at 12 -> must stop at 11 (cut - 1)
        assert win._segment_bounds(1) == (4, 11)
        # Backward from 4: previous cut at 0 -> must stop at 1 (cut + 1)
        assert win._segment_bounds(-1) == (4, 4)

        win._current_frame_idx = 13
        # Forward from 13: keyframe at 3, next cut at 25 -> stop at 24
        assert win._segment_bounds(1) == (13, 24)

        # Disabling cut-stop restores old behavior (to end of video at 29)
        win._cuts_stop_propagation = False
        assert win._segment_bounds(1) == (13, 29)
    finally:
        win.close()


if __name__ == "__main__":
    _app()
    tests = [
        test_brush_paints_and_emits,
        test_brush_stroke_on_main_window_stores_mask,
        test_points_mode_does_not_touch_mask,
        test_preview_mode_fallback,
        test_main_window_toolbar_wiring,
        test_tool_shortcuts_sync_sidebar_controls,
        test_playback_button_reflects_play_state,
        test_set_frame_clamps_to_video_end,
        test_segment_bounds,
        test_brush_size_widget_follows_tool,
        test_segment_bounds_stop_at_scene_cuts,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [OK ] {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {exc}")
    print(
        "All brush/toolbar tests passed." if not failed else f"{failed} test(s) failed."
    )
    raise SystemExit(1 if failed else 0)
