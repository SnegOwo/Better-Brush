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
        ):
            assert hasattr(win, attr), f"missing toolbar widget: {attr}"

        assert win._combo_model.itemData(0) is None  # index 0 == Auto
        assert win._config.recommendation is not None, "recommendation missing"

        # Content-mode combo matches the enum order.
        from src.hardware.profiler import ContentProfile

        assert win._combo_mode.count() == len(list(ContentProfile))
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


if __name__ == "__main__":
    _app()
    tests = [
        test_brush_paints_and_emits,
        test_brush_stroke_on_main_window_stores_mask,
        test_points_mode_does_not_touch_mask,
        test_preview_mode_fallback,
        test_main_window_toolbar_wiring,
        test_segment_bounds,
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
