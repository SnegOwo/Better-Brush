"""Offscreen GUI tests for canvas zoom / pan and wheel interactions.

Run with:
    python -m pytest tests/test_view_mode.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
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


def _canvas() -> VideoCanvas:
    canvas = VideoCanvas()
    canvas.setFixedSize(720, 480)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    canvas.set_frame(frame)
    canvas.show()
    canvas.activateWindow()
    QTest.qWaitForWindowExposed(canvas)
    canvas.grab()
    return canvas


def _win() -> MainWindow:
    win = MainWindow()
    win._canvas.setFixedSize(720, 480)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    win._canvas.set_frame(frame)
    win.show()
    win.activateWindow()
    win._canvas.setFocus()
    QTest.qWaitForWindowExposed(win)
    win._canvas.grab()
    return win


def _wheel(canvas: VideoCanvas, delta: int = 120, pos: QPoint | None = None,
           ctrl: bool = False) -> None:
    p = pos or QPoint(canvas.width() // 2, canvas.height() // 2)
    mods = (
        Qt.KeyboardModifier.ControlModifier
        if ctrl
        else Qt.KeyboardModifier.NoModifier
    )
    ev = QWheelEvent(
        QPointF(p), QPointF(p), QPoint(), QPoint(0, delta),
        Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False,
        Qt.MouseEventSource.MouseEventSynthesizedByApplication,
    )
    QCoreApplication.sendEvent(canvas, ev)


def _img_point(canvas: VideoCanvas, wx: int, wy: int) -> tuple[float, float]:
    return (
        (wx - canvas._offset.x()) / canvas._scale,  # noqa: SLF001
        (wy - canvas._offset.y()) / canvas._scale,  # noqa: SLF001
    )


def test_fit_frame_state() -> None:
    _app()
    c = _canvas()
    dividend = min(c.width() / 160, c.height() / 100)
    assert c._scale > 0  # noqa: SLF001
    assert abs(c._scale - dividend) < 1e-6  # noqa: SLF001


def test_ctrl_scroll_zooms_in_points_mode() -> None:
    _app()
    c = _canvas()
    c.set_edit_mode("points")
    before = c._zoom  # noqa: SLF001
    _wheel(c, delta=120, ctrl=True)
    assert c._zoom > before, "Ctrl+scroll up must zoom in"


def test_ctrl_scroll_zooms_in_brush_mode() -> None:
    _app()
    c = _canvas()
    c.set_edit_mode("brush")
    before = c._zoom  # noqa: SLF001
    _wheel(c, delta=120, ctrl=True)
    assert c._zoom > before, "Ctrl+scroll up must zoom in (brush mode)"


def test_ctrl_scroll_zoom_out_clamped() -> None:
    _app()
    c = _canvas()
    c.set_edit_mode("points")
    for _ in range(80):
        _wheel(c, delta=-120, ctrl=True)
    assert c._zoom >= 0.05, "zoom must clamp at the lower bound"  # noqa: SLF001


def test_plain_scroll_navigates_frames() -> None:
    _app()
    c = _canvas()
    nav: list[int] = []
    c.frame_navigate.connect(nav.append)
    c.set_edit_mode("points")
    _wheel(c, delta=-120)
    assert nav == [1], "plain scroll down must navigate to next frame"
    _wheel(c, delta=120)
    assert nav == [1, -1], "plain scroll up must navigate to previous frame"


def test_plain_scroll_navigates_in_brush_mode() -> None:
    _app()
    c = _canvas()
    nav: list[int] = []
    c.frame_navigate.connect(nav.append)
    c.set_edit_mode("brush")
    _wheel(c, delta=-120)
    assert nav == [1], "plain scroll must navigate in brush mode"


def test_ctrl_scroll_does_not_navigate() -> None:
    _app()
    c = _canvas()
    nav: list[int] = []
    c.frame_navigate.connect(nav.append)
    c.set_edit_mode("points")
    _wheel(c, delta=-120, ctrl=True)
    _wheel(c, delta=120, ctrl=True)
    assert nav == [], "Ctrl+scroll must not navigate frames"


def test_middle_drag_pans() -> None:
    _app()
    c = _canvas()
    pan0 = c._pan  # noqa: SLF001
    QTest.mousePress(c, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    QTest.mouseMove(c, QPoint(132, 70))
    QTest.mouseRelease(c, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(132, 70))
    assert c._pan == pan0 + QPoint(32, -30), "drag did not pan"  # noqa: SLF001


def test_zoom_keeps_cursor_point_stable() -> None:
    _app()
    c = _canvas()
    c.set_edit_mode("points")
    pos = QPoint(c.width() // 2, c.height() // 2)
    img_before = _img_point(c, pos.x(), pos.y())
    _wheel(c, delta=120, pos=pos, ctrl=True)
    img_after = _img_point(c, pos.x(), pos.y())
    assert abs(img_before[0] - img_after[0]) <= 1
    assert abs(img_before[1] - img_after[1]) <= 1


def test_no_view_tool_button() -> None:
    """The View tool button no longer exists."""
    _app()
    win = _win()
    try:
        assert not hasattr(win, "_btn_view") or not hasattr(win, "_btn_view")
        assert win._canvas.get_edit_mode() in ("points", "brush")
    finally:
        win.close()


def test_v_key_does_not_activate_view() -> None:
    _app()
    win = _win()
    try:
        win._set_tool("points")
        QTest.keyClick(win._canvas, Qt.Key.Key_V)
        assert win._canvas.get_edit_mode() == "points", "V must not change mode"
    finally:
        win.close()


if __name__ == "__main__":
    _app()
    tests = [
        test_fit_frame_state,
        test_ctrl_scroll_zooms_in_points_mode,
        test_ctrl_scroll_zooms_in_brush_mode,
        test_ctrl_scroll_zoom_out_clamped,
        test_plain_scroll_navigates_frames,
        test_plain_scroll_navigates_in_brush_mode,
        test_ctrl_scroll_does_not_navigate,
        test_middle_drag_pans,
        test_zoom_keeps_cursor_point_stable,
        test_no_view_tool_button,
        test_v_key_does_not_activate_view,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [OK ] {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {exc}")
    print("All view-mode tests passed." if not failed else f"{failed} test(s) failed.")
    raise SystemExit(1 if failed else 0)
