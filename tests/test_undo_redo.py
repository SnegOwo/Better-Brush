"""Offscreen GUI tests: undo/redo (Ctrl+Z / Ctrl+Y) system.

No GPU / SAM 2 weights are required — only Qt's event machinery.

Run with:
    python -m pytest tests/test_undo_redo.py -v
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

from src.models.sam2_manager import PointPrompt
from src.ui.main_window import MainWindow

_APP = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _win() -> MainWindow:
    win = MainWindow()
    win._canvas.setFixedSize(720, 480)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    win._canvas.set_frame(frame)
    win.show()
    win.activateWindow()
    win._canvas.setFocus()
    QTest.qWaitForWindowExposed(win)
    return win


def _img_center(canvas, ix: int, iy: int) -> QPoint:
    ox, oy = canvas._offset.x(), canvas._offset.y()
    return QPoint(int(ox + ix * canvas._scale), int(oy + iy * canvas._scale))


def _click_point(canvas, ix: int, iy: int, button=Qt.MouseButton.LeftButton) -> None:
    QTest.mouseClick(
        canvas, button, Qt.KeyboardModifier.NoModifier, _img_center(canvas, ix, iy)
    )


def test_undo_redo_point() -> None:
    _app()
    win = _win()
    try:
        win._on_canvas_point(10, 20, 1)
        assert len(win._points) == 1
        win._undo()
        assert len(win._points) == 0, "undo did not remove the point"
        win._redo()
        assert len(win._points) == 1, "redo did not restore the point"
    finally:
        win.close()


def test_undo_redo_brush_stroke() -> None:
    _app()
    win = _win()
    try:
        win._canvas.set_edit_mode("brush")
        win._canvas.set_brush_size(10)
        _click_point(win._canvas, 80, 50)
        assert 0 in win._masks and win._masks[0].any(), "brush not stored"
        count = int(win._masks[0].sum())
        win._undo()
        assert 0 not in win._masks, "undo did not remove the mask"
        win._redo()
        assert 0 in win._masks and win._masks[0].sum() == count, "redo lost the mask"
    finally:
        win.close()


def test_undo_clear_mask() -> None:
    _app()
    win = _win()
    try:
        win._masks[0] = np.ones((100, 160), dtype=bool)
        win._keyframes.add(0)
        win._undo_manager.reset(win._points, win._masks, win._keyframes)
        win._clear_mask()
        assert 0 not in win._masks, "clear mask did not remove the mask"
        win._undo()
        assert 0 in win._masks, "undo did not restore the cleared mask"
    finally:
        win.close()


def test_undo_clear_points() -> None:
    _app()
    win = _win()
    try:
        win._points.append(PointPrompt(x=1, y=2, label=1))
        win._undo_manager.reset(win._points, win._masks, win._keyframes)
        win._clear_points()
        assert not win._points, "points not cleared"
        win._undo()
        assert len(win._points) == 1, "undo did not restore the points"
    finally:
        win.close()


def test_undo_redo_shortcuts() -> None:
    _app()
    win = _win()
    try:
        _click_point(win._canvas, 40, 30)
        assert len(win._points) == 1, "point via click not added"

        QTest.keyClick(win._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert len(win._points) == 0, "Ctrl+Z did not undo"

        QTest.keyClick(win._canvas, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert len(win._points) == 1, "Ctrl+Y did not redo"

        QTest.keyClick(win._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(
            win._canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        assert len(win._points) == 1, "Ctrl+Shift+Z did not redo"
    finally:
        win.close()


def test_new_action_clears_redo() -> None:
    _app()
    win = _win()
    try:
        _click_point(win._canvas, 40, 30)
        QTest.keyClick(win._canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert not win._points
        assert win._undo_manager.can_redo, (
            "redo should be available before a new action"
        )
        win._on_canvas_point(50, 60, 1)
        assert not win._undo_manager.can_redo, "new action must clear redo history"
    finally:
        win.close()


def test_multi_step_undo_redo() -> None:
    _app()
    win = _win()
    try:
        for i in range(3):
            win._on_canvas_point(10 + i * 5, 20, 1)
        assert len(win._points) == 3
        win._undo()
        win._undo()
        assert len(win._points) == 1, "multi-undo moved to wrong state"
        win._redo()
        assert len(win._points) == 2, "multi-redo moved to wrong state"
    finally:
        win.close()


if __name__ == "__main__":
    _app()
    tests = [
        test_undo_redo_point,
        test_undo_redo_brush_stroke,
        test_undo_clear_mask,
        test_undo_clear_points,
        test_undo_redo_shortcuts,
        test_new_action_clears_redo,
        test_multi_step_undo_redo,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [OK ] {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {exc}")
    print("All undo/redo tests passed." if not failed else f"{failed} test(s) failed.")
    raise SystemExit(1 if failed else 0)
