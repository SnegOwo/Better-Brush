"""Headless regression tests for the matte inspector and Inspect menu."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

from src.processing.qc_coach import QCTipsDatabase
from src.ui.components.matte_inspector import MatteInspectorDialog
from src.ui.components.tutorial_dialog import TutorialDialog
from src.ui.main_window import MainWindow

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def test_inspector_uses_distinct_source_and_qc_views(tmp_path) -> None:
    _app()
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :, 0] = 30
    mask = np.zeros((40, 60), dtype=bool)
    mask[8:32, 16:44] = True
    dialog = MatteInspectorDialog(
        frame,
        mask,
        mask,
        tips_database=QCTipsDatabase(tmp_path / "tips.json"),
    )

    assert not np.array_equal(
        dialog.source_preview_image(),
        dialog.current_preview_image(),
    )
    assert dialog.delta_pixels == 0
    assert not dialog.masks_changed
    dialog._before_preview.set_view(2.0, (12, -7))
    assert dialog._after_preview.zoom() == 2.0
    assert dialog._after_preview.pan() == (12, -7)
    dialog._btn_link_views.setChecked(False)
    dialog._after_preview.set_view(3.0, (40, 50))
    assert dialog._before_preview.zoom() == 2.0
    dialog._question.setText("it's too sharp")
    dialog._ask_question()
    assert "Raise Edge smoothness" in dialog._answer.text()
    dialog.close()


def test_inspect_menu_exposes_qc_shortcut() -> None:
    _app()
    win = MainWindow()
    try:
        assert "Ctrl+Shift+M" in win._act_matte_inspector.shortcut().toString()
        assert "Ctrl+Shift+G" in win._act_gpu_monitor.shortcut().toString()
        assert "Ctrl+Shift+P" in win._act_shot_profiles.shortcut().toString()
        assert "Ctrl+Shift+H" in win._act_history.shortcut().toString()
        assert not hasattr(win, "_btn_matte_inspector")
    finally:
        win._confirm_discard = lambda: True
        win.close()


def test_tutorial_tracks_progress_and_can_try_action() -> None:
    _app()
    dialog = TutorialDialog()
    actions: list[str] = []
    dialog.action_requested.connect(actions.append)
    dialog._btn_next.click()

    assert dialog._step == 1
    assert dialog._progress.value() == 2
    assert not hasattr(dialog, "_done")

    dialog._step = 0
    dialog._show_step()
    dialog._btn_try.click()
    assert actions == ["open_video"]
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.close()
