"""Regression tests for the manual edge-smoothness and mask-inset controls."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from src.core.project import ProjectManager
from src.ui.main_window import MainWindow

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def test_slider_updates_current_shot_profile(tmp_path, monkeypatch) -> None:
    _app()
    monkeypatch.setattr(ProjectManager, "settings_dir", lambda: tmp_path)
    win = MainWindow()
    try:
        win._smooth_slider.setValue(8)
        profile = win._shot_profiles.get_for_shot(1, 1)
        assert profile.edge_smoothness == 8
        assert win._smooth_strength == 8
        assert win._smooth_slider.value() == 8
    finally:
        win._confirm_discard = lambda: True
        win.close()


def test_mask_inset_slider_updates_profile_and_label(tmp_path, monkeypatch) -> None:
    _app()
    monkeypatch.setattr(ProjectManager, "settings_dir", lambda: tmp_path)
    win = MainWindow()
    try:
        win._inset_slider.setValue(2)
        assert win._mask_inset == 2
        assert win._lbl_inset_val.text() == "2 px"
        assert win._shot_profiles.get_for_shot(1, 1).mask_inset == 2
        assert win._shot_profiles.default.mask_inset == 2

        win._inset_slider.setValue(0)
        assert win._mask_inset == 0
        assert win._lbl_inset_val.text() == "0 px"
        assert win._shot_profiles.get_for_shot(1, 1).mask_inset == 0
    finally:
        win._confirm_discard = lambda: True
        win.close()


def test_mask_inset_rerenders_canvas_without_touching_stored_mask(
    tmp_path, monkeypatch
) -> None:
    """Dragging the slider repaints from the cached mask; it is not destructive."""
    _app()
    monkeypatch.setattr(ProjectManager, "settings_dir", lambda: tmp_path)
    win = MainWindow()
    try:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        win._masks = {0: mask}

        assert np.array_equal(win._display_mask(0), mask)

        win._inset_slider.setValue(2)

        shown = win._display_mask(0)
        assert shown is not None
        assert int(shown.sum()) < int(mask.sum())
        # The cached mask is untouched, so the value stays reversible and
        # re-adjustable without re-running SAM.
        assert np.array_equal(win._masks[0], mask)

        win._inset_slider.setValue(0)
        assert np.array_equal(win._display_mask(0), mask)
    finally:
        win._confirm_discard = lambda: True
        win.close()
