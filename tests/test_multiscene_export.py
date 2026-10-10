"""Multi-scene propagation/export plus the export cancel button.

Regression tests for two bugs:

1. With automatic scene-change detection, propagation stopped at the first
   detected cut, so only the opening shot ever owned a mask.  Both "selection"
   and "everything" exports then silently produced scene 1 only, because the
   exporter walks the frames that actually own a mask.
2. The export progress dialog is a child of the main window, so
   ``MainWindow.setEnabled(False)`` force-disabled it and the Cancel button
   stopped responding.

No GPU / SAM 2 weights are required - the predictor is stubbed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from src.models.sam2_manager import PointPrompt
from src.processing.inference_worker import PropagationWorker
from src.ui.main_window import ExportWorker, MainWindow

_APP = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


class _FakeSAM:
    """SAM2Manager stand-in that records how the worker drove it."""

    def __init__(self, total: int = 40) -> None:
        self.total = total
        self.initialised: list[str] = []
        self.prompts: list[tuple[str, int]] = []
        self.seeds: list[tuple[int, np.ndarray]] = []
        self.runs: list[tuple[int, int | None, bool]] = []

    def init_video(self, video_path, progress_cb=None):
        self.initialised.append(str(video_path))
        return object()

    def add_points_to_video(self, frame_idx, obj_id, points, clear_old_points=True):
        self.prompts.append(("points", frame_idx))

    def add_mask_to_video(self, frame_idx, obj_id, mask):
        self.prompts.append(("mask", frame_idx))
        self.seeds.append((frame_idx, np.array(mask, copy=True)))

    def _mask(self, frame_idx: int) -> np.ndarray:
        """A mask tagged with the frame it came from, to trace the carry."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = frame_idx % 2 == 0
        return mask

    def propagate_in_video(
        self,
        start_frame_idx=0,
        max_frames=None,
        reverse=False,
        on_frame=None,
        cancel_event=None,
    ):
        count = self.total if max_frames is None else int(max_frames)
        self.runs.append((start_frame_idx, max_frames, reverse))
        order = (
            range(start_frame_idx, start_frame_idx - count, -1)
            if reverse
            else range(start_frame_idx, start_frame_idx + count)
        )
        out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for frame_idx in order:
            if not 0 <= frame_idx < self.total:
                continue
            if cancel_event is not None and cancel_event.is_set():
                break
            entry = (frame_idx, 1, self._mask(frame_idx), np.array([1.0]))
            out.append(entry)
            if on_frame is not None:
                on_frame(entry[0], entry[1], entry[2])
        return out


def _worker(sam: _FakeSAM, segments, start: int, reverse: bool) -> PropagationWorker:
    worker = PropagationWorker(None, sam)  # type: ignore[arg-type]
    worker.set_task(
        "clip.mp4",
        start_frame=start,
        points=[PointPrompt(1.0, 1.0, 1)],
        total_frames=sam.total,
    )
    worker.set_segments(segments)
    worker.set_range(reverse=reverse)
    return worker


def _covered(results) -> set[int]:
    return {entry[0] for entry in results}


# ------------------------------------------------------------------ #
# Propagation chains every shot                                       #
# ------------------------------------------------------------------ #
def test_propagation_covers_every_shot_forward() -> None:
    _app()
    sam = _FakeSAM(total=40)
    results: list = []
    worker = _worker(sam, [(0, 9), (10, 19), (20, 39)], start=0, reverse=False)
    worker.finished.connect(results.append)
    worker.run()

    assert len(results) == 1
    tracked = _covered(results[0])
    assert tracked == set(range(40)), (
        "every shot must be tracked - the export only sees frames that own a mask"
    )
    assert [tuple(r) for r in sam.runs] == [
        (0, 10, False),
        (10, 10, False),
        (20, 20, False),
    ]
    # Only the first shot is prompted by the user; the rest are re-seeded from
    # the last mask of the previous shot, so a cut is never crossed in one run.
    assert sam.prompts[0] == ("points", 0)
    assert [frame for kind, frame in sam.prompts[1:]] == [10, 20]
    assert all(kind == "mask" for kind, _ in sam.prompts[1:])


def test_propagation_covers_every_shot_backward() -> None:
    _app()
    sam = _FakeSAM(total=40)
    results: list = []
    worker = _worker(sam, [(29, 39), (19, 28), (0, 18)], start=39, reverse=True)
    worker.finished.connect(results.append)
    worker.run()

    tracked = _covered(results[0])
    assert tracked == set(range(40))
    # Backward segments are re-seeded on the *first* frame of the next segment
    # down - the one the tracker is about to cross.
    assert [frame for kind, frame in sam.prompts[1:]] == [19, 0]
    # The carried mask is the one on the segment edge being crossed (29 here,
    # the lowest frame of 29..39), whichever order the backend returned.
    assert np.array_equal(sam.seeds[0][1], sam._mask(29))


def test_propagation_reseeds_from_previous_shot_edge() -> None:
    _app()
    sam = _FakeSAM(total=40)
    _worker(sam, [(0, 9), (10, 19)], start=0, reverse=False).run()

    assert len(sam.seeds) == 1, "two segments means a single re-seed"
    frame, mask = sam.seeds[0]
    assert frame == 10
    # The seed is the mask produced for the last frame of shot 1.
    assert np.array_equal(mask, sam._mask(9))


def test_propagation_stops_the_chain_when_a_shot_yields_nothing() -> None:
    _app()

    class _BlindAfterFirst(_FakeSAM):
        def propagate_in_video(self, *a, **k):
            if len(self.runs) >= 1:
                self.runs.append((k.get("start_frame_idx", 0), None, False))
                return []
            return super().propagate_in_video(*a, **k)

    sam = _BlindAfterFirst(total=40)
    results: list = []
    worker = _worker(sam, [(0, 9), (10, 19), (20, 39)], start=0, reverse=False)
    worker.finished.connect(results.append)
    worker.run()

    assert _covered(results[0]) == set(range(10))
    assert len(sam.runs) == 2, "must not keep chasing shots once tracking is lost"


def test_propagation_without_segments_keeps_single_run() -> None:
    _app()
    sam = _FakeSAM(total=12)
    results: list = []
    worker = PropagationWorker(None, sam)  # type: ignore[arg-type]
    worker.set_task(
        "clip.mp4", start_frame=0, points=[PointPrompt(1.0, 1.0, 1)], total_frames=12
    )
    worker.set_range(max_frames=5)
    worker.finished.connect(results.append)
    worker.run()

    assert sam.runs == [(0, 5, False)]
    assert _covered(results[0]) == set(range(5))


def test_propagation_cancel_stops_between_shots() -> None:
    _app()
    sam = _FakeSAM(total=40)
    results: list = []
    worker = _worker(sam, [(0, 9), (10, 19), (20, 39)], start=0, reverse=False)
    worker.finished.connect(results.append)
    worker.cancel()
    worker.run()
    assert results[0] == []


# ------------------------------------------------------------------ #
# Segment planning                                                   #
# ------------------------------------------------------------------ #
def _window(cuts: list[int], total: int = 30) -> MainWindow:
    win = MainWindow()
    win._video_reader = type("Stub", (), {"total_frames": total})()
    win._cuts_stop_propagation = True
    win._scene_cuts = list(cuts)
    return win


def test_shot_segments_chain_across_cuts_forward() -> None:
    _app()
    win = _window([12, 25])
    try:
        win._current_frame_idx = 4
        win._keyframes = {0}
        assert win._shot_segments(1) == [(4, 11), (12, 24), (25, 29)]
    finally:
        win.close()


def test_shot_segments_chain_across_cuts_backward() -> None:
    _app()
    win = _window([12, 25])
    try:
        win._current_frame_idx = 20
        win._keyframes = {20}
        assert win._shot_segments(-1) == [(12, 20), (0, 11)]
    finally:
        win.close()


def test_shot_segments_single_run_without_cuts() -> None:
    _app()
    win = _window([])
    try:
        win._current_frame_idx = 4
        win._keyframes = {0}
        assert win._shot_segments(1) == [(4, 29)]
        win._current_frame_idx = 20
        win._keyframes = {20}
        assert win._shot_segments(-1) == [(0, 20)]
    finally:
        win.close()


def test_shot_segments_ignore_frames_written_by_propagation() -> None:
    """Dense propagation output must not shred the clip into 1-frame runs."""
    _app()
    win = _window([20])
    try:
        win._current_frame_idx = 5
        # A first pass masked 0..19, so every one of those is a keyframe.
        win._keyframes = set(range(20))
        win._propagated_frames = set(range(20))
        assert win._shot_segments(1) == [(5, 19), (20, 29)]
    finally:
        win.close()


def test_shot_segments_honour_hand_placed_prompts() -> None:
    _app()
    win = _window([20])
    try:
        win._current_frame_idx = 0
        win._keyframes = {0, 8}  # frame 8 is a hand-placed correction
        assert win._shot_segments(1) == [(0, 7), (8, 19), (20, 29)]
    finally:
        win.close()


def test_shot_segments_respect_disabled_cut_stop() -> None:
    _app()
    win = _window([12, 25])
    try:
        win._cuts_stop_propagation = False
        win._current_frame_idx = 4
        win._keyframes = {0}
        assert win._shot_segments(1) == [(4, 29)]
    finally:
        win.close()


# ------------------------------------------------------------------ #
# Export covers every scene                                          #
# ------------------------------------------------------------------ #
def test_export_writes_every_masked_scene() -> None:
    """The exporter must emit frames from every shot, not just the first."""
    _app()
    from src.export.exporter import Exporter

    masks = {idx: np.zeros((8, 8), dtype=bool) for idx in range(30)}
    for idx in range(30):
        masks[idx][idx % 8, idx % 8] = True
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp)
        worker = ExportWorker(Exporter(output_dir=out), "png", masks)
        worker.run()
        written = sorted(p.stem for p in out.rglob("*.png"))
        assert len(written) == 30
        # Frames 20+ live in the second and third shot.
        assert any(name.endswith("_000020") for name in written)
        assert any(name.endswith("_000029") for name in written)


# ------------------------------------------------------------------ #
# Export cancel button                                               #
# ------------------------------------------------------------------ #
def _defuse_dialogs(monkeypatch) -> list[str]:
    """Silence modal popups (restored by monkeypatch) and record which fired."""
    seen: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: seen.append("critical"))
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: seen.append("information")),
    )
    return seen


def _export_settings(out: Path) -> dict:
    return {
        "format": "png",
        "file_name": "out",
        "output_dir": out,
        "fps": 30.0,
        "resolution": None,
        "range_mode": "all",
        "start_frame": 0,
        "end_frame": 0,
        "include_audio": False,
        "background": None,
        "object_mode": "combined",
    }


def _prepared_export(win: MainWindow, out: Path, monkeypatch) -> None:
    """Drive the real export path with the worker start neutered."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    win._masks.clear()
    win._masks[0] = mask
    monkeypatch.setattr(ExportWorker, "start", lambda self: None, raising=False)
    win._run_export_from_settings(_export_settings(out))


def _close_export(win: MainWindow) -> None:
    """Tear the progress dialog down.

    It is application-modal, so leaving it open would swallow the keyboard
    input of every later test in the session.
    """
    dlg = win._export_progress_dlg
    if dlg is not None:
        dlg.close()
        win._export_progress_dlg = None
    win.setEnabled(True)


def test_export_cancel_button_responds_while_exporting(
    tmp_path: Path, monkeypatch
) -> None:
    """The regression: the dialog is a child, so disabling the window killed it."""
    _app()
    _defuse_dialogs(monkeypatch)
    win = MainWindow()
    try:
        _prepared_export(win, tmp_path, monkeypatch)
        dlg = win._export_progress_dlg
        assert dlg is not None
        assert win.isEnabled() is False, "main window is disabled during export"
        assert dlg._btn_cancel.isEnabled(), "Cancel must stay live while exporting"

        QTest.mouseClick(dlg._btn_cancel, Qt.MouseButton.LeftButton)

        assert win._export_worker.cancelled, "cancel must reach the worker"
    finally:
        _close_export(win)
        win.close()


def test_export_cancel_drops_queued_jobs(tmp_path: Path, monkeypatch) -> None:
    _app()
    _defuse_dialogs(monkeypatch)
    win = MainWindow()
    try:
        _prepared_export(win, tmp_path, monkeypatch)
        # Simulate "separate objects" mode: a second job is still queued.
        win._export_jobs.append(({0: np.zeros((8, 8), dtype=bool)}, "out_second"))

        QTest.mouseClick(
            win._export_progress_dlg._btn_cancel, Qt.MouseButton.LeftButton
        )

        assert win._export_jobs == [], "cancelling must not roll into the next job"
    finally:
        _close_export(win)
        win.close()


def test_export_cancel_is_not_reported_as_failure(tmp_path: Path, monkeypatch) -> None:
    _app()
    seen = _defuse_dialogs(monkeypatch)
    win = MainWindow()
    try:
        _prepared_export(win, tmp_path, monkeypatch)
        QTest.mouseClick(
            win._export_progress_dlg._btn_cancel, Qt.MouseButton.LeftButton
        )

        # The exporter raises on cancellation; that must not surface as an error.
        win._on_export_finished(RuntimeError("Export cancelled"))

        assert win._export_results == [("cancelled", "")]
        assert "critical" not in seen
        assert win._export_progress_dlg is None
        assert win.isEnabled() is True
    finally:
        _close_export(win)
        win.close()


def test_export_progress_dialog_reports_cancellation() -> None:
    _app()
    from src.ui.components.export_progress import ExportProgressDialog

    dlg = ExportProgressDialog()
    try:
        dlg.update_progress(5, 40, stage="Writing PNG masks", elapsed=12.0)
        dlg.set_finished(success=False, cancelled=True)
        assert dlg._lbl_title.text() == "Export cancelled"
        assert dlg._btn_cancel.isEnabled(), "Close must be clickable afterwards"
    finally:
        dlg.close()


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
        kwargs = {}
        if "tmp_path" in inspect.signature(test).parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp())
        if "monkeypatch" in inspect.signature(test).parameters:
            from _pytest.monkeypatch import MonkeyPatch

            kwargs["monkeypatch"] = MonkeyPatch()
        try:
            test(**kwargs)
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
