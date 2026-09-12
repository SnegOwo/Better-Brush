"""Headless GUI smoke test (offscreen).

Exercises the real MainWindow wiring: video loading, point prompting,
worker-based segmentation, worker-based propagation and PNG export.
QThread.run() is invoked synchronously so no event loop is required;
the underlying SAM 2 inference is entirely real.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication, QMessageBox

from src.export.exporter import Exporter
from src.processing.inference_worker import (
    InferenceWorker,
    PropagationWorker,
)
from src.ui.main_window import ExportWorker, MainWindow
from tests.test_integration import create_synthetic_video


def _defuse_dialogs() -> None:
    """Replace modal dialogs by printed messages (offscreen test only)."""
    QMessageBox.critical = staticmethod(  # type: ignore[assignment]
        lambda *a, **k: print(f"  DIALOG(critical): {a[2] if len(a) > 2 else a}")
    )
    QMessageBox.warning = staticmethod(  # type: ignore[assignment]
        lambda *a, **k: print(f"  DIALOG(warning): {a[1] if len(a) > 1 else a}")
    )
    QMessageBox.information = staticmethod(  # type: ignore[assignment]
        lambda *a, **k: None
    )


def main() -> int:
    _defuse_dialogs()
    QApplication.instance() or QApplication(sys.argv)

    print("=" * 60)
    print("Better Brush — GUI smoke test (offscreen)")
    print("=" * 60)

    win = MainWindow()
    assert win._config.recommendation is not None, "hardware recommendation missing"
    print("[1/6] MainWindow created, hardware recommendation OK")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "smoke.mp4"
        create_synthetic_video(video, frames=4, size=(256, 256))

        win._load_video(video)
        assert win._video_reader is not None
        assert win._frame_buffer is not None
        assert win._canvas._frame is not None, "canvas has no frame"
        assert win._timeline._total_frames == 4, "timeline duration not set"
        print("[2/6] Video loaded, canvas + timeline updated")

        win._on_canvas_point(76, 128, 1)  # circle center at frame 0
        assert len(win._points) == 1
        print("[3/6] Point prompt added via canvas handler")

        try:
            worker = InferenceWorker(
                win._config.recommendation, sam_manager=win._ensure_sam_manager()
            )
            worker.set_task(win._frame_buffer.get(0), list(win._points))
            worker.finished.connect(win._on_segment_finished)
            worker.run()  # synchronous execution of the real SAM 2 inference
            if 0 not in win._masks:
                print("  FAIL: segmentation stored no mask")
                return 1
            print("[4/6] Segmentation worker -> mask stored on frame 0")

            pworker = PropagationWorker(
                win._config.recommendation, win._ensure_sam_manager()
            )
            pworker.set_task(
                str(video), start_frame=0, points=list(win._points), total_frames=4
            )
            pworker.progress.connect(win._on_worker_progress)
            pworker.frame_ready.connect(win._on_propagation_frame_ready)
            pworker.finished.connect(win._on_propagation_finished)
            pworker.run()  # synchronous: JPEG extraction + video propagation
            if len(win._masks) < 4:
                print(f"  FAIL: propagation stored {len(win._masks)} mask(s)")
                return 1
            print(f"[5/6] Propagation worker -> {len(win._masks)}/4 frames masked")

            exp_dir = Path(tmp) / "exp"
            eworker = ExportWorker(
                Exporter(output_dir=exp_dir), "png", dict(win._masks)
            )
            eworker.run()
            pngs = list(exp_dir.rglob("*.png"))
            if len(pngs) != len(win._masks):
                print(f"  FAIL: expected {len(win._masks)} PNGs, got {len(pngs)}")
                return 1
            print(f"[6/6] Export worker -> {len(pngs)} PNG mask(s)")
        finally:
            win.close()

    print("\nGUI smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
