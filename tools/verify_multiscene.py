"""End-to-end check: real SAM 2, synthetic 2-scene video, chained propagation.

Not part of the test suite - run manually to confirm that re-seeding a shot
mid-stream (add_new_mask after a propagate_in_video run) really works against
the installed predictor.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from src.models.sam2_manager import PointPrompt
from src.processing.inference_worker import PropagationWorker
from src.ui.main_window import MainWindow

SIZE = (256, 256)
TOTAL = 24
CUT = 12


def make_video(path: Path) -> None:
    """Two shots: a blue disc on grey, then the same disc on a green field."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, SIZE)
    for i in range(TOTAL):
        frame = np.full((SIZE[1], SIZE[0], 3), 90, dtype=np.uint8)
        frame[:] = (60, 40, 30) if i < CUT else (40, 160, 60)
        cv2.circle(frame, (100, 120), 34, (30, 60, 220), -1)
        writer.write(frame)
    writer.release()


def main() -> int:
    QApplication.instance() or QApplication(sys.argv)
    app = MainWindow()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        video = Path(tmp) / "twoshots.mp4"
        make_video(video)
        app._load_video(video)
        app._scene_cuts = [CUT]
        app._cuts_stop_propagation = True
        app._current_frame_idx = 0
        app._points = [PointPrompt(100.0, 120.0, 1)]

        segments = app._shot_segments(1)
        print("segments:", segments)
        assert segments == [(0, CUT - 1), (CUT, TOTAL - 1)], segments

        sam = app._ensure_sam_manager(app._config.recommendation)
        worker = PropagationWorker(app._config.recommendation, sam)
        worker.set_task(
            str(video),
            start_frame=0,
            points=list(app._points),
            total_frames=TOTAL,
        )
        worker.set_segments(segments)
        worker.set_range(reverse=False)

        out: list = []
        worker.finished.connect(out.append)
        worker.run()

        assert len(out) == 1, out
        result = out[0]
        if isinstance(result, Exception):
            print("FAIL: propagation raised", result)
            return 1

        tracked = sorted({e[0] for e in result})
        print("tracked frames:", tracked)
        shot1 = [f for f in tracked if f < CUT]
        shot2 = [f for f in tracked if f >= CUT]
        print(f"shot 1: {len(shot1)} frames, shot 2: {len(shot2)} frames")

        for idx in (0, CUT - 1, CUT, TOTAL - 1):
            entry = next((e for e in result if e[0] == idx), None)
            if entry is None:
                print(f"  frame {idx}: MISSING")
                continue
            area = int(entry[2].sum())
            print(f"  frame {idx}: mask area = {area}")

        ok = len(shot1) == CUT and len(shot2) == TOTAL - CUT
        areas = [int(e[2].sum()) for e in result]
        print("non-empty masks:", sum(1 for a in areas if a > 0), "/", len(areas))
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
