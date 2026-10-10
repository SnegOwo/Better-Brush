"""Render the violet clip through the real pipeline with the edge fix on.

Segments and propagates with SAM 2, then exports ProRes 4444 twice - once with
the hard binary matte (the old behaviour) and once with soft coverage plus edge
decontamination - so the two can be compared directly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from src.export.exporter import Exporter
from src.models.sam2_manager import PointPrompt
from src.processing.inference_worker import InferenceWorker, PropagationWorker
from src.ui.main_window import MainWindow

VIDEO = Path("clip/HardSky-Light-VioletBetter05.mp4")
OUT = Path("src/exports")
PROMPTS = [
    PointPrompt(1000.0, 620.0, 1),
    PointPrompt(980.0, 250.0, 1),
    PointPrompt(1150.0, 300.0, 1),
    PointPrompt(200.0, 120.0, 0),
    PointPrompt(1820.0, 300.0, 0),
]


def main() -> int:
    QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    rec = win._config.recommendation
    print("model:", rec.model_size, "| device:", rec.device)

    win._load_video(VIDEO)
    total = win._video_reader.total_frames
    print(f"video: {total} frames @ {win._video_reader.fps:.0f} fps")

    # 1. Segment the first frame for real.
    t0 = time.monotonic()
    worker = InferenceWorker(rec, sam_manager=win._ensure_sam_manager())
    worker.set_task(win._frame_buffer.get(0), list(PROMPTS))
    res: list = []
    worker.finished.connect(res.append)
    worker.run()
    if isinstance(res[0], Exception) or not res[0]:
        print("FAIL segmentation:", res[0])
        return 1
    best = max(res[0], key=lambda r: r.mask.sum())
    win._masks[0] = best.mask.astype(bool)
    win._keyframes.add(0)
    print(f"[1/3] segmented frame 0: {win._masks[0].sum()} px "
          f"({time.monotonic() - t0:.1f}s)")

    # 2. Propagate the matte across the clip.
    t0 = time.monotonic()
    sam = win._ensure_sam_manager(rec)
    prop = PropagationWorker(rec, sam)
    prop.set_task(str(VIDEO), start_frame=0, points=list(PROMPTS), total_frames=total)
    prop.set_range(reverse=False)
    done: list = []
    prop.finished.connect(done.append)
    prop.run()
    if isinstance(done[0], Exception):
        print("FAIL propagation:", done[0])
        return 1
    for idx, _oid, m, _s in done[0]:
        win._masks[idx] = m.astype(bool)
        win._propagated_frames.add(idx)
    print(f"[2/3] propagated: {len(win._masks)}/{total} frames masked "
          f"({time.monotonic() - t0:.1f}s)")

    # 3. Export both ways.
    frames = {i: win._frame_buffer.get(i) for i in sorted(win._masks)}
    frames = {i: f for i, f in frames.items() if f is not None}
    OUT.mkdir(parents=True, exist_ok=True)
    exporter = Exporter(output_dir=OUT)
    jobs = [
        ("HardSky-light-output", 0, "hard matte (previous behaviour)"),
        ("HardSky-light-output-soft", 2, "soft edge + decontaminated"),
    ]
    for name, feather, label in jobs:
        t0 = time.monotonic()
        path = exporter.export_video(
            frames,
            dict(win._masks),
            win._video_reader.fps,
            base_name=name,
            fmt="prores",
            edge_softness=feather,
        )
        print(f"[3/3] {label}: {path} ({time.monotonic() - t0:.1f}s)")

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
