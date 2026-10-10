"""Run the spill cleaner on the violet clip and write before/after proofs.

Not part of the test suite - a manual demonstration harness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from src.models.sam2_manager import PointPrompt
from src.processing.despill import despill_frame
from src.processing.inference_worker import InferenceWorker
from src.ui.main_window import MainWindow

OUT = Path("logs/spill_demo")
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = Path("clip/HardSky-Light-VioletBetter05.mp4")
FRAME = 60
EYE = (816, 406)  # the cyan eye that matches the sky


def save(name: str, img: np.ndarray) -> None:
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("  wrote", name)


def checker(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((xx // 16) + (yy // 16)) % 2
    return np.where(c[..., None] == 0, 60, 40).astype(np.uint8) * np.ones(3, np.uint8)


def cutout(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bg = checker(*frame.shape[:2])
    return np.where(mask[..., None], frame, bg).astype(np.uint8)


def main() -> int:
    QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    rec = win._config.recommendation
    print("model:", rec.model_size, "device:", rec.device)

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    print("frame:", frame.shape)

    # --- real SAM 2 matte -------------------------------------------------
    points = [
        PointPrompt(1000.0, 620.0, 1),  # cheek
        PointPrompt(980.0, 250.0, 1),   # hair
        PointPrompt(1150.0, 300.0, 1),  # hair right
        PointPrompt(200.0, 120.0, 0),   # sky, exclude
        PointPrompt(1820.0, 300.0, 0),  # sky right, exclude
    ]
    worker = InferenceWorker(rec, sam_manager=win._ensure_sam_manager())
    worker.set_task(frame, points)
    res: list = []
    worker.finished.connect(res.append)
    worker.run()
    if isinstance(res[0], Exception):
        print("FAIL segmentation:", res[0])
        return 1
    if not res[0]:
        print("FAIL: no mask candidates")
        return 1
    # The prompt-aware picker already scores candidates; take the best area.
    best = max(res[0], key=lambda r: r.mask.sum())
    mask = best.mask.astype(bool)
    print(f"mask: {mask.sum()} px ({100 * mask.mean():.1f}% of frame)")

    sky = (hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 115) & (hsv[:, :, 1] > 120)
    spill = tuple(int(c) for c in frame[20:90, 20:90].reshape(-1, 3).mean(0))
    print("sky mean rgb:", spill, "sky hue:", round(float(hsv[20:90, 20:90, 0].mean())))

    ey, ex = EYE
    def eye_rgb(img):
        return img[ey - 5:ey + 6, ex - 5:ex + 6].reshape(-1, 3).mean(0)

    # --- the comparison sheet -------------------------------------------
    save("01_original.png", frame)
    save("02_matte.png", np.where(mask[..., None], np.array([0, 255, 120], np.uint8), frame * 0.35))
    save("03_cutout_plain.png", cutout(frame, mask))

    eye_region = np.zeros(mask.shape, bool)
    eye_region[ey - 40:ey + 40, ex - 60:ex + 60] = True
    print(f"\neye is {'INSIDE' if mask[ey, ex] else 'OUTSIDE'} the matte")

    rows = []
    cases = [
        ("band tol 20 r2", dict(tolerance=20, radius=2), None, "clean"),
        ("band tol 60 r4", dict(tolerance=60, radius=4), None, "clean"),
        ("band tol 100 r10", dict(tolerance=100, radius=10), None, "clean"),
        ("skybox tol 60 r4", dict(tolerance=60, radius=4), (0, 0, 1920, 1080), "clean"),
        ("protect eyes tol 100 r10", dict(tolerance=100, radius=10), (ex - 90, ey - 60, 180, 120), "protect"),
    ]
    for label, kw, region, mode in cases:
        out = despill_frame(frame, mask, spill, region=region, region_mode=mode, **kw)
        before = eye_rgb(frame)
        after = eye_rgb(out)
        delta = float(np.abs(after - before).max())
        print(f"  {label:26} eye rgb {tuple(before.astype(int))} -> {tuple(after.astype(int))}  maxdelta {delta:5.1f}")
        tile = out.copy()
        if region is not None:
            x, y, w_, h_ = region
            cv2.rectangle(tile, (x, y), (x + w_, y + h_), (255, 255, 0), 5)
        cv2.putText(tile, label, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
        cv2.putText(tile, label, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        rows.append(cv2.resize(tile, (640, 360)))

    save("04_settings.png", np.vstack([np.hstack(rows[:2]), np.hstack(rows[2:4]), np.hstack(rows[4:] + [np.zeros_like(rows[0])])]))

    # --- zoom on the eye + a hair edge -----------------------------------
    ey, ex = EYE
    zy, zx0 = max(0, ey - 110), max(0, ex - 150)
    zw = min(300, frame.shape[1] - zx0)
    zy2 = min(zy + 240, frame.shape[0])
    tiles = []
    for label, kw, region, mode in cases:
        out = despill_frame(frame, mask, spill, region=region, region_mode=mode, **kw)
        crop = cv2.resize(out[zy:zy2, zx0:zx0 + zw], (300, 240))
        cv2.putText(crop, label[:22], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(crop, label[:22], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(crop)
    save("05_eye_zoom.png", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:] + [np.zeros_like(tiles[0])])]))

    win.close()
    print("\ndemo images in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
