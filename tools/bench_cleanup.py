"""Benchmark the Clean Up path on a real video (text energy + speed + crops).

Usage:
    python tools/bench_cleanup.py <video> [frames] [--cpu-onnx|--no-ai]

Writes side-by-side before/after crops to logs/bench/ so the repair quality
can be judged visually, and prints the residual text energy per region.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing import plates as P  # noqa: E402


def text_energy(region: np.ndarray) -> float:
    """Percent of caption-ink pixels — a proxy for burned-in subtitle ink.

    Uses the same thin-bright-stroke-with-dark-outline signature as the
    detector, so bright scenery (shirts, sparks) does not pollute the
    metric.
    """
    ink = P._caption_ink_mask(region)
    if ink is None:
        return 0.0
    return float(cv2.countNonZero(ink)) / max(1, region.shape[0] * region.shape[1]) * 100.0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__)
        return 2
    video = args[0]
    n_frames = int(args[1]) if len(args) > 1 else 6

    if "--no-ai" in flags:
        P._LAMA_SLOW = True  # force the OpenCV path
        print("engine forced: OpenCV donor/Telea")
    elif "--cpu-onnx" in flags:
        P._find_lama_torch = lambda: None  # type: ignore[assignment]
        print("engine forced: ONNX (CPU)")
    else:
        print("engine:", P.inpaint_engine_status())

    t0 = time.perf_counter()
    frames = P.sample_frames(video, n=24)
    detection_ms = int((time.perf_counter() - t0) * 1000)
    if not frames:
        print("could not read frames from", video)
        return 1
    h, w = frames[0].shape[:2]
    print(f"video: {w}x{h}, sampled {len(frames)} frames, detection {detection_ms} ms")

    result = P.detect_cleanup_regions(video, targets="both")
    plates = [p for p in (result.subtitle_plate, result.logo_plate) if p is not None]
    print(f"plates: {len(plates)}")
    for p in plates:
        print(
            f"  {p.kind:6s} mode={p.mode:9s} track={p.track} region={p.region} "
            f"ref_mask={'yes' if p.ref_mask is not None else 'no'}"
        )
    if not plates:
        return 1

    out_dir = Path("logs/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full frames at a few positions for orientation (always useful when the
    # detector picks the wrong band).
    cap = cv2.VideoCapture(video)
    total0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, frac in enumerate((0.05, 0.33, 0.66, 0.95)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total0 * frac))
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"probe_{k}_{frac:.2f}.png"), fr)
    cap.release()

    # Sample frames for the visual + metric pass.
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    times: list[float] = []
    before_e: list[float] = []
    after_e: list[float] = []
    for k, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        cleaned = P.apply_clean(rgb, plates)
        times.append(time.perf_counter() - t0)
        for p in plates:
            if p.kind != "text":
                continue
            x, y, rw, rh = p.region
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + rw), min(h, y + rh)
            before_e.append(text_energy(rgb[y0:y1, x0:x1]))
            after_e.append(text_energy(cleaned[y0:y1, x0:x1]))
            if k == 0:
                crop_b = rgb[y0:y1, x0:x1]
                crop_a = cleaned[y0:y1, x0:x1]
                cv2.imwrite(
                    str(out_dir / f"band_before_{x0}_{y0}.png"),
                    cv2.cvtColor(crop_b, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(out_dir / f"band_after_{x0}_{y0}.png"),
                    cv2.cvtColor(crop_a, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(out_dir / f"frame_before_{k}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(out_dir / f"frame_after_{k}.png"),
                    cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR),
                )
    cap.release()

    if times:
        print(
            "per-frame apply: min %.3fs mean %.3fs max %.3fs (%d frames)"
            % (min(times), sum(times) / len(times), max(times), len(times))
        )
    if before_e:
        print(
            "caption ink percent: before %.3f -> after %.3f (removed %.0f%%)"
            % (
                float(np.mean(before_e)),
                float(np.mean(after_e)),
                100.0 * (1.0 - float(np.mean(after_e)) / max(1e-6, float(np.mean(before_e)))),
            )
        )
    print("crops written to", out_dir.resolve())
    print("engine now:", P.inpaint_engine_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
