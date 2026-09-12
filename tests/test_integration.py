"""Integration test: synthetic video → SAM 2 mask → export."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.exporter import Exporter
from src.hardware import ContentProfile, HardwareDetector, SAM2Profiler
from src.models.sam2_manager import PointPrompt, SAM2Manager
from src.processing.frame_buffer import FrameBuffer
from src.processing.video_reader import VideoReader
from src.utils.logging_config import configure_logging


def create_synthetic_video(
    path: Path, frames: int = 10, size: tuple[int, int] = (256, 256)
) -> None:
    """Create a synthetic video with a moving circle for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cx = int(size[0] * 0.3 + (size[0] * 0.4) * (i / frames))
        cy = size[1] // 2
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def main() -> int:
    # Windows consoles may use legacy codepages — force UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging(level=logging.INFO)

    print("=" * 60)
    print("Better Brush — Integration Test")
    print("=" * 60)

    # 1. Hardware detection
    print("\n[1/7] Hardware detection...")
    profile = HardwareDetector().detect()
    rec = SAM2Profiler().recommend(profile, content=ContentProfile.GENERAL)
    print(f"  Device: {rec.device}, Model: {rec.model_size.value}")

    # 2. Create synthetic video
    print("\n[2/7] Creating synthetic video...")
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        create_synthetic_video(video_path, frames=5, size=(256, 256))
        print(f"  Created: {video_path} ({video_path.stat().st_size} bytes)")

        # 3. Video reader
        print("\n[3/7] Video reader...")
        reader = VideoReader(video_path)
        print(
            f"  {reader.width}x{reader.height}, {reader.fps} fps, {reader.total_frames} frames"
        )
        buf = FrameBuffer(reader, max_frames=10)

        if not SAM2Manager(rec).is_available:
            print("\nSKIPPED: SAM 2 is not installed — pipeline not validated.")
            buf.close()
            return 0

        # Circle center at frame 0: cx = int(256*0.3) = 76, cy = 128, radius 30
        points = [PointPrompt(x=76, y=128, label=1)]

        # 4. SAM 2 image inference
        print("\n[4/7] SAM 2 image inference...")
        sam = SAM2Manager(rec)
        frame = buf.get(0)
        assert frame is not None
        try:
            results = sam.predict_frame(frame, points, multimask_output=True)
        except Exception as exc:
            print(f"  FAIL: image inference raised: {exc}")
            logger.exception("Image inference failed")
            buf.close()
            return 1
        best = max(results, key=lambda m: m.score)
        print(f"  {len(results)} candidate mask(s), best score={best.score:.3f}")
        if best.score < 0.5:
            print(f"  WARNING: low IoU prediction score ({best.score:.3f} < 0.5)")

        # 5. SAM 2 video propagation (file -> JPEG folder -> init_state)
        print("\n[5/7] SAM 2 video propagation...")
        try:
            sam.init_video(video_path)
            sam.add_points_to_video(frame_idx=0, obj_id=1, points=points)
            propagated = sam.propagate_in_video(start_frame_idx=0)
        except Exception as exc:
            print(f"  FAIL: propagation raised: {exc}")
            logger.exception("Propagation failed")
            buf.close()
            return 1
        masks_per_frame: dict[int, np.ndarray] = {}
        for f_idx, _obj, m, _s in propagated:
            masks_per_frame[f_idx] = m
        covered = sum(1 for m in masks_per_frame.values() if m.any())
        print(
            f"  {len(masks_per_frame)} frame(s) returned, {covered} with non-empty mask"
        )
        if covered < reader.total_frames:
            print(f"  WARNING: only {covered}/{reader.total_frames} frames have a mask")
            buf.close()
            return 1

        # 6. Export PNG from propagated masks
        print("\n[6/7] Export PNG sequence...")
        exporter = Exporter(output_dir=Path(tmp) / "exports")
        png_dir = exporter.export_png_sequence(masks_per_frame, base_name="test")
        n_pngs = len(list(Path(png_dir).glob("*.png")))
        print(f"  Exported {n_pngs} PNG(s) to {png_dir}")
        if n_pngs != reader.total_frames:
            print(f"  FAIL: expected {reader.total_frames} PNGs, got {n_pngs}")
            buf.close()
            return 1

        # 7. Export transparent video
        print("\n[7/7] Export transparent video...")
        frames_dict = {i: buf.get(i) for i in range(reader.total_frames)}
        try:
            mov_path = exporter.export_transparent_video(
                frames_dict, masks_per_frame, reader.fps, base_name="test"
            )
            size_kb = mov_path.stat().st_size / 1024
            print(f"  Exported {mov_path.name} ({size_kb:.1f} KB)")
        except Exception as exc:
            print(f"  FAIL: transparent video export raised: {exc}")
            logger.exception("Video export failed")
            buf.close()
            return 1

        buf.close()

    print("\n" + "=" * 60)
    print("Integration test PASSED (video -> SAM 2 -> masks -> exports).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
