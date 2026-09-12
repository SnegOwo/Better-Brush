"""Video reading via OpenCV with frame indexing and metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable video properties."""

    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str


class VideoReader:
    """Thread-safe-ish frame-level video reader."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._open()

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0.0
        codec = self._get_codec_name()
        self._metadata = VideoMetadata(
            path=self.path,
            width=w,
            height=h,
            fps=fps,
            total_frames=total,
            duration_sec=duration,
            codec=codec,
        )
        logger.info(
            "Video opened: %s (%dx%d @ %.2f fps, %d frames, %.1f s)",
            self.path.name,
            w,
            h,
            fps,
            total,
            duration,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("VideoReader closed.")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("Reader not opened")
        return self._metadata

    @property
    def total_frames(self) -> int:
        return self.metadata.total_frames

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    def read_frame(self, idx: int) -> np.ndarray | None:
        """Read a single frame by index. Returns RGB uint8 array or None."""
        if self._cap is None:
            return None
        if idx < 0 or idx >= self.metadata.total_frames:
            logger.warning(
                "Frame index %d out of range [0, %d)", idx, self.metadata.total_frames
            )
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning("Failed to read frame %d", idx)
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_next(self) -> np.ndarray | None:
        """Read next frame sequentially (for playback)."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def export_frames_jpeg(
        self,
        dest_dir: Path | str,
        quality: int = 95,
        on_progress: callable | None = None,
    ) -> Path:
        """Dump every frame as ``%05d.jpg`` into *dest_dir*.

        This is the format SAM 2's video loader expects (JPEG folder).
        Returns the destination directory.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        total = self.metadata.total_frames
        for idx in range(total):
            frame = self.read_frame(idx)
            if frame is None:
                logger.warning("Stopped JPEG export at frame %d", idx)
                break
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(dest / f"{idx:05d}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if on_progress is not None:
                on_progress(idx + 1, total)
        logger.info("Exported %d JPEG frames to %s", total, dest)
        return dest

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_codec_name(self) -> str:
        if self._cap is None:
            return "unknown"
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
