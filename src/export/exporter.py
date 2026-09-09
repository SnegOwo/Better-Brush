"""Export masks as PNG sequences and transparent video via FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Exporter:
    """Handle export of mask sequences to PNG alpha and ProRes 4444."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path.cwd() / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"

    # ------------------------------------------------------------------ #
    # PNG sequence
    # ------------------------------------------------------------------ #
    def export_png_sequence(
        self,
        masks: dict[int, np.ndarray],
        base_name: str = "mask",
    ) -> Path:
        """Save each mask as PNG with alpha channel.

        Returns:
            Directory where PNGs were written.
        """
        dest = self.output_dir / f"{base_name}_png"
        dest.mkdir(parents=True, exist_ok=True)
        for idx, mask in sorted(masks.items()):
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[mask] = [255, 255, 255, 255]  # white visible
            rgba[~mask] = [0, 0, 0, 0]  # transparent
            img = Image.fromarray(rgba, mode="RGBA")
            img.save(dest / f"{base_name}_{idx:06d}.png")
        logger.info("Exported %d PNG masks to %s", len(masks), dest)
        return dest

    # ------------------------------------------------------------------ #
    # Transparent video (ProRes 4444 or fallback)
    # ------------------------------------------------------------------ #
    def export_transparent_video(
        self,
        frames: dict[int, np.ndarray],
        masks: dict[int, np.ndarray],
        fps: float,
        base_name: str = "output",
        codec: str = "prores_ks",
        pix_fmt: str = "yuva444p10le",
    ) -> Path:
        """Compose RGB frames + masks into a transparent video.

        Falls back to QuickTime PNG codec if ProRes is unavailable.
        """
        if not self._ffmpeg_available():
            raise RuntimeError("ffmpeg not found in PATH")

        dest = self.output_dir / f"{base_name}_video"
        dest.mkdir(parents=True, exist_ok=True)
        out_file = dest / f"{base_name}.mov"

        # Build RGBA frames in temp dir
        with tempfile.TemporaryDirectory(prefix="bb_export_") as tmp:
            tmp_path = Path(tmp)
            for idx in sorted(frames.keys()):
                frame = frames[idx]
                mask = masks.get(idx)
                if mask is None:
                    mask = np.ones(frame.shape[:2], dtype=bool)
                rgba = self._compose_rgba(frame, mask)
                Image.fromarray(rgba, mode="RGBA").save(tmp_path / f"frm_{idx:06d}.png")

            # Determine if ProRes is supported
            if codec == "prores_ks" and not self._prores_available():
                logger.warning("ProRes encoder not found, falling back to qtrle")
                codec = "qtrle"
                pix_fmt = "rgba"

            cmd = [
                self._ffmpeg_path,
                "-y",
                "-framerate", str(fps),
                "-i", str(tmp_path / "frm_%06d.png"),
                "-c:v", codec,
                "-pix_fmt", pix_fmt,
                "-an",
                str(out_file),
            ]
            logger.info("Running FFmpeg: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("FFmpeg stderr: %s", result.stderr)
                raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

        logger.info("Exported transparent video to %s", out_file)
        return out_file

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_rgba(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgb = frame[:, :, :3] if frame.shape[2] >= 3 else np.stack([frame[:, :, 0]] * 3, axis=-1)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba

    def _ffmpeg_available(self) -> bool:
        return shutil.which(self._ffmpeg_path) is not None

    def _prores_available(self) -> bool:
        try:
            result = subprocess.run(
                [self._ffmpeg_path, "-codecs"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "prores_ks" in result.stdout or "prores" in result.stdout
        except Exception:  # noqa: BLE001
            return False
