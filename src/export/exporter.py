"""Export masks as PNG sequences and transparent video via FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_ffmpeg() -> str:
    """Prefer the ffmpeg shipped inside the app bundle, then system PATH."""
    if getattr(sys, "frozen", False):
        bases = [
            Path(getattr(sys, "_MEIPASS", "") or ""),
            Path(sys.executable).resolve().parent,
        ]
        for base in bases:
            for name in ("ffmpeg.exe", "ffmpeg"):
                if (base / name).is_file():
                    return str(base / name)
    return shutil.which("ffmpeg") or "ffmpeg"


class Exporter:
    """Handle export of mask sequences to PNG alpha and video via FFmpeg."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path.cwd() / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_path = _resolve_ffmpeg()

    # ------------------------------------------------------------------ #
    # PNG sequence
    # ------------------------------------------------------------------ #
    def export_png_sequence(
        self,
        masks: dict[int, np.ndarray],
        base_name: str = "mask",
        on_progress: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Save each mask as PNG with alpha channel directly into output_dir.

        Returns:
            Output directory where PNGs were written.
        """
        dest = self.output_dir
        total = len(masks)
        for i, idx in enumerate(sorted(masks)):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Export cancelled")
            mask = masks[idx]
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[mask] = [255, 255, 255, 255]  # white visible
            rgba[~mask] = [0, 0, 0, 0]  # transparent
            img = Image.fromarray(rgba, mode="RGBA")
            img.save(dest / f"{base_name}_{idx:06d}.png")
            if on_progress is not None:
                on_progress(i + 1, total)
        logger.info("Exported %d PNG masks to %s", total, dest)
        return dest

    # ------------------------------------------------------------------ #
    # Video export
    # ------------------------------------------------------------------ #
    def export_video(
        self,
        frames: dict[int, np.ndarray],
        masks: dict[int, np.ndarray],
        fps: float,
        base_name: str = "output",
        fmt: str = "prores",
        resolution: Optional[tuple[int, int]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        audio_source: Optional[Path] = None,
    ) -> Path:
        """Compose frames + masks into a video written directly into output_dir.

        Formats:
          - ``prores``  : ProRes 4444 (MOV, alpha) — falls back to qtrle
          - ``qtrle``   : QuickTime Animation (MOV, alpha)
          - ``h264``    : H.264 (MP4, no alpha — preview / sharing)

        Resolution is optional; ffmpeg scales if provided.

        ``audio_source`` optionally points at the original video file whose
        audio track should be muxed into the export (kept lossless-ish as
        AAC, chosen for mp4/mov compatibility).
        """
        if not self._ffmpeg_available():
            raise RuntimeError(
                "ffmpeg not found — reinstall Better Brush or add ffmpeg to PATH"
            )

        if fmt == "prores":
            codec, pix_fmt, ext = "prores_ks", "yuva444p10le", "mov"
            if not self._prores_available():
                logger.warning("ProRes encoder not found, falling back to qtrle")
                codec, pix_fmt, ext = "qtrle", "rgba", "mov"
        elif fmt == "qtrle":
            codec, pix_fmt, ext = "qtrle", "rgba", "mov"
        elif fmt == "h264":
            codec, pix_fmt, ext = "libx264", "yuv420p", "mp4"
        else:
            raise ValueError(f"Unknown export format: {fmt}")

        out_file = self.output_dir / f"{base_name}.{ext}"

        # Build RGBA frames in temp dir
        with tempfile.TemporaryDirectory(prefix="bb_export_") as tmp:
            tmp_path = Path(tmp)
            total = len(frames)
            for i, idx in enumerate(sorted(frames)):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Export cancelled")
                frame = frames[idx]
                mask = masks.get(idx)
                if mask is None:
                    mask = np.ones(frame.shape[:2], dtype=bool)
                rgba = self._compose_rgba(frame, mask)
                Image.fromarray(rgba, mode="RGBA").save(tmp_path / f"frm_{idx:06d}.png")
                if on_progress is not None:
                    on_progress(i + 1, total)

            cmd = [
                self._ffmpeg_path,
                "-y",
                "-framerate", str(fps),
                "-i", str(tmp_path / "frm_%06d.png"),
            ]
            if resolution:
                w, h = resolution
                cmd += ["-vf", f"scale={w}:{h}"]
            cmd += [
                "-c:v", codec,
                "-pix_fmt", pix_fmt,
                "-an",
                str(out_file),
            ]
            logger.info("Running FFmpeg: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = proc.communicate(timeout=3600)
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Export cancelled")
            if proc.returncode != 0:
                logger.error("FFmpeg stderr: %s", stderr)
                raise RuntimeError(f"FFmpeg failed with code {proc.returncode}")
            del stdout

        # ── Mux original audio if requested ───────────────────────────────
        if audio_source is not None and Path(audio_source).is_file():
            out_file = self._mux_audio(out_file, audio_source)

        logger.info("Exported video to %s", out_file)
        return out_file

    def _mux_audio(self, video_file: Path, audio_source: Path) -> Path:
        """Re-mux *audio_source*'s audio into *video_file* (fast, no re-encode)."""
        # Keep a real container extension so ffmpeg can infer the output format
        # (a bare ".mux.tmp" suffix makes ffmpeg fail with "Invalid argument").
        tmp = video_file.with_name(video_file.stem + ".mux" + video_file.suffix)
        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(video_file),
            "-i", str(audio_source),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(tmp),
        ]
        logger.info("Muxing audio: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate(timeout=3600)
        if proc.returncode != 0:
            logger.error("Audio mux failed: %s", stderr)
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg audio mux failed with code {proc.returncode}")
        del stdout
        video_file.unlink(missing_ok=True)
        tmp.replace(video_file)
        return video_file

    def export_transparent_video(
        self,
        frames: dict[int, np.ndarray],
        masks: dict[int, np.ndarray],
        fps: float,
        base_name: str = "output",
        codec: str = "prores_ks",
        pix_fmt: str = "yuva444p10le",
    ) -> Path:
        """Backwards-compatible wrapper — exports ProRes 4444 (alpha)."""
        return self.export_video(
            frames, masks, fps, base_name=base_name, fmt="prores"
        )

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
