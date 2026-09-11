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
from PIL import Image, ImageDraw, ImageFont

from src.core import licensing

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
        background: Optional[tuple[int, int, int]] = None,
        watermark: Optional[bool] = None,
    ) -> Path:
        """Compose frames + masks into a video written directly into output_dir.

        Formats:
          - ``prores``  : ProRes 4444 (MOV, alpha) — falls back to qtrle
          - ``qtrle``   : QuickTime Animation (MOV, alpha)
          - ``h264``    : H.264 (MP4, no alpha — preview / sharing)

        Resolution is optional; ffmpeg scales if provided.

        ``audio_source`` optionally points at the original video file whose
        audio track should be muxed into the export.

        ``background`` (an RGB tuple) replaces the transparent areas with a
        solid color and disables alpha — for H.264 MP4 exports that will be
        chroma-keyed (green/blue screen) by editors without alpha support.

        ``watermark`` defaults to the edition: Lite exports carry a small
        "Better Brush" watermark; Pro exports are clean.
        """
        if not self._ffmpeg_available():
            raise RuntimeError(
                "ffmpeg not found — reinstall Better Brush or add ffmpeg to PATH"
            )

        pro = licensing.is_pro()

        # ── Lite: cap resolution + audio, add watermark (codecs are shared) ──
        if not pro:
            if resolution:
                w, h = resolution
                capped = (
                    min(int(w), licensing.LITE_MAX_WIDTH),
                    min(int(h), licensing.LITE_MAX_HEIGHT),
                )
                if capped != (int(w), int(h)):
                    logger.info(
                        "Lite edition: scaling %sx%s -> %sx%s",
                        w, h, *capped,
                    )
                resolution = capped
            audio_source = None  # original-audio muxing is a Pro feature
        if watermark is None:
            watermark = not pro

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
                rgba = self._compose_rgba(frame, mask, background=background)
                if watermark:
                    rgba = self._draw_watermark(rgba)
                Image.fromarray(rgba, mode="RGBA").save(tmp_path / f"frm_{idx:06d}.png")
                if on_progress is not None:
                    on_progress(i + 1, total)

            cmd = [
                self._ffmpeg_path,
                "-y",
                "-nostdin",
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
        """Mux *audio_source*'s audio into *video_file* (fast, copy-first).

        Audio is copied losslessly when the codec is container-compatible,
        otherwise re-encoded to AAC.  If audio cannot be muxed for any reason
        the export is kept silent instead of failing — audio must never
        brick an otherwise-finished render.
        """
        # Keep a real container extension so ffmpeg can infer the output format
        # (a bare ".mux.tmp" suffix makes ffmpeg fail with "Invalid argument").
        tmp = video_file.with_name(video_file.stem + ".mux" + video_file.suffix)

        for audio_args in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):
            cmd = [
                self._ffmpeg_path,
                "-y",
                "-nostdin",
                "-i", str(video_file),
                "-i", str(audio_source),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                *audio_args,
                "-shortest",
                str(tmp),
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = proc.communicate(timeout=3600)
            if proc.returncode == 0:
                del stdout
                video_file.unlink(missing_ok=True)
                tmp.replace(video_file)
                return video_file
            # Log detail, then fall through to the next strategy.
            tail = " | ".join(stderr.strip().splitlines()[-4:]) if stderr.strip() else "no stderr"
            logger.warning(
                "Audio mux attempt failed (code %s, args %s): %s",
                proc.returncode, " ".join(audio_args), tail,
            )
            tmp.unlink(missing_ok=True)

        logger.warning("Could not mux original audio; keeping export silent: %s", video_file)
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
    def _compose_rgba(
        frame: np.ndarray, mask: np.ndarray, background: Optional[tuple[int, int, int]] = None
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgb = frame[:, :, :3] if frame.shape[2] >= 3 else np.stack([frame[:, :, 0]] * 3, axis=-1)
        if background is not None:
            # Flatten the alpha channel into a solid chroma-key background
            # (kept fully opaque for yuv420p): subject pixels keep their color,
            # everything else becomes the key color.
            fill = np.zeros((h, w, 3), dtype=np.uint8)
            fill[:, :] = background
            out = np.where(mask[:, :, None], rgb, fill)
            rgba[:, :, :3] = out
            rgba[:, :, 3] = 255
            return rgba
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba

    @staticmethod
    def _draw_watermark(rgba: np.ndarray) -> np.ndarray:
        """Burn a small 'Better Brush · Lite' mark into the bottom-right.

        Text is drawn over the subject only (masked pixels), so transparent
        exports keep a clean alpha channel while every Lite video still
        carries a modest, non-destructive watermark.
        """
        h, w = rgba.shape[:2]
        if h < 60 or w < 120:
            return rgba  # too small to watermark legibly
        img = Image.fromarray(rgba, mode="RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        size = max(9, min(14, h // 60))
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:  # noqa: BLE001 - fall back to the default bitmap font
            font = ImageFont.load_default()
        text = "Better Brush"
        text_w = draw.textlength(text, font=font)
        pad = max(6, h // 40)
        top = h - size - pad
        left = w - int(text_w) - pad - 8
        shadow = (0, 0, 0, 160)
        white = (235, 235, 238, 200)
        draw.text((left + 1, top + 1), text, font=font, fill=shadow)
        draw.text((left, top), text, font=font, fill=white)
        return np.asarray(Image.alpha_composite(img, overlay), dtype=np.uint8)

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
