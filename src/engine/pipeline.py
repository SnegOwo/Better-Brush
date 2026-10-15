"""Headless background removal: a source file in, alpha footage out.

This is the engine the After Effects panel talks to. It reuses the same
building blocks as the desktop app (``VideoReader``, ``SAM2Manager``,
``auto_subject``, ``edge_finish``) rather than duplicating the pipeline, and it
writes its result by **streaming** frames through ffmpeg so a long shot never
has to be held in RAM all at once.

The one deliberate restriction: this path removes the background behind a
*single character*. ``no_character`` / ``multiple_subjects`` are surfaced as
errors so the panel can hand the user off to the full app.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.engine import auto_subject
from src.engine.schemas import (
    ERR_INTERNAL,
    ERR_MULTIPLE_SUBJECTS,
    ERR_NO_CHARACTER,
    ERR_UNSUPPORTED,
    JobCancelled,
    JobProgress,
    JobRequest,
    JobResult,
    SubjectInfo,
)

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr"}
_SAMPLE_FRAMES = 3
_EDGE_FEATHER = 1.0
_PROGRESS_EVERY = 5


class PipelineError(RuntimeError):
    """A job that failed for a reason the panel can explain to the user."""

    def __init__(self, message: str, kind: str = ERR_INTERNAL) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class _SourceInfo:
    count: int
    fps: float
    width: int
    height: int
    frame_in: int


def default_output_root() -> Path:
    """Where a job writes its result when the host supplies no directory.

    Deliberately *not* the OS temp folder: the After Effects project links to
    the alpha movie, and Windows can clear ``%TEMP%`` out from under it. A
    per-user application-data folder survives that.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "BetterBrush" / "AfterEffects"


# --------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------- #
def _makedirs(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PipelineError(
            f"Could not create the output folder {path}: {exc}",
            ERR_UNSUPPORTED,
        ) from exc
    return str(path)


def _emit(on_progress, stage: str, current: int, total: int, message: str = "") -> None:
    if on_progress is not None:
        try:
            on_progress(stage, int(current), int(total), message)
        except Exception:
            logger.debug("progress callback raised", exc_info=True)


def _write_jpg(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise PipelineError(f"could not read frame {path}", ERR_INTERNAL)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _extract_range(
    request: JobRequest,
    source: Path,
    dest: Path,
    on_progress,
    cancel_event: threading.Event | None,
) -> _SourceInfo:
    """Materialise the requested frame range as ``dest/%05d.jpg``."""
    dest.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        files = sorted(
            p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not files:
            raise PipelineError(f"no image files in {source}", ERR_UNSUPPORTED)
        end = len(files) - 1 if request.frame_out < 0 else min(
            request.frame_out, len(files) - 1
        )
        start = max(0, min(request.frame_in, end))
        first = cv2.imread(str(files[start]))
        if first is None:
            raise PipelineError(f"could not read {files[start]}", ERR_INTERNAL)
        height, width = first.shape[:2]
        for i, src in enumerate(files[start : end + 1]):
            _check_cancel(cancel_event)
            shutil.copyfile(src, dest / f"{i:05d}.jpg")
            if i % _PROGRESS_EVERY == 0 or i == end - start:
                _emit(on_progress, "extract", i + 1, end - start + 1, "")
        return _SourceInfo(
            count=end - start + 1,
            fps=request.fps or 24.0,
            width=int(width),
            height=int(height),
            frame_in=start,
        )

    if source.suffix.lower() in _IMAGE_SUFFIXES:
        rgb = _read_rgb(source)
        _write_jpg(dest / "00000.jpg", rgb)
        return _SourceInfo(
            count=1,
            fps=request.fps or 24.0,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            frame_in=0,
        )

    from src.processing.video_reader import VideoReader

    reader = VideoReader(source)
    try:
        total = reader.total_frames
        if total <= 0:
            raise PipelineError(f"no decodable frames in {source}", ERR_UNSUPPORTED)
        end = total - 1 if request.frame_out < 0 else min(request.frame_out, total - 1)
        start = max(0, min(request.frame_in, end))
        span = end - start + 1
        for i, idx in enumerate(range(start, end + 1)):
            _check_cancel(cancel_event)
            frame = reader.read_frame(idx)
            if frame is None:
                logger.warning("decoding stopped early at frame %d", idx)
                span = i
                break
            _write_jpg(dest / f"{i:05d}.jpg", frame)
            if i % _PROGRESS_EVERY == 0 or i == span - 1:
                _emit(on_progress, "extract", i + 1, span, "")
        if span <= 0:
            raise PipelineError(f"no decodable frames in {source}", ERR_UNSUPPORTED)
        return _SourceInfo(
            count=span,
            fps=reader.fps or request.fps or 24.0,
            width=int(reader.width),
            height=int(reader.height),
            frame_in=start,
        )
    finally:
        reader.close()


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled()


# --------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------- #
def _sample_indices(count: int) -> list[int]:
    if count <= 1:
        return [0]
    picks = {0, count // 2, count - 1}
    if count > 6:
        picks = {count // 4, count // 2, (3 * count) // 4}
    return sorted(picks)


def _analyze(
    manager,
    frames_dir: Path,
    count: int,
    on_progress,
    cancel_event: threading.Event | None,
) -> tuple[auto_subject.SubjectSelection, int]:
    """Probe a few frames and pick the strongest single-character hit."""
    indices = _sample_indices(count)
    best: auto_subject.SubjectSelection | None = None
    best_idx = indices[len(indices) // 2]
    saw_multiple = False

    for k, idx in enumerate(indices):
        _check_cancel(cancel_event)
        frame = _read_rgb(frames_dir / f"{idx:05d}.jpg")
        face = auto_subject.detect_face_center(frame)
        selection = auto_subject.find_subject(manager, frame, face_center=face)
        _emit(
            on_progress,
            "analyze",
            k + 1,
            len(indices),
            f"probing frame {idx}",
        )
        if selection.multiple:
            saw_multiple = True
        if selection.found and (best is None or selection.confidence > best.confidence):
            best = selection
            best_idx = idx

    if best is not None:
        return best, best_idx
    if saw_multiple:
        raise PipelineError(
            "More than one character is visible — open Better Brush to pick "
            "the subjects you want.",
            ERR_MULTIPLE_SUBJECTS,
        )
    raise PipelineError(
        "No clear character found in this clip — open Better Brush to paint a "
        "brush stroke on the subject.",
        ERR_NO_CHARACTER,
    )


# --------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------- #
def _propagate(
    manager,
    frames_dir: Path,
    key_idx: int,
    seed_mask: np.ndarray,
    count: int,
    on_progress,
    cancel_event: threading.Event | None,
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    seen = 0

    def _on_frame(frame_idx: int, _obj_id: int, mask: np.ndarray) -> None:
        nonlocal seen
        if frame_idx < 0 or frame_idx >= count:
            return
        masks[frame_idx] = np.asarray(mask).astype(bool)
        seen += 1
        if seen % _PROGRESS_EVERY == 0:
            _emit(on_progress, "propagate", seen, count, f"tracking frame {frame_idx}")

    _emit(on_progress, "propagate", 0, count, "tracking…")
    manager.init_video(
        frames_dir,
        progress_cb=lambda c, t: _emit(on_progress, "extract", c, t, "preparing"),
    )
    manager.add_mask_to_video(key_idx, 1, seed_mask)

    forward = manager.propagate_in_video(
        start_frame_idx=key_idx,
        on_frame=_on_frame,
        cancel_event=cancel_event,
    )
    for frame_idx, _obj, mask, _score in forward:
        if 0 <= frame_idx < count:
            masks[frame_idx] = np.asarray(mask).astype(bool)

    if key_idx > 0:
        backward = manager.propagate_in_video(
            start_frame_idx=key_idx,
            reverse=True,
            on_frame=_on_frame,
            cancel_event=cancel_event,
        )
        for frame_idx, _obj, mask, _score in backward:
            if 0 <= frame_idx < count:
                masks.setdefault(frame_idx, np.asarray(mask).astype(bool))

    _emit(on_progress, "propagate", count, count, "propagation complete")
    return masks


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
def _alpha_for(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from src.processing.edge_finish import soft_coverage

    return soft_coverage(mask, feather=_EDGE_FEATHER)


def _rgba(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = _alpha_for(frame_rgb, mask)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([frame_rgb, alpha_u8])


def _streaming_mov(
    frames_dir: Path,
    masks: dict[int, np.ndarray],
    count: int,
    fps: float,
    width: int,
    height: int,
    out_path: Path,
    on_progress,
    cancel_event: threading.Event | None,
    preview_path: Path | None = None,
    preview_idx: int | None = None,
) -> None:
    from src.export.exporter import _resolve_ffmpeg

    ffmpeg = _resolve_ffmpeg()
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).is_file():
        raise PipelineError(
            "ffmpeg was not found, so the alpha movie could not be written.",
            ERR_UNSUPPORTED,
        )

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4444",
        "-pix_fmt",
        "yuva444p10le",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    last_mask: np.ndarray | None = None
    try:
        assert proc.stdin is not None
        for i in range(count):
            _check_cancel(cancel_event)
            frame = _read_rgb(frames_dir / f"{i:05d}.jpg")
            mask = masks.get(i)
            if mask is None:
                mask = last_mask
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype=bool)
            else:
                last_mask = mask
            rgba = _rgba(frame, mask)
            proc.stdin.write(rgba.tobytes())
            if preview_path is not None and preview_idx is not None and i == preview_idx:
                cv2.imwrite(str(preview_path), rgba[:, :, ::-1])
            if i % _PROGRESS_EVERY == 0 or i == count - 1:
                _emit(on_progress, "export", i + 1, count, "writing alpha movie")
    except (BrokenPipeError, OSError) as exc:
        raise PipelineError(f"ffmpeg failed: {exc}", ERR_INTERNAL) from exc
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
    code = proc.wait()
    if code != 0:
        err = b""
        if proc.stderr is not None:
            err = proc.stderr.read()[-400:]
        raise PipelineError(
            f"ffmpeg exited with code {code}: {err.decode(errors='replace').strip()}",
            ERR_INTERNAL,
        )


def _png_sequence(
    frames_dir: Path,
    masks: dict[int, np.ndarray],
    count: int,
    out_dir: Path,
    on_progress,
    cancel_event: threading.Event | None,
    preview_path: Path | None = None,
    preview_idx: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    last_mask: np.ndarray | None = None
    for i in range(count):
        _check_cancel(cancel_event)
        frame = _read_rgb(frames_dir / f"{i:05d}.jpg")
        mask = masks.get(i, last_mask)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=bool)
        else:
            last_mask = mask
        rgba = _rgba(frame, mask)
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), rgba[:, :, ::-1])
        if preview_path is not None and preview_idx is not None and i == preview_idx:
            cv2.imwrite(str(preview_path), rgba[:, :, ::-1])
        if i % _PROGRESS_EVERY == 0 or i == count - 1:
            _emit(on_progress, "export", i + 1, count, "writing PNG frames")


# --------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------- #
def _quality_report(masks: dict[int, np.ndarray], count: int) -> dict:
    ratios = []
    for i in range(count):
        mask = masks.get(i)
        if mask is None:
            continue
        ratios.append(float(mask.mean()))
    if not ratios:
        return {"coverage": 0.0, "flicker": 0.0, "state": "no_masks"}
    arr = np.asarray(ratios, dtype=np.float32)
    flicker = float(np.mean(np.abs(np.diff(arr)))) if arr.size > 1 else 0.0
    return {
        "coverage": round(float(arr.mean()), 4),
        "flicker": round(flicker, 4),
        "state": "ok",
    }


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #
def run_job(
    request: JobRequest,
    on_progress=None,
    cancel_event: threading.Event | None = None,
) -> JobResult:
    """Run one background-removal job to completion."""
    source = Path(request.input_path)
    if not source.exists():
        raise PipelineError(f"input not found: {source}", ERR_UNSUPPORTED)

    from src.engine import model_setup

    recommendation = model_setup.build_recommendation(
        quality=request.quality, preset=request.preset, model=request.model
    )

    if request.output_dir:
        work = Path(request.output_dir)
    else:
        root = default_output_root()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        work = Path(tempfile.mkdtemp(prefix=f"{stamp}_", dir=str(_makedirs(root))))
    work.mkdir(parents=True, exist_ok=True)
    frames_dir = work / "frames"

    try:
        _emit(on_progress, "extract", 0, 0, "reading source")
        info = _extract_range(request, source, frames_dir, on_progress, cancel_event)
        if info.count <= 0:
            raise PipelineError("source has no frames", ERR_UNSUPPORTED)

        from src.models.sam2_manager import SAM2Manager

        manager = SAM2Manager(recommendation)
        if not manager.is_available:
            raise PipelineError(
                "SAM 2 is not installed for this environment.", ERR_UNSUPPORTED
            )

        selection, key_idx = _analyze(
            manager, frames_dir, info.count, on_progress, cancel_event
        )
        masks = _propagate(
            manager,
            frames_dir,
            key_idx,
            selection.mask,
            info.count,
            on_progress,
            cancel_event,
        )
        manager.reset_video_state()

        fps = request.fps or info.fps or 24.0
        preview = work / "preview.png"
        mov = work / "alpha.mov"
        output_kind = "mov_alpha"
        try:
            _streaming_mov(
                frames_dir,
                masks,
                info.count,
                fps,
                info.width,
                info.height,
                mov,
                on_progress,
                cancel_event,
                preview_path=preview,
                preview_idx=key_idx,
            )
            output_path = mov
        except PipelineError as exc:
            if exc.kind != ERR_UNSUPPORTED:
                raise
            logger.warning("falling back to PNG sequence: %s", exc)
            png_dir = work / "png"
            _png_sequence(
                frames_dir,
                masks,
                info.count,
                png_dir,
                on_progress,
                cancel_event,
                preview_path=preview,
                preview_idx=key_idx,
            )
            output_path = png_dir
            output_kind = "png_sequence"

        subject = SubjectInfo(
            confidence=selection.confidence,
            box=selection.box,
            face_seeded=selection.face_seeded,
            reasons=selection.reasons,
        )
        result = JobResult(
            output_path=str(output_path),
            output_kind=output_kind,
            frame_in=info.frame_in,
            frame_out=info.frame_in + info.count - 1,
            frame_count=info.count,
            fps=float(fps),
            width=info.width,
            height=info.height,
            preview_path=str(preview) if preview.is_file() else None,
            subject=subject,
            qc=_quality_report(masks, info.count),
            model=recommendation.model_size.value,
        )
        _emit(on_progress, "done", info.count, info.count, "finished")
        return result
    except JobCancelled:
        raise
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("job failed")
        raise PipelineError(str(exc), ERR_INTERNAL) from exc
    finally:
        # The extracted frames are disposable; the alpha movie next to them is
        # not, so only the frames sub-folder is removed.
        shutil.rmtree(frames_dir, ignore_errors=True)


__all__ = ["JobProgress", "PipelineError", "default_output_root", "run_job"]
