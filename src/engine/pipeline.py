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
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.hardware.profiler import SAM2Recommendation
    from src.models.sam2_manager import SAM2Manager

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
        progress_cb=lambda c, t: _emit(on_progress, "propagate", c, t, "reading frames"),
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


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write an RGBA frame to a PNG.

    ``_rgba`` builds RGBA while OpenCV reads and writes colour as BGR(A), so
    only the three colour channels are swapped — the alpha has to stay last.
    Reversing the whole array instead (as this used to) puts a colour channel
    into alpha: the frame renders opaque with rotated colours, both in the
    panel preview and in After Effects for a PNG-sequence result.
    """
    cv2.imwrite(str(path), rgba[:, :, [2, 1, 0, 3]])


def _stop_process(proc: subprocess.Popen) -> None:
    """Stop an ffmpeg we can no longer feed, and reap it.

    Without this the child outlives the failed job with the output file still
    open, so the next attempt cannot replace it.
    """
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:  # pragma: no cover - already gone
            pass
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):  # pragma: no cover
        logger.warning("ffmpeg did not exit after being killed")


def _ffmpeg_stderr(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    try:
        return proc.stderr.read()[-400:].decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):  # pragma: no cover - closed pipe
        return ""


def _ffmpeg_failure(
    proc: subprocess.Popen,
    error: Exception | None = None,
    code: int | None = None,
) -> str:
    """Explain a failed export using ffmpeg's *own* diagnostics.

    When the frame pipe breaks, the write error is the symptom, not the cause:
    on Windows a dead ffmpeg surfaces as ``[Errno 22] Invalid argument``, which
    says nothing about why it died. ffmpeg's stderr is the part that names the
    reason, so it is always folded into the message.
    """
    detail = _ffmpeg_stderr(proc)
    parts = ["ffmpeg could not write the alpha movie"]
    if error is not None:
        parts.append(f"({type(error).__name__}: {error})")
    if code is None:
        code = proc.poll()
    if code is not None:
        parts.append(f"— exit code {code}")
    if detail:
        parts.append(f"— {detail}")
    return " ".join(parts)


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
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    if width <= 0 or height <= 0:
        raise PipelineError(
            "the source frames have no usable size, so the alpha movie "
            "could not be written.",
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
    failure: BaseException | None = None
    try:
        assert proc.stdin is not None
        for i in range(count):
            _check_cancel(cancel_event)
            frame = _read_rgb(frames_dir / f"{i:05d}.jpg")
            if frame.shape[1] != width or frame.shape[0] != height:
                raise PipelineError(
                    f"frame {i} is {frame.shape[1]}x{frame.shape[0]} but the movie "
                    f"was started as {width}x{height}: the source changes size "
                    "mid-clip, which cannot be muxed into one file.",
                    ERR_INTERNAL,
                )
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
                _write_rgba_png(preview_path, rgba)
            if i % _PROGRESS_EVERY == 0 or i == count - 1:
                _emit(on_progress, "export", i + 1, count, "writing alpha movie")
    except (BrokenPipeError, OSError) as exc:
        failure = exc
    except BaseException as exc:  # noqa: BLE001 - re-raised below, after cleanup
        # Cancellation or a bad frame: re-raised below, but only after the
        # child is stopped, so no ffmpeg is left holding the output file.
        failure = exc
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    if failure is not None:
        _stop_process(proc)
        if isinstance(failure, (PipelineError, JobCancelled)):
            raise failure
        raise PipelineError(_ffmpeg_failure(proc, failure), ERR_INTERNAL)

    code = proc.wait()
    if code != 0:
        raise PipelineError(_ffmpeg_failure(proc, None, code), ERR_INTERNAL)


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
        _write_rgba_png(out_dir / f"{i:05d}.png", rgba)
        if preview_path is not None and preview_idx is not None and i == preview_idx:
            _write_rgba_png(preview_path, rgba)
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
#: The engine serves one job at a time — ``JobRegistry.submit`` refuses a second
#: — so a loaded model can be kept warm between runs. Loading a checkpoint costs
#: ten to twenty seconds and a fresh CUDA context costs more; that is the whole
#: difference between "adjust and run it again" and "wait again".
_MANAGER_CACHE: dict[tuple[str, str, bool, bool], SAM2Manager] = {}


def _manager_key(recommendation: SAM2Recommendation) -> tuple[str, str, bool, bool]:
    """Identity of a *loaded model*: the checkpoint, device and offload flags.

    Built field by field rather than hashing the recommendation itself so any
    recommendation object works, including the small stand-ins the tests use.
    """
    return (
        str(getattr(recommendation, "model_size", "")),
        str(getattr(recommendation, "device", "")),
        bool(getattr(recommendation, "offload_state_to_cpu", False)),
        bool(getattr(recommendation, "offload_video_to_cpu", False)),
    )


def _manager_for(recommendation: SAM2Recommendation) -> SAM2Manager:
    """Return a SAM 2 manager for *recommendation*, reusing a loaded model."""
    key = _manager_key(recommendation)
    cached = _MANAGER_CACHE.get(key)
    if cached is not None:
        logger.info("reusing the loaded %s model", key[0])
        return cached

    from src.models.sam2_manager import SAM2Manager

    manager = SAM2Manager(recommendation)
    if not manager.is_available:
        raise PipelineError(
            "SAM 2 is not installed for this environment.", ERR_UNSUPPORTED
        )
    # Load the weights here rather than lazily inside the first analysis call:
    # otherwise the slowest part of a cold run is invisible to the progress
    # stages (the panel would sit on "finding the character"), the phase timings
    # lie, and a checkpoint that fails to load reads like a detection failure.
    manager.load_image_predictor()
    manager.load_video_predictor()
    # One model at a time: a second would put two checkpoints on the same GPU,
    # and the previous job's is the one worth keeping warm.
    _MANAGER_CACHE.clear()
    _MANAGER_CACHE[key] = manager
    return manager


def run_job(
    request: JobRequest,
    on_progress=None,
    cancel_event: threading.Event | None = None,
) -> JobResult:
    """Run one background-removal job to completion."""
    job_started = time.time()
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
    #: Names the step a failure happened in. The panel shows a single line of
    #: text, so "… (while tracking the subject)" saves a whole round trip.
    phase = "starting up"

    try:
        # The model is loaded first, before frames are extracted, so the panel's
        # first reported step is the one that takes longest and otherwise looks
        # like nothing happening — and so the stage order matches the panel's
        # checklist, which only ever moves forwards.
        phase = "loading the model"
        started = time.time()
        _emit(on_progress, "load", 0, 0, f"loading {recommendation.model_size.value}")
        manager = _manager_for(recommendation)
        logger.info("phase model: %.1fs", time.time() - started)

        phase = "reading the source"
        started = time.time()
        _emit(on_progress, "extract", 0, 0, "reading source")
        info = _extract_range(request, source, frames_dir, on_progress, cancel_event)
        if info.count <= 0:
            raise PipelineError("source has no frames", ERR_UNSUPPORTED)
        logger.info(
            "phase source: %.1fs (%d frames @ %dx%d)",
            time.time() - started,
            info.count,
            info.width,
            info.height,
        )

        phase = "finding the character"
        started = time.time()
        selection, key_idx = _analyze(
            manager, frames_dir, info.count, on_progress, cancel_event
        )
        logger.info(
            "phase subject: %.1fs (key frame %d, confidence %.2f, face_seeded %s)",
            time.time() - started,
            key_idx,
            selection.confidence,
            selection.face_seeded,
        )

        phase = "tracking the subject"
        started = time.time()
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
        logger.info(
            "phase tracking: %.1fs (%d/%d frames masked)",
            time.time() - started,
            len(masks),
            info.count,
        )

        phase = "writing the alpha movie"
        started = time.time()
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
        logger.info("phase export: %.1fs", time.time() - started)
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
        logger.info(
            "job finished: %s frames in %.1fs (%s, %s)",
            info.count,
            time.time() - job_started,
            recommendation.model_size.value,
            output_kind,
        )
        _emit(on_progress, "done", info.count, info.count, "finished")
        return result
    except JobCancelled:
        raise
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("job failed while %s", phase)
        raise PipelineError(f"{exc} (while {phase})", ERR_INTERNAL) from exc
    finally:
        # The extracted frames are disposable; the alpha movie next to them is
        # not, so only the frames sub-folder is removed.
        shutil.rmtree(frames_dir, ignore_errors=True)


__all__ = ["JobProgress", "PipelineError", "default_output_root", "run_job"]
