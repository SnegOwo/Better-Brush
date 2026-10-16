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

import base64
import binascii
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.engine import auto_subject
from src.engine.schemas import (
    ERR_BAD_REQUEST,
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
from src.utils.proc import no_window

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.hardware.profiler import SAM2Recommendation
    from src.models.sam2_manager import SAM2Manager

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr"}
_SAMPLE_FRAMES = 3
_EDGE_FEATHER = 1.0
_PROGRESS_EVERY = 5

#: How many JPEGs the export loop keeps decoding ahead of the ffmpeg pipe.
#: Decoding a frame and compositing its RGBA are CPU work that releases the GIL
#: (both are OpenCV calls), while the pipe write is the part that has to happen
#: in order. Overlapping them costs a few frames of RAM and takes the decode off
#: the critical path, which is most of the export stage on a long shot.
_EXPORT_PREFETCH = 4

#: JPEG encodes run in the extraction loop's shadow, for the same reason.
_EXTRACT_WORKERS = 4

#: Brush frames the panel has already asked for, newest first. The brush surface
#: re-renders whenever the frame slider moves, so a scrub back and forth would
#: otherwise pay for a full mask unpack and PNG encode every time.
_FRAME_CACHE_LIMIT = 8
_FRAME_CACHE: dict[tuple[str, str, int, float], bytes] = {}

#: Written next to the result so a later brush correction can re-track from the
#: exact key frame the first pass picked.
_SEED_MASK = "seed.png"
_SEED_META = "seed.json"

#: The per-frame masks, packed and compressed. They are what lets the panel
#: render *any* frame for the brush and correct the frame the user is looking
#: at, without keeping a matte-sized float image per frame.
_MASKS_FILE = "masks.npz"

#: Interior background regions up to this fraction of the frame are treated as
#: holes and filled. SAM 2 leaves gaps in dark hair, inside a curled hand and
#: between fingers; those composite as see-through patches. The app's own
#: default cap (~0.2 %) is too small for hair, which is exactly the defect the
#: panel brush was brought in to fix.
_FILL_HOLE_RATIO = 0.01

#: Extracted frames are kept so the brush can render any frame and a correction
#: can skip re-extraction. Beyond this many frames the disk cost stops being
#: worth it, so they are dropped and the brush decodes the source on demand.
_KEEP_FRAMES_LIMIT = 1800

#: Last packed mask set, keyed by path. The brush hits /frame once per scrub,
#: and re-reading a multi-megabyte archive each time would make scrubbing janky.
_MASK_CACHE: dict[str, tuple[float, tuple[int, int], np.ndarray]] = {}


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

    Deliberately *not* the OS temp folder: the host's project links to the
    alpha movie, and Windows can clear ``%TEMP%`` out from under it. A per-user
    application-data folder survives that. Shared by every host panel — the
    After Effects and Premiere Pro panels run the same engine.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "BetterBrush" / "Plugins"


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
        # The encode of frame i is independent of the decode of frame i+1 and
        # OpenCV releases the GIL while writing, so the encode rides along in the
        # next decode's shadow instead of doubling the stage.
        inflight: deque[Future[None]] = deque()
        with ThreadPoolExecutor(max_workers=_EXTRACT_WORKERS) as pool:
            for i, idx in enumerate(range(start, end + 1)):
                _check_cancel(cancel_event)
                frame = reader.read_frame(idx)
                if frame is None:
                    logger.warning("decoding stopped early at frame %d", idx)
                    span = i
                    break
                inflight.append(pool.submit(_write_jpg, dest / f"{i:05d}.jpg", frame))
                if len(inflight) >= _EXTRACT_WORKERS:
                    inflight.popleft().result()
                if i % _PROGRESS_EVERY == 0 or i == span - 1:
                    _emit(on_progress, "extract", i + 1, span, "")
            for future in inflight:
                future.result()
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


def _decoded_frames(
    frames_dir: Path,
    count: int,
    cancel_event: threading.Event | None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, rgb)`` in order, decoding ahead on a thread pool.

    The consumer (an ffmpeg pipe write, or a PNG write) stays strictly in order;
    only the JPEG decode is overlapped. ``VideoReader`` hands back fresh arrays,
    so nothing is shared with the decoder.
    """
    workers = max(1, min(_EXPORT_PREFETCH, count))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple[int, Future[np.ndarray]]] = deque(
            (i, pool.submit(_read_rgb, frames_dir / f"{i:05d}.jpg"))
            for i in range(min(workers, count))
        )
        for i in range(count):
            _check_cancel(cancel_event)
            index, future = pending.popleft()
            ahead = i + workers
            if ahead < count:
                pending.append(
                    (ahead, pool.submit(_read_rgb, frames_dir / f"{ahead:05d}.jpg"))
                )
            yield index, future.result()


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
    """Probe a few frames and pick the strongest single-character hit.

    The winner is then re-prompted once with a box around it: a point prompt
    only segments the part it happened to land on, and the character's hair is
    what it most often leaves behind.
    """
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
            refined = auto_subject.refine_mask(
                manager, frame, selection.mask, face_center=face
            )
            if refined is not None:
                selection.mask = refined
                selection.box = auto_subject.bbox_of(refined)
                selection.reasons.append("grown to the full silhouette")
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
# Seed persistence + brush correction
# --------------------------------------------------------------------- #
def _save_seed(work: Path, mask: np.ndarray, meta: dict) -> None:
    """Persist the key-frame seed so a brush correction can re-use it.

    The mask is tiny and the metadata is a few numbers, so this costs nothing
    next to the alpha movie sitting beside it — and it is what lets a correction
    skip the subject search entirely.
    """
    cv2.imwrite(str(work / _SEED_MASK), (np.asarray(mask).astype(np.uint8) * 255))
    (work / _SEED_META).write_text(json.dumps(meta), encoding="utf-8")


def _load_seed(resume_dir: str) -> tuple[np.ndarray, dict]:
    root = Path(resume_dir)
    mask_path = root / _SEED_MASK
    meta_path = root / _SEED_META
    if not mask_path.is_file() or not meta_path.is_file():
        raise PipelineError(
            "The earlier result is no longer on disk, so there is nothing to "
            "correct — run Remove Background again.",
            ERR_UNSUPPORTED,
        )
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise PipelineError(
            "The stored mask could not be read back — run the removal again.",
            ERR_UNSUPPORTED,
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    return gray > 127, meta


def _decode_stroke(payload: str | None, name: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Decode a base64 PNG brush stroke into a bool mask at *shape* (h, w)."""
    if not payload:
        return None
    # Tolerate a data: URL prefix, which canvas.toDataURL() produces.
    text = payload.split(",")[-1].strip()
    try:
        raw = base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise PipelineError(f"'{name}' is not valid base64 data.", ERR_BAD_REQUEST) from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise PipelineError(
            f"'{name}' could not be read as an image.", ERR_BAD_REQUEST
        )
    if image.shape[:2] != shape:
        image = cv2.resize(
            image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return image > 127


def _apply_correction(
    seed: np.ndarray, add: np.ndarray | None, remove: np.ndarray | None
) -> np.ndarray:
    """Layer painted strokes over the stored seed: add wins, then remove trims."""
    corrected = np.asarray(seed).astype(bool).copy()
    if add is not None:
        corrected |= add
    if remove is not None:
        corrected &= ~remove
    return corrected


def _read_meta(work: Path) -> dict:
    path = work / _SEED_META
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close enclosed gaps (dark hair, finger webbing) that SAM 2 leaves open."""
    from src.processing.mask_refine import fill_mask_holes

    if mask is None or not np.asarray(mask).any():
        return mask
    area = int(mask.shape[0] * mask.shape[1])
    return fill_mask_holes(
        mask, max_hole_area=max(4096, int(area * _FILL_HOLE_RATIO))
    )


def _close_holes(
    masks: dict[int, np.ndarray],
    count: int,
    on_progress,
    cancel_event: threading.Event | None,
) -> None:
    """Fill enclosed gaps in every tracked mask, in place."""
    for n, index in enumerate(sorted(masks)):
        _check_cancel(cancel_event)
        masks[index] = _fill_holes(masks[index])
        if n % _PROGRESS_EVERY == 0:
            _emit(on_progress, "propagate", n + 1, count, "closing gaps in the mask")
    _emit(on_progress, "propagate", count, count, "closing gaps in the mask")


def _save_masks(
    work: Path, masks: dict[int, np.ndarray], count: int, shape: tuple[int, int]
) -> None:
    """Persist the per-frame masks as one packed, compressed archive."""
    packed = np.zeros((count, (shape[0] * shape[1] + 7) // 8), dtype=np.uint8)
    for i in range(count):
        mask = masks.get(i)
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.shape[:2] != shape:
            mask = cv2.resize(
                (mask.astype(np.uint8) * 255),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        packed[i] = np.packbits(mask.reshape(-1).astype(bool))
    np.savez_compressed(
        work / _MASKS_FILE, shape=np.array(shape, dtype=np.int64), packed=packed
    )


def _masks_packed(work: Path) -> tuple[tuple[int, int], np.ndarray]:
    path = work / _MASKS_FILE
    if not path.is_file():
        raise PipelineError(
            "This result has no stored masks, so it cannot be brushed — run the "
            "removal again.",
            ERR_UNSUPPORTED,
        )
    stamp = path.stat().st_mtime
    cached = _MASK_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1], cached[2]
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        packed = np.ascontiguousarray(data["packed"])
    _MASK_CACHE.clear()  # only the newest result is ever worth caching
    _MASK_CACHE[str(path)] = (stamp, shape, packed)
    return shape, packed


def _mask_at(work: Path, index: int) -> np.ndarray | None:
    shape, packed = _masks_packed(work)
    if packed.shape[0] == 0 or index < 0 or index >= packed.shape[0]:
        return None
    bits = np.unpackbits(packed[index], count=shape[0] * shape[1])
    return bits.reshape(shape).astype(bool)


def _read_source_rgb(source: Path, index: int) -> np.ndarray:
    """Decode one frame of the source by absolute frame index."""
    if source.is_dir():
        files = sorted(
            p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not files:
            raise PipelineError(f"no image files in {source}", ERR_UNSUPPORTED)
        return _read_rgb(files[max(0, min(index, len(files) - 1))])
    if source.suffix.lower() in _IMAGE_SUFFIXES:
        return _read_rgb(source)

    from src.processing.video_reader import VideoReader

    reader = VideoReader(source)
    try:
        frame = reader.read_frame(index)
    finally:
        reader.close()
    if frame is None:
        raise PipelineError(
            f"could not decode frame {index} from {source.name}", ERR_INTERNAL
        )
    return frame


def _frame_at(work: Path, index: int) -> np.ndarray:
    """The source frame for *index*, from the kept frames or the source file."""
    kept = work / "frames" / f"{index:05d}.jpg"
    if kept.is_file():
        return _read_rgb(kept)
    meta = _read_meta(work)
    source = Path(str(meta.get("source", "")))
    if not source.exists():
        raise PipelineError(
            "The source clip is no longer available for this result.",
            ERR_UNSUPPORTED,
        )
    return _read_source_rgb(source, int(meta.get("frame_in", 0)) + index)


def _frame_cache_key(work: Path, index: int, mode: str) -> tuple[str, str, int, float]:
    """Identity of a rendered frame: result, frame, and the mask file's stamp.

    The stamp is what makes a stale entry impossible: a correction rewrites
    ``masks.npz``, so every frame rendered before it is keyed differently and
    quietly becomes unreachable.
    """
    try:
        stamp = (work / _MASKS_FILE).stat().st_mtime
    except OSError:
        stamp = 0.0
    return (mode, str(work), int(index), stamp)


def _frame_cache_get(key: tuple[str, str, int, float]) -> bytes | None:
    body = _FRAME_CACHE.pop(key, None)
    if body is not None:
        _FRAME_CACHE[key] = body  # re-insert: dict order is the LRU order
    return body


def _frame_cache_put(key: tuple[str, str, int, float], body: bytes) -> None:
    work = key[1]
    for stale in [k for k in _FRAME_CACHE if k[1] == work and k != key]:
        del _FRAME_CACHE[stale]
    _FRAME_CACHE[key] = body
    while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
        _FRAME_CACHE.pop(next(iter(_FRAME_CACHE)))


def render_frame(work_dir: str, index: int) -> bytes:
    """PNG (RGBA) of the cut-out at *index*, for the panel's brush surface."""
    work = Path(work_dir)
    key = _frame_cache_key(work, index, "rgba")
    cached = _frame_cache_get(key)
    if cached is not None:
        return cached

    mask = _mask_at(work, index)
    if mask is None:
        raise PipelineError("That frame is outside this result.", ERR_BAD_REQUEST)
    frame = _frame_at(work, index)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            (mask.astype(np.uint8) * 255),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 127
    from src.processing.edge_finish import soft_coverage

    rgba = _rgba(frame, mask, soft_coverage(mask, feather=_EDGE_FEATHER))
    # Compression 1 rather than the default 6: this image only ever crosses the
    # loopback to the panel, so trading a bigger buffer for a three-times-faster
    # encode is a straight win on scrub latency.
    ok, buf = cv2.imencode(
        ".png",
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
    )
    if not ok:
        raise PipelineError("could not encode the preview frame.", ERR_INTERNAL)
    body = buf.tobytes()
    _frame_cache_put(key, body)
    return body


def render_source_frame(work_dir: str, index: int) -> tuple[bytes, str]:
    """JPEG of the *untouched* source frame at *index*, plus its media type.

    The brush surface paints on the cut-out, which makes it hard to judge an
    edge: the user cannot see what the background looked like. This is the other
    half of the compare toggle in the panel.
    """
    work = Path(work_dir)
    key = _frame_cache_key(work, index, "source")
    cached = _frame_cache_get(key)
    if cached is not None:
        return cached, "image/jpeg"

    frame = _frame_at(work, index)
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    if not ok:
        raise PipelineError("could not encode the source frame.", ERR_INTERNAL)
    body = buf.tobytes()
    _frame_cache_put(key, body)
    return body, "image/jpeg"


def _resume_seed(
    request: JobRequest,
    info: _SourceInfo,
    on_progress,
) -> tuple[auto_subject.SubjectSelection, int]:
    """Rebuild the seed from an earlier result, applying the panel's strokes.

    The frame the user painted becomes the new propagation seed, so a wrong
    frame is fixed where it was seen rather than only at the original key frame.
    """
    work = Path(request.resume_dir or "")
    fallback, meta = _load_seed(str(work))
    frame = request.correction_frame
    if frame is None:
        frame = int(meta.get("key_idx", 0))
    frame = max(0, min(int(frame), info.count - 1))

    base = _mask_at(work, frame)
    if base is None:
        base = fallback
        shape = (info.height, info.width)
        if base.shape[:2] != shape:
            base = cv2.resize(
                base.astype(np.uint8) * 255,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 127
    # Fill the automatic holes *before* the strokes land, so a user "remove"
    # that closes a gap is never quietly re-filled underneath them.
    base = _fill_holes(base)
    add = _decode_stroke(request.correction_add, "correction_add", base.shape[:2])
    remove = _decode_stroke(
        request.correction_remove, "correction_remove", base.shape[:2]
    )
    corrected = _apply_correction(base, add, remove)
    if not corrected.any():
        raise PipelineError(
            "That correction erased the whole subject, so there is nothing left "
            "to track — undo a few strokes and try again.",
            ERR_UNSUPPORTED,
        )
    _emit(on_progress, "analyze", 1, 1, "using your correction")
    box = meta.get("box")
    selection = auto_subject.SubjectSelection(
        mask=corrected,
        confidence=float(meta.get("confidence", 0.0)),
        box=tuple(box) if isinstance(box, (list, tuple)) and len(box) == 4 else None,
        reasons=["your brush correction"],
    )
    return selection, frame


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
# Matting
# --------------------------------------------------------------------- #
def _matte(
    frames_dir: Path,
    masks: dict[int, np.ndarray],
    key_idx: int,
    count: int,
    device: str,
    on_progress,
    cancel_event: threading.Event | None,
) -> dict[int, np.ndarray]:
    """Refine the tracked outline into a soft alpha with MatAnyone 2.

    The propagated mask is the prompt on the key frame and the clamp on every
    other, so the learned matte can only soften the edge the animator already
    agreed with — it cannot wander into background.
    """
    from src.processing.matting import MattingRunner, MattingSettings

    runner = MattingRunner("matanyone2", device=device, cancel_event=cancel_event)
    alphas: dict[int, np.ndarray] = {}
    seen = 0

    def _frame_getter(index: int) -> np.ndarray:
        return _read_rgb(frames_dir / f"{index:05d}.jpg")

    def _on_alpha(index: int, alpha: np.ndarray) -> None:
        nonlocal seen
        alphas[index] = alpha
        seen += 1
        if seen % _PROGRESS_EVERY == 0 or seen == count:
            _emit(on_progress, "matte", seen, count, "refining edges")

    _emit(on_progress, "matte", 0, count, "refining edges")
    try:
        runner.matte(
            list(range(count)),
            _frame_getter,
            {key_idx: masks[key_idx]},
            # The panel's mask is the contract: the matte may soften its edge,
            # but it must never hollow out its interior, and in a flat colour
            # region it must not leave a ghost of the background either. Both
            # of those are the anime-specific half of "better matting".
            MattingSettings(clamp_solid_interior=True, flat_snap=True),
            on_alpha=_on_alpha,
            mask_getter=lambda i: masks.get(i),
        )
    except Exception as exc:
        # The runner signals cancellation with a RuntimeError of its own; the
        # job must see a real JobCancelled or the record would claim success.
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled() from exc
        raise
    finally:
        # Two networks on one card is the one thing this GPU cannot afford, so
        # the matting model is released the moment its work is done.
        runner.unload()
    _emit(on_progress, "matte", count, count, "refining edges")
    return alphas


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
def _alpha_for(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from src.processing.edge_finish import soft_coverage

    return soft_coverage(mask, feather=_EDGE_FEATHER)


def _rgba(
    frame_rgb: np.ndarray, mask: np.ndarray, alpha: np.ndarray | None = None
) -> np.ndarray:
    """Compose RGBA. *alpha* (a learned matte) wins over the segment outline."""
    from src.processing.edge_finish import decontaminate_edges, finish_edges

    if alpha is None:
        # No matte: derive coverage from the outline and clean the fringe the
        # ramp exposes, which is the whole edge treatment in one call.
        frame_rgb, alpha = finish_edges(frame_rgb, mask, feather=_EDGE_FEATHER)
    else:
        # The matte's own ramp also has partially covered pixels, and those
        # carry background colour at nearly full strength. Subtract it so a
        # hair tip keeps the hair's colour instead of a blend of hair and sky.
        frame_rgb = decontaminate_edges(frame_rgb, mask, alpha)
    # SAM 2 can hand back a mask below source resolution; the alpha must line up
    # with the frame whatever it is measured against.
    if alpha.shape[:2] != frame_rgb.shape[:2]:
        alpha = cv2.resize(
            alpha.astype(np.float32),
            (frame_rgb.shape[1], frame_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
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
    alphas: dict[int, np.ndarray] | None = None,
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
        **no_window(),
    )
    last_mask: np.ndarray | None = None
    failure: BaseException | None = None
    try:
        assert proc.stdin is not None
        for i, frame in _decoded_frames(frames_dir, count, cancel_event):
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
            rgba = _rgba(frame, mask, alphas.get(i) if alphas else None)
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
    alphas: dict[int, np.ndarray] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    last_mask: np.ndarray | None = None
    for i, frame in _decoded_frames(frames_dir, count, cancel_event):
        mask = masks.get(i, last_mask)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=bool)
        else:
            last_mask = mask
        rgba = _rgba(frame, mask, alphas.get(i) if alphas else None)
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


def _add_source_audio(
    movie: Path,
    source: Path,
    frame_in: int,
    count: int,
    fps: float,
) -> Path:
    """Hand the alpha movie back the audio of the clip it was cut from.

    The engine read the whole source, so the cut-out is the same footage minus
    the background — and a cut-out that quietly loses the sound is the wrong
    answer. Premiere is the host that notices: ``changeMediaPath`` refuses a file
    with no audio where the media it replaces had some, and deletes the audio
    references if it is forced through.

    The mux is a stream copy, and it gives up quietly rather than failing a
    render that is otherwise finished.
    """
    from src.export.exporter import has_audio_stream, mux_audio

    try:
        if not has_audio_stream(source):
            logger.info("source has no audio to keep: %s", source)
            return movie
        start = max(0.0, frame_in / fps) if fps > 0 else 0.0
        duration = count / fps if fps > 0 else None
        return mux_audio(movie, source, start=start, duration=duration)
    except Exception as exc:  # noqa: BLE001 - audio is never worth a failed job
        logger.warning("could not add the source audio (%s); keeping it silent", exc)
        return movie


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
        if request.resume_dir:
            selection, key_idx = _resume_seed(request, info, on_progress)
        else:
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

        # A solid seed gives a solid track: close the enclosed gaps in the
        # prompt itself before SAM 2 ever sees it, at source resolution.
        seed = selection.mask
        if seed is None:
            seed = np.zeros((info.height, info.width), dtype=bool)
        if seed.shape[:2] != (info.height, info.width):
            seed = cv2.resize(
                seed.astype(np.uint8) * 255,
                (info.width, info.height),
                interpolation=cv2.INTER_NEAREST,
            ) > 127
        seed = _fill_holes(seed)
        selection.mask = seed

        # Persist the seed before tracking: a later brush correction re-tracks
        # from exactly this frame, and the panel only needs the directory back.
        _save_seed(
            work,
            seed,
            {
                "key_idx": int(key_idx),
                "confidence": float(selection.confidence),
                "box": list(selection.box) if selection.box else None,
                "count": int(info.count),
                "frame_in": int(info.frame_in),
                "source": str(source),
            },
        )

        phase = "tracking the subject"
        started = time.time()
        masks = _propagate(
            manager,
            frames_dir,
            key_idx,
            seed,
            info.count,
            on_progress,
            cancel_event,
        )
        manager.reset_video_state()
        # Close the interior gaps SAM leaves behind, then make the frame the
        # user painted authoritative: a disconnected patch that SAM does not
        # believe belongs to the subject must still survive there.
        _close_holes(masks, info.count, on_progress, cancel_event)
        masks[key_idx] = seed
        _save_masks(work, masks, info.count, (info.height, info.width))
        logger.info(
            "phase tracking: %.1fs (%d/%d frames masked)",
            time.time() - started,
            len(masks),
            info.count,
        )

        alphas: dict[int, np.ndarray] | None = None
        if request.matting and key_idx in masks:
            phase = "refining the edges"
            started = time.time()
            try:
                alphas = _matte(
                    frames_dir,
                    masks,
                    key_idx,
                    info.count,
                    str(getattr(recommendation, "device", "cuda")),
                    on_progress,
                    cancel_event,
                )
                logger.info(
                    "phase matte: %.1fs (%d/%d frames matted)",
                    time.time() - started,
                    len(alphas),
                    info.count,
                )
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - matting must never sink a job
                # A missing download or an out-of-memory card should still ship
                # the outline matte rather than fail the whole removal.
                logger.warning("matting failed (%s); using the segment outline", exc)
                _emit(on_progress, "matte", info.count, info.count, "matting unavailable")
                alphas = None

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
                alphas=alphas,
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
                alphas=alphas,
            )
            output_path = png_dir
            output_kind = "png_sequence"

        if request.keep_audio and output_kind == "mov_alpha":
            output_path = _add_source_audio(
                output_path, source, info.frame_in, info.count, fps
            )

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
            work_dir=str(work),
            matting=alphas is not None,
            key_idx=int(key_idx),
        )
        logger.info(
            "job finished: %s frames in %.1fs (%s, %s, matting=%s)",
            info.count,
            time.time() - job_started,
            recommendation.model_size.value,
            output_kind,
            alphas is not None,
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
        # Frames are kept so the panel's brush can render any frame and a
        # correction can skip re-extraction — both are what make brushing feel
        # instant rather than a re-run. On a very long clip that disk cost stops
        # being worth it, so those fall back to decoding the source on demand.
        try:
            frame_files = list(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
        except OSError:
            frame_files = []
        if len(frame_files) > _KEEP_FRAMES_LIMIT:
            shutil.rmtree(frames_dir, ignore_errors=True)


__all__ = ["JobProgress", "PipelineError", "default_output_root", "render_frame", "run_job"]
