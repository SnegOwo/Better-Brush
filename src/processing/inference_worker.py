"""QThread workers for SAM 2 inference and mask propagation."""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

from src.hardware.profiler import SAM2Recommendation
from src.models.sam2_manager import PointPrompt, SAM2Manager
from src.processing.mask_selection import trim_prompt_background

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Run SAM 2 image inference off the GUI thread.

    A shared :class:`SAM2Manager` can be injected so the model weights are
    loaded once and reused across calls; otherwise a temporary manager is
    created (and released) per task.
    """

    finished = Signal(object)  # list[MaskResult] or Exception
    progress = Signal(str)

    def __init__(
        self,
        recommendation: SAM2Recommendation,
        sam_manager: SAM2Manager | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.rec = recommendation
        self._sam: SAM2Manager | None = sam_manager
        self._owns_manager = sam_manager is None
        self._frame: np.ndarray | None = None
        self._points: list[PointPrompt] = []
        self._mask: np.ndarray | None = None
        self._mask_points: list = []

    def set_task(self, frame: np.ndarray, points: list[PointPrompt]) -> None:
        self._frame = frame
        self._points = list(points)
        self._mask = None

    def set_task_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        points: list | None = None,
    ) -> None:
        """Run a refinement pass using a hand-drawn mask as the prompt.

        *points* are the user's points on this frame: they ride along so the
        refinement cannot be confined to a stale mask extent.
        """
        self._frame = frame
        self._points = []
        self._mask = mask
        self._mask_points = list(points or [])

    def run(self) -> None:
        try:
            if self._sam is None:
                self.progress.emit("Loading model (first run)...")
                self._sam = SAM2Manager(self.rec)
            if self._frame is None:
                raise ValueError("No frame set")
            if self._mask is not None:
                self.progress.emit("Refining brush mask with SAM 2...")
                results = self._sam.predict_frame_from_mask(
                    self._frame, self._mask, extra_points=self._mask_points
                )
                self.finished.emit(results)
                return
            if not self._points:
                raise ValueError("No prompt points set")
            self.progress.emit("Running inference...")
            results = self._sam.predict_frame(
                self._frame,
                self._points,
                multimask_output=True,
            )
            self.finished.emit(results)
        except torch.cuda.OutOfMemoryError:
            logger.error("Inference ran out of GPU memory")
            if self._sam is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001, S110 - cache clear is best effort
                    pass
            self.finished.emit(
                RuntimeError(
                    "Out of GPU memory while segmenting. Close other GPU apps, "
                    "or pick a smaller model in AI Model (e.g. SAM 2.1 Tiny)."
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via finished().emit(exc)
            logger.error("Inference failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)

    def unload(self) -> None:
        if self._sam is not None and self._owns_manager:
            self._sam.unload()
            self._sam = None


class PropagationWorker(QThread):
    """Run SAM 2 video propagation off the GUI thread.

    Uses a shared :class:`SAM2Manager` (passed by the caller) so prompts added
    on the main thread and the loaded model weights are reused — the worker
    never creates a competing model instance.
    """

    finished = Signal(object)  # list[tuple] or Exception
    progress = Signal(str, int, int)  # stage, current, total
    frame_ready = Signal(int, int, object)  # frame_idx, obj_id, mask

    def __init__(
        self,
        recommendation: SAM2Recommendation,
        sam_manager: SAM2Manager,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.rec = recommendation
        self._sam = sam_manager
        self._video_path: str | None = None
        self._start_frame = 0
        self._points: list[PointPrompt] = []
        self._mask: np.ndarray | None = None
        self._total_frames = 0
        self._max_frames: int | None = None
        self._reverse = False
        self._segments: list[tuple[int, int]] = []
        self._cancel = threading.Event()

    def set_task(
        self,
        video_path: str,
        start_frame: int = 0,
        points: list[PointPrompt] | None = None,
        total_frames: int = 0,
        mask: np.ndarray | None = None,
    ) -> None:
        self._video_path = video_path
        self._start_frame = start_frame
        self._points = list(points or [])
        self._total_frames = total_frames
        self._mask = mask

    def set_range(self, max_frames: int | None = None, reverse: bool = False) -> None:
        """Restrict propagation to a segment (for partial corrections).

        Args:
            max_frames: Number of frames to track (None = until the end).
            reverse: Track backwards from the start frame.
        """
        self._max_frames = max_frames
        self._reverse = bool(reverse)

    def set_segments(self, segments: Sequence[tuple[int, int]]) -> None:
        """Track each ``(start, end)`` shot as its own predictor run.

        A detected scene cut must never be crossed inside a single tracking
        run, so every shot is propagated separately and each one after the
        first is re-seeded from the last mask of the previous shot.  Chaining
        the segments is what keeps the shots after the first from being left
        unmasked - without it the exporter, which walks the frames that own a
        mask, only ever saw the opening scene.

        Args:
            segments: Contiguous, non-overlapping ``(first, last)`` frame
                pairs in propagation order.  Empty keeps the single unbounded
                run driven by :meth:`set_range`.
        """
        self._segments = [(int(first), int(last)) for first, last in segments]

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        try:
            if self._video_path is None:
                raise ValueError("No video path set")

            def stage(msg: str, cur: int = 0, total: int = 0) -> None:
                self.progress.emit(msg, cur, total)

            stage("Preparing frames...")
            self._sam.init_video(
                self._video_path,
                progress_cb=lambda cur, total: stage("Preparing frames", cur, total),
            )

            if self._segments:
                plan = [
                    (first, last, abs(last - first) + 1)
                    for first, last in self._segments
                ]
            else:
                plan = [(self._start_frame, self._start_frame, self._max_frames)]

            if self._mask is not None:
                stage("Applying brush mask...")
                self._sam.add_mask_to_video(
                    frame_idx=self._start_frame,
                    obj_id=1,
                    mask=self._mask,
                )
                # A painted mask and points are not exclusive: the mask is
                # the user's latest correction, the points refine it. Using
                # one and dropping the other is how brush work used to
                # disappear the moment a point existed.
                if self._points:
                    stage("Applying prompts...")
                    self._sam.add_points_to_video(
                        frame_idx=self._start_frame,
                        obj_id=1,
                        points=self._points,
                        clear_old_points=False,
                    )
            elif self._points:
                stage("Applying prompts...")
                self._sam.add_points_to_video(
                    frame_idx=self._start_frame,
                    obj_id=1,
                    points=self._points,
                )

            results: list[tuple[int, int, np.ndarray, Any]] = []
            carry: np.ndarray | None = None

            for index, (first, last, count) in enumerate(plan):
                if self._cancel.is_set():
                    logger.info("Propagation cancelled before shot %d", index + 1)
                    break
                if index and carry is not None and carry.any():
                    # Anchor the new shot on the previous shot's last mask so
                    # the tracker is re-grounded instead of drifting across the
                    # camera change.
                    stage(f"Re-seeding shot at frame {first}...")
                    self._sam.add_mask_to_video(
                        frame_idx=first,
                        obj_id=1,
                        mask=carry,
                    )
                stage("Propagating", 0, max(self._total_frames, 1))
                chunk = self._sam.propagate_in_video(
                    # Backward runs leave the segment from its far end, so the
                    # anchor is the last frame rather than the first.
                    start_frame_idx=last if self._reverse else first,
                    max_frames=count,
                    reverse=self._reverse,
                    on_frame=self._on_frame,
                    cancel_event=self._cancel,
                )
                if not chunk:
                    logger.warning(
                        "Shot starting at frame %d produced no mask - "
                        "stopping after %d shot(s)",
                        first,
                        index,
                    )
                    break
                results.extend(chunk)
                # The mask to carry forward is the one on the segment edge we
                # are about to cross; the two backends return frames in
                # different orders, so pick it by index rather than position.
                edge = min if self._reverse else max
                carry = edge(chunk, key=lambda entry: entry[0])[2]
            self.finished.emit(results)
        except torch.cuda.OutOfMemoryError:
            logger.error("Propagation ran out of GPU memory")
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001, S110 - cache clear is best effort
                pass
            self.finished.emit(
                RuntimeError(
                    "Out of GPU memory while propagating. Close other GPU apps, "
                    "or pick a smaller model in AI Model (e.g. SAM 2.1 Tiny)."
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via finished().emit(exc)
            logger.error("Propagation failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)

    def _on_frame(self, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        self.frame_ready.emit(frame_idx, obj_id, mask)
        self.progress.emit("Propagating", frame_idx + 1, max(self._total_frames, 1))

    def unload(self) -> None:
        # Shared manager: only reset the per-video state, never free the
        # weights owned by MainWindow.
        if self._sam is not None:
            self._sam.reset_video_state()


# ------------------------------------------------------------------ #
# Scene-cut detection worker                                           #
# ------------------------------------------------------------------ #
class SceneCutWorker(QThread):
    """Detect scene cuts in the loaded video (off the GUI thread)."""

    finished = Signal(object)  # list[int] or Exception
    progress = Signal(str, int, int)

    def __init__(
        self,
        video_path: str,
        threshold: float = 27.0,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._threshold = threshold
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from src.processing.scene_detector import detect_scene_cuts

            self.progress.emit("Detecting scene cuts…", 0, 0)
            cuts = detect_scene_cuts(
                self._video_path,
                threshold=self._threshold,
                cancel_event=self._cancel,
            )
            self.finished.emit(cuts)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Scene cut detection failed: %s\n%s", exc, traceback.format_exc()
            )
            self.finished.emit(exc)


# ------------------------------------------------------------------ #
# Face-parts isolation worker                                          #
# ------------------------------------------------------------------ #
class FacePartsWorker(QThread):
    """Detect a face and build prompt points for a specific part."""

    finished = Signal(object)  # list[tuple[int,int,int]] or Exception

    def __init__(
        self, frame_rgb: np.ndarray, part: str, parent: Any | None = None
    ) -> None:
        super().__init__(parent)
        self._frame = frame_rgb
        self._part = part

    def run(self) -> None:
        try:
            from src.processing.face_parts import detect_landmarks, face_part_points

            landmarks = detect_landmarks(self._frame)
            if landmarks is None:
                self.finished.emit([])
                return
            points = face_part_points(landmarks, self._part, self._frame.shape[:2])
            self.finished.emit(points)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Face part detection failed: %s\n%s", exc, traceback.format_exc()
            )
            self.finished.emit(exc)


# ------------------------------------------------------------------ #
# Post-processing worker (edge refinement + temporal smoothing)        #
# ------------------------------------------------------------------ #
#: Canonical shape of a per-frame profile setting:
#: ``(edge_smoothness, temporal_smoothing, detail_preservation,
#:   temporal_window, mask_inset)``.
_ProfileSettings = tuple[float, bool, float, int, int]

#: Accepted for backwards compatibility with resolvers written before
#: ``mask_inset`` was threaded through. The inset falls back to 0, which is the
#: same as "no inset" rather than an error.
_LEGACY_PROFILE_ARITY = 4


def _normalize_profile_resolver(
    resolver: Callable[[int], Any],
) -> Callable[[int], _ProfileSettings]:
    """Coerce a per-frame profile resolver to the canonical 5-tuple.

    The resolver contract is positional, so an arity mistake is invisible until
    the result is destructured. Validating it at construction time means a bad
    resolver fails on the first frame - reported through the normal ``finished``
    path - rather than after the whole clip has been smoothed.

    Args:
        resolver: Callable mapping a frame index to per-frame settings.

    Returns:
        A wrapper that always yields a 5-tuple.

    Raises:
        ValueError: Only when the resolver is actually called, so constructing
            a worker with an idle resolver stays free of side effects.
    """

    def normalized(index: int) -> _ProfileSettings:
        settings = resolver(index)
        if not isinstance(settings, tuple):
            raise TypeError(
                "profile_resolver must return a tuple of per-frame settings, "
                f"got {type(settings).__name__}"
            )
        if len(settings) == _LEGACY_PROFILE_ARITY:
            # Legacy shape had no mask_inset; append the neutral value.
            return (*settings, 0)
        if len(settings) != 5:
            raise ValueError(
                "profile_resolver must return 5 values "
                "(edge_smoothness, temporal_smoothing, detail_preservation, "
                f"temporal_window, mask_inset), got {len(settings)}"
            )
        return settings  # type: ignore[return-value]

    return normalized


class PostProcessWorker(QThread):
    """Run guided-filter refinement and/or temporal smoothing."""

    finished = Signal(object)  # dict[int, np.ndarray] or Exception
    progress = Signal(str, int, int)

    def __init__(
        self,
        frame_loader: Callable[[int], np.ndarray | None],
        masks: dict[int, np.ndarray],
        refine_edges: bool = False,
        temporal_smooth: bool = False,
        smooth_strength: float = 5.0,
        detail_preservation: float = 0.75,
        mask_inset: int = 0,
        temporal_window: int = 1,
        profile_resolver: Callable[[int], Any] | None = None,
        prompt_points: Sequence[PointPrompt] | None = None,
        cel_hold: bool = False,
        line_lock: bool = False,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame_loader = frame_loader
        self._masks = dict(masks)
        self._refine_edges = refine_edges
        self._temporal_smooth = temporal_smooth
        self._smooth_strength = smooth_strength
        self._detail_preservation = detail_preservation
        self._mask_inset = mask_inset
        self._temporal_window = temporal_window
        self._prompt_points = list(prompt_points or [])
        self._cel_hold = bool(cel_hold)
        self._line_lock = bool(line_lock)
        self._cancel = threading.Event()
        # Normalise the resolver once, here, instead of trusting a positional
        # tuple contract at the point of use.  A wrong arity used to be
        # discovered only when groups.items() was unpacked - after every frame
        # had been smoothed, deep inside a worker thread, where the ValueError
        # was merely logged and the user just saw "post-processing failed".
        self._profile_resolver = (
            None
            if profile_resolver is None
            else _normalize_profile_resolver(profile_resolver)
        )

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        try:
            result = dict(self._masks)

            frames: dict[int, np.ndarray] = {}
            for idx in sorted(result):
                f = self._frame_loader(idx)
                if f is not None:
                    frames[idx] = f
            if self._refine_edges or self._temporal_smooth:
                from src.processing.mask_refine import fill_mask_holes

                # Hole filling runs before any edge work: the guided filter
                # and the temporal smoother both treat an interior gap as
                # real background and would keep it stable.
                idxs = sorted(result)
                for idx in idxs:
                    result[idx] = fill_mask_holes(result[idx])

            if self._cel_hold:
                from src.processing.cel_hold import apply_cel_hold

                # Held cels reuse the previous mask verbatim: on an anime hold
                # the drawing has not changed, so any difference is boiling.
                self.progress.emit("Reusing held cels", 0, 0)
                result = apply_cel_hold(result, frames).masks

            if self._temporal_smooth:
                from src.processing.temporal_smooth import temporal_smooth_masks

                if self._profile_resolver is None:
                    result = temporal_smooth_masks(
                        frames,
                        result,
                        window=max(1, int(self._temporal_window)),
                        on_progress=lambda stage, cur, tot: self.progress.emit(
                            stage, cur, tot
                        ),
                        cancel_event=self._cancel,
                    )
                else:
                    groups: dict[tuple[float, bool, float, int, int], list[int]] = {}
                    for idx in sorted(result):
                        settings = self._profile_resolver(idx)
                        groups.setdefault(settings, []).append(idx)
                    for (
                        _edge,
                        temporal,
                        _detail,
                        window,
                        _inset,
                    ), indices in groups.items():
                        if not temporal:
                            continue
                        subset_masks = {idx: result[idx] for idx in indices}
                        subset_frames = {
                            idx: frames[idx] for idx in indices if idx in frames
                        }
                        smoothed = temporal_smooth_masks(
                            subset_frames,
                            subset_masks,
                            window=max(1, int(window)),
                            on_progress=lambda stage, cur, tot: self.progress.emit(
                                stage, cur, tot
                            ),
                            cancel_event=self._cancel,
                        )
                        result.update(smoothed)

            if self._refine_edges:
                from src.processing.mask_refine import refine_mask

                idxs = sorted(result)
                total = len(idxs)
                for k, idx in enumerate(idxs):
                    if self._cancel.is_set():
                        break
                    self.progress.emit("Refining mask edges", k + 1, total)
                    frame = self._frame_loader(idx)
                    if frame is not None:
                        if self._profile_resolver is None:
                            strength = self._smooth_strength
                            detail = self._detail_preservation
                            inset = self._mask_inset
                        else:
                            strength, _temporal, detail, _window, inset = (
                                self._profile_resolver(idx)
                            )
                        result[idx] = refine_mask(
                            frame,
                            result[idx],
                            smooth_strength=strength,
                            detail_preservation=detail,
                            mask_inset=inset,
                        )

            if self._line_lock:
                from src.processing.line_prior import snap_to_anime_lines

                # After refinement the edge is smooth but may float off the
                # pencil line; snap it back onto the line art.
                idxs = sorted(result)
                total = len(idxs)
                for k, idx in enumerate(idxs):
                    if self._cancel.is_set():
                        break
                    self.progress.emit("Locking edges to line art", k + 1, total)
                    frame = frames.get(idx)
                    if frame is None:
                        frame = self._frame_loader(idx)
                    if frame is not None:
                        result[idx] = snap_to_anime_lines(result[idx], frame)

            if any(point.label == 0 for point in self._prompt_points):
                idxs = sorted(result)
                total = len(idxs)
                for k, idx in enumerate(idxs):
                    if self._cancel.is_set():
                        break
                    self.progress.emit("Protecting excluded background", k + 1, total)
                    frame = self._frame_loader(idx)
                    if frame is not None:
                        result[idx] = trim_prompt_background(
                            frame,
                            result[idx],
                            self._prompt_points,
                        )

            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Post-processing failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)


class CleanUpWorker(QThread):
    """Background worker for subtitle/logo cleanup detection.

    ``target`` selects which clean-up region to look for:
    ``"subtitles"``, ``"logo"`` or ``"both"``.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)

    def __init__(
        self,
        video_path: str,
        target: str = "both",
        parent: Any | None = None,
        subtitle_rect: tuple[int, int, int, int] | None = None,
        logo_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._target = target
        self._subtitle_rect = subtitle_rect
        self._logo_rect = logo_rect
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from src.processing.plates import detect_cleanup_regions

        try:
            result = detect_cleanup_regions(
                self._video_path,
                targets=self._target,
                progress_cb=lambda msg, cur, total: self.progress.emit(msg, cur, total),
                subtitle_rect=self._subtitle_rect,
                logo_rect=self._logo_rect,
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Cleanup detection failed")
            self.finished.emit(exc)


# ------------------------------------------------------------------ #
# Cleanup bake worker ("Mask to End")                                 #
# ------------------------------------------------------------------ #
class CleanBakeWorker(QThread):
    """Pre-inpaint every frame of the video into a :class:`CleanFrameStore`.

    The bake runs in a background thread with its *own* ``VideoCapture`` so
    the GUI stays responsive; repaired frames are written into ``store`` as
    they complete, which is what makes timeline scrubbing smooth afterwards
    (frames are already cleaned instead of being repaired live per frame).
    Cancellation is cooperative between frames.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)  # None | "cancelled" | Exception

    def __init__(
        self,
        video_path: str,
        plates: list,
        seed: int = 0,
        store=None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._plates = list(plates)
        self._seed = int(seed)
        self._store = store
        self._cancel = threading.Event()
        self.cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        try:
            from src.processing.plates import clean_video_frames

            clean_video_frames(
                self._video_path,
                self._plates,
                seed=self._seed,
                on_frame=self._store_frame,
                progress_cb=lambda msg, cur, tot: self.progress.emit(msg, cur, tot),
                cancel_event=self._cancel,
            )
            if self._cancel.is_set():
                self.cancelled = True
                self.finished.emit("cancelled")
                return
            self.finished.emit(None)
        except Exception as exc:  # noqa: BLE001 - surfaced via finished().emit(exc)
            logger.error("Cleanup bake failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)

    def _store_frame(self, idx: int, frame) -> None:
        if self._store is not None:
            self._store.set(idx, frame)


class RemovalWorker(QThread):
    """Run MiniMax-Remover object removal off the GUI thread.

    The worker opens its *own* VideoCapture (the shared frame buffer is not
    thread-safe) and pulls frames on demand, so clips of any length are
    processed with bounded memory. Every cleaned frame is written into the
    frame store as soon as it is ready, which makes the preview and the
    export agree on the result. Cancellation is cooperative between frames.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)  # int | "cancelled" | Exception

    def __init__(
        self,
        video_path: str,
        indices: list[int],
        mask_getter: Callable[[int], np.ndarray],
        settings: object,
        model_dir: str,
        device: str = "cuda",
        cleanup_plates: list | None = None,
        cleanup_seed: int = 0,
        store=None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = str(video_path)
        self._indices = list(indices)
        self._mask_getter = mask_getter
        self._settings = settings
        self._model_dir = model_dir
        self._device = device
        self._cleanup_plates = list(cleanup_plates or [])
        self._cleanup_seed = int(cleanup_seed)
        self._store = store
        self._cancel = threading.Event()
        self.cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        reader = None
        try:
            from src.processing.removal import MinimaxRemover
            from src.processing.video_reader import VideoReader

            reader = VideoReader(Path(self._video_path))

            def frame_getter(idx: int) -> np.ndarray:
                frame = reader.read_frame(idx)
                if frame is None:
                    raise RuntimeError(f"Frame {idx} could not be read")
                if self._cleanup_plates:
                    from src.processing.plates import apply_clean

                    frame = apply_clean(frame, self._cleanup_plates, seed=self._cleanup_seed)
                return frame

            remover = MinimaxRemover(
                model_dir=Path(self._model_dir),
                device=self._device,
                cancel_event=self._cancel,
            )
            total = len(self._indices)
            done = 0
            seen: set[int] = set()

            def on_frame(_idx: int, frame) -> None:
                nonlocal done
                if self._store is not None:
                    self._store.set(_idx, frame)
                if _idx in seen:
                    # Overlap frames are delivered twice (the second time as
                    # the cross-faded blend); store it, but count it once.
                    return
                seen.add(_idx)
                done += 1
                self.progress.emit("Removing object", done, total)

            def on_status(message: str) -> None:
                self.progress.emit(message, done, total)

            remover.remove_frames(
                self._indices,
                frame_getter,
                self._mask_getter,
                self._settings,
                on_frame=on_frame,
                on_status=on_status,
            )
            if self._cancel.is_set():
                self.cancelled = True
                self.finished.emit("cancelled")
                return
            self.finished.emit(done)
        except torch.cuda.OutOfMemoryError:
            self.finished.emit(
                RuntimeError(
                    "Out of GPU memory. Close other GPU-heavy apps, or pick a "
                    "lower Resolution (352 or 480) in Object Removal."
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via finished().emit(exc)
            logger.error("Object removal failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)
        finally:
            if reader is not None:
                reader.close()


class MattingWorker(QThread):
    """Run learned video matting (MatAnyone / MatAnyone 2) off the GUI thread.

    The worker opens its *own* VideoCapture (the shared frame buffer is not
    thread-safe), so clips of any length are processed with bounded memory.
    Each completed soft matte is emitted and stored by the GUI into the
    object's ``mattes`` map. Cancellation is cooperative between frames.
    """

    progress = Signal(str, int, int)
    alpha_ready = Signal(int, object)  # frame index, float32 alpha [0, 1]
    finished = Signal(object)  # int | "cancelled" | Exception

    def __init__(
        self,
        video_path: str,
        indices: list[int],
        prompt_masks: dict[int, np.ndarray],
        settings: object,
        device: str = "cuda",
        model: str = "matanyone2",
        mask_getter: Callable[[int], np.ndarray] | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = str(video_path)
        self._mask_getter = mask_getter
        self._indices = list(indices)
        self._prompt_masks = dict(prompt_masks)
        self._settings = settings
        self._device = device
        self._model = model
        self._cancel = threading.Event()
        self.cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel.set()

    def run(self) -> None:
        reader = None
        try:
            from src.processing.matting import MattingRunner
            from src.processing.video_reader import VideoReader

            reader = VideoReader(Path(self._video_path))

            def frame_getter(idx: int) -> np.ndarray:
                frame = reader.read_frame(idx)
                if frame is None:
                    raise RuntimeError(f"Frame {idx} could not be read")
                return frame

            runner = MattingRunner(
                model=self._model, device=self._device, cancel_event=self._cancel
            )
            mask_getter = self._mask_getter
            total = len(self._indices)
            done = 0

            def on_alpha(index: int, alpha) -> None:
                nonlocal done
                self.alpha_ready.emit(index, alpha)
                done += 1
                self.progress.emit("Matting", done, total)

            runner.matte(
                self._indices,
                frame_getter,
                self._prompt_masks,
                self._settings,
                on_alpha=on_alpha,
                mask_getter=mask_getter,
            )
            runner.unload()
            if self._cancel.is_set():
                self.cancelled = True
                self.finished.emit("cancelled")
                return
            self.finished.emit(done)
        except Exception as exc:  # noqa: BLE001 - surfaced via finished().emit(exc)
            logger.error("Matting failed: %s\n%s", exc, traceback.format_exc())
            self.finished.emit(exc)
        finally:
            if reader is not None:
                reader.close()
