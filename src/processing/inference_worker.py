"""QThread workers for SAM 2 inference and mask propagation."""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.hardware.profiler import SAM2Recommendation
from src.models.sam2_manager import MaskResult, PointPrompt, SAM2Manager

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
        sam_manager: Optional[SAM2Manager] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.rec = recommendation
        self._sam: Optional[SAM2Manager] = sam_manager
        self._owns_manager = sam_manager is None
        self._frame: Optional[np.ndarray] = None
        self._points: list[PointPrompt] = []
        self._mask: Optional[np.ndarray] = None

    def set_task(self, frame: np.ndarray, points: list[PointPrompt]) -> None:
        self._frame = frame
        self._points = list(points)
        self._mask = None

    def set_task_mask(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Run a refinement pass using a hand-drawn mask as the prompt."""
        self._frame = frame
        self._points = []
        self._mask = mask

    def run(self) -> None:
        try:
            if self._sam is None:
                self.progress.emit("Loading model (first run)...")
                self._sam = SAM2Manager(self.rec)
            if self._frame is None:
                raise ValueError("No frame set")
            if self._mask is not None:
                self.progress.emit("Refining brush mask with SAM 2...")
                results = self._sam.predict_frame_from_mask(self._frame, self._mask)
                self.finished.emit(results)
                return
            if not self._points:
                raise ValueError("No prompt points set")
            self.progress.emit("Running inference...")
            results = self._sam.predict_frame(self._frame, self._points)
            self.finished.emit(results)
        except Exception as exc:
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
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.rec = recommendation
        self._sam = sam_manager
        self._video_path: Optional[str] = None
        self._start_frame = 0
        self._points: list[PointPrompt] = []
        self._mask: Optional[np.ndarray] = None
        self._total_frames = 0
        self._max_frames: Optional[int] = None
        self._reverse = False
        self._cancel = threading.Event()

    def set_task(
        self,
        video_path: str,
        start_frame: int = 0,
        points: Optional[list[PointPrompt]] = None,
        total_frames: int = 0,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self._video_path = video_path
        self._start_frame = start_frame
        self._points = list(points or [])
        self._total_frames = total_frames
        self._mask = mask

    def set_range(self, max_frames: Optional[int] = None, reverse: bool = False) -> None:
        """Restrict propagation to a segment (for partial corrections).

        Args:
            max_frames: Number of frames to track (None = until the end).
            reverse: Track backwards from the start frame.
        """
        self._max_frames = max_frames
        self._reverse = bool(reverse)

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

            if self._mask is not None:
                stage("Applying brush mask...")
                self._sam.add_mask_to_video(
                    frame_idx=self._start_frame,
                    obj_id=1,
                    mask=self._mask,
                )
            elif self._points:
                stage("Applying prompts...")
                self._sam.add_points_to_video(
                    frame_idx=self._start_frame,
                    obj_id=1,
                    points=self._points,
                )

            stage("Propagating", 0, max(self._total_frames, 1))
            results = self._sam.propagate_in_video(
                start_frame_idx=self._start_frame,
                max_frames=self._max_frames,
                reverse=self._reverse,
                on_frame=self._on_frame,
                cancel_event=self._cancel,
            )
            self.finished.emit(results)
        except Exception as exc:
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
