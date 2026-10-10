"""Frame cache / buffer for smooth random access during editing."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from src.processing.video_reader import VideoReader

logger = logging.getLogger(__name__)


class FrameBuffer:
    """LRU frame cache backed by VideoReader.

    Limits memory usage by capping either max frames or max megabytes.
    """

    def __init__(
        self,
        reader: VideoReader,
        max_frames: int = 64,
        max_mb: int = 512,
    ) -> None:
        self.reader = reader
        self.max_frames = max_frames
        self.max_bytes = max_mb * 1024 * 1024
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._current_size_bytes = 0
        logger.info("FrameBuffer: max_frames=%d max_mb=%d", max_frames, max_mb)

    # ------------------------------------------------------------------ #
    # Cache API
    # ------------------------------------------------------------------ #
    def get(self, idx: int) -> np.ndarray | None:
        """Return frame *idx* from cache or read it."""
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        frame = self.reader.read_frame(idx)
        if frame is not None:
            self._put(idx, frame)
        return frame

    def preload_range(self, start: int, end: int) -> None:
        """Sequentially load a window [start, end) into cache."""
        for idx in range(start, min(end, self.reader.total_frames)):
            if idx not in self._cache:
                frame = self.reader.read_frame(idx)
                if frame is not None:
                    self._put(idx, frame)

    def invalidate(self, idx: int | None = None) -> None:
        """Clear entire cache or drop a specific frame."""
        if idx is None:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.debug("FrameBuffer cleared.")
        else:
            frame = self._cache.pop(idx, None)
            if frame is not None:
                self._current_size_bytes -= frame.nbytes

    def close(self) -> None:
        self.invalidate()
        self.reader.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _put(self, idx: int, frame: np.ndarray) -> None:
        """Insert frame, evicting LRU if over capacity."""
        nbytes = frame.nbytes
        while (
            len(self._cache) >= self.max_frames
            or (self._current_size_bytes + nbytes) > self.max_bytes
        ) and self._cache:
            _, old = self._cache.popitem(last=False)
            self._current_size_bytes -= old.nbytes
        self._cache[idx] = frame
        self._current_size_bytes += nbytes


class CleanFrameStore:
    """LRU cache for pre-cleaned ("baked") frames.

    The ``Mask to End`` bake worker writes cleaned frames from a background
    thread while the GUI reads them during playback, so every mutation is
    guarded by a lock.  Unlike :class:`FrameBuffer` there is deliberately no
    fallback read: ``get()`` returns ``None`` for a frame that has been evicted
    so the caller can re-run the live cleanup as a degraded fallback.
    """

    def __init__(
        self,
        max_frames: int = 256,
        max_mb: int = 1536,
    ) -> None:
        self.max_frames = max_frames
        self.max_bytes = max_mb * 1024 * 1024
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._used_bytes = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def get(self, idx: int) -> np.ndarray | None:
        """Return the baked frame, or ``None`` when not cached."""
        with self._lock:
            frame = self._cache.get(idx)
            if frame is not None:
                self._cache.move_to_end(idx)
            return frame

    def set(self, idx: int, frame: np.ndarray) -> None:
        """Store a baked frame, evicting LRU entries over capacity."""
        with self._lock:
            old = self._cache.pop(idx, None)
            if old is not None:
                self._used_bytes -= old.nbytes
            nbytes = frame.nbytes
            while (
                len(self._cache) >= self.max_frames
                or (self._used_bytes + nbytes) > self.max_bytes
            ) and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._used_bytes -= evicted.nbytes
            self._cache[idx] = frame
            self._used_bytes += nbytes

    def contains(self, idx: int) -> bool:
        with self._lock:
            return idx in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._used_bytes = 0

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def memory_bytes(self) -> int:
        with self._lock:
            return self._used_bytes
