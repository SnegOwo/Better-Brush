"""Scene-cut detection for automatic segment boundary detection."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import cv2

logger = logging.getLogger(__name__)


def detect_scene_cuts(
    video_path: str | bytes,
    threshold: float = 27.0,
    downscale_width: int = 32,
    min_cut_distance: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[int]:
    """Return sorted list of frame indices where scene cuts occur.

    Tries scenedetect if available, falls back to a native histogram detector.
    """
    try:
        cuts = _detect_scenedetect(str(video_path), threshold)
    except Exception:
        logger.debug("scenedetect unavailable, using native fallback", exc_info=True)
        cuts = _detect_native(
            str(video_path), threshold, downscale_width, on_progress, cancel_event
        )

    if min_cut_distance > 0 and len(cuts) > 1:
        filtered = [cuts[0]]
        for c in cuts[1:]:
            if c - filtered[-1] >= min_cut_distance:
                filtered.append(c)
        cuts = filtered

    return sorted(set(cuts))


def _detect_scenedetect(video_path: str, threshold: float = 27.0) -> list[int]:
    """Detect cuts using scenedetect's ContentDetector."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    total = video.duration.frame_num
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()

    cuts = []
    for _start_tc, end_tc in scenes:
        f = end_tc.frame_num
        if 0 <= f < total:
            cuts.append(f)

    return cuts


def _detect_native(
    video_path: str,
    threshold: float = 27.0,
    downscale_width: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[int]:
    """Fallback detector using HSV histogram comparison."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    prev_hist = None
    cuts: list[int] = []
    idx = 0

    norm_threshold = min(threshold / 100.0, 1.0)

    while True:
        if cancel_event is not None and cancel_event.is_set():
            break

        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        scale = downscale_width / w
        small = cv2.resize(
            frame, (downscale_width, int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hist = hist.flatten()

        if prev_hist is not None:
            score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if score > norm_threshold:
                cuts.append(idx)

        prev_hist = hist
        idx += 1

        if on_progress is not None:
            on_progress(idx, total_frames)

    cap.release()
    return cuts
