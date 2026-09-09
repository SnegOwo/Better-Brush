"""SAM 2 model wrapper: lazy loading, inference, video propagation."""

from __future__ import annotations

import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from src.hardware.detector import Backend
from src.hardware.profiler import ModelSize, SAM2Recommendation
from src.models.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

# Graceful degradation if SAM 2 is not installed
_SAM2_AVAILABLE = False
_BUILD_SAM2 = None
_BUILD_SAM2_VIDEO_PREDICTOR = None
_SAM2_IMAGE_PREDICTOR_CLASS = None

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_AVAILABLE = True
    _BUILD_SAM2 = build_sam2
    _BUILD_SAM2_VIDEO_PREDICTOR = build_sam2_video_predictor
    _SAM2_IMAGE_PREDICTOR_CLASS = SAM2ImagePredictor
except Exception as exc:  # noqa: BLE001
    logger.warning("SAM 2 not available (%s). Running in stub mode.", exc)

# Hydra config names *inside the installed sam2 package* (sam2/configs/).
# build_sam2() resolves them via hydra compose() with the search path that
# sam2/__init__.py registers — a local file path would NOT work here.
_HYDRA_CONFIGS: dict[ModelSize, str] = {
    ModelSize.TINY: "configs/sam2.1/sam2.1_hiera_t.yaml",
    ModelSize.SMALL: "configs/sam2.1/sam2.1_hiera_s.yaml",
    ModelSize.BASE_PLUS: "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ModelSize.LARGE: "configs/sam2.1/sam2.1_hiera_l.yaml",
}


@dataclass(frozen=True)
class PointPrompt:
    """A single prompt point on a frame."""

    x: int
    y: int
    label: int  # 1 = positive, 0 = negative


@dataclass(frozen=True)
class MaskResult:
    """Output of a single mask prediction."""

    mask: np.ndarray  # bool HxW
    score: float
    logits: Optional[np.ndarray] = None


class SAM2Manager:
    """High-level wrapper around SAM 2 image and video predictors."""

    def __init__(self, recommendation: SAM2Recommendation) -> None:
        self.rec = recommendation
        self._ckpt_mgr = CheckpointManager()
        self._image_predictor: Optional[Any] = None
        self._video_predictor: Optional[Any] = None
        self._video_state: Optional[Any] = None
        self._current_video_path: Optional[Path] = None
        logger.info("SAM2Manager created for %s on %s", self.rec.model_size.value, self.rec.device)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        return _SAM2_AVAILABLE

    @property
    def device(self) -> str:
        return self.rec.device

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def load_image_predictor(self) -> None:
        if not self.is_available:
            raise RuntimeError("SAM 2 is not installed. See requirements.txt")
        if self._image_predictor is not None:
            return
        ckpt = self._ckpt_mgr.get_checkpoint_path(self.rec.model_size)
        cfg = _HYDRA_CONFIGS[self.rec.model_size]
        logger.info("Loading image predictor: %s | %s", cfg, ckpt.name)
        model = _BUILD_SAM2(cfg, str(ckpt), device=self.device)  # type: ignore[misc]
        self._image_predictor = _SAM2_IMAGE_PREDICTOR_CLASS(model)  # type: ignore[misc]
        logger.info("Image predictor loaded.")

    def load_video_predictor(self) -> None:
        if not self.is_available:
            raise RuntimeError("SAM 2 is not installed. See requirements.txt")
        if self._video_predictor is not None:
            return
        ckpt = self._ckpt_mgr.get_checkpoint_path(self.rec.model_size)
        cfg = _HYDRA_CONFIGS[self.rec.model_size]
        logger.info("Loading video predictor: %s | %s", cfg, ckpt.name)
        self._video_predictor = _BUILD_SAM2_VIDEO_PREDICTOR(  # type: ignore[misc]
            cfg, str(ckpt), device=self.device
        )
        logger.info("Video predictor loaded.")

    def unload(self) -> None:
        """Release model weights and clear CUDA cache."""
        self._image_predictor = None
        self._video_predictor = None
        self._video_state = None
        self._current_video_path = None
        gc.collect()
        if self.rec.backend == Backend.CUDA and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("SAM2Manager unloaded.")

    # ------------------------------------------------------------------ #
    # Image inference
    # ------------------------------------------------------------------ #
    def predict_frame(
        self,
        frame: np.ndarray,
        points: list[PointPrompt],
        multimask_output: bool = False,
    ) -> list[MaskResult]:
        """Run image predictor on a single frame.

        Args:
            frame: RGB uint8 array HxWx3.
            points: Prompt points (at least one is required).
            multimask_output: Return 3 candidate masks or just the best.
        """
        if not points:
            raise ValueError("predict_frame requires at least one prompt point")
        self.load_image_predictor()
        predictor = self._image_predictor
        predictor.set_image(frame)

        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        labels = np.array([p.label for p in points], dtype=np.int32)

        # SAM 2's own predict() runs under torch.no_grad() internally.
        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=multimask_output,
            )

        # masks shape: (3, H, W) if multimask else (1, H, W)
        results: list[MaskResult] = []
        for i in range(masks.shape[0]):
            results.append(
                MaskResult(
                    mask=masks[i].astype(bool),
                    score=float(scores[i]),
                    logits=logits[i] if logits is not None else None,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Video inference
    # ------------------------------------------------------------------ #
    _FRAMES_CACHE_DIRNAME = ".better_brush_frames"

    def init_video(
        self,
        video_path: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Any:
        """Initialise video state; must be called before video propagation.

        Args:
            video_path: Path to a video file (any OpenCV-readable format) or
                directly to a directory of JPEG frames named ``%05d.jpg``.
            progress_cb: Optional ``current, total`` callback while frames are
                extracted to JPEG (no-op when a JPEG directory is given).
        """
        self.load_video_predictor()
        src = Path(video_path)
        if src.is_dir():
            frames_dir = src
        else:
            frames_dir = self._ensure_frames_dir(src, progress_cb)
        logger.info("Init video state: %s", frames_dir)
        # NOTE: this SAM 2 version loads video via decord for .mp4 files
        # (unmaintained, no Python 3.13 wheels), so we always pass a JPEG
        # frames directory — the officially supported loader path.
        self._video_state = self._video_predictor.init_state(video_path=str(frames_dir))
        self._current_video_path = src
        return self._video_state

    @classmethod
    def _ensure_frames_dir(
        cls,
        video_path: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Extract all frames of *video_path* to a cached JPEG directory.

        SAM 2's video loader only supports MP4 (via decord, unavailable on
        modern Python) or JPEG folders. We reuse the OpenCV-based VideoReader
        so every format OpenCV can decode is supported.
        """
        from src.processing.video_reader import VideoReader

        cache_dir = (
            video_path.parent
            / cls._FRAMES_CACHE_DIRNAME
            / f"{video_path.stem}_{video_path.stat().st_size}"
        )
        marker = cache_dir / ".complete"
        if marker.exists():
            logger.info("Reusing cached JPEG frames: %s", cache_dir)
            return cache_dir

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting frames to JPEG: %s", cache_dir)
        reader = VideoReader(video_path)
        try:
            reader.export_frames_jpeg(cache_dir, on_progress=progress_cb)
        finally:
            reader.close()
        marker.write_text("ok", encoding="utf-8")
        return cache_dir

    def add_points_to_video(
        self,
        frame_idx: int,
        obj_id: int,
        points: list[PointPrompt],
        clear_old_points: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add positive/negative points to a specific frame in video mode.

        Returns:
            (masks, scores, logits) from SAM 2 for the prompted frame.
        """
        if self._video_state is None:
            raise RuntimeError("Video not initialised. Call init_video() first.")
        if not points:
            raise ValueError("add_points_to_video requires at least one point")
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        labels = np.array([p.label for p in points], dtype=np.int32)
        # SAM 2 main renamed add_new_points_or_boxes -> add_new_points_or_box
        fn = getattr(self._video_predictor, "add_new_points_or_box", None)
        if fn is None:
            fn = self._video_predictor.add_new_points_or_boxes  # type: ignore[attr-defined]
        return fn(
            inference_state=self._video_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=coords,
            labels=labels,
            clear_old_points=clear_old_points,
        )

    def propagate_in_video(
        self,
        start_frame_idx: int = 0,
        max_frames: Optional[int] = None,
        reverse: bool = False,
        on_frame: Optional[Callable[[int, int, np.ndarray], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        """Propagate mask forward (or backward) through video.

        Args:
            start_frame_idx: Frame holding the prompts to propagate from.
            max_frames: Limit the number of tracked frames (None = all).
            reverse: Track backwards from start_frame_idx.
            on_frame: Optional callback(frame_idx, obj_id, mask) per result,
                called as frames are produced (enables live UI updates).
            cancel_event: Optional threading.Event — checked between frames.

        Returns:
            List of (frame_idx, obj_id, mask, score).
        """
        if self._video_state is None:
            raise RuntimeError("Video not initialised.")
        logger.info("Propagating from frame %d (reverse=%s) ...", start_frame_idx, reverse)
        kwargs: dict[str, Any] = {"start_frame_idx": start_frame_idx}
        if max_frames is not None:
            kwargs["max_frame_num_to_track"] = max_frames
        if reverse:
            kwargs["reverse"] = True

        results: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for frame_idx, obj_ids, mask_logits in self._video_predictor.propagate_in_video(
            self._video_state, **kwargs
        ):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Propagation cancelled at frame %d", frame_idx)
                break
            # mask_logits shape: (N_obj, 1, H, W) at original video resolution
            masks = (mask_logits > 0.0).cpu().numpy()[:, 0]
            for i, obj_id in enumerate(obj_ids):
                # SAM 2 video predictor does not expose IoU scores; use 1.0
                entry = (int(frame_idx), int(obj_id), masks[i], np.array([1.0]))
                results.append(entry)
                if on_frame is not None:
                    on_frame(entry[0], entry[1], entry[2])
        logger.info("Propagation complete: %d frame entries", len(results))
        return results

    def reset_video_state(self) -> None:
        """Clear current video state (keeps model loaded)."""
        self._video_state = None
        self._current_video_path = None
        gc.collect()
