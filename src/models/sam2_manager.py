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
        self._enable_fast_cuda_paths()
        logger.info("SAM2Manager created for %s on %s", self.rec.model_size.value, self.rec.device)

    # ------------------------------------------------------------------ #
    # Speed helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _enable_fast_cuda_paths() -> None:
        """Turn on cuDNN autotuning / TF32 to speed up SAM 2 on tensor cores."""
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:  # pragma: no cover - non-CUDA build
            pass

    def _speed_context(self):
        """Return an autocast context for fast inference.

        Uses bf16 mixed precision on CUDA (Ampere+ tensor cores give a
        near-2x speed-up) and a no-op context everywhere else.
        """
        if self.rec.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

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
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        labels = np.array([p.label for p in points], dtype=np.int32)

        # SAM 2's own predict() runs under torch.no_grad() internally;
        # mixed precision (bf16 on CUDA) is OFF by default -> we enable it.
        with self._speed_context(), torch.inference_mode():
            predictor.set_image(frame)
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

    def predict_frame_from_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        n_points: int = 4,
    ) -> list[MaskResult]:
        """Refine a hand-drawn (brush) mask on a single frame with SAM 2.

        The drawn mask is used to build a box prompt plus a handful of
        positive points sampled inside the painted region, then SAM 2 is
        asked to recover the true object silhouette. Candidate masks are
        ranked by overlap (IoU) with the drawn mask so the result "follows"
        the user's intention.

        Args:
            frame: RGB uint8 array HxWx3.
            mask: Bool HxW user-painted mask.
            n_points: Number of interior points to sample as prompts.

        Returns:
            A list with a single :class:`MaskResult` (the best candidate).
        """
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("predict_frame_from_mask requires a non-empty mask")

        # Bounding box with a small padding
        pad = max(2, int(0.04 * max(w, h)))
        x1, y1 = int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad))
        x2, y2 = int(min(w - 1, xs.max() + pad)), int(min(h - 1, ys.max() + pad))
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Sample points spread across the painted region
        coords = self._sample_mask_points(xs, ys, w, h, n_points)
        labels = np.ones(len(coords), dtype=np.int32)

        self.load_image_predictor()
        predictor = self._image_predictor
        with self._speed_context(), torch.inference_mode():
            predictor.set_image(frame)
            masks_arr, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )

        # Prefer the candidate that best matches the user's drawn mask
        best_idx = 0
        best_iou = -1.0
        for i in range(masks_arr.shape[0]):
            cand = masks_arr[i].astype(bool)
            intersection = np.logical_and(cand, mask).sum()
            union = np.logical_or(cand, mask).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou < 0.0:  # no overlap at all -> fall back to highest score
            best_idx = int(np.argmax(scores))

        return [
            MaskResult(
                mask=masks_arr[best_idx].astype(bool),
                score=float(scores[best_idx]),
            )
        ]

    @staticmethod
    def _sample_mask_points(
        ys: np.ndarray, xs: np.ndarray, w: int, h: int, n_points: int
    ) -> np.ndarray:
        """Pick evenly-spread points inside the painted mask region."""
        pts = [(int(xs.mean()), int(ys.mean()))]
        if n_points > 1 and len(xs) >= n_points:
            step = max(1, len(xs) // (n_points - 1))
            order = np.argsort(xs)
            for k in range(1, n_points):
                i = order[min(len(xs) - 1, k * step)]
                pts.append((int(xs[i]), int(ys[i])))
        return np.array(pts, dtype=np.float32)

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
        if self._video_state is not None and self._current_video_path == src:
            # Re-use the already-extracted frames + memory bank: iterative
            # "correct a frame, propagate again" workflows skip re-init cost.
            logger.info("Reusing cached video state for %s", src)
            return self._video_state
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
        with self._speed_context(), torch.inference_mode():
            return fn(
                inference_state=self._video_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=coords,
                labels=labels,
                clear_old_points=clear_old_points,
            )

    def add_mask_to_video(
        self,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> Any:
        """Add a hand-drawn mask as the propagation source for video mode.

        Lets the user propagate from a brush-painted (or refined) mask
        instead of prompt points.
        """
        if self._video_state is None:
            raise RuntimeError("Video not initialised. Call init_video() first.")
        fn = getattr(self._video_predictor, "add_new_mask", None)
        if fn is None:
            raise RuntimeError("add_new_mask is not available in the installed SAM 2")
        with self._speed_context(), torch.inference_mode():
            return fn(
                inference_state=self._video_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
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
        with self._speed_context(), torch.inference_mode():
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
