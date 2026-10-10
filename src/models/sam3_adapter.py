"""Lazily-imported adapter for Meta's SAM 3 / SAM 3.1 (``sam3`` package).

Better Brush runs on SAM 2 today. This module is a thin, dependency-free
bridge to the newer ``sam3`` release so the same UI can drive SAM 3
instance prompts and SAM 3.1 Object Multiplex video tracking without
touching the rest of the codebase.

Two adapters are exposed:

* :class:`Sam3ImageAdapter` — a ``SAM2ImagePredictor``-like wrapper around
  ``build_sam3_image_model`` / ``Sam3Processor``. It reuses the SAM 2 tracker's
  instance-interactivity head (``enable_inst_interactivity=True``) and exposes
  ``set_image()`` / ``predict()`` with the same contract the rotoscoping UI
  already calls.
* :class:`Sam3VideoAdapter` — a ``SAM2Manager``-like wrapper around
  ``build_sam3_multiplex_video_predictor``, which is request/response based
  (``start_session`` / ``add_prompt`` / ``propagate_in_video``) rather than an
  inference-state object.

Nothing in this module imports ``sam3`` at import time. The package is only
touched when a model is actually built, so this module imports cleanly on
installs without SAM 3 and reports a precise, actionable error when the user
tries to enable it. Both SAM 3 checkpoints are gated on Hugging Face
(``facebook/sam3`` and ``facebook/sam3.1``); the error messages say so.
"""

from __future__ import annotations

import gc
import importlib.util
import inspect
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAM3_INSTALL_HINT = (
    "SAM 3 support is optional. Install it with:\n"
    "    pip install 'sam3 @ git+https://github.com/facebookresearch/sam3.git'"
)

SAM3_ACCESS_HINT = (
    "SAM 3 / SAM 3.1 weights are published on gated Hugging Face repos "
    "('facebook/sam3' and 'facebook/sam3.1'). Request access on the model page, "
    "then authenticate with 'hf auth login' (or set HF_TOKEN) before "
    "downloading — or pass checkpoint_path=<local .pt file> to use a copy you "
    "already have."
)

_ACCESS_ERROR_MARKERS = (
    "gated",
    "401",
    "403",
    "repository not found",
    "access to model",
    "huggingface",
    "hf_hub",
    "token",
)

_IMAGE_CHECKPOINT_NAMES = ("sam3.pt", "sam3.1_multiplex.pt", "sam3.1.pt")
_VIDEO_CHECKPOINT_NAMES = ("sam3.1_multiplex.pt", "sam3.pt")

__all__ = [
    "SAM3_ACCESS_HINT",
    "SAM3_INSTALL_HINT",
    "PointPrompt",
    "Sam3AdapterError",
    "Sam3CheckpointError",
    "Sam3DependencyError",
    "Sam3ImageAdapter",
    "Sam3StateError",
    "Sam3VideoAdapter",
    "default_checkpoint_dir",
    "default_device",
    "sam3_is_available",
]


class Sam3AdapterError(RuntimeError):
    """Base class for every SAM 3 adapter failure."""


class Sam3DependencyError(Sam3AdapterError):
    """The optional ``sam3`` package is missing or unusable."""


class Sam3CheckpointError(Sam3AdapterError):
    """The SAM 3 checkpoint is missing locally or blocked by the HF gate."""


class Sam3StateError(Sam3AdapterError):
    """A method was called before its prerequisite (image / session)."""


@dataclass(frozen=True)
class PointPrompt:
    """A single prompt point in frame pixel coordinates."""

    x: float
    y: float
    label: int = 1


def sam3_is_available() -> bool:
    """Return True when the optional ``sam3`` package can be imported."""
    try:
        return importlib.util.find_spec("sam3") is not None
    except (ImportError, ValueError):
        return False


def default_device() -> str:
    """Return the device SAM 3 should run on by default."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_checkpoint_dir() -> Path:
    """Return the repository ``checkpoints/`` directory used for local weights."""
    return Path(__file__).resolve().parents[2] / "checkpoints"


def _import_sam3_symbols() -> tuple[Any, Any, Any]:
    """Import the SAM 3 builders and processor, or raise a helpful error."""
    try:
        from sam3.model_builder import (
            build_sam3_image_model,
            build_sam3_multiplex_video_predictor,
        )
    except Exception as exc:
        raise Sam3DependencyError(
            f"Could not import sam3.model_builder: {exc}\n\n{SAM3_INSTALL_HINT}"
        ) from exc
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        raise Sam3DependencyError(
            f"Could not import sam3.model.sam3_image_processor: {exc}\n\n"
            f"{SAM3_INSTALL_HINT}"
        ) from exc
    return build_sam3_image_model, build_sam3_multiplex_video_predictor, Sam3Processor


def _raise_build_error(kind: str, exc: Exception) -> None:
    """Translate a builder failure into a checkpoint or dependency error."""
    text = str(exc).lower()
    gated = isinstance(exc, (FileNotFoundError, PermissionError, OSError)) or any(
        marker in text for marker in _ACCESS_ERROR_MARKERS
    )
    if gated:
        raise Sam3CheckpointError(
            f"Failed to load the SAM 3 {kind} checkpoint: {exc}\n\n{SAM3_ACCESS_HINT}"
        ) from exc
    raise Sam3DependencyError(
        f"Failed to build the SAM 3 {kind} model: {exc}\n\n{SAM3_INSTALL_HINT}"
    ) from exc


def _resolve_checkpoint(
    explicit: Path | str | None,
    filenames: Sequence[str],
    search_dir: Path | str | None,
) -> Path:
    """Return a usable local checkpoint path or explain how to obtain one."""
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise Sam3CheckpointError(
                f"SAM 3 checkpoint not found at {candidate}.\n\n{SAM3_ACCESS_HINT}"
            )
        return candidate
    roots = [Path(search_dir)] if search_dir is not None else [default_checkpoint_dir()]
    for root in roots:
        for name in filenames:
            candidate = root / name
            if candidate.is_file():
                return candidate
    searched = ", ".join(str(root) for root in roots)
    raise Sam3CheckpointError(
        f"No local SAM 3 checkpoint found (looked for "
        f"{', '.join(filenames)} in {searched}).\n\n{SAM3_ACCESS_HINT}"
    )


def _accepts_keyword(target: Callable[..., Any], name: str) -> bool:
    """Return True when *target* accepts the keyword *name*."""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def _to_numpy(value: Any) -> np.ndarray | None:
    """Convert a torch tensor / array-like to numpy, passing None through."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _as_rgb_frame(frame: Any) -> np.ndarray:
    """Validate and normalise an incoming frame to a contiguous RGB uint8 array."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(
            f"frame must be an RGB array shaped (H, W, 3), got {array.shape}"
        )
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0, 255)
        array = array.astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array)


def _prompt_arrays(points: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert prompt points to ``(coords Nx2 float32, labels N int32)``.

    Accepts :class:`PointPrompt` instances, any object exposing ``x``/``y``
    (+ optional ``label``), plain ``(x, y)`` / ``(x, y, label)`` sequences, and
    a single flat ``Nx2`` / ``Nx3`` array.
    """
    rows: list[tuple[float, float]] = []
    labels: list[int] = []
    array = (
        np.asarray(points, dtype=object)
        if not isinstance(points, np.ndarray)
        else points
    )
    if array.ndim == 2 and array.shape[1] in (2, 3):
        for row in array:
            values = [float(v) for v in np.asarray(row).reshape(-1)]
            rows.append((values[0], values[1]))
            labels.append(int(values[2]) if len(values) >= 3 else 1)
    else:
        for item in array.reshape(-1):
            if hasattr(item, "x") and hasattr(item, "y"):
                rows.append((float(item.x), float(item.y)))
                labels.append(int(getattr(item, "label", 1)))
            else:
                values = [float(v) for v in np.asarray(item).reshape(-1)]
                if len(values) < 2:
                    raise ValueError(
                        "each prompt point must be (x, y) or (x, y, label)"
                    )
                rows.append((values[0], values[1]))
                labels.append(int(values[2]) if len(values) >= 3 else 1)
    if not rows:
        raise ValueError("at least one prompt point is required")
    coords = np.asarray(rows, dtype=np.float32)
    return coords, np.asarray(labels, dtype=np.int32)


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 3x3 binary erosion of *mask* using pure numpy shifts."""
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return eroded


def _sample_interior_points(mask: np.ndarray, n_points: int) -> np.ndarray:
    """Sample up to *n_points* positive prompt points inside *mask*.

    The mask is eroded first so prompts land strictly inside the painted
    region instead of on its boundary; thin strokes that vanish under erosion
    fall back to the original mask. Points are spread deterministically along
    the horizontal extent of the region and always include the centroid, which
    keeps repeated calls reproducible.
    """
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {binary.shape}")
    ys, xs = np.nonzero(_erode_mask(binary))
    if xs.size == 0:
        ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("mask prompt requires a non-empty mask")
    total = max(1, int(n_points))
    order = np.argsort(xs, kind="stable")
    chosen: list[tuple[float, float]] = [(float(xs.mean()), float(ys.mean()))]
    for k in range(1, total):
        index = order[min(xs.size - 1, round(k * (xs.size - 1) / (total - 1)))]
        chosen.append((float(xs[index]), float(ys[index])))
    unique: list[tuple[float, float]] = []
    for point in chosen:
        if point not in unique:
            unique.append(point)
    return np.asarray(unique, dtype=np.float32)


def _mask_iou(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Intersection-over-union of two binary masks."""
    union = int(np.count_nonzero(candidate | reference))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(candidate & reference) / union)


def _release_memory() -> None:
    """Collect garbage and return freed CUDA blocks to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            logger.debug("torch.cuda.empty_cache() failed", exc_info=True)


class Sam3ImageAdapter:
    """``SAM2ImagePredictor``-like wrapper around a SAM 3 image model.

    The model is built lazily on first use, from a local checkpoint, with
    ``load_from_HF=False`` (no gated download at runtime) and
    ``enable_inst_interactivity=True`` so the SAM 2 tracker head is wired in
    and point prompts work.

    Args:
        checkpoint_path: Local ``.pt`` checkpoint. When omitted, a known
            filename is looked up in ``checkpoint_dir``.
        device: Torch device string; defaults to CUDA when available.
        checkpoint_dir: Directory searched for a checkpoint when
            ``checkpoint_path`` is not given.
        enable_inst_interactivity: Build with the SAM 1/2 instance tracker.
        model_builder: Injected replacement for ``build_sam3_image_model``
            (used by tests and by callers with a custom build).
        processor_factory: Injected replacement for ``Sam3Processor``.
        build_kwargs: Extra keyword arguments forwarded to the model builder.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        *,
        checkpoint_dir: Path | str | None = None,
        enable_inst_interactivity: bool = True,
        model_builder: Callable[..., Any] | None = None,
        processor_factory: Callable[..., Any] | None = None,
        build_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or default_device()
        self.enable_inst_interactivity = enable_inst_interactivity
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._model_builder = model_builder
        self._processor_factory = processor_factory
        self._build_kwargs = dict(build_kwargs or {})
        self._model: Any | None = None
        self._processor: Any | None = None
        self._state: Any | None = None
        self._frame_hw: tuple[int, int] = (0, 0)
        self._uses_state_kwarg = True

    @property
    def is_loaded(self) -> bool:
        """True once the model and processor exist."""
        return self._model is not None

    @property
    def frame_hw(self) -> tuple[int, int]:
        """Height and width of the frame currently held by ``set_image``."""
        return self._frame_hw

    def ensure_loaded(self) -> None:
        """Build the model and processor if they are not loaded yet."""
        if self._model is not None:
            return
        builder = self._model_builder
        processor_factory = self._processor_factory
        if builder is None or processor_factory is None:
            image_builder, _video_builder, processor_cls = _import_sam3_symbols()
            builder = builder or image_builder
            processor_factory = processor_factory or processor_cls
        checkpoint = _resolve_checkpoint(
            self.checkpoint_path, _IMAGE_CHECKPOINT_NAMES, self._checkpoint_dir
        )
        logger.info(
            "Building SAM 3 image model from %s on %s (inst_interactivity=%s)",
            checkpoint,
            self.device,
            self.enable_inst_interactivity,
        )
        try:
            model = builder(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                enable_inst_interactivity=self.enable_inst_interactivity,
                device=self.device,
                **self._build_kwargs,
            )
        except Exception as exc:
            _raise_build_error("image", exc)
            raise
        if _accepts_keyword(processor_factory, "device"):
            processor = processor_factory(model, device=self.device)
        else:
            processor = processor_factory(model)
        self._model = model
        self._processor = processor

    def set_image(self, frame: np.ndarray) -> Any:
        """Encode an RGB frame and remember the processor state.

        Args:
            frame: RGB uint8 array shaped ``(H, W, 3)``.

        Returns:
            The processor state dict produced by ``Sam3Processor.set_image``.
        """
        array = _as_rgb_frame(frame)
        self.ensure_loaded()
        assert self._processor is not None
        with torch.inference_mode():
            self._state = self._processor.set_image(array)
        self._frame_hw = (int(array.shape[0]), int(array.shape[1]))
        return self._state

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray | None = None,
        multimask_output: bool = False,
        box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Predict masks for point prompts on the frame set by ``set_image``.

        Args:
            point_coords: ``(N, 2)`` float32 pixel coordinates ``(x, y)``.
            point_labels: ``(N,)`` int32 labels; 1 = positive, 0 = negative.
                Defaults to all-positive.
            multimask_output: Return the three ambiguous-prompt candidates
                instead of a single mask.
            box: Optional ``(4,)`` float32 ``XYXY`` pixel box prompt.

        Returns:
            ``(masks, scores, logits)`` where *masks* is a ``(C, H, W)`` bool
            array, *scores* is a ``(C,)`` float32 array of quality predictions,
            and *logits* is a ``(C, 256, 256)`` float32 low-resolution logit
            array (or None when the model returns none).
        """
        if self._state is None:
            raise Sam3StateError("set_image() must be called before predict()")
        coords, labels = _prompt_arrays(point_coords)
        if point_labels is not None:
            labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
            if labels.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"point_labels has {labels.shape[0]} entries but "
                    f"point_coords has {coords.shape[0]}"
                )
        self.ensure_loaded()
        kwargs: dict[str, Any] = {
            "point_coords": coords,
            "point_labels": labels,
            "multimask_output": bool(multimask_output),
            "normalize_coords": False,
        }
        if box is not None:
            kwargs["box"] = np.asarray(box, dtype=np.float32).reshape(-1)
        masks, scores, logits = self._call_predict_inst(**kwargs)
        return _normalise_image_outputs(masks, scores, logits)

    def predict_from_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        n_points: int = 4,
    ) -> tuple[np.ndarray, float]:
        """Refine a hand-drawn mask with SAM 3 instance prompts.

        Positive points are sampled from the interior of *mask*, all three
        candidates are requested, and the candidate with the highest IoU
        against the drawn mask wins so the result follows the user's intent
        rather than the model's own confidence ordering.

        Args:
            frame: RGB uint8 array shaped ``(H, W, 3)``.
            mask: Boolean ``(H, W)`` user-painted mask.
            n_points: Number of interior points to sample.

        Returns:
            ``(mask, score)`` for the best-matching candidate.
        """
        self.set_image(frame)
        coords = _sample_interior_points(mask, n_points)
        labels = np.ones(coords.shape[0], dtype=np.int32)
        masks, scores, _logits = self.predict(coords, labels, multimask_output=True)
        reference = np.asarray(mask).astype(bool)
        best_index = 0
        best_iou = -1.0
        for index in range(masks.shape[0]):
            iou = _mask_iou(masks[index], reference)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou <= 0.0:
            best_index = int(np.argmax(scores))
        return masks[best_index], float(scores[best_index])

    def unload(self) -> None:
        """Drop the model, processor and cached state."""
        self._model = None
        self._processor = None
        self._state = None
        self._frame_hw = (0, 0)
        _release_memory()
        logger.info("SAM 3 image adapter unloaded.")

    def _call_predict_inst(self, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Call ``predict_inst`` handling both state keyword spellings."""
        assert self._model is not None
        if self._uses_state_kwarg:
            try:
                return self._model.predict_inst(state=self._state, **kwargs)
            except TypeError as exc:
                message = str(exc)
                if "state" not in message:
                    raise
                self._uses_state_kwarg = False
                logger.debug(
                    "predict_inst does not accept 'state', retrying with "
                    "'inference_state' (%s)",
                    message,
                )
        return self._model.predict_inst(inference_state=self._state, **kwargs)


def _normalise_image_outputs(
    masks: Any, scores: Any, logits: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Coerce SAM 3 instance outputs into the SAM 2 predictor contract."""
    mask_array = np.asarray(_to_numpy(masks))
    if mask_array.ndim == 2:
        mask_array = mask_array[None]
    if mask_array.ndim == 4:
        mask_array = mask_array[:, 0]
    if mask_array.dtype != bool:
        mask_array = mask_array > 0.0
    score_array = np.asarray(_to_numpy(scores), dtype=np.float32).reshape(-1)
    logit_array = _to_numpy(logits)
    if logit_array is not None:
        logit_array = np.asarray(logit_array, dtype=np.float32)
        if logit_array.ndim == 4:
            logit_array = logit_array[:, 0]
    return mask_array, score_array, logit_array


class Sam3VideoAdapter:
    """``SAM2Manager``-like wrapper around the SAM 3.1 multiplex predictor.

    The multiplex predictor is request/response based: sessions are opened
    with a ``start_session`` request and frames stream out of
    ``handle_stream_request``. This adapter hides that protocol behind the
    point-in / masks-out API the rotoscoping UI already uses.

    Args:
        checkpoint_path: Local SAM 3.1 ``.pt`` checkpoint. When omitted, a
            known filename is looked up in ``checkpoint_dir``.
        device: Torch device used for the prompt tensors.
        checkpoint_dir: Directory searched for a checkpoint when
            ``checkpoint_path`` is not given.
        predictor_builder: Injected replacement for
            ``build_sam3_multiplex_video_predictor``.
        build_kwargs: Extra keyword arguments forwarded to the builder.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        *,
        checkpoint_dir: Path | str | None = None,
        predictor_builder: Callable[..., Any] | None = None,
        build_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or default_device()
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._predictor_builder = predictor_builder
        self._build_kwargs = dict(build_kwargs or {})
        self._predictor: Any | None = None
        self._session_id: str | None = None
        self._frame_hw: tuple[int, int] = (0, 0)

    @property
    def is_loaded(self) -> bool:
        """True once the multiplex predictor exists."""
        return self._predictor is not None

    @property
    def session_id(self) -> str | None:
        """Identifier of the open video session, or None."""
        return self._session_id

    @property
    def frame_hw(self) -> tuple[int, int]:
        """Video frame size ``(height, width)`` used to normalise points."""
        return self._frame_hw

    def set_frame_size(self, height: int, width: int) -> None:
        """Declare the video frame size so points can be normalised."""
        if height <= 0 or width <= 0:
            raise ValueError(f"frame size must be positive, got {(height, width)}")
        self._frame_hw = (int(height), int(width))

    def ensure_loaded(self) -> None:
        """Build the multiplex predictor if it is not loaded yet."""
        if self._predictor is not None:
            return
        builder = self._predictor_builder
        if builder is None:
            _image_builder, builder, _processor_cls = _import_sam3_symbols()
        checkpoint = _resolve_checkpoint(
            self.checkpoint_path, _VIDEO_CHECKPOINT_NAMES, self._checkpoint_dir
        )
        logger.info("Building SAM 3.1 multiplex video predictor from %s", checkpoint)
        try:
            self._predictor = builder(
                checkpoint_path=str(checkpoint), **self._build_kwargs
            )
        except Exception as exc:
            _raise_build_error("video", exc)
            raise

    def init_video(
        self,
        video_path_or_jpeg_dir: Path | str,
        frame_size: tuple[int, int] | None = None,
    ) -> str:
        """Start a tracking session for a video file or JPEG frame directory.

        Args:
            video_path_or_jpeg_dir: Path to an MP4 or to a directory of frames.
            frame_size: Optional ``(height, width)`` of the video, required to
                normalise the pixel points given to ``add_points_to_video``.

        Returns:
            The new session id.
        """
        resource = Path(video_path_or_jpeg_dir)
        if not resource.exists():
            raise Sam3StateError(f"video path does not exist: {resource}")
        if frame_size is not None:
            self.set_frame_size(*frame_size)
        self.ensure_loaded()
        response = self._require_predictor().handle_request(
            {"type": "start_session", "resource_path": str(resource)}
        )
        session_id = response["session_id"]
        self._session_id = str(session_id)
        logger.info("SAM 3 video session %s on %s", session_id, resource)
        return self._session_id

    def add_points_to_video(
        self,
        frame_idx: int,
        obj_id: int,
        points: Iterable[Any],
        clear_old_points: bool = True,
    ) -> dict[str, Any]:
        """Prompt one object on one frame with pixel-space points.

        The points are converted to the ``[0, 1]`` normalised ``(x, y)`` tensors
        the multiplex model expects and sent as an ``add_prompt`` request.

        Args:
            frame_idx: Frame the prompt is attached to.
            obj_id: Tracker object id to create or refine.
            points: :class:`PointPrompt` objects or ``(x, y[, label])`` tuples
                in pixel coordinates.
            clear_old_points: Replace any prompts already on that object/frame.

        Returns:
            The predictor's ``add_prompt`` response.
        """
        session_id = self._require_session()
        coords, labels = _prompt_arrays(points)
        normalized = self._normalize_coords(coords)
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_idx),
            "points": torch.tensor(normalized, dtype=torch.float32, device=self.device),
            "point_labels": torch.tensor(labels, dtype=torch.int32, device=self.device),
            "obj_id": int(obj_id),
            "clear_old_points": bool(clear_old_points),
        }
        return self._require_predictor().handle_request(request)

    def add_mask_to_video(
        self,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
        n_points: int = 4,
    ) -> dict[str, Any]:
        """Seed a tracker object from a hand-drawn mask.

        SAM 3.1's multiplex predictor has no mask-prompt entry point, so the
        drawn region is converted into positive interior points and forwarded
        to :meth:`add_points_to_video`.

        Args:
            frame_idx: Frame the mask was painted on.
            obj_id: Tracker object id to create or refine.
            mask: Boolean ``(H, W)`` user-painted mask.
            n_points: Number of interior points to sample.

        Returns:
            The predictor's ``add_prompt`` response.
        """
        coords = _sample_interior_points(mask, n_points)
        prompts = [PointPrompt(float(x), float(y), 1) for x, y in coords]
        return self.add_points_to_video(frame_idx, obj_id, prompts)

    def propagate_in_video(
        self,
        start_frame_idx: int = 0,
        max_frames: int | None = None,
        reverse: bool = False,
        on_frame: Callable[[int, int, np.ndarray], None] | None = None,
        cancel_event: Any | None = None,
        *,
        obj_id: int | None = 1,
    ) -> list[tuple[int, int, np.ndarray, float]]:
        """Track the prompted object through the clip.

        Frames stream out of the predictor's ``propagate_in_video`` request.
        Only the entry whose object id matches *obj_id* is kept; frames outside
        the requested direction and beyond *max_frames* are dropped. Backward
        propagation is buffered and replayed in ascending frame order so the
        callback always sees the timeline forwards.

        Args:
            start_frame_idx: Frame carrying the prompts.
            max_frames: Maximum number of frames to return (None = all).
            reverse: Propagate backwards from ``start_frame_idx``.
            on_frame: Optional ``callback(frame_idx, obj_id, mask)`` invoked
                per accepted frame.
            cancel_event: Optional ``threading.Event`` checked between frames.
            obj_id: Object to extract, or None for every tracked object.

        Returns:
            List of ``(frame_idx, obj_id, mask, score)`` in ascending frame
            order.
        """
        session_id = self._require_session()
        request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "backward" if reverse else "forward",
            "start_frame_index": int(start_frame_idx),
        }
        if max_frames is not None:
            request["max_frame_num_to_track"] = int(max_frames)

        results: list[tuple[int, int, np.ndarray, float]] = []
        buffered: list[tuple[int, int, np.ndarray, float]] = []
        limit = int(max_frames) if max_frames is not None else None
        stop = False
        for chunk in self._require_predictor().handle_stream_request(request):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("SAM 3 propagation cancelled by the caller.")
                break
            frame_index, outputs = _split_stream_chunk(chunk)
            for entry_id, mask, score in _select_object_masks(outputs, obj_id):
                if reverse:
                    if frame_index > start_frame_idx:
                        continue
                elif frame_index < start_frame_idx:
                    continue
                entry = (frame_index, entry_id, mask, score)
                if reverse:
                    buffered.append(entry)
                else:
                    _emit(entry, results, on_frame)
                if limit is not None and len(results) + len(buffered) >= limit:
                    stop = True
                    break
            if stop:
                break
        if reverse:
            for entry in reversed(buffered):
                _emit(entry, results, on_frame)
        logger.info("SAM 3 propagation produced %d frame entries", len(results))
        return results

    def reset(self) -> bool:
        """Reset the tracked state of the open session, keeping it usable."""
        if self._session_id is None:
            return False
        try:
            self._require_predictor().handle_request(
                {"type": "reset_session", "session_id": self._session_id}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM 3 reset_session failed: %s", exc)
            return False
        return True

    def close(self) -> bool:
        """Close the open session and drop it. Idempotent."""
        session_id = self._session_id
        if session_id is None:
            return False
        self._session_id = None
        try:
            self._require_predictor().handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM 3 close_session failed: %s", exc)
        _release_memory()
        return True

    def unload(self) -> None:
        """Close any session and release the predictor."""
        self.close()
        self._predictor = None
        self._frame_hw = (0, 0)
        _release_memory()
        logger.info("SAM 3 video adapter unloaded.")

    def _require_predictor(self) -> Any:
        if self._predictor is None:
            raise Sam3StateError(
                "SAM 3 video predictor is not loaded. Call ensure_loaded() or "
                f"init_video() first.\n\n{SAM3_INSTALL_HINT}"
            )
        return self._predictor

    def _require_session(self) -> str:
        if self._session_id is None:
            raise Sam3StateError(
                "No SAM 3 video session is open. Call init_video() first."
            )
        return self._session_id

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        height, width = self._frame_hw
        if height <= 0 or width <= 0:
            raise Sam3StateError(
                "Video frame size is unknown, so points cannot be normalised. "
                "Call set_frame_size(height, width) or pass frame_size= to "
                "init_video()."
            )
        normalized = np.empty_like(coords, dtype=np.float32)
        normalized[:, 0] = np.clip(coords[:, 0] / float(width), 0.0, 1.0)
        normalized[:, 1] = np.clip(coords[:, 1] / float(height), 0.0, 1.0)
        return normalized


def _emit(
    entry: tuple[int, int, np.ndarray, float],
    results: list[tuple[int, int, np.ndarray, float]],
    on_frame: Callable[[int, int, np.ndarray], None] | None,
) -> None:
    """Append an entry to *results* and forward it to the callback."""
    results.append(entry)
    if on_frame is not None:
        on_frame(entry[0], entry[1], entry[2])


def _split_stream_chunk(chunk: Any) -> tuple[int, dict[str, Any]]:
    """Split a streamed frame into ``(frame_index, outputs)``."""
    if not isinstance(chunk, dict):
        raise Sam3AdapterError(
            f"unexpected propagate_in_video payload of type {type(chunk).__name__}"
        )
    if "outputs" in chunk:
        frame_index = chunk.get("frame_index", chunk.get("frame_idx", 0))
        return int(frame_index), dict(chunk["outputs"])
    frame_index = chunk.get("frame_index", chunk.get("frame_idx", 0))
    return int(frame_index), chunk


def _select_object_masks(
    outputs: dict[str, Any], obj_id: int | None
) -> list[tuple[int, np.ndarray, float]]:
    """Pull the masks belonging to *obj_id* out of one propagated frame."""
    masks = _to_numpy(outputs.get("out_binary_masks"))
    if masks is None or masks.size == 0:
        return []
    mask_array = np.asarray(masks)
    if mask_array.ndim == 4:
        mask_array = mask_array[:, 0]
    mask_array = mask_array.astype(bool)
    ids = _to_numpy(outputs.get("out_obj_ids"))
    if ids is None:
        obj_ids = np.arange(mask_array.shape[0], dtype=np.int64)
    else:
        obj_ids = np.asarray(ids).reshape(-1).astype(np.int64)
    scores = _to_numpy(outputs.get("out_probs"))
    if scores is None:
        scores = _to_numpy(outputs.get("out_obj_scores"))
    score_array = (
        np.ones(obj_ids.shape[0], dtype=np.float32)
        if scores is None
        else np.asarray(scores, dtype=np.float32).reshape(-1)
    )
    selected: list[tuple[int, np.ndarray, float]] = []
    for index, candidate_id in enumerate(obj_ids):
        if obj_id is not None and int(candidate_id) != int(obj_id):
            continue
        if index >= mask_array.shape[0]:
            continue
        score = float(score_array[index]) if index < score_array.size else 1.0
        selected.append((int(candidate_id), mask_array[index], score))
    return selected
