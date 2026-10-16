"""MatAnyone / MatAnyone 2 video matting (soft alpha mattes).

The two models are memory-propagation matting networks: a SAM 2-style
binary mask defines the subject on a prompt frame, and the network produces
a soft alpha matte for every frame of the clip (hair, motion blur and
partial coverage), using the previous frames as memory.

Prompt frames are used one by one: the tracker is re-seeded on each prompt,
propagates forward and backward until it reaches the neighbouring prompt,
and the overlap around each prompt is re-anchored to that prompt's mask —
which keeps long clips from drifting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Matting models shipped in the UI, in display order.
MATTING_MODELS = ("matanyone", "matanyone2")

#: Hugging Face repos holding the weights + configs.
_MODEL_REPOS = {
    "matanyone": ("PeiqingYang/MatAnyone", "matanyone"),
    "matanyone2": ("PeiqingYang/MatAnyone2", "matanyone2"),
}

#: Internal resolutions offered in the UI (longest side, in pixels).
MATTING_RESOLUTIONS = (352, 480, 576, 720, 1080, 1440, 2160)

#: Mask cleanup radii applied to the prompt mask before matting (upstream
#: defaults): a small dilation followed by the same erosion smooths the
#: binary outline so the network only has to refine the edge.
_PROMPT_ERODE = 10
_PROMPT_DILATE = 10


@dataclass(frozen=True)
class MattingSettings:
    """User-facing parameters for a matting run."""

    model: str = "matanyone2"
    resolution: int = 720
    gamma: float = 1.0
    grow: int = 0
    #: Constrain the matte to the segmentation mask (+ ``mask_clamp_pad``).
    clamp_to_mask: bool = True
    #: Radius of the dilated mask used as the clamp boundary.
    mask_clamp_pad: int = 6
    #: Keep the *interior* of the segmentation mask opaque whatever the network
    #: says there. The matte refines the edge; it must not hollow out what the
    #: animator already marked as the subject (see ``_clamp_alpha_to_mask``).
    clamp_solid_interior: bool = False

    def __post_init__(self) -> None:
        model = str(self.model or "matanyone2").strip().lower()
        object.__setattr__(
            self, "model", model if model in MATTING_MODELS else "matanyone2"
        )
        resolution = int(self.resolution)
        object.__setattr__(
            self,
            "resolution",
            min(MATTING_RESOLUTIONS, key=lambda r: abs(r - resolution)),
        )
        object.__setattr__(
            self, "gamma", max(0.1, min(10.0, float(self.gamma)))
        )
        object.__setattr__(self, "grow", max(-20, min(20, int(self.grow))))


def default_model_dir(model: str) -> Path:
    """Return the default weights directory for a matting model."""
    from src.core.constants import CHECKPOINTS_DIR

    return CHECKPOINTS_DIR / model


def _install_topk_guard(memory_manager_module) -> None:
    """Clamp the memory top-k softmax to the number of stored entries.

    The MatAnyone models always ask for ``top_k=30`` memory entries, but a
    top-k over fewer entries than ``k`` raises "selected index k out of
    range" — which happens on small frames (few prototypes per frame) and on
    the very first reads of a short clip.  Clamping ``k`` to the available
    count is mathematically identical: the top-k softmax over ``N <= k``
    entries is exactly the full softmax.
    """
    if getattr(memory_manager_module, "_bb_topk_guard", False):
        return
    original = memory_manager_module.do_softmax

    def guarded_do_softmax(
        similarity, top_k=None, inplace=False, return_usage=False
    ):
        if top_k is not None:
            top_k = min(top_k, similarity.shape[1])
        return original(
            similarity, top_k=top_k, inplace=inplace, return_usage=return_usage
        )

    memory_manager_module.do_softmax = guarded_do_softmax
    memory_manager_module._bb_topk_guard = True


class MattingRunner:
    """Loads a MatAnyone model and produces soft alpha mattes."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.model_key = model if model in MATTING_MODELS else "matanyone2"
        self.device = device
        self._cancel = cancel_event or threading.Event()
        self._core = None
        # Filled by matte(): returns the animator's segmentation mask for a
        # frame, used to keep the learned matte off background.
        self._mask_getter: Callable[[int], np.ndarray] | None = None
        # One-time flag so a broken clamp lookup is reported, not swallowed.
        self._clamp_miss_warned = False

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Download (once) and load the matting network."""
        if self._core is not None:
            return
        if self.model_key == "matanyone2":
            from src.vendor.matanyone2 import InferenceCore
            from src.vendor.matanyone2.inference import (
                memory_manager as _memory_manager,
            )
            from src.vendor.matanyone2.model.matanyone2 import MatAnyone2

            network = self._load_network(MatAnyone2)
        else:
            from src.vendor.matanyone import InferenceCore
            from src.vendor.matanyone.inference import memory_manager as _memory_manager
            from src.vendor.matanyone.model.matanyone import MatAnyone

            network = self._load_network(MatAnyone)
        _install_topk_guard(_memory_manager)
        self._core = InferenceCore(network, device=self.device)
        logger.info("Matting model %s loaded on %s", self.model_key, self.device)

    def _load_network(self, cls) -> object:
        local_dir = default_model_dir(self.model_key)
        if not (local_dir / "model.safetensors").is_file():
            self._download(local_dir)
        network = cls.from_pretrained(local_dir)
        network.to(self.device)
        network.eval()
        return network

    def _download(self, local_dir: Path) -> None:
        repo_id, _folder = _MODEL_REPOS[self.model_key]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to download the matting model"
            ) from exc
        logger.info("Downloading %s weights from %s ...", self.model_key, repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "model.safetensors"):
            cached = hf_hub_download(repo_id=repo_id, filename=filename)
            target = local_dir / filename
            if not target.exists():
                import shutil

                shutil.copy2(cached, target)

    def unload(self) -> None:
        """Release the model from memory (safe on a broken CUDA context)."""
        self._core = None
        import gc

        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - never crash the app on exit
                logger.debug("empty_cache during unload failed (%s)", exc)

    # ------------------------------------------------------------------ #
    # Matting
    # ------------------------------------------------------------------ #
    def matte(
        self,
        indices: Sequence[int],
        frame_getter: Callable[[int], np.ndarray],
        prompt_masks: dict[int, np.ndarray],
        settings: MattingSettings,
        on_alpha: Callable[[int, np.ndarray], None] | None = None,
        mask_getter: Callable[[int], np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        """Produce a soft alpha matte for every frame index.

        Frames are fetched on demand from *frame_getter* so clips of any
        length are processed with bounded memory.

        Args:
            indices: Frame indices to matte, in playback order.
            frame_getter: Returns the RGB uint8 frame for an index.
            prompt_masks: Bool subject masks at the prompt frames.
            settings: Matting parameters.
            on_alpha: Optional callback(index, float32 alpha in [0, 1]).
            mask_getter: Optional callback returning the segmentation
                mask for a frame; when present the matte is clamped to it.

        Returns:
            Mapping of frame index -> float32 alpha (H, W) at source size.
        """
        self.load()
        self._mask_getter = mask_getter
        frames = list(indices)
        prompts = [p for p in sorted(prompt_masks) if p in set(frames)]
        if not frames or not prompts:
            return {}

        alphas: dict[int, np.ndarray] = {}
        for position, prompt in enumerate(prompts):
            self._check_cancel()
            next_prompt = (
                prompts[position + 1] if position + 1 < len(prompts) else None
            )
            prev_prompt = prompts[position - 1] if position > 0 else None
            forward_end = (
                next_prompt - 1 if next_prompt is not None else frames[-1]
            )
            backward_end = (
                prev_prompt + 1 if prev_prompt is not None else frames[0]
            )
            forward = [i for i in frames if prompt <= i <= forward_end]
            backward = [i for i in reversed(frames) if backward_end <= i < prompt]
            results = self._propagate(
                frame_getter, prompt_masks[prompt], prompt, forward, backward, settings
            )
            for index, alpha in results.items():
                alphas[index] = alpha
                if on_alpha is not None:
                    on_alpha(index, alpha)
        return alphas

    def _propagate(
        self,
        frame_getter: Callable[[int], np.ndarray],
        prompt_mask: np.ndarray,
        prompt: int,
        forward: list[int],
        backward: list[int],
        settings: MattingSettings,
    ) -> dict[int, np.ndarray]:
        """Seed the tracker on *prompt* and walk forward and backward.

        The prompt frame is memorised, then re-fed as the first frame of the
        temporal sequence (the upstream convention: ``first_frame_pred``
        restarts the time counter so the prompt frame is frame zero, and the
        clip's frames follow it in order).  Upstream then also repeats the
        prompt frame a few "warmup" times, but those repeats pin the frame
        counter at zero and progressively starve the memory bank — measured
        collapse to an empty matte on MatAnyone 1 — so no warmup is used.
        """
        core = self._core
        assert core is not None

        source_h, source_w = frame_getter(prompt).shape[:2]
        height, width = _fit_resolution(source_h, source_w, settings.resolution)
        prompt_image = self._image_tensor(frame_getter(prompt), height, width)
        prompt_tensor = self._mask_tensor(
            prompt_mask, height, width, erode=_PROMPT_ERODE, dilate=_PROMPT_DILATE
        )

        autocast_enabled = self.device.startswith("cuda")
        results: dict[int, np.ndarray] = {}
        for sequence in (forward, backward):
            core.clear_memory()
            with torch.inference_mode(), torch.autocast(
                device_type=self._autocast_device(),
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                core.step(prompt_image, prompt_tensor, objects=[1])
                core.step(prompt_image, first_frame_pred=True)
                prob = core.step(prompt_image)
                results[prompt] = self._postprocess(
                    prob, settings, source_h, source_w, prompt_mask
                )
                for index in sequence:
                    self._check_cancel()
                    if index == prompt:
                        continue
                    image = self._image_tensor(frame_getter(index), height, width)
                    prob = core.step(image)
                    results[index] = self._postprocess(
                        prob, settings, source_h, source_w, self._mask_at(index)
                    )
        return results

    def _mask_at(self, index: int) -> np.ndarray | None:
        """The animator seg_mask for *index*, when known."""
        if self._mask_getter is None:
            return None
        try:
            return self._mask_getter(index)
        except Exception as exc:  # noqa: BLE001 - a miss must not kill the run
            # The clamp silently degrades to the matte's own threshold when
            # this returns None, which lets background scenery into the matte.
            # Still never crash, but surface the first failure so a genuinely
            # broken getter is not invisible.
            if not self._clamp_miss_warned:
                self._clamp_miss_warned = True
                logger.warning(
                    "Matte clamp lookup failed for %s (%s); the matte is not "
                    "constrained to the segmentation mask this run.",
                    index,
                    exc,
                )
            else:
                logger.debug("Matte clamp lookup failed for %s (%s)", index, exc)
            return None

    def _autocast_device(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("Matting cancelled")

    # ------------------------------------------------------------------ #
    # Tensor helpers
    # ------------------------------------------------------------------ #
    def _image_tensor(
        self, frame: np.ndarray, height: int, width: int
    ) -> torch.Tensor:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(
            device=self.device, dtype=torch.float32
        )
        return (tensor.permute(2, 0, 1) / 255.0)

    def _mask_tensor(
        self,
        mask: np.ndarray,
        height: int,
        width: int,
        erode: int = 0,
        dilate: int = 0,
    ) -> torch.Tensor:
        """Prompt mask as a float32 tensor in [0, 255] (upstream convention:
        the network divides by 255 internally)."""
        cleaned = mask.astype(np.uint8)
        if cleaned.max() > 1:
            cleaned = (cleaned > 127).astype(np.uint8) * 255
        if dilate > 0:
            kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
            cleaned = cv2.dilate(cleaned, kernel)
        if erode > 0:
            kernel = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
            cleaned = cv2.erode(cleaned, kernel)
        resized = cv2.resize(cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(
            device=self.device, dtype=torch.float32
        )
        return (tensor > 0).float() * 255.0

    def _postprocess(
        self,
        prob: torch.Tensor,
        settings: MattingSettings,
        source_h: int,
        source_w: int,
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a raw network output to a full-resolution float alpha.

        *seg_mask* is the animator's binary segmentation mask for this frame;
        the clamp uses it so the matte cannot wander into background.
        """
        core = self._core
        assert core is not None
        soft = core.output_prob_to_mask(prob, matting=True)
        alpha = soft.detach().float().cpu().numpy()
        alpha = np.clip(alpha, 0.0, 1.0)
        if (source_h, source_w) != alpha.shape[:2]:
            alpha = cv2.resize(
                alpha, (source_w, source_h), interpolation=cv2.INTER_LINEAR
            )
        if settings.gamma != 1.0:
            alpha = np.power(np.clip(alpha, 0.0, 1.0), settings.gamma)
        if settings.grow != 0:
            alpha = _grow_alpha(alpha, settings.grow)
        # The matting network is told only "this character is the subject";
        # it has no idea where the sized-down SAM mask ended, so it happily
        # mattes hair, cloth and background it finds plausible. Constrain it
        # to a slightly dilated version of the segmentation mask: anything
        # beyond that is background, not matte.
        if settings.clamp_to_mask:
            alpha = _clamp_alpha_to_mask(
                alpha,
                settings.mask_clamp_pad,
                seg_mask,
                solid_interior=settings.clamp_solid_interior,
            )
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _fit_resolution(height: int, width: int, resolution: int) -> tuple[int, int]:
    """Return (h, w) within *resolution* on the long side."""
    scale = min(1.0, resolution / max(height, width))
    out_h = max(64, round(height * scale))
    out_w = max(64, round(width * scale))
    return out_h, out_w


def _clamp_alpha_to_mask(
    alpha: np.ndarray,
    pad: int,
    seg_mask: np.ndarray | None = None,
    solid_interior: bool = False,
) -> np.ndarray:
    """Keep *alpha* inside the segmentation mask, plus a small margin.

    The learned matting network is free to put partial coverage anywhere it
    likes, which on busy backgrounds pulls in scenery the animator never
    selected. Intersecting with a dilated copy of the SAM mask bounds the
    matte to the subject the user actually drew.

    Args:
        alpha: Soft matte in [0, 1].
        pad: Dilation radius in pixels, so a matte may still reach slightly
            outside the segmentation border (where hair really does end).
        seg_mask: The binary segmentation mask for this frame, when the
            caller has one. Without it the matte's own 0.5 threshold is used,
            which cannot distinguish detail from background.
        solid_interior: Keep everything a little inside the mask opaque. With
            this off the clamp is one-sided: it can only *remove* coverage, so
            a matte that under-reads a region — hair the network is unsure of,
            a patch the animator has just painted in — comes back transparent
            and the work looks like it did nothing. With it on, the mask is
            authoritative in its interior and the network owns only the edge.

    Returns:
        The clamped float matte.
    """
    exact: np.ndarray | None = None
    if seg_mask is not None and seg_mask.any():
        # The animator's own mask: an exact boundary, not the network's guess.
        exact = seg_mask.astype(bool)
        keep = exact
    else:
        soft = cv2.GaussianBlur(alpha, (0, 0), 1.0)
        keep = soft > 0.5
    if not keep.any():
        return alpha
    if pad > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)
        )
        keep = cv2.dilate(keep.astype(np.uint8), kernel).astype(bool)
    # Feather the clamp edge so the matte does not get a hard cut.
    weight = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5)
    clamped = np.clip(alpha * weight, 0.0, 1.0)
    if solid_interior and exact is not None:
        clamped = np.maximum(clamped, _solid_core(exact))
    return np.clip(clamped, 0.0, 1.0)


#: Share of the short side eroded off a mask to find its "definitely the
#: subject" interior. Thin structures (hair strands, fingertips) fall outside
#: the core and are left to the network, which is exactly where a soft matte
#: earns its keep.
_SOLID_CORE_RATIO = 0.008


def _solid_core(mask: np.ndarray) -> np.ndarray:
    """A feathered float weight: 1 deep inside *mask*, 0 outside it."""
    height, width = mask.shape[:2]
    radius = max(2, round(min(height, width) * _SOLID_CORE_RATIO))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not core.any():
        return np.zeros(mask.shape[:2], dtype=np.float32)
    return cv2.GaussianBlur(core.astype(np.float32), (0, 0), 1.0)


def _grow_alpha(alpha: np.ndarray, grow: int) -> np.ndarray:
    """Grow (>0) or shrink (<0) a soft matte, keeping its edge gradients."""
    if grow == 0:
        return alpha
    size = abs(int(grow)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if grow > 0:
        grown = cv2.dilate(alpha, kernel)
    else:
        grown = cv2.erode(alpha, kernel)
    # Blend a fraction of the blurred edge back so growth stays soft.
    grown = cv2.addWeighted(grown, 0.75, cv2.GaussianBlur(grown, (0, 0), 1.0), 0.25, 0.0)
    return np.clip(grown, 0.0, 1.0)
