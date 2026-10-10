"""MiniMax-Remover video object removal engine.

A distilled video inpainting model (6 sampling steps, no classifier-free
guidance) that removes the masked subject and reconstructs the background,
using Stable-Video-Diffusion-style Wan VAE latents as conditioning.

The engine runs whole windows of frames through the model, then composites
the reconstructed frames back over the originals with a grown, feathered
matte so unremoved pixels are never touched.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Internal resolutions offered in the UI (longest side, in pixels).
REMOVAL_RESOLUTIONS = (352, 480, 720, 1080)

#: Frames per inference window. The Wan VAE compresses time by 4x, so a
#: window of 81 frames maps to 21 latent frames. Windows overlap and the
#: overlap is cross-faded so the final clip has no visible seams.
_WINDOW_FRAMES = 81
_WINDOW_OVERLAP = 8
#: The remover never goes below this many frames per window (the Wan VAE
#: needs at least 5 frames for its temporal compression).
_MIN_WINDOW = 5

#: Window sizes by *free* VRAM (MiB) when the run starts. Bigger windows are
#: faster per frame (the fixed encode/decode cost is amortised), but the DiT
#: attention grows with the number of latent frames, so an 8 GB laptop GPU
#: must stay small. The engine still halves the window and retries if a run
#: unexpectedly runs out of memory.
_VRAM_WINDOW_TABLE: tuple[tuple[int, int], ...] = (
    (12_000, 81),
    (9_000, 33),
    (7_000, 17),
    (5_500, 13),
    (4_000, 9),
    (0, 5),
)


def _free_vram_mb() -> int | None:
    """Best-effort free-VRAM probe (NVML, no CUDA initialisation needed)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free / (1024 * 1024))
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 - probe is advisory only
        logger.debug("Free-VRAM probe failed (%s)", exc)
        return None


def _window_for_vram() -> int:
    """Return the window size that safely fits the current GPU."""
    free = _free_vram_mb()
    if free is None:
        return 17
    for threshold, window in _VRAM_WINDOW_TABLE:
        if free >= threshold:
            return window
    return _MIN_WINDOW

#: Coefficients the Wan VAE uses to normalise latents (safety fallback - the
#: real values ship in the VAE config).
_LATENTS_MEAN = -0.22893865196990967
_LATENTS_STD = 0.6470513343811035

_MODEL_REPO = "zibojia/minimax-remover"
_MODEL_PARTS = ("vae", "transformer", "scheduler")


def _is_out_of_memory(exc: BaseException) -> bool:
    """True when *exc* is a GPU out-of-memory failure (recoverable or not)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cudaErrorMemoryAllocation" in message


def default_model_dir() -> Path:
    """Return the default MiniMax-Remover weights directory."""
    from src.core.constants import CHECKPOINTS_DIR

    return CHECKPOINTS_DIR / "minimax"


@dataclass(frozen=True)
class RemovalSettings:
    """User-facing parameters for a removal run."""

    resolution: int = 480
    steps: int = 6
    mask_grow: int = 21
    feather: int = 21
    seed: int = 42
    #: Frames per inference window; 0 = auto (picked from free VRAM). The
    #: engine also halves it and retries if a window unexpectedly OOMs.
    window_frames: int = 0
    #: Tiled VAE decode/encode. Slightly slower, but the Wan VAE decoder
    #: otherwise holds every frame at full internal resolution at once,
    #: which does not fit on 8 GB laptop GPUs. On by default.
    vae_tiling: bool = True

    def __post_init__(self) -> None:
        resolution = int(self.resolution)
        object.__setattr__(
            self,
            "resolution",
            min(REMOVAL_RESOLUTIONS, key=lambda r: abs(r - resolution)),
        )
        object.__setattr__(self, "steps", max(4, min(12, int(self.steps))))
        object.__setattr__(self, "mask_grow", max(0, min(64, int(self.mask_grow))))
        object.__setattr__(self, "feather", max(0, min(64, int(self.feather))))


class MinimaxRemover:
    """Loads MiniMax-Remover and removes the masked subject from frames."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda",
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._cancel = cancel_event or threading.Event()
        self._transformer: torch.nn.Module | None = None
        self._vae: torch.nn.Module | None = None
        self._scheduler: Any = None

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        """True when the platform can run the model (fp16 CUDA)."""
        return self.device.startswith("cuda") and torch.cuda.is_available()

    def load(self, vae_tiling: bool = False) -> None:
        """Download (once) and load the model components in fp16."""
        if self._transformer is not None:
            return
        if not self.available:
            raise RuntimeError(
                "MiniMax-Remover needs a CUDA-capable NVIDIA GPU "
                "(the model runs in half precision)."
            )
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler

        from src.vendor.minimax import transformer as minimax_transformer

        self._ensure_downloaded()

        logger.info("Loading MiniMax-Remover from %s", self.model_dir)
        self._vae = AutoencoderKLWan.from_pretrained(
            self.model_dir / "vae", torch_dtype=torch.float16
        ).to(self.device)
        self._vae.eval()
        self._vae.enable_slicing()
        if vae_tiling:
            self._vae.enable_tiling()
        self._transformer = (
            minimax_transformer.MinimaxRemoverTransformer3DModel.from_pretrained(
                self.model_dir / "transformer",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        )
        self._transformer.eval()
        self._scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_dir / "scheduler"
        )
        logger.info("MiniMax-Remover loaded.")

    def unload(self) -> None:
        """Release weights and free the GPU (safe on a broken CUDA context)."""
        self._transformer = None
        self._vae = None
        self._scheduler = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            # A failed inference can leave the allocator in a state where
            # empty_cache() itself raises; closing the app must never crash.
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.debug("empty_cache during unload failed (%s)", exc)

    # ------------------------------------------------------------------ #
    # Checkpoint download
    # ------------------------------------------------------------------ #
    def _ensure_downloaded(self) -> None:
        if (self.model_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists():
            return
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to download MiniMax-Remover"
            ) from exc
        logger.info("Downloading MiniMax-Remover weights from %s ...", _MODEL_REPO)
        snapshot_download(
            repo_id=_MODEL_REPO,
            allow_patterns=[f"{part}/*" for part in _MODEL_PARTS],
            local_dir=str(self.model_dir),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def remove(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        settings: RemovalSettings,
    ) -> list[np.ndarray]:
        """Remove the masked subject from every frame.

        Args:
            frames: RGB uint8 frames at the source resolution.
            masks: Bool masks (same size as frames) marking the subject.
            settings: Removal parameters.

        Returns:
            Cleaned RGB uint8 frames at the source resolution.
        """
        results = self.remove_frames(
            range(len(frames)),
            frame_getter=lambda i: frames[i],
            mask_getter=lambda i: masks[i],
            settings=settings,
        )
        return [results[i] for i in range(len(frames))]

    def remove_frames(
        self,
        indices: Sequence[int],
        frame_getter: Callable[[int], np.ndarray],
        mask_getter: Callable[[int], np.ndarray],
        settings: RemovalSettings,
        on_frame: Callable[[int, np.ndarray], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[int, np.ndarray]:
        """Remove the subject from the frames at *indices*.

        Frames are fetched on demand from *frame_getter* / *mask_getter* so
        clips of any length can be processed with bounded memory. Consecutive
        windows overlap and the overlap is cross-faded, so the reconstructed
        clip has no visible seams or temporal pops.

        When the GPU runs out of memory the window is automatically halved
        (down to :data:`_MIN_WINDOW` frames) and the run continues, so an
        8 GB laptop GPU finishes a long clip instead of failing.

        Args:
            indices: Frame indices to process, in playback order.
            frame_getter: Returns the RGB uint8 frame for an index.
            mask_getter: Returns the bool subject mask for an index.
            settings: Removal parameters.
            on_frame: Optional callback(frame_index, cleaned_frame) fired as
                each frame is completed.
            on_status: Optional callback(message) fired when the engine
                adapts itself (e.g. shrinks the window after an OOM).

        Returns:
            Mapping of frame index -> cleaned RGB uint8 frame.
        """
        self.load(settings.vae_tiling)
        order = list(indices)
        if not order:
            return {}
        window = max(
            _MIN_WINDOW, int(settings.window_frames) or _window_for_vram()
        )
        # Small windows need a proportionally small overlap, or the stride
        # between windows would shrink to (nearly) zero.
        overlap = min(_WINDOW_OVERLAP, max(1, window // 2))
        results: dict[int, np.ndarray] = {}
        position = 0
        while position < len(order):
            if self._cancel.is_set():
                raise RuntimeError("Removal cancelled")
            take = min(window, len(order) - position)
            window_indices = order[position : position + take]
            try:
                cleaned = self._run_window(
                    [frame_getter(i) for i in window_indices],
                    [mask_getter(i) for i in window_indices],
                    settings,
                )
            except Exception as exc:
                if not _is_out_of_memory(exc) or window <= _MIN_WINDOW:                    raise
                # Halve the window and redo the last overlap so the crossfade
                # between the two windows stays continuous.
                window = max(_MIN_WINDOW, window // 2)
                overlap = min(_WINDOW_OVERLAP, max(1, window // 2))
                self._release_gpu()
                if on_status is not None:
                    on_status(
                        f"Removing object · {window}-frame windows "
                        "(auto-fit for GPU memory)"
                    )
                position = max(0, position - overlap)
                continue
            for offset, frame in enumerate(cleaned):
                index = window_indices[offset]
                if offset < overlap and index in results:
                    # Overlapping region of the previous window: cross-fade
                    # the two reconstructions instead of cutting between them.
                    fade = (offset + 1) / (overlap + 1)
                    results[index] = _blend(results[index], frame, fade)
                else:
                    results[index] = frame
                if on_frame is not None:
                    on_frame(index, results[index])
            position += max(1, window - overlap)
            self._release_gpu()
        return results

    def _release_gpu(self) -> None:
        """Return unused blocks to the driver between windows (best effort)."""
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - a broken context must not kill the app
            logger.debug("empty_cache failed (%s)", exc)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def _run_window(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        settings: RemovalSettings,
    ) -> list[np.ndarray]:
        """Remove the subject from one window and composite it back.

        The Wan VAE compresses time 4x, so a window length must satisfy
        ``(F - 1) % 4 == 0``. The final (short) window is padded by repeating
        its last frame and the padded tail is dropped from the result.
        """
        height, width = frames[0].shape[:2]
        proc_h, proc_w = _fit_resolution(height, width, settings.resolution)

        original_count = len(frames)
        pad_to = max(5, ((original_count - 1 + 3) // 4) * 4 + 1)
        if pad_to > original_count:
            pad = [frames[-1]] * (pad_to - original_count)
            pad_masks = [masks[-1]] * (pad_to - original_count)
            frames = [*frames, *pad]
            masks = [*masks, *pad_masks]

        images = _to_model_tensor(frames, proc_h, proc_w, self.device)
        mask_tensor = _to_mask_tensor(masks, proc_h, proc_w, self.device)
        masked_images = images * (1.0 - mask_tensor)

        latents_mean, latents_std = self._latent_norm()
        with torch.no_grad():
            masked_latents = (
                self._vae.encode(masked_images).latent_dist.mode()
            )
            mask_latents = self._vae.encode(2.0 * mask_tensor - 1.0).latent_dist.mode()
            masked_latents = (masked_latents - latents_mean) * latents_std
            mask_latents = (mask_latents - latents_mean) * latents_std

        num_frames = images.shape[2]
        num_latent_frames = (num_frames - 1) // 4 + 1
        shape = (1, 16, num_latent_frames, proc_h // 8, proc_w // 8)
        generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        latents = torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float16
        )

        self._scheduler.set_timesteps(settings.steps, device=self.device)
        for t in self._scheduler.timesteps:
            if self._cancel.is_set():
                raise RuntimeError("Removal cancelled")
            model_input = torch.cat([latents, masked_latents, mask_latents], dim=1)
            timestep = t.expand(latents.shape[0])
            noise_pred = self._transformer(
                hidden_states=model_input.half(), timestep=timestep
            )[0]
            latents = self._scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

        latents = latents.half() / latents_std + latents_mean
        with torch.no_grad():
            video = self._vae.decode(latents, return_dict=False)[0]
        cleaned = _to_uint8_frames(video)[:original_count]
        # The VAE decodes at the internal resolution; bring the rebuilt frames
        # back to the source size before compositing them over the originals.
        cleaned = [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            for frame in cleaned
        ]
        return [
            _composite(original, removed, mask, settings)
            for original, removed, mask in zip(frames[:original_count], cleaned, masks[:original_count])
        ]

    def _latent_norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Wan VAE latent normalisation tensors, shaped (1, C, 1, 1, 1)."""
        config = getattr(self._vae, "config", None)
        mean = getattr(config, "latents_mean", None) or [_LATENTS_MEAN] * 16
        std = getattr(config, "latents_std", None) or [_LATENTS_STD] * 16
        z_dim = len(mean)
        mean_t = (
            torch.tensor(mean, dtype=torch.float16, device=self.device)
            .view(1, z_dim, 1, 1, 1)
        )
        std_t = 1.0 / torch.tensor(std, dtype=torch.float16, device=self.device).view(
            1, z_dim, 1, 1, 1
        )
        return mean_t, std_t


# ---------------------------------------------------------------------- #
# Tensor helpers
# ---------------------------------------------------------------------- #
def _fit_resolution(height: int, width: int, resolution: int) -> tuple[int, int]:
    """Return (h, w) within *resolution* on the long side, both /16 aligned."""
    scale = min(1.0, resolution / max(height, width))
    out_h = max(64, round(height * scale / 16) * 16)
    out_w = max(64, round(width * scale / 16) * 16)
    return out_h, out_w


def _to_model_tensor(
    frames: list[np.ndarray], height: int, width: int, device: str
) -> torch.Tensor:
    """Stack frames into a (1, 3, F, H, W) fp16 tensor in [-1, 1]."""
    planes = []
    for frame in frames:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(
            device=device, dtype=torch.float16
        )
        tensor = tensor.permute(2, 0, 1).float() / 127.5 - 1.0
        planes.append(tensor)
    stacked = torch.stack(planes, dim=1)  # (3, F, H, W)
    return stacked.unsqueeze(0).half()


def _to_mask_tensor(
    masks: list[np.ndarray], height: int, width: int, device: str
) -> torch.Tensor:
    """Stack masks into a (1, 3, F, H, W) fp16 binary tensor.

    The mask is replicated to three channels so it can pass through the
    same (3-channel) VAE encoder as the video frames.
    """
    planes = []
    for mask in masks:
        resized = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float16)
        planes.append(tensor[None])
    stacked = torch.stack(planes, dim=1)  # (1, F, H, W)
    stacked = (stacked > 0).half()
    return stacked.repeat(1, 3, 1, 1, 1)


def _to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    """Convert a (1, 3, F, H, W) model output back to RGB uint8 frames."""
    frames = video.float().clamp(-1.0, 1.0)
    frames = ((frames[0].permute(1, 2, 3, 0) + 1.0) * 127.5).to(torch.uint8)
    return [frame.cpu().numpy() for frame in frames]


def _blend(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend of two uint8 frames; *alpha* weights *second*."""
    weight = float(np.clip(alpha, 0.0, 1.0))
    blended = (
        first.astype(np.float32) * (1.0 - weight)
        + second.astype(np.float32) * weight
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _composite(
    original: np.ndarray,
    removed: np.ndarray,
    mask: np.ndarray,
    settings: RemovalSettings,
) -> np.ndarray:
    """Blend the reconstructed frame over the original with a feathered matte.

    The removal matte is grown and feathered so the transition from
    reconstructed to untouched pixels never shows a hard seam, while pixels
    far outside the subject are copied unchanged.
    """
    if settings.mask_grow <= 0 and settings.feather <= 0:
        return removed
    if not mask.any():
        return original
    grown = mask.astype(np.uint8)
    if settings.mask_grow > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (settings.mask_grow * 2 + 1, settings.mask_grow * 2 + 1),
        )
        grown = cv2.dilate(grown, kernel)
    if settings.feather > 0:
        sigma = max(1.0, settings.feather / 3.0)
        grown = cv2.GaussianBlur(grown, (0, 0), sigma)
        grown = cv2.normalize(grown, None, 0, 255, cv2.NORM_MINMAX)
    alpha = (grown.astype(np.float32) / 255.0)[:, :, None]
    blended = (
        removed.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
