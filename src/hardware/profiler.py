"""Profiler: recommend SAM 2 model, backend config and content profile."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from src.hardware.detector import Backend, GPUInfo, HardwareProfile

logger = logging.getLogger(__name__)


class ContentProfile(Enum):
    """Semantic content categories influencing SAM 2 parameters."""

    ANIME = "anime"  # anime / AMV: crisp lines, flat shading
    HUMAN = "human"  # human subjects: hair edges, fine details
    OBJECT = "object"  # generic objects / products

    # Backward-compatible alias (previously named "general").
    GENERAL = OBJECT


class ModelSize(Enum):
    """SAM model variants (smaller → faster, less accurate)."""

    SAM3_1 = "sam3.1_multiplex"
    LARGE = "sam2.1_hiera_large"
    BASE_PLUS = "sam2.1_hiera_base_plus"
    SMALL = "sam2.1_hiera_small"
    TINY = "sam2.1_hiera_tiny"
    #: A user-supplied fine-tuned checkpoint (see docs/FINE-TUNING.md).
    #: Checkpoint and config live in ``checkpoints/custom/`` with a JSON
    #: sidecar; picked manually, never auto-recommended.
    CUSTOM = "custom_finetune"
    # Lightweight "track anything" family (EfficientTAM, Meta) — plain ViT
    # image encoder + efficient memory attention.  Only ~100-140 MB on disk,
    # runs comfortably on integrated GPUs and old hardware.
    EFFICIENTTAM_TI = "efficienttam_ti_512x512"
    EFFICIENTTAM_S = "efficienttam_s_512x512"


@dataclass(frozen=True)
class SAM2Recommendation:
    """Complete recommendation for a SAM 2 session."""

    model_size: ModelSize
    model_config: str
    checkpoint_url: str
    device: str
    backend: Backend
    mixed_precision: bool
    content_profile: ContentProfile
    num_mask_tokens: int
    pred_iou_thresh: float
    stability_score_offset: float
    max_frames_in_memory: int
    #: Low-VRAM cards benefit from keeping the video-predictor's memory bank
    #: and frame buffer on the CPU.  Off by default so bigger cards stay fast.
    offload_state_to_cpu: bool
    offload_video_to_cpu: bool
    user_message: str


def _sam3_package_available() -> bool:
    try:
        return importlib.util.find_spec("sam3") is not None
    except (ImportError, ValueError):
        return False


def _sam2_package_available() -> bool:
    """Return whether the SAM 2 package can be imported at all."""
    try:
        return importlib.util.find_spec("sam2") is not None
    except (ImportError, ValueError):
        return False


def _efficienttam_package_available() -> bool:
    """Return whether the vendored EfficientTAM package is present."""
    root = Path(__file__).resolve().parents[1]  # .../src
    return (
        root / "vendor" / "efficienttam" / "build_efficienttam.py"
    ).is_file()


class SAM2Profiler:
    """Map hardware capabilities to conservative SAM model settings.

    Priority is GPU stability over raw speed: a model tier is only
    recommended when the free VRAM leaves ~35 % headroom. This avoids pushing
    modest cards to 100 % utilisation, which can overheat the GPU and crash
    the driver (black screen on laptops/desktops with weak cooling).
    """

    # Minimum VRAM REQUIRED (MB) to run a tier *comfortably* in video
    # propagation mode (weights + activations + SAM 2's per-frame memory
    # bank). Deliberately higher than the raw checkpoint size on disk.
    _VRAM_THRESHOLDS: ClassVar[dict[ModelSize, int]] = {
        ModelSize.SAM3_1: 15_000,
        ModelSize.TINY: 2_800,
        ModelSize.SMALL: 5_000,
        ModelSize.BASE_PLUS: 8_500,
        ModelSize.LARGE: 13_500,
        # A fine-tuned checkpoint inherits its tier's footprint.
        ModelSize.CUSTOM: 8_500,
        # EfficientTAM runs at a 512×512 input resolution: the flat ViT encoder
        # is far cheaper than Hiera at any tier, so it fits where even SAM 2.1
        # Tiny does not (older GTX cards, integrated graphics, small VRAM).
        ModelSize.EFFICIENTTAM_S: 1_800,
        ModelSize.EFFICIENTTAM_TI: 1_100,
    }

    # Only recommend a tier when free VRAM × this factor clears the
    # threshold above (keeps ~35 % of VRAM free at runtime).
    _VRAM_SAFETY_FACTOR: float = 0.65

    #: A card with less total VRAM than this cannot host SAM 2.1 Tiny
    #: comfortably, so the lightweight EfficientTAM family is used instead.
    _SAM2_MIN_TOTAL_MB: int = 3_500

    #: Cards below this total VRAM keep the video predictor's state (memory
    #: bank + frames) on the CPU to stay clear of the VRAM ceiling.
    _OFFLOAD_BELOW_MB: int = 10_000

    _CONFIG_MAP: ClassVar[dict[ModelSize, str]] = {
        ModelSize.SAM3_1: "sam3.1_multiplex.pt",
        ModelSize.LARGE: "sam2.1_hiera_l.yaml",
        ModelSize.BASE_PLUS: "sam2.1_hiera_b+.yaml",
        ModelSize.SMALL: "sam2.1_hiera_s.yaml",
        ModelSize.TINY: "sam2.1_hiera_t.yaml",
        ModelSize.EFFICIENTTAM_TI: "efficienttam_ti_512x512.yaml",
        ModelSize.EFFICIENTTAM_S: "efficienttam_s_512x512.yaml",
    }

    _URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{checkpoint}"
    )

    _HF_URL_TEMPLATE = (
        "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/{checkpoint}"
    )

    _CHECKPOINT_MAP: ClassVar[dict[ModelSize, str]] = {
        ModelSize.SAM3_1: "sam3.1_multiplex.pt",
        ModelSize.LARGE: "sam2.1_hiera_large.pt",
        ModelSize.BASE_PLUS: "sam2.1_hiera_base_plus.pt",
        ModelSize.SMALL: "sam2.1_hiera_small.pt",
        ModelSize.TINY: "sam2.1_hiera_tiny.pt",
        ModelSize.EFFICIENTTAM_TI: "efficienttam_ti_512x512.pt",
        ModelSize.EFFICIENTTAM_S: "efficienttam_s_512x512.pt",
    }

    _PROFILE_PRESETS: ClassVar[dict[ContentProfile, dict[str, float | int]]] = {
        # Anime / AMV: crisp outlines, flat colours — thinner line art needs
        # a slightly lower IoU bar, long clips deserve a generous memory bank.
        ContentProfile.ANIME: {
            "num_mask_tokens": 3,
            "pred_iou_thresh": 0.82,
            "stability_score_offset": 0.8,
            "max_frames_in_memory": 64,
        },
        # Human: hair / skin edges need extra mask tokens and a high IoU bar
        # so partial / wobbly selections get rejected.
        ContentProfile.HUMAN: {
            "num_mask_tokens": 4,
            "pred_iou_thresh": 0.90,
            "stability_score_offset": 1.0,
            "max_frames_in_memory": 48,
        },
        # Generic objects / products: intermediate tolerance, 3 tokens.
        ContentProfile.OBJECT: {
            "num_mask_tokens": 3,
            "pred_iou_thresh": 0.86,
            "stability_score_offset": 0.9,
            "max_frames_in_memory": 48,
        },
    }

    def recommend(
        self,
        profile: HardwareProfile,
        content: ContentProfile = ContentProfile.GENERAL,
        force_model: ModelSize | None = None,
    ) -> SAM2Recommendation:
        target_gpu = self._pick_target_gpu(profile)
        model_size = force_model or self._pick_model(target_gpu)
        device = profile.recommended_device
        backend = profile.best_backend
        mixed_precision = self._supports_amp(backend, target_gpu)
        presets = self._PROFILE_PRESETS[content]
        # Small cards get the memory bank offloaded to system RAM.  This is
        # what keeps a 6-8 GB laptop GPU from OOMing during long propagation
        # runs, at the cost of a little host<->device traffic.
        low_vram = (
            target_gpu is not None
            and 0 < target_gpu.total_vram_mb < self._OFFLOAD_BELOW_MB
        )
        msg = self._build_user_message(model_size, backend, mixed_precision, target_gpu)

        return SAM2Recommendation(
            model_size=model_size,
            model_config=self._CONFIG_MAP[model_size],
            checkpoint_url=self._url_for(model_size),
            device=device,
            backend=backend,
            mixed_precision=mixed_precision,
            content_profile=content,
            num_mask_tokens=presets["num_mask_tokens"],  # type: ignore[arg-type]
            pred_iou_thresh=presets["pred_iou_thresh"],  # type: ignore[arg-type]
            stability_score_offset=presets["stability_score_offset"],  # type: ignore[arg-type]
            max_frames_in_memory=presets["max_frames_in_memory"],  # type: ignore[arg-type]
            offload_state_to_cpu=low_vram,
            offload_video_to_cpu=low_vram,
            user_message=msg,
        )

    @staticmethod
    def _pick_target_gpu(profile: HardwareProfile) -> GPUInfo | None:
        cuda = [g for g in profile.gpus if g.backend == Backend.CUDA]
        if cuda:
            return max(cuda, key=lambda g: g.total_vram_mb)
        dml = [g for g in profile.gpus if g.backend == Backend.DIRECTML]
        if dml:
            return dml[0]
        return None

    def _pick_model(self, gpu: GPUInfo | None) -> ModelSize:
        if gpu is None or gpu.total_vram_mb == 0:
            logger.warning("No GPU detected — falling back to TINY on CPU")
            return ModelSize.TINY
        # The tier is chosen from the card's TOTAL VRAM, never from the free
        # VRAM observed at call time.  Free VRAM moves while the app runs
        # (another window opens, the loaded model itself occupies memory), so
        # the old free-VRAM heuristic silently picked a different - and
        # occasionally much weaker (EfficientTAM) - model between runs and
        # even mid-session.  Users saw that as "sometimes it just cannot find
        # the character"; a stable total-VRAM decision never switches a good
        # model out from under them.  Runtime OOM is handled by the workers,
        # which report it clearly instead of guessing the tier up front.
        capacity = int(gpu.total_vram_mb)
        # Conservative headroom on top of the (already padded) tier
        # thresholds: never run a card at its limit, which overheats modestly
        # cooled laptops and can reset the driver (black screen).
        effective = int(capacity * self._VRAM_SAFETY_FACTOR)
        logger.debug(
            "Pick model: %d MB total -> %d MB effective (factor x%.2f)",
            capacity,
            effective,
            self._VRAM_SAFETY_FACTOR,
        )
        # The light EfficientTAM family is only used when SAM 2 cannot be used
        # at all: not installed, or the card is genuinely too small.  It is
        # markedly weaker on anime line art, so it must never be a silent
        # "downgrade" a slightly busy GPU can trigger.
        if _sam2_package_available() and capacity >= self._SAM2_MIN_TOTAL_MB:
            for size in (
                ModelSize.SAM3_1,
                ModelSize.LARGE,
                ModelSize.BASE_PLUS,
                ModelSize.SMALL,
                ModelSize.TINY,
            ):
                if size is ModelSize.SAM3_1 and not _sam3_package_available():
                    continue
                if effective >= self._VRAM_THRESHOLDS[size]:
                    logger.info(
                        "Recommended %s (threshold %d MB, effective %d MB)",
                        size.value,
                        self._VRAM_THRESHOLDS[size],
                        effective,
                    )
                    return size
        if _efficienttam_package_available():
            light = (
                ModelSize.EFFICIENTTAM_S
                if effective >= self._VRAM_THRESHOLDS[ModelSize.EFFICIENTTAM_S]
                else ModelSize.EFFICIENTTAM_TI
            )
            logger.info(
                "SAM 2 does not fit (%d MB total); using %s",
                capacity,
                light.value,
            )
            return light
        logger.warning(
            "Neither SAM 2 nor EfficientTAM is importable — defaulting to "
            "SAM 2.1 Tiny (loading will fail with a clear message)."
        )
        return ModelSize.TINY

    def largest_model_for_profile(self, profile: HardwareProfile) -> ModelSize | None:
        """Return the largest tier that fits current GPU memory, if any."""
        gpus = [gpu for gpu in profile.gpus if gpu.total_vram_mb > 0]
        if not gpus:
            return None
        target = max(gpus, key=lambda gpu: gpu.available_vram_mb or gpu.total_vram_mb)
        available = target.available_vram_mb or target.total_vram_mb
        for size in (
            ModelSize.SAM3_1,
            ModelSize.LARGE,
            ModelSize.BASE_PLUS,
            ModelSize.SMALL,
            ModelSize.TINY,
            ModelSize.EFFICIENTTAM_S,
            ModelSize.EFFICIENTTAM_TI,
        ):
            if available >= self._VRAM_THRESHOLDS[size]:
                return size
        return None

    @staticmethod
    def _supports_amp(backend: Backend, gpu: GPUInfo | None) -> bool:
        if backend != Backend.CUDA or gpu is None:
            return False
        return bool(gpu.compute_capability and gpu.compute_capability[0] >= 8)

    def _url_for(self, size: ModelSize) -> str:
        if size is ModelSize.SAM3_1:
            return "https://huggingface.co/facebook/sam3.1"
        if size in (ModelSize.EFFICIENTTAM_S, ModelSize.EFFICIENTTAM_TI):
            return self._HF_URL_TEMPLATE.format(
                checkpoint=self._CHECKPOINT_MAP[size]
            )
        ckpt = self._CHECKPOINT_MAP[size]
        return self._URL_TEMPLATE.format(checkpoint=ckpt)

    @staticmethod
    def _build_user_message(
        model: ModelSize, backend: Backend, amp: bool, gpu: GPUInfo | None
    ) -> str:
        # The model pick is deliberately conservative: staying well under the
        # GPU's limits prevents overheating (black screen) on modest cards.
        parts = [
            f"Modèle recommandé : **{model.value}** (choix conservateur, stabilité GPU)",
            f"Backend : **{backend.value}**",
        ]
        if gpu:
            parts.append(f"GPU détecté : {gpu.name} ({gpu.total_vram_mb} MB VRAM)")
        else:
            parts.append("Aucun GPU détecté — inférence sur CPU (lente)")
        if amp:
            parts.append("Mixed precision (bfloat16) activée")
        return " | ".join(parts)
