"""Profiler: recommend SAM 2 model, backend config and content profile."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
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
    """SAM 2.1 Hiera variants (smaller → faster, less accurate)."""

    LARGE = "sam2.1_hiera_large"
    BASE_PLUS = "sam2.1_hiera_base_plus"
    SMALL = "sam2.1_hiera_small"
    TINY = "sam2.1_hiera_tiny"


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
    user_message: str


class SAM2Profiler:
    """Map hardware capabilities to safe, *conservative* SAM 2 settings.

    Priority is GPU stability over raw speed: a model tier is only
    recommended when the free VRAM leaves ~35 % headroom. This avoids pushing
    modest cards to 100 % utilisation, which can overheat the GPU and crash
    the driver (black screen on laptops/desktops with weak cooling).
    """

    # Minimum VRAM REQUIRED (MB) to run a tier *comfortably* in video
    # propagation mode (weights + activations + SAM 2's per-frame memory
    # bank). Deliberately higher than the raw checkpoint size on disk.
    _VRAM_THRESHOLDS: ClassVar[dict[ModelSize, int]] = {
        ModelSize.TINY: 2_800,
        ModelSize.SMALL: 5_000,
        ModelSize.BASE_PLUS: 8_500,
        ModelSize.LARGE: 13_500,
    }

    # Only recommend a tier when free VRAM × this factor clears the
    # threshold above (keeps ~35 % of VRAM free at runtime).
    _VRAM_SAFETY_FACTOR: float = 0.65

    _CONFIG_MAP: ClassVar[dict[ModelSize, str]] = {
        ModelSize.LARGE: "sam2.1_hiera_l.yaml",
        ModelSize.BASE_PLUS: "sam2.1_hiera_b+.yaml",
        ModelSize.SMALL: "sam2.1_hiera_s.yaml",
        ModelSize.TINY: "sam2.1_hiera_t.yaml",
    }

    _URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{checkpoint}"
    )

    _CHECKPOINT_MAP: ClassVar[dict[ModelSize, str]] = {
        ModelSize.LARGE: "sam2.1_hiera_large.pt",
        ModelSize.BASE_PLUS: "sam2.1_hiera_base_plus.pt",
        ModelSize.SMALL: "sam2.1_hiera_small.pt",
        ModelSize.TINY: "sam2.1_hiera_tiny.pt",
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
        # Free VRAM is the live signal; total VRAM is a fallback when the
        # driver does not report free memory (DirectML, some WSL setups).
        available = gpu.available_vram_mb or gpu.total_vram_mb
        # Conservative headroom: we must stay comfortably below the card's
        # limit. The previous "* 0.8" heuristic recommended the largest tier
        # that *barely* fit -> GPU at 100 % -> overheating -> driver crash
        # (black screen) on modestly-cooled cards.
        effective = int(available * self._VRAM_SAFETY_FACTOR)
        logger.debug(
            "Pick model: %d MB free -> %d MB effective (factor x%.2f)",
            available,
            effective,
            self._VRAM_SAFETY_FACTOR,
        )
        for size in (
            ModelSize.LARGE,
            ModelSize.BASE_PLUS,
            ModelSize.SMALL,
            ModelSize.TINY,
        ):
            if effective >= self._VRAM_THRESHOLDS[size]:
                logger.info(
                    "Recommended %s (threshold %d MB, effective %d MB)",
                    size.value,
                    self._VRAM_THRESHOLDS[size],
                    effective,
                )
                return size
        logger.info("Free VRAM below the smallest tier — falling back to TINY")
        return ModelSize.TINY

    @staticmethod
    def _supports_amp(backend: Backend, gpu: GPUInfo | None) -> bool:
        if backend != Backend.CUDA or gpu is None:
            return False
        return bool(gpu.compute_capability and gpu.compute_capability[0] >= 8)

    def _url_for(self, size: ModelSize) -> str:
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
