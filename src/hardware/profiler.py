"""Profiler: recommend SAM 2 model, backend config and content profile."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.hardware.detector import Backend, GPUInfo, HardwareProfile

logger = logging.getLogger(__name__)


class ContentProfile(Enum):
    """Semantic content categories influencing SAM 2 parameters."""

    ANIME = "anime"
    HUMAN = "human"
    GENERAL = "general"


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
    """Map hardware capabilities to optimal SAM 2 settings."""

    _VRAM_THRESHOLDS: dict[ModelSize, int] = {
        ModelSize.TINY: 4_096,
        ModelSize.SMALL: 6_144,
        ModelSize.BASE_PLUS: 8_192,
        ModelSize.LARGE: 12_288,
    }

    _CONFIG_MAP: dict[ModelSize, str] = {
        ModelSize.LARGE: "sam2.1_hiera_l.yaml",
        ModelSize.BASE_PLUS: "sam2.1_hiera_b+.yaml",
        ModelSize.SMALL: "sam2.1_hiera_s.yaml",
        ModelSize.TINY: "sam2.1_hiera_t.yaml",
    }

    _URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{checkpoint}"
    )

    _CHECKPOINT_MAP: dict[ModelSize, str] = {
        ModelSize.LARGE: "sam2.1_hiera_large.pt",
        ModelSize.BASE_PLUS: "sam2.1_hiera_base_plus.pt",
        ModelSize.SMALL: "sam2.1_hiera_small.pt",
        ModelSize.TINY: "sam2.1_hiera_tiny.pt",
    }

    _PROFILE_PRESETS: dict[ContentProfile, dict[str, float | int]] = {
        ContentProfile.ANIME: {
            "num_mask_tokens": 3,
            "pred_iou_thresh": 0.82,
            "stability_score_offset": 0.8,
            "max_frames_in_memory": 64,
        },
        ContentProfile.HUMAN: {
            "num_mask_tokens": 3,
            "pred_iou_thresh": 0.88,
            "stability_score_offset": 1.0,
            "max_frames_in_memory": 48,
        },
        ContentProfile.GENERAL: {
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
        force_model: Optional[ModelSize] = None,
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
    def _pick_target_gpu(profile: HardwareProfile) -> Optional[GPUInfo]:
        cuda = [g for g in profile.gpus if g.backend == Backend.CUDA]
        if cuda:
            return max(cuda, key=lambda g: g.total_vram_mb)
        dml = [g for g in profile.gpus if g.backend == Backend.DIRECTML]
        if dml:
            return dml[0]
        return None

    def _pick_model(self, gpu: Optional[GPUInfo]) -> ModelSize:
        if gpu is None or gpu.total_vram_mb == 0:
            logger.warning("No GPU detected — falling back to TINY on CPU")
            return ModelSize.TINY
        vram = gpu.available_vram_mb
        effective = int(vram * 0.8)
        for size in (ModelSize.LARGE, ModelSize.BASE_PLUS, ModelSize.SMALL, ModelSize.TINY):
            if effective >= self._VRAM_THRESHOLDS[size]:
                return size
        return ModelSize.TINY

    @staticmethod
    def _supports_amp(backend: Backend, gpu: Optional[GPUInfo]) -> bool:
        if backend != Backend.CUDA or gpu is None:
            return False
        if gpu.compute_capability and gpu.compute_capability[0] >= 8:
            return True
        return False

    def _url_for(self, size: ModelSize) -> str:
        ckpt = self._CHECKPOINT_MAP[size]
        return self._URL_TEMPLATE.format(checkpoint=ckpt)

    @staticmethod
    def _build_user_message(
        model: ModelSize, backend: Backend, amp: bool, gpu: Optional[GPUInfo]
    ) -> str:
        parts = [f"Modèle recommandé : **{model.value}**", f"Backend : **{backend.value}**"]
        if gpu:
            parts.append(f"GPU détecté : {gpu.name} ({gpu.total_vram_mb} MB VRAM)")
        else:
            parts.append("Aucun GPU détecté — inférence sur CPU (lente)")
        if amp:
            parts.append("Mixed precision (bfloat16) activée")
        return " | ".join(parts)
