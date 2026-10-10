"""Checkpoint download and verification."""

from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.hardware.profiler import ModelSize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CustomModel:
    """A user-supplied fine-tuned SAM 2 checkpoint."""

    checkpoint: Path
    config: str
    label: str

# SAM 2.1 official Meta URLs (092824 release)
_BASE_CKPT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
_BASE_CFG = (
    "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1"
)

_CHECKPOINT_URLS: dict[ModelSize, str] = {
    ModelSize.TINY: f"{_BASE_CKPT}/sam2.1_hiera_tiny.pt",
    ModelSize.SMALL: f"{_BASE_CKPT}/sam2.1_hiera_small.pt",
    ModelSize.BASE_PLUS: f"{_BASE_CKPT}/sam2.1_hiera_base_plus.pt",
    ModelSize.LARGE: f"{_BASE_CKPT}/sam2.1_hiera_large.pt",
    ModelSize.EFFICIENTTAM_TI: (
        "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
        "/efficienttam_ti_512x512.pt"
    ),
    ModelSize.EFFICIENTTAM_S: (
        "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
        "/efficienttam_s_512x512.pt"
    ),
}

_CONFIG_URLS: dict[ModelSize, str] = {
    ModelSize.TINY: f"{_BASE_CFG}/sam2.1_hiera_t.yaml",
    ModelSize.SMALL: f"{_BASE_CFG}/sam2.1_hiera_s.yaml",
    ModelSize.BASE_PLUS: f"{_BASE_CFG}/sam2.1_hiera_b%2B.yaml",
    ModelSize.LARGE: f"{_BASE_CFG}/sam2.1_hiera_l.yaml",
}

_SAM3_REPO = "facebook/sam3.1"
_SAM3_CHECKPOINT = "sam3.1_multiplex.pt"
_SAM3_CONFIG = "config.json"

# Config file names shipped inside the vendored EfficientTAM package.
_EFFICIENTTAM_CONFIGS: dict[ModelSize, str] = {
    ModelSize.EFFICIENTTAM_S: "efficienttam_s_512x512.yaml",
    ModelSize.EFFICIENTTAM_TI: "efficienttam_ti_512x512.yaml",
}

# A fine-tuned checkpoint lives in checkpoints/custom/ with a sidecar that
# says which base architecture and display name to use, e.g.
#   {"checkpoint": "anime_amv_v1.pt", "config": "sam2.1_hiera_b+.yaml",
#    "label": "Anime AMV v1"}
_CUSTOM_DIR_NAME = "custom"
_CUSTOM_SIDECAR = "sam2_custom.json"


class CheckpointManager:
    """Handle downloading and caching of SAM 2 weights and configs."""

    def __init__(self, checkpoints_dir: Path | None = None) -> None:
        # src.core.constants resolves the storage root for both the source
        # tree and the frozen bundle; the previous relative walk landed one
        # folder above the repo when the app ran from a nested path.
        from src.core.constants import CHECKPOINTS_DIR

        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.checkpoints_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager: %s", self.checkpoints_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_checkpoint_path(self, size: ModelSize) -> Path:
        """Return local path to checkpoint, downloading if necessary."""
        if size is ModelSize.SAM3_1:
            local = self.checkpoints_dir / _SAM3_CHECKPOINT
            if not local.exists():
                self._download_hf(_SAM3_REPO, _SAM3_CHECKPOINT, local)
            return local
        if size is ModelSize.CUSTOM:
            return self.custom_model().checkpoint
        if size in _CHECKPOINT_URLS:
            filename = _CHECKPOINT_URLS[size].split("/")[-1]
            local = self.checkpoints_dir / filename
            if not local.exists():
                self._download(
                    _CHECKPOINT_URLS[size], local, desc=f"Checkpoint {size.value}"
                )
            return local
        raise RuntimeError(f"No checkpoint source for model tier: {size.value}")

    def get_config_path(self, size: ModelSize) -> Path:
        """Return local path to config, downloading if necessary.

        EfficientTAM configs ship inside the vendored package
        (``src/vendor/efficienttam/configs/``), so nothing is downloaded for
        those variants.
        """
        if size is ModelSize.SAM3_1:
            local = self.configs_dir / "sam3.1_config.json"
            if not local.exists():
                self._download_hf(_SAM3_REPO, _SAM3_CONFIG, local)
            return local
        if size in (ModelSize.EFFICIENTTAM_S, ModelSize.EFFICIENTTAM_TI):
            vendored = (
                Path(__file__).resolve().parents[1]
                / "vendor"
                / "efficienttam"
                / "configs"
                / "efficienttam"
                / _EFFICIENTTAM_CONFIGS[size]
            )
            if not vendored.is_file():
                raise RuntimeError(
                    f"Vendored EfficientTAM config missing: {vendored}"
                )
            return vendored
        if size is ModelSize.CUSTOM:
            return Path(self.custom_model().config)
        filename = _CONFIG_URLS[size].split("/")[-1]
        local = self.configs_dir / filename
        if not local.exists():
            self._download(_CONFIG_URLS[size], local, desc=f"Config {size.value}")
        return local

    def custom_dir(self) -> Path:
        """Directory holding user-supplied fine-tuned checkpoints."""
        return self.checkpoints_dir / _CUSTOM_DIR_NAME

    def custom_model(self) -> CustomModel:
        """Read the ``checkpoints/custom`` sidecar (raises when absent)."""
        sidecar = self.custom_dir() / _CUSTOM_SIDECAR
        if not sidecar.is_file():
            raise RuntimeError(
                "No fine-tuned model installed. Drop a checkpoint into "
                f"{self.custom_dir()} together with {_CUSTOM_SIDECAR} "
                "(see docs/FINE-TUNING.md)."
            )
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable {_CUSTOM_SIDECAR}: {exc}") from exc
        checkpoint = str(data.get("checkpoint", "")).strip()
        if not checkpoint:
            raise RuntimeError(
                f"{_CUSTOM_SIDECAR} is missing the 'checkpoint' file name."
            )
        path = self.custom_dir() / checkpoint
        if not path.is_file():
            raise RuntimeError(
                f"Fine-tuned checkpoint not found: {path}. "
                f"Expected it inside {self.custom_dir()}."
            )
        return CustomModel(
            checkpoint=path,
            config=str(data.get("config", "sam2.1_hiera_b+.yaml")),
            label=str(data.get("label") or f"Custom model ({checkpoint})"),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for SAM 3.1 downloads"
            ) from exc
        logger.info("Downloading %s from Hugging Face repo %s ...", filename, repo_id)
        try:
            token = os.environ.get("HF_TOKEN") or None
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from {repo_id}: {exc}. "
                "The SAM 3.1 repository is gated; request access and run "
                "'hf auth login' (or set HF_TOKEN)."
            ) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + ".part")
        shutil.copy2(cached, temporary)
        temporary.replace(dest)
        logger.info("Saved to %s (%d bytes)", dest, dest.stat().st_size)

    @staticmethod
    def _download(url: str, dest: Path, desc: str = "file") -> None:
        logger.info("Downloading %s from %s ...", desc, url)
        dest_part = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, str(dest_part))
            dest_part.rename(dest)
            logger.info("Saved to %s (%d bytes)", dest, dest.stat().st_size)
        except Exception:
            if dest_part.exists():
                dest_part.unlink()
            raise
