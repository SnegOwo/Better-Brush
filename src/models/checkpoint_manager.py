"""Checkpoint download and verification."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

from src.hardware.profiler import ModelSize

logger = logging.getLogger(__name__)

# SAM 2.1 official Meta URLs (092824 release)
_BASE_CKPT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
_BASE_CFG = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1"

_CHECKPOINT_URLS: dict[ModelSize, str] = {
    ModelSize.TINY: f"{_BASE_CKPT}/sam2.1_hiera_tiny.pt",
    ModelSize.SMALL: f"{_BASE_CKPT}/sam2.1_hiera_small.pt",
    ModelSize.BASE_PLUS: f"{_BASE_CKPT}/sam2.1_hiera_base_plus.pt",
    ModelSize.LARGE: f"{_BASE_CKPT}/sam2.1_hiera_large.pt",
}

_CONFIG_URLS: dict[ModelSize, str] = {
    ModelSize.TINY: f"{_BASE_CFG}/sam2.1_hiera_t.yaml",
    ModelSize.SMALL: f"{_BASE_CFG}/sam2.1_hiera_s.yaml",
    ModelSize.BASE_PLUS: f"{_BASE_CFG}/sam2.1_hiera_b%2B.yaml",
    ModelSize.LARGE: f"{_BASE_CFG}/sam2.1_hiera_l.yaml",
}


class CheckpointManager:
    """Handle downloading and caching of SAM 2 weights and configs."""

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.checkpoints_dir = checkpoints_dir or Path(__file__).resolve().parents[3] / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.checkpoints_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager: %s", self.checkpoints_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_checkpoint_path(self, size: ModelSize) -> Path:
        """Return local path to checkpoint .pt, downloading if necessary."""
        filename = _CHECKPOINT_URLS[size].split("/")[-1]
        local = self.checkpoints_dir / filename
        if not local.exists():
            self._download(_CHECKPOINT_URLS[size], local, desc=f"Checkpoint {size.value}")
        return local

    def get_config_path(self, size: ModelSize) -> Path:
        """Return local path to config .yaml, downloading if necessary."""
        filename = _CONFIG_URLS[size].split("/")[-1]
        local = self.configs_dir / filename
        if not local.exists():
            self._download(_CONFIG_URLS[size], local, desc=f"Config {size.value}")
        return local

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
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
