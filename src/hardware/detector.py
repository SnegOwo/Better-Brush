"""Hardware detection: GPU, VRAM, CPU, RAM and available backends."""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from enum import Enum

import psutil
import torch

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Supported compute backends ordered by preference."""

    CUDA = "cuda"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class GPUInfo:
    """Metadata for a single GPU."""

    index: int
    name: str
    total_vram_mb: int
    available_vram_mb: int
    backend: Backend
    compute_capability: tuple[int, int] | None = None
    driver_version: str | None = None


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of the machine capabilities."""

    cpu_name: str
    cpu_count_physical: int
    cpu_count_logical: int
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    os_name: str = field(default_factory=lambda: platform.system())
    python_version: str = field(default_factory=lambda: platform.python_version())
    torch_version: str = field(default_factory=lambda: torch.__version__)
    best_backend: Backend = Backend.CPU
    recommended_device: str = "cpu"

    @property
    def total_vram_mb(self) -> int:
        """Sum VRAM across all detected GPUs."""
        return sum(g.total_vram_mb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def has_cuda(self) -> bool:
        return any(g.backend == Backend.CUDA for g in self.gpus)

    @property
    def sam3_1_eligible(self) -> bool:
        """Whether detected hardware clears the conservative SAM 3.1 VRAM floor."""
        return any(
            gpu.backend == Backend.CUDA
            and (gpu.available_vram_mb or gpu.total_vram_mb) >= 15_000
            for gpu in self.gpus
        )


class HardwareDetector:
    """Detect hardware capabilities and available deep-learning backends."""

    def detect(self) -> HardwareProfile:
        """Run full detection and return an immutable profile."""
        logger.info("Starting hardware detection...")
        cpu_name, cpu_phys, cpu_log = self._detect_cpu()
        ram_total, ram_avail = self._detect_ram()
        gpus = self._detect_gpus()
        best_backend, recommended_device = self._pick_best_backend(gpus)

        profile = HardwareProfile(
            cpu_name=cpu_name,
            cpu_count_physical=cpu_phys,
            cpu_count_logical=cpu_log,
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
            gpus=gpus,
            best_backend=best_backend,
            recommended_device=recommended_device,
        )
        logger.info("Hardware detection complete: %s", profile)
        return profile

    # ------------------------------------------------------------------ #
    # CPU / RAM
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_cpu() -> tuple[str, int, int]:
        name = platform.processor() or "Unknown CPU"
        if not name or name.strip() == "":
            # Fallback for some Windows builds
            name = os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU")
        phys = psutil.cpu_count(logical=False) or 1
        log = psutil.cpu_count(logical=True) or 1
        return name, phys, log

    @staticmethod
    def _detect_ram() -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))

    # ------------------------------------------------------------------ #
    # GPUs
    # ------------------------------------------------------------------ #
    def _detect_gpus(self) -> list[GPUInfo]:
        gpus: list[GPUInfo] = []
        gpus.extend(self._detect_cuda_gpus())
        gpus.extend(self._detect_directml_gpus())
        # Deduplicate by index — CUDA always wins if both report same index
        seen = set()
        unique: list[GPUInfo] = []
        for g in gpus:
            if g.index not in seen:
                seen.add(g.index)
                unique.append(g)
        return unique

    def _detect_cuda_gpus(self) -> list[GPUInfo]:
        """Enumerate NVIDIA GPUs via PyTorch + NVML for accurate VRAM."""
        if not torch.cuda.is_available():
            logger.debug("CUDA not available")
            return []

        gpus: list[GPUInfo] = []
        nvml_available = False
        try:
            import pynvml  # nvidia-ml-py

            pynvml.nvmlInit()
            nvml_available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("NVML init failed (%s). Falling back to torch.cuda.", exc)

        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cc = torch.cuda.get_device_capability(i)

            if nvml_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = int(mem.total // (1024 * 1024))
                    avail_mb = int(mem.free // (1024 * 1024))
                    drv = pynvml.nvmlSystemGetDriverVersion()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("NVML query failed for GPU %d: %s", i, exc)
                    total_mb, avail_mb, drv = self._fallback_vram(i)
            else:
                total_mb, avail_mb, drv = self._fallback_vram(i)

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_vram_mb=total_mb,
                    available_vram_mb=avail_mb,
                    backend=Backend.CUDA,
                    compute_capability=cc,
                    driver_version=str(drv) if drv else None,
                )
            )

        if nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: S110, BLE001
                pass

        return gpus

    @staticmethod
    def _fallback_vram(device_index: int) -> tuple[int, int, str | None]:
        """Use torch.cuda properties when NVML is unreachable."""
        props = torch.cuda.get_device_properties(device_index)
        total_mb = int(props.total_memory // (1024 * 1024))
        # torch does not expose free memory directly without caching allocator
        reserved = torch.cuda.memory_reserved(device_index)
        allocated = torch.cuda.memory_allocated(device_index)
        free = props.total_memory - reserved - allocated
        avail_mb = max(0, int(free // (1024 * 1024)))
        return total_mb, avail_mb, None

    def _detect_directml_gpus(self) -> list[GPUInfo]:
        """Detect DirectML-capable adapters (Windows, AMD/Intel)."""
        try:
            import torch_directml  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("torch_directml not installed")
            return []

        try:
            device_count = torch_directml.device_count()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DirectML device_count failed: %s", exc)
            return []

        gpus: list[GPUInfo] = []
        for i in range(device_count):
            try:
                # DirectML does not expose VRAM easily; we rely on a heuristic
                # or user override later.
                gpus.append(
                    GPUInfo(
                        index=i + 100,  # offset to avoid collision with CUDA indices
                        name=f"DirectML Adapter {i}",
                        total_vram_mb=0,  # unknown without DXGI/WMI
                        available_vram_mb=0,
                        backend=Backend.DIRECTML,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("DirectML adapter %d detection failed: %s", i, exc)
        return gpus

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_best_backend(gpus: list[GPUInfo]) -> tuple[Backend, str]:
        cuda_gpus = [g for g in gpus if g.backend == Backend.CUDA]
        dml_gpus = [g for g in gpus if g.backend == Backend.DIRECTML]

        if cuda_gpus:
            return Backend.CUDA, f"cuda:{cuda_gpus[0].index}"
        if dml_gpus:
            return Backend.DIRECTML, f"directml:{dml_gpus[0].index - 100}"
        return Backend.CPU, "cpu"
