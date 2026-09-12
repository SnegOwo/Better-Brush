"""Standalone hardware detection test (no GUI, no SAM 2 weights needed)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running from repo root without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware import (
    ContentProfile,
    HardwareDetector,
    SAM2Profiler,
)
from src.utils.logging_config import configure_logging


def main() -> int:
    # Windows consoles may use legacy codepages — force UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging(level=logging.DEBUG)
    logger = logging.getLogger("test_hardware")

    print("=" * 60)
    print("Better Brush — Hardware Detection & Profiling Test")
    print("=" * 60)

    # 1. Detect hardware
    logger.info("Step 1/3: Detecting hardware...")
    detector = HardwareDetector()
    profile = detector.detect()

    print("\n--- Hardware Profile ---")
    print(f"OS            : {profile.os_name}")
    print(f"Python        : {profile.python_version}")
    print(f"PyTorch       : {profile.torch_version}")
    print(f"CPU           : {profile.cpu_name}")
    print(
        f"CPU cores     : {profile.cpu_count_physical} physical / {profile.cpu_count_logical} logical"
    )
    print(f"RAM total     : {profile.ram_total_mb:,} MB")
    print(f"RAM available : {profile.ram_available_mb:,} MB")
    print(f"Best backend  : {profile.best_backend.value}")
    print(f"Device str    : {profile.recommended_device}")

    if profile.gpus:
        print(f"\nDetected GPUs ({len(profile.gpus)}):")
        for gpu in profile.gpus:
            cc = f" (CC {gpu.compute_capability})" if gpu.compute_capability else ""
            print(
                f"  [{gpu.index}] {gpu.name}{cc} — "
                f"{gpu.total_vram_mb:,} MB total / {gpu.available_vram_mb:,} MB free "
                f"[{gpu.backend.value}]"
            )
    else:
        print("\nNo GPU detected — CPU-only inference will be very slow.")

    # 2. Profile recommendation for each content type
    logger.info("Step 2/3: Building SAM 2 recommendations...")
    profiler = SAM2Profiler()

    print("\n--- SAM 2 Recommendations ---")
    for content in ContentProfile:
        rec = profiler.recommend(profile, content=content)
        print(f"\nProfile : {content.value.upper()}")
        print(f"  Model          : {rec.model_size.value}")
        print(f"  Config         : {rec.model_config}")
        print(f"  Checkpoint     : {rec.checkpoint_url}")
        print(f"  Device         : {rec.device}")
        print(f"  Mixed precision: {rec.mixed_precision}")
        print(f"  Max frames mem : {rec.max_frames_in_memory}")
        print(f"  Mask tokens    : {rec.num_mask_tokens}")
        print(f"  IoU thresh     : {rec.pred_iou_thresh}")
        print(f"  Stability offs : {rec.stability_score_offset}")
        print(f"  Message        : {rec.user_message}")

    # 3. JSON dump for programmatic consumption
    logger.info("Step 3/3: Exporting JSON summary...")
    summary = {
        "hardware": {
            "cpu": profile.cpu_name,
            "cpu_cores": {
                "physical": profile.cpu_count_physical,
                "logical": profile.cpu_count_logical,
            },
            "ram_mb": {
                "total": profile.ram_total_mb,
                "available": profile.ram_available_mb,
            },
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_mb": {
                        "total": g.total_vram_mb,
                        "available": g.available_vram_mb,
                    },
                    "backend": g.backend.value,
                    "compute_capability": g.compute_capability,
                }
                for g in profile.gpus
            ],
            "best_backend": profile.best_backend.value,
            "device": profile.recommended_device,
        },
        "recommendations": {
            cp.value: {
                "model": profiler.recommend(profile, content=cp).model_size.value,
                "mixed_precision": profiler.recommend(
                    profile, content=cp
                ).mixed_precision,
                "device": profiler.recommend(profile, content=cp).device,
            }
            for cp in ContentProfile
        },
    }
    json_path = Path("hardware_summary.json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nJSON summary written to: {json_path.resolve()}")

    print("\n" + "=" * 60)
    print("Hardware test completed successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
