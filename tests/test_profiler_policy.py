"""Unit tests for the *conservative* SAM 2 model recommendation policy.

This tests that the profiler never recommends a model that would push the
GPU to its VRAM limit (the old heuristic did, causing overheating / black
screens). No GPU or SAM 2 weights are required.

Run with:
    python -m pytest tests/test_profiler_policy.py -v
or standalone:
    python -m tests.test_profiler_policy
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.detector import Backend, GPUInfo, HardwareProfile
from src.hardware.profiler import ModelSize, SAM2Profiler

_SUT = SAM2Profiler()

# (free VRAM MB, total VRAM MB, expected model) — mirrors the real cases that
# crashed the user's GPU. free ≈ 11 GB on a 12 GB card used to recommend
# base_plus; it must now stay on the much lighter *small* tier.
_CASES: list[tuple[int, int, ModelSize]] = [
    (0, 0, ModelSize.TINY),  # no GPU / unknown VRAM
    (3_000, 3_000, ModelSize.EFFICIENTTAM_S),  # 3 GB card: tiny needs 2.8 GB
    (5_000, 5_000, ModelSize.TINY),  # 5 GB: 3.25 GB effective < 5 GB
    (8_000, 8_000, ModelSize.SMALL),  # 8 GB card -> small (5.2 GB eff.)
    (11_000, 12_288, ModelSize.SMALL),  # RTX 3060 12 GB, idle at launch
    (6_582, 12_288, ModelSize.TINY),  # 3060 while another app uses VRAM
    (14_000, 16_000, ModelSize.BASE_PLUS),  # 16 GB card -> base_plus
    (17_000, 18_000, ModelSize.BASE_PLUS),  # 18 GB card still base_plus
    (24_000, 24_000, ModelSize.LARGE),  # 24 GB card: SAM 3.1 skipped when optional package absent
    (32_000, 32_000, ModelSize.LARGE),  # same fallback with the optional extra
]

# Below the SAM 2.1 Tiny threshold the picker falls back to the lightweight
# EfficientTAM family, which runs on cards that cannot host Tiny at all.
_LIGHT_CASES: list[tuple[int, int, ModelSize]] = [
    (2_400, 3_000, ModelSize.EFFICIENTTAM_TI),  # 1.56 GB effective
    (1_800, 2_400, ModelSize.EFFICIENTTAM_TI),  # 1.17 GB effective
    (1_200, 1_500, ModelSize.EFFICIENTTAM_TI),  # below every threshold
]


def _gpu(total_mb: int, available_mb: int, backend: Backend = Backend.CUDA) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="Test GPU",
        total_vram_mb=total_mb,
        available_vram_mb=available_mb,
        backend=backend,
        compute_capability=(8, 6),
    )


def _profile(*gpus: GPUInfo) -> HardwareProfile:
    gpus = list(gpus)
    best = gpus[0].backend if gpus else Backend.CPU
    return HardwareProfile(
        cpu_name="Test CPU",
        cpu_count_physical=4,
        cpu_count_logical=8,
        ram_total_mb=16_384,
        ram_available_mb=8_192,
        gpus=gpus,
        best_backend=best,
        recommended_device=f"cuda:{gpus[0].index}" if gpus else "cpu",
    )


def _run_cases() -> list[str]:
    failures: list[str] = []
    for free_mb, total_mb, expected in _CASES:
        got = _SUT.recommend(_profile(_gpu(total_mb, free_mb))).model_size
        if got is not expected:
            failures.append(
                f"free={free_mb} MB total={total_mb} MB -> {got.value!r}, "
                f"expected {expected.value!r}"
            )
    return failures


def test_recommendation_stays_below_vram_limit() -> None:
    """Every case must be recoverable (never the tier that barely fits)."""
    assert not _run_cases(), "policy mismatches:\n" + "\n".join(_run_cases())


def test_weak_gpus_fall_back_to_efficienttam() -> None:
    """Cards that cannot host SAM 2.1 Tiny get a lightweight model."""
    failures: list[str] = []
    for free_mb, total_mb, expected in _LIGHT_CASES:
        got = _SUT.recommend(_profile(_gpu(total_mb, free_mb))).model_size
        if got is not expected:
            failures.append(
                f"free={free_mb} MB total={total_mb} MB -> {got.value!r}, "
                f"expected {expected.value!r}"
            )
    assert not failures, "light-model mismatches:\n" + "\n".join(failures)


def test_efficienttam_thresholds_are_below_tiny() -> None:
    """The light tiers must always fit where Tiny does not."""
    thresholds = _SUT._VRAM_THRESHOLDS
    assert thresholds[ModelSize.EFFICIENTTAM_S] < thresholds[ModelSize.TINY]
    assert thresholds[ModelSize.EFFICIENTTAM_TI] < thresholds[ModelSize.EFFICIENTTAM_S]


def test_effective_vram_keeps_headroom() -> None:
    """effective VRAM = free × 0.65, i.e. ~35 % headroom is kept."""
    assert int(11_000 * _SUT._VRAM_SAFETY_FACTOR) == 7_150


def test_cpu_fallback_is_tiny() -> None:
    assert _SUT.recommend(_profile()).model_size is ModelSize.TINY


def test_force_model_is_respected() -> None:
    rec = _SUT.recommend(_profile(_gpu(8_000, 8_000)), force_model=ModelSize.LARGE)
    assert rec.model_size is ModelSize.LARGE


def test_hardware_reports_sam3_eligibility() -> None:
    assert _profile(_gpu(12_000, 12_000)).sam3_1_eligible is False
    assert _profile(_gpu(24_000, 24_000)).sam3_1_eligible is True


def test_sam3_1_can_be_forced_and_recommended(monkeypatch) -> None:
    import src.hardware.profiler as profiler_module

    monkeypatch.setattr(profiler_module, "_sam3_package_available", lambda: True)
    forced = _SUT.recommend(
        _profile(_gpu(8_000, 8_000)), force_model=ModelSize.SAM3_1
    )
    automatic = _SUT.recommend(_profile(_gpu(24_000, 24_000)))
    assert forced.model_size is ModelSize.SAM3_1
    assert automatic.model_size is ModelSize.SAM3_1
    assert "huggingface.co/facebook/sam3.1" in automatic.checkpoint_url


def test_user_message_mentions_conservative_choice() -> None:
    rec = _SUT.recommend(_profile(_gpu(11_000, 12_288)))
    assert "conservateur" in rec.user_message
    assert "stabilité" in rec.user_message


def main() -> int:
    """Standalone runner (no pytest required)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    failures = _run_cases()
    print("SAM 2 recommendation policy checks")
    for free_mb, total_mb, expected in _CASES:
        got = _SUT.recommend(_profile(_gpu(total_mb, free_mb))).model_size
        ok = got is expected
        print(
            f"  [{'OK ' if ok else 'FAIL'}] free={free_mb:>6} MB "
            f"total={total_mb:>6} MB -> {got.value:<24} expected {expected.value}"
        )
    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(f"  - {line}")
        return 1

    checks = [
        (
            "CPU fallback -> tiny",
            _SUT.recommend(_profile()).model_size is ModelSize.TINY,
        ),
        (
            "force_model=large respected",
            _SUT.recommend(
                _profile(_gpu(8_000, 8_000)), force_model=ModelSize.LARGE
            ).model_size
            is ModelSize.LARGE,
        ),
        (
            "user message mentions conservative choice",
            "conservateur"
            in _SUT.recommend(_profile(_gpu(11_000, 12_288))).user_message,
        ),
    ]
    for label, ok in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
        if not ok:
            return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
