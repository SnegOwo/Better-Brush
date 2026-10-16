"""Model selection for the headless engine (no GPU or weights required)."""

from __future__ import annotations

import pytest

from src.engine import model_setup


def _no_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_setup, "custom_model_available", lambda: False)


def _with_custom(
    monkeypatch: pytest.MonkeyPatch, label: str = "Anime AMV v1"
) -> None:
    monkeypatch.setattr(model_setup, "custom_model_available", lambda: True)
    monkeypatch.setattr(model_setup, "custom_model_label", lambda: label)


def test_quality_loads_the_model_the_panel_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Balanced (Base+)" must load Base+, even with a fine-tuned model installed.

    It used to prefer the fine-tuned checkpoint for both ``balanced`` and
    ``best``, so the panel's own label was a lie — and, because the profiler had
    no entry for that tier, the job failed instead.
    """
    _with_custom(monkeypatch)
    assert model_setup.resolve_model_name("fast") == "small"
    assert model_setup.resolve_model_name("balanced") == "base_plus"
    assert model_setup.resolve_model_name("best") == "custom"


def test_best_falls_back_without_a_fine_tuned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_custom(monkeypatch)
    assert model_setup.resolve_model_name("best") == "base_plus"


def test_an_explicit_model_overrides_the_quality_preset() -> None:
    assert model_setup.resolve_model_name("fast", override="large") == "large"


def test_describe_names_the_model_behind_each_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_custom(monkeypatch)
    info = model_setup.describe()

    assert info["custom_model"] is True
    assert info["custom_label"] == "Anime AMV v1"
    assert info["quality_models"] == {
        "fast": "small",
        "balanced": "base_plus",
        "best": "custom",
    }
