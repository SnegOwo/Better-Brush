"""Tests for local QC image analysis and the persistent tip database."""

from __future__ import annotations

import numpy as np

from src.processing.matte_qc import analyze_matte
from src.processing.qc_coach import QCTipsDatabase, analyze_qc_image


def test_analysis_recommends_hair_detail_settings() -> None:
    frame = np.full((48, 64, 3), 32, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=bool)
    mask[8:40, 16:48] = True
    report = analyze_matte(frame, mask)

    analysis = analyze_qc_image(frame, mask, report)

    assert analysis.summary
    assert analysis.top_tip is not None
    assert analysis.top_tip.key in {"good_matte", "hair_smooth", "holes", "jagged_flicker", "spill"}


def test_hair_complaint_returns_direct_actionable_answer(tmp_path) -> None:
    database = QCTipsDatabase(tmp_path / "qc-tips.json")
    answer = database.ask("hair is way too smooth and not sharp")

    assert answer.matched
    assert "Detail preservation" in answer.answer
    assert answer.settings["edge_smoothness"] <= 3
    assert answer.settings["detail_preservation"] >= 0.85
    assert database.questions


def test_too_sharp_complaint_uses_the_opposite_direction(tmp_path) -> None:
    database = QCTipsDatabase(tmp_path / "qc-tips.json")
    answer = database.ask("it's too sharp")

    assert answer.tip.key == "hair_too_sharp"
    assert "Raise Edge smoothness" in answer.answer
    assert answer.settings["edge_smoothness"] >= 4
    assert answer.settings["temporal_smoothing"] is True
    assert database.ask("the hair is too sharp").tip.key == "hair_too_sharp"


def test_custom_tip_persists_and_is_preferred(tmp_path) -> None:
    path = tmp_path / "qc-tips.json"
    first = QCTipsDatabase(path)
    first.add(
        "my studio's fringe problem",
        "Use a clean plate and exclude the light stand.",
        key="studio_fringe",
        keywords=("studio", "fringe", "problem"),
    )
    reloaded = QCTipsDatabase(path)
    answer = reloaded.ask("studio fringe problem")

    assert answer.source == "custom"
    assert "clean plate" in answer.answer
    assert reloaded.questions
