"""Local QC guidance, complaint lookup, and persistent tip storage."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.processing.matte_qc import MatteQCReport, analyze_matte

logger = logging.getLogger(__name__)

_TIP_VERSION = 1
_MAX_STORED_QUESTIONS = 100
_WORD_RE = re.compile(r"[a-z0-9]+")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tokens(value: str) -> set[str]:
    return set(_WORD_RE.findall(str(value).lower()))


@dataclass(frozen=True)
class QCTip:
    """A question/answer pair with optional profile recommendations."""

    key: str
    title: str
    answer: str
    keywords: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    priority: int = 50

    def matches(self, question: str) -> float:
        """Return a simple relevance score for a user question."""
        question_tokens = _tokens(question)
        if not question_tokens:
            return 0.0
        keyword_tokens = set()
        for value in (*self.keywords, *self.variants):
            keyword_tokens.update(_tokens(value))
        overlap = len(question_tokens & keyword_tokens)
        normalized = str(question).lower().strip()
        phrase_bonus = 0
        for variant in self.variants:
            if variant.lower() in normalized:
                phrase_bonus = 3
        return overlap + phrase_bonus

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "key": self.key,
            "title": self.title,
            "answer": self.answer,
            "keywords": list(self.keywords),
            "variants": list(self.variants),
            "settings": dict(self.settings),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCTip:
        """Build a tip from persisted data."""
        return cls(
            key=str(data.get("key", "custom")),
            title=str(data.get("title", "Custom tip")),
            answer=str(data.get("answer", "")),
            keywords=tuple(str(value) for value in data.get("keywords", ())),
            variants=tuple(str(value) for value in data.get("variants", ())),
            settings=dict(data.get("settings") or {}),
            priority=int(data.get("priority", 50)),
        )


_BUILTIN_TIPS: tuple[QCTip, ...] = (
    QCTip(
        key="hair_smooth",
        title="Hair is too smooth",
        answer=(
            "Lower Edge smoothness to 1–3, raise Detail preservation to 85–100%, and turn "
            "temporal smoothing off for close-ups. Re-open the QC inspector to check the "
            "strands; keep the raw matte when fine hair is being lost."
        ),
        keywords=("hair", "smooth", "smoothness", "sharp", "soft", "blur", "strand"),
        variants=(
            "hair is way too smooth and not sharp",
            "hair looks blurry",
            "fine hair disappeared",
        ),
        settings={
            "edge_smoothness": 2,
            "detail_preservation": 0.95,
            "temporal_smoothing": False,
        },
        priority=95,
    ),
    QCTip(
        key="hair_too_sharp",
        title="Hair or the edge is too sharp",
        answer=(
            "Raise Edge smoothness to 4–6, lower Detail preservation to 50–70%, and enable "
            "temporal smoothing. Inspect the Cutout view at high zoom; hard, aliased edges usually "
            "need a softer boundary, not less temporal cleanup."
        ),
        keywords=("sharp", "jagged", "hard", "aliased", "harsh", "overly", "crunchy"),
        variants=(
            "it's too sharp",
            "it is too sharp",
            "the hair is too sharp",
            "the edge is too sharp",
        ),
        settings={
            "edge_smoothness": 5,
            "detail_preservation": 0.6,
            "temporal_smoothing": True,
            "temporal_window": 2,
        },
        priority=100,
    ),
    QCTip(
        key="jagged_flicker",
        title="The edge is jagged or flickers",
        answer=(
            "Raise temporal smoothing for this shot, use a two-frame window, and set Edge "
            "smoothness around 4–6. Reduce Detail preservation slightly if the correction is "
            "eating thin hair."
        ),
        keywords=("jagged", "flicker", "flickering", "shimmer", "rough", "unstable", "wobble"),
        variants=(
            "the mask flickers between frames",
            "the edge is too rough",
        ),
        settings={
            "edge_smoothness": 5,
            "detail_preservation": 0.6,
            "temporal_smoothing": True,
            "temporal_window": 2,
        },
        priority=90,
    ),
    QCTip(
        key="holes",
        title="The matte has holes",
        answer=(
            "Fill the hole with the Brush tool, then re-run Segment Frame with a positive point "
            "inside the subject. If holes repeat across a shot, add exclude points on the "
            "background and propagate a fresh keyframe."
        ),
        keywords=("hole", "holes", "gap", "gaps", "missing", "interior"),
        variants=("there are holes inside the mask", "the mask has missing pixels"),
        settings={},
        priority=100,
    ),
    QCTip(
        key="spill",
        title="Background spill or a colored fringe",
        answer=(
            "Lower Edge smoothness, raise Detail preservation, and inspect the Cutout view at "
            "high zoom. Background spill is a source-edge problem: exclude the background with "
            "a point, then re-segment the boundary instead of exporting the spill as alpha."
        ),
        keywords=("spill", "fringe", "halo", "green", "background", "chroma"),
        variants=("there is a green fringe", "background color leaked into the matte"),
        settings={"edge_smoothness": 2, "detail_preservation": 0.9},
        priority=85,
    ),
    QCTip(
        key="mask_size",
        title="The matte is too large or too small",
        answer=(
            "Add an exclude point just outside the subject for a large matte, or an include point "
            "on the missing region for a small matte. Re-run Segment Frame, then use the Brush "
            "only for small local corrections."
        ),
        keywords=("large", "small", "big", "tight", "wide", "shrink", "grow", "size"),
        variants=("the mask is too big", "the cutout is too small"),
        settings={},
        priority=70,
    ),
    QCTip(
        key="good_matte",
        title="No obvious QC issues",
        answer=(
            "No strong QC evidence was found on this frame. Keep the current profile, inspect a "
            "few neighbouring frames for flicker, and move on to the next shot."
        ),
        keywords=("good", "fine", "ok", "clean", "ready"),
        variants=("the matte looks good",),
        settings={},
        priority=10,
    ),
)


@dataclass(frozen=True)
class QCAnswer:
    """A direct answer returned by the tip database."""

    question: str
    answer: str
    tip: QCTip
    matched: bool
    settings: dict[str, Any] = field(default_factory=dict)
    source: str = "builtin"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "tip": self.tip.key,
            "matched": self.matched,
            "settings": dict(self.settings),
            "source": self.source,
        }


@dataclass(frozen=True)
class QCAnalysis:
    """Automated image/matte guidance for the inspector."""

    summary: str
    tips: tuple[QCTip, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    frame_stats: dict[str, float] = field(default_factory=dict)

    @property
    def top_tip(self) -> QCTip | None:
        """Return the highest-priority tip."""
        return self.tips[0] if self.tips else None

    def answers(self) -> tuple[str, ...]:
        """Return human-readable tip messages."""
        return tuple(f"{tip.title}: {tip.answer}" for tip in self.tips)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "summary": self.summary,
            "tips": [tip.to_dict() for tip in self.tips],
            "settings": dict(self.settings),
            "frame_stats": dict(self.frame_stats),
        }


def _frame_statistics(frame_rgb: np.ndarray | None, mask: np.ndarray | None) -> dict[str, float]:
    if frame_rgb is None:
        return {}
    try:
        frame = np.asarray(frame_rgb)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return {}
        height, width = frame.shape[:2]
        gray = frame[:, :, :3].astype(np.float32).mean(axis=2)
        gradient_y, gradient_x = np.gradient(gray) if height > 1 and width > 1 else (gray * 0, gray * 0)
        edge_density = float((np.hypot(gradient_x, gradient_y) > 12.0).mean())
        coverage = float(np.asarray(mask, dtype=bool).mean()) if mask is not None else 0.0
        return {
            "width": float(width),
            "height": float(height),
            "coverage_percent": coverage * 100.0,
            "edge_density": edge_density * 100.0,
            "mean_luma": float(gray.mean()),
        }
    except (TypeError, ValueError, IndexError, FloatingPointError):
        return {}


def _tips_for_report(report: MatteQCReport) -> list[QCTip]:
    by_key = {tip.key: tip for tip in _BUILTIN_TIPS}
    tips: list[QCTip] = []
    if not report.valid:
        tips.append(
            QCTip(
                key="invalid_input",
                title="Frame and mask need attention",
                answer=(
                    "The frame and matte shapes do not agree. Re-open the current frame, make a "
                    "fresh mask, and try the inspector again."
                ),
                priority=100,
            )
        )
    if report.empty:
        tips.append(
            QCTip(
                key="empty_mask",
                title="The matte is empty",
                answer=(
                    "No foreground pixels were found. Place an include point on the subject and "
                    "press Segment Frame before inspecting the edge."
                ),
                priority=100,
            )
        )
    issues = (
        ("holes", 100),
        ("edge_flicker", 90),
        ("background_spill", 85),
        ("hair_detail", 95),
    )
    for key, fallback_priority in issues:
        issue = report.issue(key)
        if issue.count <= 0 or not issue.warning:
            continue
        base = by_key.get({"holes": "holes", "edge_flicker": "jagged_flicker", "background_spill": "spill", "hair_detail": "hair_smooth"}[key])
        if base is not None:
            tips.append(
                QCTip(
                    key=base.key,
                    title=base.title,
                    answer=base.answer,
                    keywords=base.keywords,
                    variants=base.variants,
                    settings=base.settings,
                    priority=max(fallback_priority, base.priority),
                )
            )
    if not tips:
        good = by_key["good_matte"]
        tips.append(good)
    return sorted(tips, key=lambda tip: (-tip.priority, tip.title))


def _merged_settings(tips: tuple[QCTip, ...], profile: Any | None) -> dict[str, Any]:
    del profile
    settings: dict[str, Any] = {}
    for tip in tips:
        settings.update(tip.settings)
    return settings


def analyze_qc_image(
    frame_rgb: np.ndarray | None,
    mask: np.ndarray | None,
    report: MatteQCReport | None = None,
    previous_mask: np.ndarray | None = None,
    profile: Any | None = None,
) -> QCAnalysis:
    """Analyze a frame/matte and return prioritized, actionable guidance."""
    active_report = report or analyze_matte(frame_rgb, mask, previous_mask)
    tips = tuple(_tips_for_report(active_report))
    stats = _frame_statistics(frame_rgb, mask)
    coverage = stats.get("coverage_percent", 0.0)
    if stats:
        summary = (
            f"{int(stats.get('width', 0))}×{int(stats.get('height', 0))} · "
            f"{coverage:.0f}% coverage · {stats.get('edge_density', 0.0):.0f}% detail"
        )
    else:
        summary = active_report.message or "No frame statistics available"
    return QCAnalysis(
        summary=summary,
        tips=tips,
        settings=_merged_settings(tips, profile),
        frame_stats=stats,
    )


class QCTipsDatabase:
    """Persistent question/answer storage for matte QC coaching."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            from src.core.project import ProjectManager

            path = ProjectManager.settings_dir() / "qc_tips.json"
        self.path = Path(path)
        self._custom: list[QCTip] = []
        self._questions: list[dict[str, Any]] = []
        self._load()

    @property
    def custom_tips(self) -> tuple[QCTip, ...]:
        """Return user-added tips."""
        return tuple(self._custom)

    @property
    def questions(self) -> tuple[dict[str, Any], ...]:
        """Return recently asked questions."""
        return tuple(self._questions)

    def add(
        self,
        question: str,
        answer: str,
        *,
        key: str = "custom",
        keywords: tuple[str, ...] = (),
        variants: tuple[str, ...] = (),
        settings: dict[str, Any] | None = None,
    ) -> QCTip:
        """Add or replace a user tip and persist it."""
        tip = QCTip(
            key=str(key or "custom"),
            title=str(key or "Custom tip").replace("_", " ").title(),
            answer=str(answer),
            keywords=tuple(keywords) or tuple(_tokens(question)),
            variants=tuple(variants),
            settings=dict(settings or {}),
            priority=80,
        )
        self._custom = [existing for existing in self._custom if existing.key != tip.key]
        self._custom.append(tip)
        self.save()
        return tip

    def remember(self, question: str, answer: str, source: str = "conversation") -> None:
        """Remember a question/answer pair for later lookup."""
        entry = {
            "question": str(question),
            "answer": str(answer),
            "source": source,
            "created_at": _now(),
        }
        self._questions.append(entry)
        del self._questions[:-_MAX_STORED_QUESTIONS]
        self.save()

    def search(self, question: str) -> QCTip | None:
        """Return the best matching built-in or custom tip."""
        candidates: list[QCTip] = [*self._custom, *_BUILTIN_TIPS]
        scored = [(tip.matches(question), index, tip) for index, tip in enumerate(candidates)]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    def ask(
        self,
        question: str,
        *,
        report: MatteQCReport | None = None,
        profile: Any | None = None,
    ) -> QCAnswer:
        """Answer a typed complaint and remember the result."""
        question = str(question).strip() or "How should I tune this matte?"
        found = self.search(question)
        matched = found is not None
        tip = found
        if tip is None:
            analysis = analyze_qc_image(None, None, report, profile=profile)
            tip = analysis.top_tip or _BUILTIN_TIPS[-1]
            source = "analysis"
        else:
            source = "custom" if tip in self._custom else "builtin"
        answer = QCAnswer(
            question=question,
            answer=tip.answer,
            tip=tip,
            matched=matched,
            settings=dict(tip.settings),
            source=source,
        )
        self.remember(question, answer.answer, source)
        return answer

    def save(self) -> None:
        """Persist custom tips and recent questions atomically."""
        payload = {
            "version": _TIP_VERSION,
            "custom_tips": [tip.to_dict() for tip in self._custom],
            "questions": self._questions[-_MAX_STORED_QUESTIONS:],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        custom = payload.get("custom_tips", [])
        if isinstance(custom, list):
            for item in custom:
                if isinstance(item, dict):
                    try:
                        self._custom.append(QCTip.from_dict(item))
                    except (TypeError, ValueError):
                        logger.debug("Skipping invalid QC tip")
        questions = payload.get("questions", [])
        if isinstance(questions, list):
            self._questions = [item for item in questions if isinstance(item, dict)][-_MAX_STORED_QUESTIONS:]


QCCoach = QCTipsDatabase
QCTipDatabase = QCTipsDatabase


__all__ = [
    "QCAnalysis",
    "QCAnswer",
    "QCCoach",
    "QCTip",
    "QCTipDatabase",
    "QCTipsDatabase",
    "analyze_qc_image",
]
