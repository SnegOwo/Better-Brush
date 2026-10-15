"""Pick the single character in a frame — or refuse and let the app do it.

The After Effects panel has exactly one button, so it must be *narrow on
purpose*: it removes the background behind **one character**. Anything
ambiguous — no character found, or several plausible subjects — is handed back
to the full app instead of being guessed at. That is the whole contract this
module implements.

The scoring is deliberately plain (pure NumPy + a couple of OpenCV calls) so it
is fully unit-testable without a GPU or SAM 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Tunables (module constants so tests can lean on them explicitly)
# --------------------------------------------------------------------- #
MIN_AREA_RATIO = 0.004
MAX_AREA_RATIO = 0.80
MIN_CONFIDENCE = 0.45
MULTI_SCORE_MARGIN = 0.08
MULTI_MIN_AREA_RATIO = 0.02
MULTI_MAX_IOU = 0.25

# Weighting of the six character-likeness signals. They sum to 1.0 so the
# result reads as a confidence in [0, 1].
_W_SAM = 0.30
_W_AREA = 0.20
_W_CENTER = 0.15
_W_BORDER = 0.15
_W_SOLIDITY = 0.10
_W_FACE = 0.10


@dataclass
class Candidate:
    """One SAM 2 mask candidate plus the stats used to judge it."""

    mask: np.ndarray
    sam_score: float
    area_ratio: float = 0.0
    centroid: tuple[float, float] = (0.5, 0.5)  # normalised (cx, cy)
    border_ratio: float = 0.0
    solidity: float = 0.0
    face_inside: bool = False
    score: float = 0.0


@dataclass
class SubjectSelection:
    """Outcome of the auto-subject pass on a frame."""

    mask: np.ndarray | None = None
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None
    multiple: bool = False
    face_seeded: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.mask is not None and not self.multiple


# --------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------- #
def grid_points(
    width: int, height: int, steps: int = 4, margin: float = 0.12
) -> list[tuple[int, int]]:
    """A coarse interior grid of seed points (avoids the very edges)."""
    steps = max(2, int(steps))
    xs = np.linspace(margin, 1.0 - margin, steps)
    ys = np.linspace(margin, 1.0 - margin, steps)
    points: list[tuple[int, int]] = []
    for fy in ys:
        for fx in xs:
            points.append((int(fx * (width - 1)), int(fy * (height - 1))))
    return points


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #
def _area_pref(area_ratio: float) -> float:
    """1.0 across the 'a character' band, tapering to 0 outside it."""
    if area_ratio <= 0.0:
        return 0.0
    if 0.03 <= area_ratio <= 0.45:
        return 1.0
    if area_ratio < 0.03:
        return area_ratio / 0.03
    return max(0.0, 1.0 - (area_ratio - 0.45) / 0.35)


def _border_ratio(mask: np.ndarray) -> float:
    """Fraction of the frame's edge pixels covered by the mask."""
    ring = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    return float(ring.mean()) if ring.size else 0.0


def _solidity(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
    if hull_area <= 0:
        return 0.0
    return max(0.0, min(1.0, area / hull_area))


def make_candidate(
    mask: np.ndarray,
    sam_score: float,
    face_center: tuple[float, float] | None = None,
) -> Candidate | None:
    """Measure a raw mask; ``None`` when it is hopelessly sized."""
    mask = mask.astype(bool)
    h, w = mask.shape
    area = float(mask.sum())
    total = float(h * w)
    area_ratio = area / total if total else 0.0
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return None

    ys, xs = np.nonzero(mask)
    cx = float(xs.mean()) / max(1, w - 1)
    cy = float(ys.mean()) / max(1, h - 1)

    face_inside = False
    if face_center is not None:
        fx = int(min(max(face_center[0], 0), w - 1))
        fy = int(min(max(face_center[1], 0), h - 1))
        face_inside = bool(mask[fy, fx])

    cand = Candidate(
        mask=mask,
        sam_score=float(sam_score),
        area_ratio=area_ratio,
        centroid=(cx, cy),
        border_ratio=_border_ratio(mask),
        solidity=_solidity(mask),
        face_inside=face_inside,
    )
    cand.score = score_candidate(cand)
    return cand


def score_candidate(cand: Candidate) -> float:
    """Blend the six signals into a single confidence in [0, 1]."""
    dist = float(
        np.hypot((cand.centroid[0] - 0.5) / 0.5, (cand.centroid[1] - 0.5) / 0.5)
    ) / float(np.sqrt(2.0))
    center_pref = 1.0 - min(dist, 1.0)
    border_pref = 1.0 - min(cand.border_ratio / 0.5, 1.0)
    score = (
        _W_SAM * max(0.0, min(1.0, cand.sam_score))
        + _W_AREA * _area_pref(cand.area_ratio)
        + _W_CENTER * center_pref
        + _W_BORDER * border_pref
        + _W_SOLIDITY * cand.solidity
        + _W_FACE * (1.0 if cand.face_inside else 0.0)
    )
    return max(0.0, min(1.0, score))


def evaluate(
    candidates: list[Candidate],
    min_confidence: float = MIN_CONFIDENCE,
) -> SubjectSelection:
    """Choose the single best character, or explain why we refuse."""
    usable = [c for c in candidates if c.score > 0.0]
    if not usable:
        return SubjectSelection(reasons=["no mask candidate passed the size gate"])

    usable.sort(key=lambda c: c.score, reverse=True)
    best = usable[0]
    reasons = [
        f"best score {best.score:.2f}",
        f"area {best.area_ratio:.3f}",
    ]

    if best.score < min_confidence:
        return SubjectSelection(
            confidence=best.score,
            box=bbox_of(best.mask),
            reasons=reasons + ["below confidence gate"],
        )

    rivals = [
        c
        for c in usable[1:]
        if c.score >= best.score - MULTI_SCORE_MARGIN
        and c.area_ratio >= MULTI_MIN_AREA_RATIO
        and iou(best.mask, c.mask) < MULTI_MAX_IOU
    ]
    if rivals:
        return SubjectSelection(
            confidence=best.score,
            box=bbox_of(best.mask),
            multiple=True,
            reasons=reasons + [f"{len(rivals)} rival subject(s) detected"],
        )

    return SubjectSelection(
        mask=best.mask,
        confidence=best.score,
        box=bbox_of(best.mask),
        face_seeded=best.face_inside,
        reasons=reasons,
    )


# --------------------------------------------------------------------- #
# Face anchoring (optional aid)
# --------------------------------------------------------------------- #
def detect_face_center(
    frame_rgb: np.ndarray,
) -> tuple[float, float] | None:
    """Centre of the detected face, or ``None`` (never raises)."""
    try:
        from src.processing.face_parts import detect_landmarks

        landmarks = detect_landmarks(frame_rgb)
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        logger.debug("face detection skipped: %s", exc)
        return None
    if not landmarks:
        return None
    xs = [p[0] for p in landmarks.values()]
    ys = [p[1] for p in landmarks.values()]
    if not xs:
        return None
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


# --------------------------------------------------------------------- #
# SAM 2 driven entry point
# --------------------------------------------------------------------- #
def _prompt_sets(
    width: int, height: int, steps: int, face_center: tuple[float, float] | None
) -> list[list]:
    from src.models.sam2_manager import PointPrompt

    sets: list[list] = []
    for x, y in grid_points(width, height, steps=steps):
        sets.append([PointPrompt(x=x, y=y, label=1)])
    if face_center is not None:
        fx, fy = int(face_center[0]), int(face_center[1])
        sets.append([PointPrompt(x=fx, y=fy, label=1)])
    return sets


def find_subject(
    manager,
    frame_rgb: np.ndarray,
    *,
    face_center: tuple[float, float] | None = None,
    steps: int = 4,
    multimask: bool = True,
) -> SubjectSelection:
    """Probe a frame with SAM 2 and decide whether it holds one character."""
    height, width = frame_rgb.shape[:2]
    if width < 16 or height < 16:
        return SubjectSelection(reasons=["frame too small"])

    prompt_sets = _prompt_sets(width, height, steps, face_center)
    if not prompt_sets:
        return SubjectSelection(reasons=["no seed points"])

    candidate_masks: list[tuple[np.ndarray, float]] = []
    multi = getattr(manager, "predict_frame_multi", None)
    if callable(multi):
        try:
            batches = multi(frame_rgb, prompt_sets, multimask_output=multimask)
        except Exception as exc:  # noqa: BLE001
            logger.warning("multi-prompt probe failed (%s); falling back", exc)
            batches = None
        if batches is not None:
            for results in batches:
                for res in results:
                    candidate_masks.append((res.mask, res.score))
    else:  # pragma: no cover - only hit on very old managers
        for pts in prompt_sets:
            for res in manager.predict_frame(frame_rgb, pts, multimask):
                candidate_masks.append((res.mask, res.score))

    candidates: list[Candidate] = []
    for mask, sam_score in candidate_masks:
        cand = make_candidate(mask, sam_score, face_center)
        if cand is not None:
            candidates.append(cand)

    selection = evaluate(candidates)
    if face_center is not None:
        selection.reasons.append("face detector seeded the probe")
    logger.info(
        "auto-subject: found=%s multiple=%s conf=%.3f candidates=%d",
        selection.found,
        selection.multiple,
        selection.confidence,
        len(candidates),
    )
    return selection
