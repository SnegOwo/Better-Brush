"""Self-contained matte quality-control inspection for rotoscoping.

The inspector is designed for interactive dialogs: it accepts ordinary NumPy
arrays, never raises for an empty or mismatched current mask, and returns
inspectable dataclasses plus boolean issue masks.  Scores are issue severity
scores in the inclusive range zero to one, where larger values are worse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

BoundingBox = tuple[int, int, int, int]

_ISSUE_NAMES = (
    "hair_detail",
    "background_spill",
    "holes",
    "edge_flicker",
)
_ISSUE_ALIASES = {
    "hair": "hair_detail",
    "detail": "hair_detail",
    "hair/detail": "hair_detail",
    "hair-detail": "hair_detail",
    "spill": "background_spill",
    "background-spill": "background_spill",
    "background spill": "background_spill",
    "hole": "holes",
    "flicker": "edge_flicker",
    "edge-flicker": "edge_flicker",
    "edge flicker": "edge_flicker",
}


@dataclass
class MatteIssue:
    """One detected matte-quality issue and its pixel-level evidence.

    ``count`` is the number of issue pixels, while ``component_count`` counts
    connected regions.  ``bboxes`` use exclusive right and bottom coordinates
    in ``(x0, y0, x1, y1)`` order.  ``mask`` is always a two-dimensional
    boolean array with the same shape as the inspected matte.
    """

    kind: str
    score: float = 0.0
    count: int = 0
    bboxes: list[BoundingBox] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    component_count: int = 0
    severity: str = "none"
    message: str = ""

    def __post_init__(self) -> None:
        """Normalize user-supplied values into a safe presentation object."""
        self.kind = str(self.kind)
        try:
            score = float(self.score)
        except (TypeError, ValueError):
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        self.score = float(np.clip(score, 0.0, 1.0))
        try:
            self.count = max(0, int(self.count))
        except (TypeError, ValueError):
            self.count = 0
        try:
            self.component_count = max(0, int(self.component_count))
        except (TypeError, ValueError):
            self.component_count = 0
        try:
            values = np.asarray(self.mask)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.ndim != 2:
                values = np.zeros((0, 0), dtype=bool)
            self.mask = np.asarray(values, dtype=bool)
        except (TypeError, ValueError):
            self.mask = np.zeros((0, 0), dtype=bool)
        normalized_boxes: list[BoundingBox] = []
        try:
            for bbox in self.bboxes:
                if len(bbox) != 4:
                    continue
                normalized_boxes.append(
                    (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )
                )
            self.bboxes = normalized_boxes
        except (TypeError, ValueError, IndexError):
            self.bboxes = []
        self.severity = str(self.severity or "none")
        self.message = str(self.message or "")

    @property
    def name(self) -> str:
        """Return the issue name as a convenient alias for ``kind``."""
        return self.kind

    @property
    def pixel_count(self) -> int:
        """Return the number of issue pixels."""
        return self.count

    @property
    def detected(self) -> bool:
        """Return whether any issue pixels were found."""
        return self.count > 0

    @property
    def warning(self) -> bool:
        """Return whether this issue reaches its configured warning level."""
        return self.severity in {"warning", "error"}

    @property
    def bounding_boxes(self) -> list[BoundingBox]:
        """Return the issue bounding boxes."""
        return self.bboxes

    @property
    def boxes(self) -> list[BoundingBox]:
        """Return a short alias for :attr:`bboxes`."""
        return self.bboxes

    @property
    def bbox(self) -> BoundingBox | None:
        """Return the first issue bounding box, if one exists."""
        return self.bboxes[0] if self.bboxes else None

    @property
    def bboxes_xywh(self) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes in ``(x, y, width, height)`` form."""
        return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in self.bboxes]

    def to_dict(self, include_mask: bool = True) -> dict[str, object]:
        """Return a shallow, serialization-friendly description of the issue."""
        result: dict[str, object] = {
            "kind": self.kind,
            "name": self.name,
            "score": self.score,
            "count": self.count,
            "pixel_count": self.pixel_count,
            "component_count": self.component_count,
            "bboxes": list(self.bboxes),
            "bounding_boxes": list(self.bboxes),
            "severity": self.severity,
            "warning": self.warning,
            "message": self.message,
        }
        if include_mask:
            result["mask"] = self.mask.copy()
        return result

    def __getitem__(self, key: str) -> object:
        """Expose common issue fields through dictionary-style access."""
        values: dict[str, object] = {
            "kind": self.kind,
            "name": self.name,
            "score": self.score,
            "count": self.count,
            "pixel_count": self.pixel_count,
            "component_count": self.component_count,
            "bboxes": self.bboxes,
            "bounding_boxes": self.bounding_boxes,
            "mask": self.mask,
            "severity": self.severity,
            "warning": self.warning,
            "message": self.message,
        }
        if key in values:
            return values[key]
        raise KeyError(key)


def _empty_issue(kind: str, shape: tuple[int, int]) -> MatteIssue:
    """Create a zero-valued issue with a correctly shaped boolean mask."""
    return MatteIssue(
        kind=kind,
        mask=np.zeros(shape, dtype=bool),
    )


def _canonical_issue_name(name: str) -> str:
    """Return a canonical issue key for a user-facing issue name."""
    normalized = str(name).strip().lower().replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in _ISSUE_ALIASES:
        return _ISSUE_ALIASES[normalized]
    if normalized == "hair detail":
        return "hair_detail"
    if normalized == "background spill":
        return "background_spill"
    if normalized == "edge flicker":
        return "edge_flicker"
    return str(name)


@dataclass
class MatteQCReport:
    """Complete result of inspecting one RGB frame and its matte.

    The four entries in :attr:`issues` are ``hair_detail``,
    ``background_spill``, ``holes``, and ``edge_flicker``.  The report owns
    copies of the current matte, optional previous matte, and every issue
    mask, so callers can safely hand the result to a Qt widget or another
    worker thread.
    """

    issues: dict[str, MatteIssue] = field(default_factory=dict)
    shape: tuple[int, int] = (0, 0)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    previous_mask: np.ndarray | None = None
    overall_score: float = 0.0
    overall_warning: bool = False
    valid: bool = True
    empty: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize report arrays and scalar flags."""
        try:
            self.shape = (
                max(0, int(self.shape[0])),
                max(0, int(self.shape[1])),
            )
        except (TypeError, ValueError, IndexError):
            self.shape = (0, 0)
        try:
            values = np.asarray(self.mask)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.ndim != 2:
                values = np.zeros(self.shape, dtype=bool)
            self.mask = np.asarray(values, dtype=bool)
        except (TypeError, ValueError):
            self.mask = np.zeros(self.shape, dtype=bool)
        if self.mask.size and self.shape == (0, 0):
            self.shape = (int(self.mask.shape[0]), int(self.mask.shape[1]))
        if self.previous_mask is not None:
            try:
                values = np.asarray(self.previous_mask)
                if values.ndim == 3 and values.shape[-1] == 1:
                    values = values[..., 0]
                if values.ndim != 2:
                    self.previous_mask = None
                else:
                    self.previous_mask = np.asarray(values, dtype=bool)
            except (TypeError, ValueError):
                self.previous_mask = None
        try:
            score = float(self.overall_score)
        except (TypeError, ValueError):
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        self.overall_score = float(np.clip(score, 0.0, 1.0))
        self.overall_warning = bool(self.overall_warning)
        self.valid = bool(self.valid)
        self.empty = bool(self.empty)
        self.message = str(self.message or "")
        self.warnings = [str(warning) for warning in self.warnings]
        normalized: dict[str, MatteIssue] = {}
        for key, issue in self.issues.items():
            if isinstance(issue, MatteIssue):
                normalized[_canonical_issue_name(key)] = issue
        for name in _ISSUE_NAMES:
            normalized.setdefault(name, _empty_issue(name, self.shape))
        self.issues = normalized

    def issue(self, name: str) -> MatteIssue:
        """Return an issue by canonical name or one of its friendly aliases."""
        canonical = _canonical_issue_name(name)
        issue = self.issues.get(canonical)
        if issue is None:
            return _empty_issue(canonical, self.shape)
        return issue

    @property
    def warning(self) -> bool:
        """Return the overall warning flag."""
        return self.overall_warning

    @property
    def score(self) -> float:
        """Return the maximum issue severity score."""
        return self.overall_score

    @property
    def input_valid(self) -> bool:
        """Return whether the current frame and matte were usable."""
        return self.valid

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return the inspected spatial shape as ``(height, width)``."""
        return self.shape

    @property
    def has_previous_mask(self) -> bool:
        """Return whether a valid previous matte was available."""
        return self.previous_mask is not None

    @property
    def scores(self) -> dict[str, float]:
        """Return all issue severity scores keyed by canonical name."""
        return {name: float(self.issue(name).score) for name in _ISSUE_NAMES}

    @property
    def counts(self) -> dict[str, int]:
        """Return all issue pixel counts keyed by canonical name."""
        return {name: int(self.issue(name).count) for name in _ISSUE_NAMES}

    @property
    def component_counts(self) -> dict[str, int]:
        """Return all issue component counts keyed by canonical name."""
        return {name: int(self.issue(name).component_count) for name in _ISSUE_NAMES}

    @property
    def bboxes(self) -> dict[str, list[BoundingBox]]:
        """Return all issue bounding boxes keyed by canonical name."""
        return {name: list(self.issue(name).bboxes) for name in _ISSUE_NAMES}

    @property
    def bounding_boxes(self) -> dict[str, list[BoundingBox]]:
        """Return a descriptive alias for :attr:`bboxes`."""
        return self.bboxes

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Return copies of all boolean issue masks."""
        return {name: self.issue(name).mask.copy() for name in _ISSUE_NAMES}

    @property
    def hair_detail_mask(self) -> np.ndarray:
        """Return a copy of the boolean hair/detail evidence mask."""
        return self.hair_detail_issue.mask.copy()

    @property
    def background_spill_mask(self) -> np.ndarray:
        """Return a copy of the boolean background-spill evidence mask."""
        return self.background_spill_issue.mask.copy()

    @property
    def hole_mask(self) -> np.ndarray:
        """Return a copy of the boolean enclosed-hole evidence mask."""
        return self.hole_issue.mask.copy()

    @property
    def edge_flicker_mask(self) -> np.ndarray:
        """Return a copy of the boolean edge-flicker evidence mask."""
        return self.edge_flicker_issue.mask.copy()

    @property
    def hair_detail_issue(self) -> MatteIssue:
        """Return the hair and fine-detail issue."""
        return self.issue("hair_detail")

    @property
    def background_spill_issue(self) -> MatteIssue:
        """Return the background-spill issue."""
        return self.issue("background_spill")

    @property
    def hole_issue(self) -> MatteIssue:
        """Return the enclosed-hole issue."""
        return self.issue("holes")

    @property
    def edge_flicker_issue(self) -> MatteIssue:
        """Return the temporal edge-flicker issue."""
        return self.issue("edge_flicker")

    @property
    def hair_detail(self) -> MatteIssue:
        """Return the hair/detail issue under a concise field name."""
        return self.hair_detail_issue

    @property
    def background_spill(self) -> MatteIssue:
        """Return the background-spill issue under a concise field name."""
        return self.background_spill_issue

    @property
    def holes(self) -> MatteIssue:
        """Return the enclosed-hole issue under a concise field name."""
        return self.hole_issue

    @property
    def edge_flicker(self) -> MatteIssue:
        """Return the edge-flicker issue under a concise field name."""
        return self.edge_flicker_issue

    @property
    def hair(self) -> MatteIssue:
        """Return a short alias for :attr:`hair_detail`."""
        return self.hair_detail

    @property
    def spill(self) -> MatteIssue:
        """Return a short alias for :attr:`background_spill`."""
        return self.background_spill

    @property
    def flicker(self) -> MatteIssue:
        """Return a short alias for :attr:`edge_flicker`."""
        return self.edge_flicker

    @property
    def hair_issue(self) -> MatteIssue:
        """Return a short alias for :attr:`hair_detail_issue`."""
        return self.hair_detail_issue

    @property
    def spill_issue(self) -> MatteIssue:
        """Return a short alias for :attr:`background_spill_issue`."""
        return self.background_spill_issue

    @property
    def flicker_issue(self) -> MatteIssue:
        """Return a short alias for :attr:`edge_flicker_issue`."""
        return self.edge_flicker_issue

    @property
    def hair_score(self) -> float:
        """Return the hair/detail severity score."""
        return self.hair_detail_issue.score

    @property
    def hair_detail_score(self) -> float:
        """Return the hair/detail severity score."""
        return self.hair_score

    @property
    def detail_score(self) -> float:
        """Return an alias for the hair/detail severity score."""
        return self.hair_score

    @property
    def background_spill_score(self) -> float:
        """Return the background-spill severity score."""
        return self.background_spill_issue.score

    @property
    def spill_score(self) -> float:
        """Return an alias for the background-spill severity score."""
        return self.background_spill_score

    @property
    def hole_score(self) -> float:
        """Return the enclosed-hole severity score."""
        return self.hole_issue.score

    @property
    def holes_score(self) -> float:
        """Return an alias for the enclosed-hole severity score."""
        return self.hole_score

    @property
    def edge_flicker_score(self) -> float:
        """Return the temporal edge-flicker severity score."""
        return self.edge_flicker_issue.score

    @property
    def flicker_score(self) -> float:
        """Return an alias for the temporal edge-flicker severity score."""
        return self.edge_flicker_score

    @property
    def quality_score(self) -> float:
        """Return a quality score where one is best and zero is worst."""
        return 1.0 - self.overall_score

    def __getitem__(self, key: str) -> object:
        """Provide dictionary-style access for lightweight dialog bindings."""
        if key == "warning":
            return self.warning
        if key in {"overall_warning", "overall"}:
            return self.overall_warning
        if key == "score":
            return self.score
        if key in {"scores", "counts", "bboxes", "bounding_boxes", "masks"}:
            return getattr(self, key)
        if key == "issues":
            return self.issues
        if key == "valid":
            return self.valid
        if key == "empty":
            return self.empty
        if key == "message":
            return self.message
        if key == "warnings":
            return self.warnings
        if key in {"shape", "frame_shape"}:
            return self.shape
        if key == "mask":
            return self.mask
        if key == "previous_mask":
            return self.previous_mask
        return self.issue(key)

    def get(self, key: str, default: object = None) -> object:
        """Return a report attribute or issue using dictionary-style access."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self, include_masks: bool = True) -> dict[str, object]:
        """Return a serializable summary of the report."""
        result: dict[str, object] = {
            "shape": self.shape,
            "valid": self.valid,
            "empty": self.empty,
            "overall_score": self.overall_score,
            "score": self.score,
            "overall_warning": self.overall_warning,
            "warning": self.warning,
            "message": self.message,
            "warnings": list(self.warnings),
            "scores": self.scores,
            "counts": self.counts,
            "component_counts": self.component_counts,
            "bboxes": self.bboxes,
            "issues": {
                name: issue.to_dict(include_mask=include_masks)
                for name, issue in self.issues.items()
            },
        }
        if include_masks:
            result["mask"] = self.mask.copy()
            result["previous_mask"] = (
                None if self.previous_mask is None else self.previous_mask.copy()
            )
        return result


@dataclass
class MatteComparison:
    """Difference between two matte QC reports.

    Deltas are calculated as ``after - before``.  Since report scores are
    severities, a negative score delta means the second matte improved.
    """

    before: MatteQCReport
    after: MatteQCReport
    score_deltas: dict[str, float] = field(default_factory=dict)
    count_deltas: dict[str, int] = field(default_factory=dict)
    bbox_deltas: dict[str, int] = field(default_factory=dict)
    overall_score_delta: float = 0.0
    improved: bool = False
    worsened: bool = False
    summary: str = ""

    @property
    def deltas(self) -> dict[str, float]:
        """Return a short alias for :attr:`score_deltas`."""
        return self.score_deltas

    @property
    def score_changes(self) -> dict[str, float]:
        """Return per-issue score changes."""
        return self.score_deltas

    @property
    def before_scores(self) -> dict[str, float]:
        """Return the before report's issue scores."""
        return self.before.scores

    @property
    def after_scores(self) -> dict[str, float]:
        """Return the after report's issue scores."""
        return self.after.scores

    @property
    def before_warning(self) -> bool:
        """Return the before report's overall warning flag."""
        return self.before.warning

    @property
    def after_warning(self) -> bool:
        """Return the after report's overall warning flag."""
        return self.after.warning

    def __getitem__(self, key: str) -> object:
        """Provide dictionary-style access for summary widgets."""
        values: dict[str, object] = {
            "deltas": self.score_deltas,
            "score_deltas": self.score_deltas,
            "count_deltas": self.count_deltas,
            "bbox_deltas": self.bbox_deltas,
            "overall_score_delta": self.overall_score_delta,
            "improved": self.improved,
            "worsened": self.worsened,
            "summary": self.summary,
        }
        if key in values:
            return values[key]
        return getattr(self, key)

    def to_dict(self) -> dict[str, object]:
        """Return a compact dictionary describing the comparison."""
        return {
            "before": self.before.to_dict(include_masks=False),
            "after": self.after.to_dict(include_masks=False),
            "score_deltas": dict(self.score_deltas),
            "count_deltas": dict(self.count_deltas),
            "bbox_deltas": dict(self.bbox_deltas),
            "overall_score_delta": self.overall_score_delta,
            "improved": self.improved,
            "worsened": self.worsened,
            "summary": self.summary,
        }


MatteQCIssue = MatteIssue
MatteReport = MatteQCReport
MatteQCComparison = MatteComparison


def _coerce_mask(
    value: np.ndarray | Sequence[object] | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Convert a mask to bool when it exactly matches ``shape``."""
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            return None
        if tuple(array.shape[:2]) != tuple(shape):
            return None
        return np.asarray(array != 0, dtype=bool)
    except (TypeError, ValueError):
        return None


def _coerce_rgb(
    value: np.ndarray | Sequence[object] | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Convert an RGB-like array to uint8 without requiring an exact dtype."""
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] != 3:
            return None
        if shape is not None and tuple(array.shape[:2]) != tuple(shape):
            return None
        if array.size == 0:
            return array.astype(np.uint8, copy=False)
        if np.issubdtype(array.dtype, np.floating):
            converted = np.nan_to_num(
                array.astype(np.float32),
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )
            if float(np.nanmax(converted)) <= 1.0:
                converted = converted * 255.0
            return np.clip(converted, 0.0, 255.0).astype(np.uint8)
        converted = np.nan_to_num(
            array.astype(np.float32),
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        return np.clip(converted, 0.0, 255.0).astype(np.uint8)
    except (TypeError, ValueError, OverflowError):
        return None


def _empty_report(
    shape: tuple[int, int],
    mask: np.ndarray | None = None,
    previous_mask: np.ndarray | None = None,
    valid: bool = True,
    empty: bool = False,
    warning: bool = False,
    message: str = "",
    warnings: Sequence[str] | None = None,
) -> MatteQCReport:
    """Build a complete zero-valued report for invalid or empty inputs."""
    if mask is None:
        mask_array = np.zeros(shape, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool)
    messages = list(warnings or [])
    if message and message not in messages:
        messages.insert(0, message)
    return MatteQCReport(
        issues={name: _empty_issue(name, shape) for name in _ISSUE_NAMES},
        shape=shape,
        mask=mask_array,
        previous_mask=previous_mask,
        overall_score=0.0,
        overall_warning=warning,
        valid=valid,
        empty=empty,
        message=message
        or ("No matte QC issues detected." if not warning else "; ".join(messages)),
        warnings=messages,
    )


def _edge_band(mask: np.ndarray, width: int) -> np.ndarray:
    """Return a symmetric boolean band around a binary matte boundary."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        return np.zeros(values.shape, dtype=bool)
    try:
        radius = max(0, int(width))
    except (TypeError, ValueError):
        radius = 2
    if radius <= 0:
        return np.zeros(values.shape, dtype=bool)
    radius = min(radius, 64)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    source = values.astype(np.uint8)
    dilated = cv2.dilate(
        source,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    eroded = cv2.erode(
        source,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.logical_xor(dilated != 0, eroded != 0)


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a boolean mask with a bounded elliptical kernel."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        return np.zeros(values.shape, dtype=bool)
    try:
        radius = max(0, int(radius))
    except (TypeError, ValueError):
        radius = 1
    if radius == 0 or not values.any():
        return values.copy()
    radius = min(radius, 64)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.dilate(values.astype(np.uint8), kernel, iterations=1).astype(bool)


def _component_data(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return OpenCV connected-component labels and statistics."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.size == 0:
        return (
            0,
            np.zeros(values.shape, dtype=np.int32),
            np.zeros((1, 5), dtype=np.int32),
            np.zeros((1, 2), dtype=np.float32),
        )
    return cv2.connectedComponentsWithStats(
        values.astype(np.uint8),
        connectivity=8,
    )


def _bboxes_from_components(
    labels: np.ndarray,
    stats: np.ndarray,
    min_area: int = 1,
    max_boxes: int = 64,
) -> tuple[list[BoundingBox], int]:
    """Extract sorted exclusive-coordinate boxes and component count."""
    try:
        min_area = max(1, int(min_area))
    except (TypeError, ValueError):
        min_area = 1
    try:
        max_boxes = max(0, int(max_boxes))
    except (TypeError, ValueError):
        max_boxes = 64
    boxes: list[BoundingBox] = []
    component_count = 0
    for label in range(1, int(stats.shape[0])):
        x, y, width, height, area = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
            int(stats[label, cv2.CC_STAT_AREA]),
        )
        if area < min_area:
            continue
        component_count += 1
        boxes.append((x, y, x + width, y + height))
    boxes.sort(key=lambda box: (box[1], box[0], box[2], box[3]))
    if max_boxes > 0:
        boxes = boxes[:max_boxes]
    return boxes, component_count


def _issue_from_mask(
    kind: str,
    candidate: np.ndarray,
    score: float,
    warning_threshold: float,
    min_component_area: int = 1,
    max_boxes: int = 64,
    message: str = "",
) -> MatteIssue:
    """Turn a boolean evidence mask into a scored issue object."""
    evidence = np.asarray(candidate, dtype=bool)
    if evidence.ndim != 2:
        evidence = np.zeros((0, 0), dtype=bool)
    _, labels, stats, _ = _component_data(evidence)
    bboxes, component_count = _bboxes_from_components(
        labels,
        stats,
        min_area=min_component_area,
        max_boxes=max_boxes,
    )
    count = int(np.count_nonzero(evidence))
    try:
        normalized_score = float(np.clip(float(score), 0.0, 1.0))
    except (TypeError, ValueError):
        normalized_score = 0.0
    if not np.isfinite(normalized_score):
        normalized_score = 0.0
    if count == 0:
        severity = "none"
    elif normalized_score >= max(0.75, warning_threshold * 1.5):
        severity = "error"
    elif normalized_score >= max(0.0, warning_threshold):
        severity = "warning"
    else:
        severity = "info"
    return MatteIssue(
        kind=kind,
        score=normalized_score,
        count=count,
        bboxes=bboxes,
        mask=evidence,
        component_count=component_count,
        severity=severity,
        message=message,
    )


def _estimate_background(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    edge_width: int,
) -> np.ndarray:
    """Estimate a spatially uniform background from outside the matte."""
    frame = frame_rgb.astype(np.float32) / 255.0
    outside = ~_dilate_mask(mask, max(1, edge_width + 1))
    if int(np.count_nonzero(outside)) < 8:
        outside = ~mask
    samples = frame[outside]
    if samples.size == 0:
        samples = frame.reshape(-1, 3)
    try:
        color = np.median(samples, axis=0).astype(np.float32)
    except (TypeError, ValueError):
        color = np.full(3, 0.5, dtype=np.float32)
    return np.broadcast_to(color, frame.shape).astype(np.float32)


def _hair_detail_evidence(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    edge_width: int,
) -> tuple[np.ndarray, float]:
    """Find high-frequency image detail close to a matte boundary."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y) / 4.0
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) / 4.0
    local_mean = cv2.blur(gray, (5, 5), borderType=cv2.BORDER_REFLECT101)
    local_square_mean = cv2.blur(
        gray * gray,
        (5, 5),
        borderType=cv2.BORDER_REFLECT101,
    )
    local_variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    detail = (gradient >= 0.08) & (laplacian >= 0.018) & (local_variance >= 0.00045)
    edge = _edge_band(mask, edge_width)
    context = _dilate_mask(edge, max(2, edge_width * 2))
    context_detail_count = int(np.count_nonzero(detail & context))
    edge_area = int(np.count_nonzero(edge))
    if edge_area == 0 or context_detail_count <= edge_area * 0.8:
        return np.zeros(mask.shape, dtype=bool), 0.0
    inner = edge & mask
    outer = edge & ~mask
    inner_detail = inner & detail
    outer_detail = outer & detail
    evidence = inner_detail | outer_detail
    weighted = float(np.count_nonzero(outer_detail))
    weighted += 0.2 * float(np.count_nonzero(inner_detail))
    score = float(np.clip(weighted / max(1.0, edge_area * 1.15), 0.0, 1.0))
    return evidence, score


def _background_spill_evidence(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    background_rgb: np.ndarray | None,
    edge_width: int,
) -> tuple[np.ndarray, float]:
    """Find matte pixels connected to, or inconsistent with, the background."""
    edge = _edge_band(mask, edge_width)
    if not edge.any():
        return np.zeros(mask.shape, dtype=bool), 0.0
    reference = background_rgb
    if reference is None:
        reference_uint8 = None
    else:
        reference_uint8 = reference
    if reference_uint8 is not None:
        reference_float = reference_uint8.astype(np.float32) / 255.0
    else:
        reference_float = None
    if reference_float is None:
        return np.zeros(mask.shape, dtype=bool), 0.0
    frame_float = frame_rgb.astype(np.float32) / 255.0
    color_distance = np.linalg.norm(frame_float - reference_float, axis=2) / np.sqrt(
        3.0
    )
    tolerance = 0.10
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool), 0.0
    inside_contrast = float(np.median(color_distance[mask]))
    if inside_contrast <= tolerance:
        return np.zeros(mask.shape, dtype=bool), 0.0
    background_like = color_distance <= tolerance
    outside = ~mask
    outside_labels = np.unique(
        _component_data(background_like & outside)[1][outside & background_like]
    )
    connected_background = np.zeros(mask.shape, dtype=bool)
    if outside_labels.size:
        labels = _component_data(background_like)[1]
        connected_background = np.isin(labels, outside_labels[outside_labels != 0])
    foreground_like = ~background_like
    outside_foreground = outside & edge & foreground_like
    evidence = (mask & connected_background) | outside_foreground
    edge_area = int(np.count_nonzero(edge))
    count = int(np.count_nonzero(evidence))
    if count == 0 or edge_area == 0:
        return evidence, 0.0
    score = float(np.clip(count / max(1.0, edge_area), 0.0, 1.0))
    return evidence, score


def _hole_evidence(
    mask: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, float]:
    """Find enclosed false-background components inside the matte."""
    inverse = ~np.asarray(mask, dtype=bool)
    number, labels, stats, _ = _component_data(inverse)
    height, width = inverse.shape
    evidence = np.zeros(inverse.shape, dtype=bool)
    try:
        min_area = max(1, int(min_area))
    except (TypeError, ValueError):
        min_area = 1
    for label in range(1, number):
        x, y, component_width, component_height, area = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
            int(stats[label, cv2.CC_STAT_AREA]),
        )
        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if not touches_border and area >= min_area:
            evidence |= labels == label
    mask_area = int(np.count_nonzero(mask))
    count = int(np.count_nonzero(evidence))
    if count == 0:
        return evidence, 0.0
    score = float(np.clip(count / max(1.0, mask_area), 0.0, 1.0))
    return evidence, score


def _flicker_evidence(
    mask: np.ndarray,
    previous_mask: np.ndarray,
    frame_rgb: np.ndarray,
    background_rgb: np.ndarray | None,
    previous_background_rgb: np.ndarray | None,
    edge_width: int,
) -> tuple[np.ndarray, float]:
    """Find boundary changes not explained by a background change."""
    current_edge = _edge_band(mask, edge_width)
    previous_edge = _edge_band(previous_mask, edge_width)
    changed = np.logical_xor(mask, previous_mask)
    evidence = changed & (current_edge | previous_edge)
    if previous_background_rgb is not None:
        if background_rgb is None:
            current_reference_float = _estimate_background(
                frame_rgb,
                mask,
                edge_width,
            )
        else:
            current_reference_float = background_rgb.astype(np.float32) / 255.0
        previous_reference_float = previous_background_rgb.astype(np.float32) / 255.0
        background_change = np.linalg.norm(
            current_reference_float - previous_reference_float,
            axis=2,
        ) / np.sqrt(3.0)
        evidence &= background_change <= 0.12
    union_area = int(np.count_nonzero(np.logical_or(mask, previous_mask)))
    count = int(np.count_nonzero(evidence))
    if count == 0 or union_area == 0:
        return evidence, 0.0
    score = float(np.clip(count / max(1.0, union_area), 0.0, 1.0))
    return evidence, score


def analyze_matte(
    frame_rgb: np.ndarray | Sequence[object] | None,
    mask: np.ndarray | Sequence[object] | None,
    previous_mask: np.ndarray | Sequence[object] | None = None,
    background: np.ndarray | Sequence[object] | None = None,
    previous_background: np.ndarray | Sequence[object] | None = None,
    *,
    background_rgb: np.ndarray | Sequence[object] | None = None,
    previous_background_rgb: np.ndarray | Sequence[object] | None = None,
    edge_band_width: int = 2,
    min_hole_area: int = 1,
    max_boxes: int = 64,
    hair_threshold: float = 0.25,
    background_spill_threshold: float = 0.35,
    hole_threshold: float = 0.0,
    flicker_threshold: float = 0.08,
) -> MatteQCReport:
    """Inspect one RGB frame and binary matte for common QC problems.

    Args:
        frame_rgb: RGB image shaped ``(H, W, 3)``.  Eight-bit and floating
            point arrays are accepted; floating values in ``[0, 1]`` are
            scaled automatically.
        mask: Binary or numeric matte shaped ``(H, W)``.  A mismatched
            current matte produces a safe, zero-valued report rather than an
            exception.
        previous_mask: Optional previous-frame matte used for edge-flicker
            inspection.
        background: Optional clean RGB background reference for the current
            frame.  It is used to distinguish background spill from a valid
            matte edge.
        previous_background: Optional clean background reference for the
            previous frame.  Supplying it suppresses mask changes that are
            explained by a static-camera background change.
        background_rgb: Explicit alias for ``background``.
        previous_background_rgb: Explicit alias for ``previous_background``.
        edge_band_width: Width in pixels of the boundary inspection band.
        min_hole_area: Smallest enclosed background component to report.
        max_boxes: Maximum boxes retained per issue; non-positive means all.
        hair_threshold: Severity threshold for the hair/detail warning.
        background_spill_threshold: Severity threshold for spill warnings.
        hole_threshold: Severity threshold for hole warnings.
        flicker_threshold: Severity threshold for flicker warnings.

    Returns:
        A :class:`MatteQCReport` containing scores, pixel counts, boxes,
        boolean evidence masks, and an overall warning flag.  An empty valid
        matte is reported as a warning so an inspector cannot silently pass a
        missing subject.
    """
    raw_frame_shape = _raw_spatial_shape(frame_rgb)
    raw_mask_shape = _raw_spatial_shape(mask)
    shape = raw_frame_shape or raw_mask_shape or (0, 0)
    frame = _coerce_rgb(frame_rgb, shape)
    if frame is None:
        report_mask = _coerce_mask(mask, shape)
        if report_mask is None:
            report_mask = np.zeros(shape, dtype=bool)
        return _empty_report(
            shape,
            mask=report_mask,
            valid=False,
            warning=True,
            message="Frame is not a usable RGB array.",
        )
    shape = (int(frame.shape[0]), int(frame.shape[1]))
    current_mask = _coerce_mask(mask, shape)
    if current_mask is None:
        return _empty_report(
            shape,
            valid=False,
            warning=True,
            message="Mask shape does not match the RGB frame.",
        )

    warnings: list[str] = []
    valid = True
    if not current_mask.any():
        previous = _coerce_mask(previous_mask, shape)
        if previous_mask is not None and previous is None:
            valid = False
            warnings.append("Previous mask shape does not match the RGB frame.")
        return _empty_report(
            shape,
            mask=current_mask,
            previous_mask=previous,
            valid=valid,
            empty=True,
            warning=True,
            message="Mask is empty.",
            warnings=warnings,
        )

    previous = None
    if previous_mask is not None:
        previous = _coerce_mask(previous_mask, shape)
        if previous is None:
            valid = False
            warnings.append("Previous mask shape does not match the RGB frame.")

    current_background = background_rgb if background_rgb is not None else background
    if background is not None and background_rgb is not None:
        warnings.append("Both background names were supplied; background_rgb was used.")
    old_background = (
        previous_background_rgb
        if previous_background_rgb is not None
        else previous_background
    )
    if previous_background is not None and previous_background_rgb is not None:
        warnings.append(
            "Both previous background names were supplied; previous_background_rgb was used."
        )
    if old_background is None and current_background is not None:
        old_background = current_background
    current_background_uint8 = _coerce_rgb(current_background, shape)
    if current_background is not None and current_background_uint8 is None:
        valid = False
        warnings.append("Background shape does not match the RGB frame.")
    old_background_uint8 = _coerce_rgb(old_background, shape)
    if old_background is not None and old_background_uint8 is None:
        valid = False
        warnings.append("Previous background shape does not match the RGB frame.")

    try:
        edge_width = max(1, int(edge_band_width))
    except (TypeError, ValueError):
        edge_width = 2
    edge_width = min(edge_width, 64)
    try:
        hole_area = max(1, int(min_hole_area))
    except (TypeError, ValueError):
        hole_area = 1
    try:
        box_limit = int(max_boxes)
    except (TypeError, ValueError):
        box_limit = 64

    if current_background_uint8 is None:
        estimated_background = _estimate_background(frame, current_mask, edge_width)
        background_reference = np.clip(
            estimated_background * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    else:
        background_reference = current_background_uint8

    hair_mask, hair_score_value = _hair_detail_evidence(
        frame,
        current_mask,
        edge_width,
    )
    spill_mask, spill_score_value = _background_spill_evidence(
        frame,
        current_mask,
        background_reference,
        edge_width,
    )
    hole_mask, hole_score_value = _hole_evidence(current_mask, hole_area)
    if previous is None:
        flicker_mask = np.zeros(shape, dtype=bool)
        flicker_score_value = 0.0
    else:
        flicker_mask, flicker_score_value = _flicker_evidence(
            current_mask,
            previous,
            frame,
            background_reference,
            old_background_uint8,
            edge_width,
        )

    try:
        hair_threshold_value = float(np.clip(float(hair_threshold), 0.0, 1.0))
    except (TypeError, ValueError):
        hair_threshold_value = 0.25
    try:
        spill_threshold_value = float(
            np.clip(float(background_spill_threshold), 0.0, 1.0)
        )
    except (TypeError, ValueError):
        spill_threshold_value = 0.35
    try:
        hole_threshold_value = float(np.clip(float(hole_threshold), 0.0, 1.0))
    except (TypeError, ValueError):
        hole_threshold_value = 0.0
    try:
        flicker_threshold_value = float(np.clip(float(flicker_threshold), 0.0, 1.0))
    except (TypeError, ValueError):
        flicker_threshold_value = 0.08

    issues = {
        "hair_detail": _issue_from_mask(
            "hair_detail",
            hair_mask,
            hair_score_value,
            warning_threshold=hair_threshold_value,
            max_boxes=box_limit,
            message="Fine hair or detail may be missing at the matte boundary.",
        ),
        "background_spill": _issue_from_mask(
            "background_spill",
            spill_mask,
            spill_score_value,
            warning_threshold=spill_threshold_value,
            max_boxes=box_limit,
            message="Background-colored pixels may be present in the matte edge.",
        ),
        "holes": _issue_from_mask(
            "holes",
            hole_mask,
            hole_score_value,
            warning_threshold=hole_threshold_value,
            min_component_area=1,
            max_boxes=box_limit,
            message="The matte contains enclosed holes.",
        ),
        "edge_flicker": _issue_from_mask(
            "edge_flicker",
            flicker_mask,
            flicker_score_value,
            warning_threshold=flicker_threshold_value,
            max_boxes=box_limit,
            message="The matte boundary changes between adjacent frames.",
        ),
    }
    overall_score = max(float(issue.score) for issue in issues.values())
    warning_issues = [
        issue for issue in issues.values() if issue.warning and int(issue.count) > 0
    ]
    overall_warning = bool(not valid or warning_issues or not current_mask.any())
    if warning_issues:
        message = "; ".join(issue.message for issue in warning_issues)
    elif not valid:
        message = "; ".join(warnings)
    else:
        message = "No matte QC issues detected."
    return MatteQCReport(
        issues=issues,
        shape=shape,
        mask=current_mask,
        previous_mask=previous,
        overall_score=overall_score,
        overall_warning=overall_warning,
        valid=valid,
        empty=False,
        message=message,
        warnings=warnings,
    )


def compare_matte_reports(
    before: MatteQCReport,
    after: MatteQCReport,
) -> MatteComparison:
    """Compare two QC reports using after-minus-before severity deltas."""
    if not isinstance(before, MatteQCReport) or not isinstance(after, MatteQCReport):
        raise TypeError("before and after must be MatteQCReport objects")
    score_deltas = {
        name: float(after.scores[name] - before.scores[name]) for name in _ISSUE_NAMES
    }
    count_deltas = {
        name: int(after.counts[name] - before.counts[name]) for name in _ISSUE_NAMES
    }
    bbox_deltas = {
        name: int(len(after.bboxes[name]) - len(before.bboxes[name]))
        for name in _ISSUE_NAMES
    }
    overall_delta = float(after.overall_score - before.overall_score)
    tolerance = 1e-9
    improved = overall_delta < -tolerance
    worsened = overall_delta > tolerance
    direction = "unchanged"
    if improved:
        direction = "improved"
    elif worsened:
        direction = "worsened"
    summary = (
        f"Overall matte score {before.overall_score:.3f} -> "
        f"{after.overall_score:.3f} ({direction})"
    )
    return MatteComparison(
        before=before,
        after=after,
        score_deltas=score_deltas,
        count_deltas=count_deltas,
        bbox_deltas=bbox_deltas,
        overall_score_delta=overall_delta,
        improved=improved,
        worsened=worsened,
        summary=summary,
    )


def compare_matte_before_after(
    before: MatteQCReport,
    after: MatteQCReport,
) -> MatteComparison:
    """Descriptive alias for :func:`compare_matte_reports`."""
    return compare_matte_reports(before, after)


def _color_triplet(value: Sequence[object] | np.ndarray) -> tuple[int, int, int]:
    """Validate an RGB color and return integer channel values."""
    try:
        array = np.asarray(value)
        if array.size != 3:
            return (255, 64, 64)
        channels = array.reshape(-1)[:3].astype(np.float32)
        channels = np.nan_to_num(channels, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.nanmax(channels)) <= 1.0:
            channels = channels * 255.0
        return tuple(int(value) for value in np.clip(channels, 0, 255))
    except (TypeError, ValueError, OverflowError):
        return (255, 64, 64)


def _blend_color(
    image: np.ndarray,
    selection: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend one RGB color into selected pixels in place."""
    if not selection.any() or alpha <= 0.0:
        return
    alpha_value = float(np.clip(alpha, 0.0, 1.0))
    blend = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    region = image[selection].astype(np.float32)
    blended = region * (1.0 - alpha_value) + blend * alpha_value
    image[selection] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def render_mask_overlay(
    frame_rgb: np.ndarray | Sequence[object] | None,
    mask: np.ndarray | Sequence[object] | None,
    report: MatteQCReport | None = None,
    *,
    color: Sequence[object] | np.ndarray = (255, 64, 64),
    alpha: float = 0.45,
    outline: bool = True,
    issue_alpha: float = 0.42,
) -> np.ndarray:
    """Render a mask and optional QC evidence for display in a Qt dialog.

    The result is a contiguous RGB ``uint8`` array with the same ``(H, W, 3)``
    shape as the input frame.  It can be wrapped directly with Qt's
    ``QImage.Format_RGB888`` and ``bytesPerLine=width * 3`` without importing
    Qt into this processing module.  A mismatched mask is treated as empty;
    invalid frames produce a correctly shaped black RGB image.
    """
    frame = _coerce_rgb(frame_rgb)
    if frame is None:
        shape = _raw_spatial_shape(frame_rgb) or _raw_spatial_shape(mask) or (0, 0)
        return np.zeros(shape + (3,), dtype=np.uint8)
    shape = (int(frame.shape[0]), int(frame.shape[1]))
    matte = _coerce_mask(mask, shape)
    if matte is None:
        matte = np.zeros(shape, dtype=bool)
    output = frame.copy()
    base_color = _color_triplet(color)
    try:
        alpha_value = float(np.clip(float(alpha), 0.0, 1.0))
    except (TypeError, ValueError):
        alpha_value = 0.45
    try:
        issue_alpha_value = float(np.clip(float(issue_alpha), 0.0, 1.0))
    except (TypeError, ValueError):
        issue_alpha_value = 0.42
    _blend_color(output, matte, base_color, alpha_value)
    if outline and matte.any():
        source = matte.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(source, kernel, iterations=1).astype(bool)
        eroded = cv2.erode(source, kernel, iterations=1).astype(bool)
        boundary = np.logical_xor(dilated, eroded)
        _blend_color(output, boundary, base_color, min(1.0, alpha_value + 0.25))
    if report is not None and report.shape == shape:
        issue_colors: dict[str, tuple[int, int, int]] = {
            "hair_detail": (0, 200, 255),
            "background_spill": (255, 80, 220),
            "holes": (255, 220, 0),
            "edge_flicker": (255, 120, 0),
        }
        for name, issue_color in issue_colors.items():
            issue_mask = report.issue(name).mask
            if issue_mask.shape == shape:
                _blend_color(
                    output, issue_mask.astype(bool), issue_color, issue_alpha_value
                )
    return np.ascontiguousarray(output, dtype=np.uint8)


def overlay_mask(
    frame_rgb: np.ndarray | Sequence[object] | None,
    mask: np.ndarray | Sequence[object] | None,
    report: MatteQCReport | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Short alias for :func:`render_mask_overlay`."""
    return render_mask_overlay(frame_rgb, mask, report, **kwargs)


def _raw_spatial_shape(value: object) -> tuple[int, int] | None:
    """Return a safe ``(height, width)`` shape for an array-like value."""
    try:
        array = np.asarray(value)
        if array.ndim == 2:
            return int(array.shape[0]), int(array.shape[1])
        if array.ndim == 3 and array.shape[0] >= 0 and array.shape[1] >= 0:
            return int(array.shape[0]), int(array.shape[1])
    except (TypeError, ValueError):
        return None
    return None


inspect_matte = analyze_matte
analyze_matte_qc = analyze_matte
compare_mattes = compare_matte_reports
compare_matte_qc = compare_matte_reports
compare_before_after = compare_matte_reports
render_overlay = render_mask_overlay

__all__ = [
    "BoundingBox",
    "MatteComparison",
    "MatteIssue",
    "MatteQCComparison",
    "MatteQCIssue",
    "MatteQCReport",
    "MatteReport",
    "analyze_matte",
    "analyze_matte_qc",
    "compare_before_after",
    "compare_matte_before_after",
    "compare_matte_qc",
    "compare_matte_reports",
    "compare_mattes",
    "inspect_matte",
    "overlay_mask",
    "render_mask_overlay",
    "render_overlay",
]
