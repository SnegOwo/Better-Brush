"""Prompt-aware mask candidate selection and conservative background trimming."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.models.sam2_manager import PointPrompt


def prompt_consistency(
    mask: np.ndarray,
    points: Sequence[PointPrompt],
) -> tuple[float, int, int]:
    """Return ``(consistency, missed_positive, hit_negative)`` for a mask."""
    h, w = mask.shape[:2]
    positive_total = sum(1 for point in points if point.label == 1)
    positive_inside = 0
    negative_hits = 0
    for point in points:
        x = max(0, min(w - 1, int(point.x)))
        y = max(0, min(h - 1, int(point.y)))
        inside = bool(mask[y, x])
        if point.label == 1:
            positive_inside += int(inside)
        elif inside:
            negative_hits += 1
    consistency = (
        positive_inside / positive_total if positive_total else 1.0
    ) - negative_hits * 1.5
    return consistency, positive_total - positive_inside, negative_hits


def _positive_spread_penalty(
    mask: np.ndarray,
    points: Sequence[PointPrompt],
) -> float:
    """Return the fraction of a multi-point mask outside a prompt envelope."""
    positive = [point for point in points if point.label == 1]
    if len(positive) < 2:
        return 0.0
    xs = np.array([point.x for point in positive], dtype=np.float32)
    ys = np.array([point.y for point in positive], dtype=np.float32)
    h, w = mask.shape[:2]
    margin = 0.22 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    x0 = max(0, int(xs.min() - margin))
    x1 = min(w, int(xs.max() + margin) + 1)
    y0 = max(0, int(ys.min() - margin))
    y1 = min(h, int(ys.max() + margin) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    envelope = np.zeros_like(mask, dtype=bool)
    envelope[y0:y1, x0:x1] = True
    area = int(mask.sum())
    return float(np.count_nonzero(mask & ~envelope) / max(1, area))


def choose_prompt_mask_index(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    points: Sequence[PointPrompt],
) -> int:
    """Choose the best candidate that honors positive and negative points.

    A raw SAM score alone can prefer a broad mask that includes the sky. We
    first prefer candidates containing every positive point and no negative
    point, then fall back to the least-violating candidate.
    """
    if not masks:
        raise ValueError("at least one mask candidate is required")
    count = min(len(masks), len(scores))
    if count == 0:
        raise ValueError("mask candidates and scores cannot be empty")
    ranked: list[tuple[int, float, float, int, float]] = []
    for index in range(count):
        consistency, missed, hits = prompt_consistency(masks[index], points)
        spread_penalty = _positive_spread_penalty(masks[index], points)
        ranked.append(
            (
                index,
                consistency,
                float(scores[index]),
                missed * 10 + hits * 20,
                spread_penalty,
            )
        )
    consistent = [item for item in ranked if item[3] == 0]
    pool = consistent or ranked
    return max(pool, key=lambda item: (item[1], item[2] - item[4] * 1.5, item[2], -item[0]))[0]


def trim_prompt_background(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    points: Sequence[PointPrompt],
    *,
    tolerance: float = 22.0,
    seed_radius: int = 12,
) -> np.ndarray:
    """Remove a negative-prompt background region that visibly continues outside.

    This is deliberately conservative: only mask pixels connected to a
    negative point and similar in Lab color to its local neighborhood are
    removed. A red point on the sky therefore removes the connected sky region
    without eroding the subject or changing subtitle/logo cleanup.
    """
    negative = [point for point in points if point.label == 0]
    if not negative or frame_rgb is None or mask is None or not mask.any():
        return mask
    h, w = mask.shape[:2]
    if frame_rgb.shape[:2] != (h, w):
        return mask
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_like = np.zeros((h, w), dtype=np.uint8)
    seeds: list[tuple[int, int]] = []
    for point in negative:
        x = max(0, min(w - 1, int(point.x)))
        y = max(0, min(h - 1, int(point.y)))
        seeds.append((x, y))
        radius = max(2, int(seed_radius))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = lab[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            continue
        reference = np.median(patch, axis=0)
        local_distance = np.linalg.norm(patch - reference.reshape(1, 3), axis=1)
        local_tolerance = float(np.percentile(local_distance, 90)) + 8.0
        limit = max(float(tolerance), min(38.0, local_tolerance))
        distance = np.linalg.norm(lab - reference.reshape(1, 1, 3), axis=2)
        background_like[(distance <= limit) & (mask > 0)] = 1

    if not background_like.any():
        return mask
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(background_like, 8)
    remove = np.zeros((h, w), dtype=bool)
    for label in range(1, count):
        component = labels == label
        if any(
            bool(component[min(h - 1, max(0, y)), min(w - 1, max(0, x))])
            for x, y in seeds
        ) and not any(
            bool(component[min(h - 1, max(0, point.y)), min(w - 1, max(0, point.x))])
            for point in points
            if point.label == 1
        ):
            remove |= component
    result = mask.copy()
    result[remove] = False
    positive_points = [point for point in points if point.label == 1]
    for point in positive_points:
        x = max(0, min(w - 1, int(point.x)))
        y = max(0, min(h - 1, int(point.y)))
        if not result[y, x]:
            return mask
    if not positive_points and int(result.sum()) < max(1, int(mask.sum() * 0.05)):
        return mask
    return result


__all__ = ["choose_prompt_mask_index", "prompt_consistency", "trim_prompt_background"]
