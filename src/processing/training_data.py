"""Export a training dataset for SAM 2 fine-tuning.

Better Brush doubles as the data engine: segment a clip, correct the masks
with the brush, then export frames + masks in the DAVIS-style layout that
the official SAM 2 training code consumes (``training.dataset.vos_raw_dataset.
PNGRawDataset`` with ``is_palette=False``)::

    <dataset_root>/JPEGImages/<clip>/<frame:05d>.jpg
    <dataset_root>/Annotations/<clip>/<obj_index>/<frame:05d>.png
    <dataset_root>/ImageSets/<split>.txt

Each BB object becomes one ``Annotations`` folder (its obj_id is remapped to
a contiguous index, because deleted objects can leave gaps). Only frames that
carry at least one mask are exported, so the positive signal is not drowned
out by empty frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Layout the official SAM 2 trainer reads (``PNGRawDataset``).
IMG_SUBDIR = "JPEGImages"
GT_SUBDIR = "Annotations"
SPLIT_SUBDIR = "ImageSets"


@dataclass(frozen=True)
class TrainingExportResult:
    """Summary of one export run."""

    clip: str
    frames: int
    objects: int
    dataset_root: Path


def export_clip(
    dataset_root: Path | str,
    clip_name: str,
    frame_getter: Callable[[int], np.ndarray],
    object_masks: list[dict[int, np.ndarray]],
    split: str = "train",
    frame_quality: int = 95,
    on_progress: Callable[[int, int], None] | None = None,
) -> TrainingExportResult:
    """Write one clip's frames and masks into a training dataset.

    Args:
        dataset_root: Folder that holds the ``JPEGImages`` / ``Annotations``
            subdirectories (created when missing).
        clip_name: Name of the clip inside the dataset (filesystem-safe).
        frame_getter: Returns the RGB uint8 frame for a frame index.
        object_masks: One ``{frame_index: bool mask}`` dict per object.
        split: Split file the clip is appended to.
        frame_quality: JPEG quality for exported frames.
        on_progress: Optional callback(current, total).

    Returns:
        A :class:`TrainingExportResult` summary.
    """
    root = Path(dataset_root)
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in clip_name
    ).strip("_") or "clip"

    # Frame indices that carry at least one mask, in playback order.
    covered: set[int] = set()
    for masks in object_masks:
        for index, mask in masks.items():
            if mask is not None and np.any(mask):
                covered.add(index)
    indices = sorted(covered)
    if not indices:
        raise ValueError(
            "This clip has no masks to export - segment a subject first."
        )

    img_dir = root / IMG_SUBDIR / safe_name
    gt_dir = root / GT_SUBDIR / safe_name
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # One folder per object, contiguous from 0 (bb obj_ids can have gaps).
    for slot, masks in enumerate(object_masks):
        (gt_dir / str(slot)).mkdir(parents=True, exist_ok=True)

    total = len(indices)
    for position, index in enumerate(indices, start=1):
        frame = frame_getter(index)
        if frame is None:
            raise RuntimeError(f"Frame {index} could not be read")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(img_dir / f"{index:05d}.jpg"),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(frame_quality)],
        )
        for slot, masks in enumerate(object_masks):
            mask = masks.get(index)
            if mask is None or not np.any(mask):
                continue
            png = (mask.astype(np.uint8)) * 255
            cv2.imwrite(str(gt_dir / str(slot) / f"{index:05d}.png"), png)
        if on_progress is not None:
            on_progress(position, total)

    split_dir = root / SPLIT_SUBDIR
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / f"{split}.txt"
    existing = _read_split(split_file)
    if safe_name not in existing:
        existing.append(safe_name)
        split_file.write_text("\n".join(existing) + "\n", encoding="utf-8")

    logger.info(
        "Exported training data for %s: %d frames, %d objects -> %s",
        safe_name,
        total,
        len(object_masks),
        root,
    )
    return TrainingExportResult(
        clip=safe_name,
        frames=total,
        objects=len(object_masks),
        dataset_root=root,
    )


def _read_split(split_file: Path) -> list[str]:
    if not split_file.is_file():
        return []
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataset_summary(dataset_root: Path | str) -> dict:
    """Return clip/frame counts for a dataset (used by the UI + docs)."""
    root = Path(dataset_root)
    img_root = root / IMG_SUBDIR
    if not img_root.is_dir():
        return {"clips": 0, "frames": 0}
    clips = 0
    frames = 0
    for clip_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        clips += 1
        frames += len(list(clip_dir.glob("*.jpg")))
    return {"clips": clips, "frames": frames}
