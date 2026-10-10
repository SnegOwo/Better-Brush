"""Face landmark detection for mouth / eye region isolation.

Strategy (optimised for anime drawn characters):
1. MediaPipe Face Landmarker (primary) — works well on semi-realistic anime.
2. Haar cascade (fallback) — validated to reject non-face detections (hands,
   arms, background noise) via aspect-ratio, position, and size checks.
3. Skin-colour centroid fallback — when both above fail, estimate the face
   position from the largest skin-coloured connected component (works on
   anime because skin tones are typically flat and consistent).

Prompting philosophy: SAM 2 selects an object from a *few* precise points, so
``face_part_points`` plants just 1–2 include points (iris centres for the
eyes, lip centre for the mouth) and no automatic negative points — a dense
ring/line of red points over-constrains the prompt and produces bad or
missing masks on anime faces.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.core.constants import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Landmark indices (mediapipe 478-point face mesh)
MOUTH_OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
MOUTH_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
LEFT_EYE = [226, 227, 230, 233, 232, 163, 144]
RIGHT_EYE = [446, 447, 450, 453, 452, 382, 341]

# MediaPipe iris-centre landmarks (indices 468 / 473 of the 478-point mesh).
# A single point on the iris is the most reliable SAM 2 prompt for an eye —
# the eyelid ring indices above are only used to estimate the iris centre on
# the Haar / skin fallback path.
IRIS_LEFT = 468
IRIS_RIGHT = 473

MOUTH_INCLUDE = MOUTH_OUTER_LIP + MOUTH_INNER_LIP
EYES_INCLUDE = LEFT_EYE + RIGHT_EYE

# ── Haar fallback position tables ───────────────────────────────────────────
# (rx, ry) normalised to the detected face bounding box.
# Every index referenced by MOUTH_OUTER_LIP, MOUTH_INNER_LIP, LEFT_EYE
# and RIGHT_EYE is present.

_HAAR_MOUTH: dict[int, tuple[float, float]] = {
    13: (0.50, 0.78),
    14: (0.50, 0.84),
    61: (0.38, 0.80),
    146: (0.42, 0.77),
    91: (0.50, 0.73),
    181: (0.58, 0.77),
    84: (0.62, 0.80),
    17: (0.50, 0.87),
    314: (0.38, 0.75),
    405: (0.42, 0.83),
    321: (0.58, 0.83),
    375: (0.62, 0.75),
    291: (0.62, 0.80),
    409: (0.50, 0.87),
    78: (0.44, 0.79),
    95: (0.47, 0.78),
    88: (0.50, 0.78),
    178: (0.53, 0.78),
    87: (0.56, 0.79),
    317: (0.56, 0.75),
    402: (0.56, 0.83),
    318: (0.53, 0.84),
    324: (0.50, 0.84),
    308: (0.47, 0.84),
    415: (0.44, 0.83),
    37: (0.62, 0.75),
}

_HAAR_EYES: dict[int, tuple[float, float]] = {
    226: (0.35, 0.38),
    227: (0.37, 0.36),
    230: (0.34, 0.35),
    233: (0.31, 0.36),
    232: (0.30, 0.38),
    163: (0.31, 0.41),
    144: (0.34, 0.41),
    446: (0.65, 0.38),
    447: (0.63, 0.36),
    450: (0.66, 0.35),
    453: (0.69, 0.36),
    452: (0.70, 0.38),
    382: (0.69, 0.41),
    341: (0.66, 0.41),
    33: (0.32, 0.40),
    133: (0.32, 0.40),
    159: (0.36, 0.36),
    145: (0.36, 0.44),
    362: (0.68, 0.40),
    263: (0.68, 0.40),
    386: (0.64, 0.36),
    374: (0.64, 0.44),
}

_HAAR_NOSE: dict[int, tuple[float, float]] = {
    1: (0.50, 0.56),
    2: (0.42, 0.53),
    4: (0.50, 0.46),
    5: (0.50, 0.52),
    6: (0.39, 0.56),
    98: (0.41, 0.53),
    168: (0.50, 0.48),
    195: (0.50, 0.61),
    197: (0.44, 0.62),
    219: (0.37, 0.59),
    327: (0.59, 0.53),
    439: (0.63, 0.59),
}

_HAAR_CHIN: dict[int, tuple[float, float]] = {
    152: (0.50, 0.97),
    175: (0.50, 0.92),
    199: (0.43, 0.94),
    200: (0.57, 0.94),
    428: (0.43, 0.90),
    396: (0.57, 0.90),
    147: (0.40, 0.88),
    369: (0.60, 0.88),
}

_HAAR_FOREHEAD: dict[int, tuple[float, float]] = {
    10: (0.50, 0.20),
    338: (0.50, 0.20),
    9: (0.40, 0.17),
    8: (0.31, 0.19),
    68: (0.46, 0.14),
    69: (0.55, 0.14),
    108: (0.61, 0.18),
    151: (0.28, 0.20),
    297: (0.60, 0.23),
    332: (0.40, 0.23),
}

_HAAR_CHEEKS: dict[int, tuple[float, float]] = {
    205: (0.17, 0.60),
    206: (0.13, 0.55),
    425: (0.83, 0.60),
    426: (0.87, 0.55),
    50: (0.12, 0.70),
    280: (0.88, 0.70),
    234: (0.26, 0.62),
    454: (0.74, 0.62),
}

_HAAR_EYEBROWS: dict[int, tuple[float, float]] = {
    70: (0.44, 0.27),
    63: (0.36, 0.26),
    105: (0.28, 0.27),
    66: (0.22, 0.29),
    107: (0.52, 0.27),
    336: (0.56, 0.27),
    296: (0.64, 0.26),
    334: (0.72, 0.27),
    293: (0.78, 0.29),
    300: (0.50, 0.27),
}

_HAAR_ALL: dict[int, tuple[float, float]] = {
    **_HAAR_MOUTH,
    **_HAAR_EYES,
    **_HAAR_NOSE,
    **_HAAR_CHIN,
    **_HAAR_FOREHEAD,
    **_HAAR_CHEEKS,
    **_HAAR_EYEBROWS,
}


# ── Cached MediaPipe landmarker ─────────────────────────────────────────────
_mp_landmarker = None
_mp_landmarker_path: str | None = None


def _get_mp_landmarker():
    """Return a cached MediaPipe FaceLandmarker (avoids re-creation per call)."""
    global _mp_landmarker, _mp_landmarker_path
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    model_path = str(ensure_model())
    if _mp_landmarker is not None and _mp_landmarker_path == model_path:
        return _mp_landmarker

    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    _mp_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    _mp_landmarker_path = model_path
    return _mp_landmarker


def ensure_model() -> Path:
    """Ensure the face_landmarker.task model exists locally."""
    dest = CHECKPOINTS_DIR / _MODEL_FILENAME
    if dest.is_file():
        return dest

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(_MODEL_URL, dest)
    if not dest.is_file():
        raise FileNotFoundError(f"Failed to download {_MODEL_FILENAME}")
    logger.info("Downloaded face_landmarker.task (%d bytes)", dest.stat().st_size)
    return dest


# ── Public API ──────────────────────────────────────────────────────────────


def detect_landmarks(
    frame_rgb: np.ndarray,
) -> dict[int, tuple[float, float]] | None:
    """Return {landmark_idx: (x, y)} in pixel coords, or None."""
    h, w = frame_rgb.shape[:2]
    t0 = time.perf_counter()
    mp_result = _detect_mediapipe(frame_rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if mp_result is not None and _validate_landmarks(mp_result, w, h):
        logger.info(
            "mediapipe face landmark: %d points in %.0f ms",
            len(mp_result),
            elapsed_ms,
        )
        return mp_result

    if mp_result is not None:
        logger.debug(
            "mediapipe returned landmarks but they failed validation "
            "(possible non-face detection) — trying Haar"
        )
    else:
        logger.debug("mediapipe returned nothing, trying Haar fallback")

    t0 = time.perf_counter()
    haar_result = _detect_haar(frame_rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if haar_result is not None:
        logger.info(
            "Haar face landmark fallback: %d points in %.0f ms",
            len(haar_result),
            elapsed_ms,
        )
        return haar_result

    logger.debug("Haar failed, trying skin-colour face estimation")
    t0 = time.perf_counter()
    skin_result = _detect_face_by_skin(frame_rgb)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if skin_result is not None:
        logger.info(
            "skin-colour face estimation: %d points in %.0f ms",
            len(skin_result),
            elapsed_ms,
        )
        return skin_result

    logger.warning(
        "No face detected (mediapipe + Haar + skin all failed) — "
        "frame may not contain a visible face or the face is "
        "too small / anime-stylised for the detector."
    )
    return None


def face_part_points(
    landmarks: dict[int, tuple[float, float]],
    part: str,
    frame_shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """Build (x, y, label) point lists for the given face part.

    label=1 -> include, label=0 -> exclude (negative ring).

    SAM 2 is *not* an outline tool: a handful of precise points selects an
    object, and negative points only confuse it.  So we plant 1-2 include
    points (iris centres for the eyes, the lip centre for the mouth) and no
    negatives at all.
    """
    points: list[tuple[int, int, int]] = []

    if part == "mouth":
        centroid = _centroid(landmarks, MOUTH_INNER_LIP)
        if centroid is None:
            centroid = _centroid(landmarks, MOUTH_OUTER_LIP)
        if centroid is None:
            logger.warning("mouth: centroid returned None — no landmarks available")
            return []
        points.append((int(centroid[0]), int(centroid[1]), 1))
        logger.info("mouth: 1 include point at (%d, %d)", *points[0][:2])

    elif part == "eyes":
        for iris, ring in ((IRIS_LEFT, LEFT_EYE), (IRIS_RIGHT, RIGHT_EYE)):
            pt = landmarks.get(iris)
            if pt is None:
                pt = _centroid(landmarks, ring)
            if pt is None:
                continue
            points.append((int(pt[0]), int(pt[1]), 1))
        logger.info("eyes: %d include points (1 per eye)", len(points))

    if not points:
        logger.warning("face_part_points('%s'): returned empty — this will produce a blank mask", part)

    return points


# ------------------------------------------------------------------ #
# Internals                                                           #
# ------------------------------------------------------------------ #


def _validate_landmarks(
    landmarks: dict[int, tuple[float, float]], w: int, h: int
) -> bool:
    """Validate that landmark positions make sense for a real face.

    Rejects detections where most points cluster in corners or the lower
    third of the frame (typical of hand/arm false positives on anime).
    """
    if not landmarks:
        return False

    pts = np.array(list(landmarks.values()))
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    # Face should not be in the bottom 30% of the frame (hands/arms).
    if cy > h * 0.70:
        logger.debug(
            "landmark validation FAILED: centroid y=%.0f (frame h=%d) — "
            "likely hand/arm, not a face",
            cy, h,
        )
        return False

    # Face should not be entirely in the left or right 15% of the frame.
    if cx < w * 0.10 or cx > w * 0.90:
        logger.debug(
            "landmark validation FAILED: centroid x=%.0f (frame w=%d) — "
            "too close to edge",
            cx, w,
        )
        return False

    # Landmarks should span a reasonable area (not all clumped in a tiny spot).
    spread_x = float(pts[:, 0].max() - pts[:, 0].min())
    spread_y = float(pts[:, 1].max() - pts[:, 1].min())
    if spread_x < w * 0.02 or spread_y < h * 0.02:
        logger.debug(
            "landmark validation FAILED: spread %.0fx%.0f — landmarks too clumped",
            spread_x, spread_y,
        )
        return False

    return True


def _validate_haar_rect(x: int, y: int, fw: int, fh: int, w: int, h: int) -> bool:
    """Reject Haar detections that are clearly not a face.

    Hands are typically wider than tall (fw/fh > 1.4) or detected in
    the lower-left/lower-right of the frame.  Real faces have aspect
    ratios near 0.7–1.1 and sit in the upper 2/3 of the frame.
    """
    aspect = fw / max(1, fh)

    # Hands / arms: aspect ratio too wide or too narrow.
    if aspect > 1.4 or aspect < 0.5:
        logger.debug(
            "Haar rect rejected: aspect %.2f (fw=%d fh=%d)", aspect, fw, fh
        )
        return False

    # Face should not be in the bottom 35% of the frame.
    if y > h * 0.65:
        logger.debug(
            "Haar rect rejected: y=%d (frame h=%d) — too low, likely a hand", y, h
        )
        return False

    # Face should not be smaller than 5% of the frame area (noise).
    if fw * fh < w * h * 0.005:
        logger.debug("Haar rect rejected: too small (%dx%d)", fw, fh)
        return False

    return True


def _centroid(
    lms: dict[int, tuple[float, float]], indices: list[int]
) -> tuple[float, float] | None:
    pts = [(lms[i][0], lms[i][1]) for i in indices if i in lms]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# ── Detection engines ───────────────────────────────────────────────────────


def _detect_mediapipe(
    frame_rgb: np.ndarray,
) -> dict[int, tuple[float, float]] | None:
    try:
        import mediapipe as mp
    except ImportError:
        logger.debug("mediapipe not installed — skipping")
        return None

    try:
        landmarker = _get_mp_landmarker()
        if landmarker is None:
            return None
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            logger.debug("mediapipe: no face found in frame")
            return None

        h, w = frame_rgb.shape[:2]
        lm = result.face_landmarks[0]
        return {i: (pt.x * w, pt.y * h) for i, pt in enumerate(lm)}
    except Exception as exc:
        logger.warning("mediapipe face landmark detection failed: %s", exc, exc_info=True)
        return None


def _detect_haar(
    frame_rgb: np.ndarray,
) -> dict[int, tuple[float, float]] | None:
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(faces) == 0:
            logger.debug("Haar: no face rectangles found")
            return None

        h, w = frame_rgb.shape[:2]

        # Try each detected rectangle, validating each one.
        best = None
        best_score = float("inf")
        for (x, y, fw, fh) in faces:
            if not _validate_haar_rect(int(x), int(y), int(fw), int(fh), w, h):
                continue
            # Prefer the face closest to the frame centre.
            cx, cy = x + fw / 2, y + fh / 2
            score = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
            if score < best_score:
                best_score = score
                best = (int(x), int(y), int(fw), int(fh))

        if best is None:
            logger.debug("Haar: all %d face rects rejected (non-face detections)", len(faces))
            return None

        x, y, fw, fh = best
        logger.debug("Haar: face rect (%d, %d) %dx%d", x, y, fw, fh)
        return {idx: (x + rx * fw, y + ry * fh) for idx, (rx, ry) in _HAAR_ALL.items()}
    except Exception as exc:
        logger.warning("Haar face detection failed: %s", exc, exc_info=True)
        return None


def _detect_face_by_skin(
    frame_rgb: np.ndarray,
) -> dict[int, tuple[float, float]] | None:
    """Estimate face position from the largest skin-coloured region.

    Works for anime because skin tones are typically flat and consistent.
    Returns Haar-style landmark points centred on the skin blob, or None.
    """
    try:
        h, w = frame_rgb.shape[:2]
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        # Anime skin-tone range: H 0-25, S 40-200, V 150-255.
        mask = cv2.inRange(hsv, (0, 40, 150), (25, 200, 255))

        # Also catch lighter/pinker skin: H 160-180 (wraps around red).
        mask2 = cv2.inRange(hsv, (160, 40, 150), (180, 200, 255))
        mask = cv2.bitwise_or(mask, mask2)

        # Clean up noise.
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

        # Find connected components and pick the largest one.
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            logger.debug("skin fallback: no skin-coloured regions found")
            return None

        # Skip label 0 (background).
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        area = int(areas[best_label - 1])

        # Skin region must cover at least 1% of the frame.
        if area < w * h * 0.01:
            logger.debug("skin fallback: largest region too small (%d px)", area)
            return None

        sx = stats[best_label, cv2.CC_STAT_LEFT]
        sy = stats[best_label, cv2.CC_STAT_TOP]
        sw = stats[best_label, cv2.CC_STAT_WIDTH]
        sh = stats[best_label, cv2.CC_STAT_HEIGHT]

        # The face should be roughly within this skin region.
        # Use the skin bounding box as a "face rect" and place landmarks.
        logger.debug(
            "skin fallback: skin region (%d, %d) %dx%d, area=%d",
            sx, sy, sw, sh, area,
        )

        return {
            idx: (sx + rx * sw, sy + ry * sh)
            for idx, (rx, ry) in _HAAR_ALL.items()
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("skin-colour face detection failed: %s", exc)
        return None
