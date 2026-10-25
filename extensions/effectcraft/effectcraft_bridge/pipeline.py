"""Turn a job document into an alpha PNG sequence.

Two modes:

``simulate``  no torch, no model, no GPU — a deterministic feathered ellipse
              tracked across the range. It exists so the whole bridge (job
              file → status polling → command generation → import → track
              matte) can be exercised end to end in seconds, and so the tests
              never need a model. The status says ``simulated`` so nobody can
              mistake it for real segmentation.

``engine``    the real Better Brush pipeline: ``src.engine.pipeline.run_job``.
              That entry point prefers a ProRes 4444 movie, so this module
              normalises its output down to the PNG sequence EffectCraft wants.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    STATE_CANCELLED,
    STATE_DONE,
    STATE_ERROR,
    STATE_RUNNING,
    Job,
    Status,
    cancel_requested,
    touch_alive,
    write_status,
)

Progress = Callable[[str, int, int, str], None]
CancelProbe = Callable[[], bool]


class JobCancelled(Exception):
    """Raised when the panel dropped a cancel flag for this job."""


class PipelineFailure(RuntimeError):
    """A job that could not produce a matte. Carries a user-facing message."""

    def __init__(self, message: str, kind: str = "internal") -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------- #
# Reading the source
# --------------------------------------------------------------------- #
def _read_range(job: Job) -> list[np.ndarray]:
    """Frames for ``[frame_in, frame_out]``; a still image is one frame."""
    import cv2

    src = Path(job.source_path)
    if not src.is_file():
        raise PipelineFailure(f"source not found: {src}", "unsupported")

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is not None and image.ndim in (2, 3):
        return [image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)]

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise PipelineFailure(f"cannot decode {src.name}", "unsupported")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        end = job.frame_out
        if end < 0:
            end = max(0, total - 1)
        if job.frame_in > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, job.frame_in)
        frames: list[np.ndarray] = []
        index = job.frame_in
        while index <= end:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            index += 1
        if not frames:
            raise PipelineFailure("the requested frame range is empty", "unsupported")
        return frames
    finally:
        cap.release()


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# --------------------------------------------------------------------- #
# Simulated segmentation
# --------------------------------------------------------------------- #
def _simulated_alpha(index: int, width: int, height: int) -> np.ndarray:
    """A soft, moving ellipse. Deterministic, so tests can assert on it.

    The wobble is closed-form: frame 0 and frame 30 differ, so a scrub in
    EffectCraft shows the matte tracking rather than a static cut-out.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width * (0.5 + 0.035 * math.sin(index * 0.15))
    cy = height * (0.5 + 0.025 * math.cos(index * 0.11))
    rx = max(4.0, width * 0.26)
    ry = max(4.0, height * 0.42)
    dist = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    # 0.5 sits on the contour; the slope sets the feather to roughly 1/18 of
    # the radius, which reads as a soft edge instead of a staircase.
    return np.clip((1.0 - dist) * 18.0 + 0.5, 0.0, 1.0).astype(np.float32)


def _write_png(path: Path, bgr: np.ndarray, alpha: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack([bgr, (alpha * 255.0 + 0.5).astype(np.uint8)])
    if not cv2.imwrite(str(path), rgba):
        raise PipelineFailure(f"could not write {path.name}")


def _stroke_polys(polys: Any) -> list[np.ndarray]:
    import cv2

    return [
        np.array([[int(round(x)), int(round(y))] for x, y in poly], np.int32)
        for poly in polys
    ]


def apply_strokes(alpha: np.ndarray, job: Job) -> np.ndarray:
    """Layer the correction's polygons over an alpha, engine semantics.

    Mirrors ``src.engine.pipeline._apply_correction``: the *add* polygons are
    unioned in first and the *remove* polygons trim afterwards, so a region
    both strokes cover ends up background. Used by ``simulate`` mode, where
    there is no engine to do it; engine mode sends the same polygons as PNG
    masks instead and lets ``JobRequest`` apply them to the stored seed.
    """
    corr = job.correction
    if corr is None or (not corr.add and not corr.remove):
        return alpha
    import cv2

    out = np.asarray(alpha, dtype=np.float32).copy()
    if corr.add:
        mask = np.zeros(out.shape, dtype=np.uint8)
        cv2.fillPoly(mask, _stroke_polys(corr.add), 255)
        out[mask > 0] = 1.0
    if corr.remove:
        mask = np.zeros(out.shape, dtype=np.uint8)
        cv2.fillPoly(mask, _stroke_polys(corr.remove), 255)
        out[mask > 0] = 0.0
    return out


def run_simulate(
    job: Job,
    out_dir: Path,
    on_progress: Progress,
    cancelled: CancelProbe,
) -> dict[str, Any]:
    frames = _read_range(job)
    total = len(frames)
    on_progress("extract", 0, total, "reading source")

    prepared: list[tuple[int, np.ndarray, np.ndarray]] = []
    for offset, raw in enumerate(frames):
        if cancelled():
            raise JobCancelled()
        bgr = _as_bgr(raw)
        h, w = bgr.shape[:2]
        if offset == 0:
            on_progress("analyze", 1, total, "simulated subject (no model)")
        prepared.append((offset, bgr, apply_strokes(_simulated_alpha(offset, w, h), job)))
    on_progress("analyze", total, total, "simulated subject (no model)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (offset, bgr, alpha) in enumerate(prepared):
        if cancelled():
            raise JobCancelled()
        _write_png(out_dir / f"{offset:05d}.png", bgr, alpha)
        if i % 5 == 0 or i == total - 1:
            on_progress("export", i + 1, total, "writing PNG frames")

    post = _post_process(job, out_dir, on_progress, cancelled)
    h, w = prepared[0][1].shape[:2]
    return {
        "png_dir": str(out_dir),
        "first_frame": str(out_dir / "00000.png"),
        "frame_count": total,
        "frame_in": job.frame_in,
        "frame_out": job.frame_in + total - 1,
        "fps": float(job.fps or 24.0),
        "width": int(w),
        "height": int(h),
        "engine": "simulate",
        "model": "none (simulate mode)",
        "simulated": True,
        # Same folder layout as the engine run, so a later correction has a
        # resume_dir to send even when this was only a plumbing check.
        "work_dir": str(out_dir.parent),
        "post_processed": post is not None,
        "post": post,
    }


def _post_process(
    job: Job,
    png_dir: Path,
    on_progress: Progress,
    cancelled: CancelProbe,
) -> dict[str, Any] | None:
    """The adapter's own stage ``post``; ``None`` when the toggle is off."""
    if not job.post_process or not png_dir.is_dir():
        return None
    from .postprocess import PostProcessCancelled, post_process_sequence

    on_progress("post", 0, 1, "post-processing the matte")
    try:
        info = post_process_sequence(
            png_dir,
            on_progress=lambda cur, tot, msg: on_progress("post", cur, tot, msg),
            cancelled=cancelled,
        )
    except PostProcessCancelled as exc:
        raise JobCancelled() from exc
    on_progress("post", info["frames"], info["frames"], "post-processing done")
    return info


# --------------------------------------------------------------------- #
# Real pipeline
# --------------------------------------------------------------------- #
def find_ffmpeg() -> str | None:
    """FFmpeg for ``alpha.mov`` → PNG. Three sources, cheapest first."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:  # bundled with imageio-ffmpeg, a declared dependency
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except Exception:  # noqa: BLE001 - optional at runtime
        pass
    # The repo's own copy (gitignored; present next to a local checkout).
    repo = Path(__file__).resolve().parents[2] / "For-Agent" / "ffmpeg.exe"
    return str(repo) if repo.is_file() else None


def transcode_to_png_sequence(mov: Path, out_dir: Path) -> int:
    """Decode an alpha movie into ``%05d.png`` RGBA frames. Returns the count.

    ``run_job`` writes ProRes 4444 by default; OpenCV cannot be trusted to
    carry a fourth channel through that decode, so the conversion goes to
    FFmpeg — which produced the file in the first place.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise PipelineFailure(
            "the pipeline wrote an alpha movie but no FFmpeg was found to "
            "convert it into a PNG sequence. Install FFmpeg (or `pip install "
            f"imageio-ffmpeg`) and re-run. The movie is at: {mov}",
            "unsupported",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # -y: the folder is ours. format=rgba forces the alpha plane to survive.
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(mov),
        "-vf", "format=rgba", str(out_dir / "%05d.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:]
        raise PipelineFailure(
            "FFmpeg could not convert the alpha movie: "
            + (" ".join(tail) or "unknown error")
        )
    return len(list(out_dir.glob("*.png")))


class _FlagEvent:
    """``threading.Event``-shaped cancel flag backed by a file on disk.

    ``run_job`` only ever calls ``is_set()``, and the panel's cancel signal is
    a file, because the panel must not need a socket to stop a job.
    """

    def __init__(self, root: Path, job_id: str) -> None:
        from .protocol import cancel_path

        self._path = cancel_path(root, job_id)

    def is_set(self) -> bool:
        return self._path.exists()

    def set(self) -> None:  # the engine never sets it; keep the protocol shape
        return None


def run_engine(
    job: Job,
    out_dir: Path,
    on_progress: Progress,
    root: Path,
) -> dict[str, Any]:
    """Run the real Better Brush pipeline and normalise its output to PNGs."""
    try:
        from src.engine.pipeline import run_job
        from src.engine.schemas import JobRequest
    except ImportError as exc:  # pragma: no cover - needs the real env
        raise PipelineFailure(
            "the Better Brush engine is not importable from here "
            f"({exc}). Run the adapter from the repository root, or use "
            "--mode simulate.",
            "unsupported",
        ) from exc

    flag = _FlagEvent(root, job.id)
    request = JobRequest(
        input_path=job.source_path,
        frame_in=job.frame_in,
        frame_out=job.frame_out if job.frame_out >= 0 else -1,
        fps=job.fps,
        width=job.width,
        height=job.height,
        preset=job.preset,
        quality=job.quality,
        output_dir=str(out_dir.parent),
        matting=job.matting,
        resume_dir=_resume_dir_for(job, root),
        correction_add=_stroke_png(job, "add"),
        correction_remove=_stroke_png(job, "remove"),
        correction_frame=job.correction.frame if job.correction else None,
    )
    result = run_job(
        request,
        on_progress=on_progress,
        cancel_event=flag,
    )

    if result.output_kind == "png_sequence":
        png_dir = Path(result.output_path)
    else:
        mov = Path(result.output_path)
        if not mov.is_file():
            raise PipelineFailure(f"expected an alpha movie at {mov}")
        png_dir = out_dir
        transcode_to_png_sequence(mov, png_dir)

    post = _post_process(job, png_dir, on_progress, flag.is_set)
    first = _first_png(png_dir)
    if first is None:
        raise PipelineFailure("the pipeline produced no PNG frames")
    return {
        "png_dir": str(png_dir),
        "first_frame": str(first),
        "frame_count": result.frame_count,
        "frame_in": result.frame_in,
        "frame_out": result.frame_out,
        "fps": float(result.fps or job.fps or 24.0),
        "width": int(result.width),
        "height": int(result.height),
        "engine": "betterbrush",
        "model": result.model,
        "simulated": False,
        "qc": dict(result.qc),
        "matting": bool(result.matting),
        # The folder a *later* correction resumes from: seed, masks and frames.
        "work_dir": str(result.work_dir or out_dir.parent),
        "quality": job.quality,
        "post_processed": post is not None,
        "post": post,
    }


def _resume_dir_for(job: Job, root: Path) -> str | None:
    """Validate the earlier run's folder before the engine is pointed at it.

    The panel sends a path it found in a previous status document; it is read
    back from disk here, so it has to be (a) inside this bridge's ``out``
    folder — a job file must not be able to aim the adapter at arbitrary
    filesystem locations — and (b) still present.
    """
    corr = job.correction
    if corr is None:
        return None
    candidate = Path(corr.resume_dir)
    allowed = (root / "out").resolve()
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - unreadable path
        raise PipelineFailure(f"cannot reach the earlier run: {exc}", "bad_request")
    if resolved != allowed and allowed not in resolved.parents:
        raise PipelineFailure(
            "the correction points at a folder outside this bridge's out/ "
            "directory; re-run Remove Background first.",
            "bad_request",
        )
    if not resolved.is_dir():
        raise PipelineFailure(
            f"the earlier run's folder is gone: {resolved.name} — re-run "
            "Remove Background before correcting it.",
            "bad_request",
        )
    return str(resolved)


def _stroke_png(job: Job, kind: str) -> str | None:
    """Rasterise one stroke set to a base64 PNG at source resolution.

    ``JobRequest.correction_add`` / ``_remove`` take masks, not polygons, so
    the conversion happens here — in the adapter, where it can be tested,
    rather than in the panel.
    """
    corr = job.correction
    if corr is None:
        return None
    polys = corr.add if kind == "add" else corr.remove
    if not polys:
        return None
    import base64

    import cv2

    mask = np.zeros((job.height, job.width), dtype=np.uint8)
    if kind == "remove":
        # Exactly the region erase_removed() stamps out afterwards, so the
        # engine's seed and the finished frames agree on what is deleted.
        region = remove_region(job)
        if region is not None:
            mask[region] = 255
    else:
        cv2.fillPoly(mask, _stroke_polys(polys), 255)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise PipelineFailure(f"could not encode the {kind} strokes", "internal")
    return base64.b64encode(buf.tobytes()).decode("ascii")


#: How far each step of ``correction.remove_force`` (1-5) grows a red stroke,
#: as a fraction of the frame width. Force 3, the default, adds ~0.5% of the
#: width: enough to take an edge painted just inside the subject.
_REMOVE_GROWTH = 0.0025


def remove_region(job: Job) -> np.ndarray | None:
    """The red strokes as a boolean mask, grown by ``remove_force``.

    ``None`` when the job carries no remove stroke.
    """
    corr = job.correction
    if corr is None or not corr.remove:
        return None

    import cv2

    region = np.zeros((job.height, job.width), dtype=np.uint8)
    cv2.fillPoly(region, _stroke_polys(corr.remove), 255)
    step = max(0, int(corr.remove_force) - 1)
    if step:
        px = max(1, int(round(job.width * _REMOVE_GROWTH * step)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (px * 2 + 1, px * 2 + 1))
        region = cv2.dilate(region, kernel)
    return region > 0


def erase_removed(job: Job, png_dir: Path, on_progress: Progress) -> int:
    """Stamp the red strokes out of every finished frame they still cover.

    The engine applies a ``remove`` stroke to the *seed* frame and re-propagates
    from there, so a large object can creep back frame by frame — which is why
    the same spot had to be painted again and again in the field. Here the
    painted region is also erased from the finished frames, but **only where
    the matte is still opaque**: an object that has moved away leaves the
    background underneath untouched, and already-transparent pixels are not
    "erased" twice.

    Returns how many frames changed.
    """
    region = remove_region(job)
    if region is None:
        return 0

    import cv2

    frames = sorted(png_dir.glob("*.png"))
    if not frames:
        return 0
    erased = 0
    for n, path in enumerate(frames):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] < 4:
            continue
        if image.shape[0] != region.shape[0] or image.shape[1] != region.shape[1]:
            # Frames at another size cannot be cut with a mask drawn at this
            # one; leave them alone instead of cropping blindly.
            continue
        alpha = image[:, :, 3]
        hit = region & (alpha > 0)
        if not hit.any():
            continue
        alpha[hit] = 0
        cv2.imwrite(str(path), image)
        erased += 1
        if n % 8 == 0:
            on_progress("post", n + 1, len(frames), "erasing what you painted")
    if erased:
        on_progress("post", len(frames), len(frames), "erasing what you painted")
    return erased


def _first_png(png_dir: Path) -> Path | None:
    if not png_dir.is_dir():
        return None
    for p in sorted(png_dir.glob("*.png")):
        return p
    return None


#: The preview's fixed canvas. ScriptUI's image control renders an image at its
#: natural size (it does not scale), so the adapter — not the panel — decides
#: the size, and every panel layout sees the same box.
PREVIEW_SIZE = (300, 169)


def write_preview(png_dir: Path, job: Job) -> str:
    """A small letterboxed JPEG of the frame a run is *about*, for the panel.

    The frame chosen is the correction's key frame when there is one, else the
    first: those are the frames the user is looking at when they judge the
    result. Alpha is composited over a checkerboard so a cut-out reads as a
    cut-out instead of as a black rectangle.

    A preview is decoration: any failure returns ``""`` rather than failing a
    job that already rendered.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is required to render anyway
        return ""
    try:
        index = job.correction.frame if job.correction else 0
        path = png_dir / f"{index:05d}.png"
        if not path.is_file():
            path = _first_png(png_dir) or Path()
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            return ""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:h, 0:w]
        squares = ((xx // 10 + yy // 10) % 2).astype(np.float32)
        checker = (90.0 + 50.0 * squares)[:, :, None]
        flat = rgba[:, :, :3].astype(np.float32) * alpha + checker * (1.0 - alpha)

        pw, ph = PREVIEW_SIZE
        scale = min(pw / w, ph / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(flat, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((ph, pw, 3), 40.0, np.float32)
        y0, x0 = (ph - nh) // 2, (pw - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized

        out = png_dir.parent / "preview.jpg"
        if not cv2.imwrite(str(out), np.clip(canvas, 0, 255).astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
            return ""
        return str(out)
    except Exception:  # noqa: BLE001 - a preview must never fail a job
        return ""


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #
def execute(job: Job, root: Path, *, mode: str | None = None,
            log: Callable[[str], None] | None = None,
            on_done: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
            ) -> Status:
    """Run one job to a terminal state, keeping its status file current.

    ``on_done`` decorates the result dict immediately before the final status
    write. The adapter uses it to attach the EffectCraft command list, so the
    panel never sees a ``done`` status that is still missing its commands.
    """
    status = Status(id=job.id, state=STATE_RUNNING, started=_now())
    write_status(root, status)
    # The panel watches adapter.alive to decide whether EffectCraft needs to
    # start an adapter: a job in flight means this one is very much alive,
    # even though the watch loop is blocked inside it.
    touch_alive(root)
    out_dir = job.png_dir(root)
    chosen = mode or job.mode
    started = time.time()
    last_write = [0.0]

    def on_progress(stage: str, current: int, total: int, message: str) -> None:
        # Every report is proof the adapter is working, so stamp the heartbeat
        # here too: the watch loop is blocked inside this call chain, and the
        # panel must never read that silence as "no adapter".
        touch_alive(root)
        status.stage = stage
        status.current = int(current)
        status.total = int(total)
        status.message = message or stage
        # Throttle: the pipeline can report every frame and the panel is
        # reading this same file off the same disk.
        now = time.monotonic()
        finished = bool(total) and current >= total
        if now - last_write[0] >= 0.25 or finished:
            last_write[0] = now
            write_status(root, status)

    def cancelled() -> bool:
        return cancel_requested(root, job.id)

    try:
        if chosen == "simulate":
            result = run_simulate(job, out_dir, on_progress, cancelled)
        else:
            result = run_engine(job, out_dir, on_progress, root)
        # Severity: a red stroke is meant to *delete*, and the engine only trims
        # the seed. Stamp the painted region out of the finished frames too.
        erased = erase_removed(job, Path(result["png_dir"]), on_progress)
        if erased:
            result["erased_frames"] = erased
        result["mode"] = chosen
        result["duration_s"] = round(time.time() - started, 3)
        # The panel's result image: written after the post pass so it shows
        # the alpha the user will actually get.
        result["preview"] = write_preview(Path(result["png_dir"]), job)
        if on_done is not None:
            result = on_done(result)
        status.state = STATE_DONE
        status.stage = "done"
        if not status.total:
            status.total = result["frame_count"]
            status.current = result["frame_count"]
        status.message = (
            f"{'simulated' if result.get('simulated') else 'rendered'} "
            f"{result['frame_count']} frames in {result['duration_s']}s"
        )
        status.result = result
    except Exception as exc:  # noqa: BLE001 - one job must not kill the watcher
        if type(exc).__name__ == "JobCancelled":
            status.state = STATE_CANCELLED
            status.message = "cancelled"
        else:
            status.state = STATE_ERROR
            status.error = {
                "kind": str(getattr(exc, "kind", "internal")),
                "message": str(exc),
            }
            status.message = str(exc)
        if log:
            log(f"job {job.id}: {type(exc).__name__}: {exc}")
    write_status(root, status)
    # Drop any cancel flag once the job is terminal. Leaving it behind would
    # make the next job with the same id — or a manual retry of this one —
    # cancel itself before it started.
    try:
        from .protocol import cancel_path

        flag = cancel_path(root, job.id)
        if flag.is_file():
            flag.unlink()
    except OSError:
        pass
    return status


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


__all__ = [
    "JobCancelled",
    "PipelineFailure",
    "execute",
    "find_ffmpeg",
    "run_engine",
    "run_simulate",
    "transcode_to_png_sequence",
]
