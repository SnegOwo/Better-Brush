"""Build the looping Better Brush demo GIF from the real application.

Two stages, both reproducible:

    python tools/make_demo_gif.py capture   # drives MainWindow, writes PNGs
    python tools/make_demo_gif.py compose   # PNGs -> docs/images/demo.gif

``capture`` opens a real ``MainWindow`` on the bundled Big Buck Bunny example,
clicks a prompt, runs the actual SAM 2 segmentation and propagation workers,
switches to the cutout preview, paints a brush correction and grabs the real
export dialog. Nothing is mocked: every frame is ``QWidget.grab()`` of the
running app. ``compose`` adds the cursor, click ripple and step captions and
assembles an optimised, seamlessly looping GIF.

The intermediate frames live in ``build/demo-gif/`` (git-ignored) so composing
can be tweaked without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build" / "demo-gif"
RAW_DIR = BUILD_DIR / "raw"
CLIP_SRC = REPO_ROOT / "Sammie-Roto-2-main" / "examples" / "example_bunny.mp4"
CLIP_TRIMMED = BUILD_DIR / "demo_clip.mp4"
OUTPUT_GIF = REPO_ROOT / "docs" / "images" / "demo.gif"

# The bunny at frame 0 (1280x720 source).
PROMPT_POINT = (636, 470)
BRUSH_IMAGE_POINT = (900, 560)

WINDOW_SIZE = (1400, 860)
GIF_WIDTH = 760
FPS = 12
GIF_COLORS = 96


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
@dataclass
class Capture:
    name: str
    path: Path
    kind: str
    meta: dict = field(default_factory=dict)


def _pump(app, seconds: float = 0.0, until=None, timeout: float = 300.0) -> bool:
    """Run the Qt event loop without entering exec()."""
    from PySide6.QtCore import QElapsedTimer

    timer = QElapsedTimer()
    timer.start()
    deadline = seconds * 1000 if seconds else timeout * 1000
    while True:
        app.processEvents()
        if until is not None and until():
            return True
        if seconds and timer.elapsed() >= seconds * 1000:
            return True
        if not seconds and timer.elapsed() >= deadline:
            return False
        time.sleep(0.01)


def _trim_clip() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not CLIP_SRC.is_file():
        raise SystemExit(f"sample clip missing: {CLIP_SRC}")
    if CLIP_TRIMMED.is_file() and CLIP_TRIMMED.stat().st_mtime >= CLIP_SRC.stat().st_mtime:
        return CLIP_TRIMMED
    ffmpeg = shutil.which("ffmpeg") or str(REPO_ROOT / "For-Agent" / "ffmpeg.exe")
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(CLIP_SRC), "-t", "4", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            str(CLIP_TRIMMED),
        ],
        check=True,
    )
    return CLIP_TRIMMED


def capture() -> list[Capture]:
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    sys.path.insert(0, str(REPO_ROOT))

    from PySide6.QtWidgets import QApplication

    from src.processing.inference_worker import InferenceWorker
    from src.ui.components.export_dialog import ExportDialog
    from src.ui.main_window import MainWindow

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True)

    clip = _trim_clip()
    app = QApplication.instance() or QApplication(sys.argv)

    print("building MainWindow ...", flush=True)
    win = MainWindow()
    win.resize(*WINDOW_SIZE)
    win.show()
    _pump(app, 0.4)

    print(f"loading {clip.name} ...", flush=True)
    win._load_video(clip)
    _pump(app, until=lambda: win._scene_cut_worker is None, timeout=120)

    print(f"model: {win._config.recommendation}", flush=True)
    shots: list[Capture] = []

    def shot(name: str, kind: str, **meta) -> None:
        path = RAW_DIR / f"{name}.png"
        win.grab().save(str(path))
        shots.append(Capture(name, path, kind, meta))
        print(f"  captured {name} ({path.stat().st_size // 1024} KB)", flush=True)

    shot("01_start", "start")

    print("adding prompt point ...", flush=True)
    win._on_canvas_point(*PROMPT_POINT, 1)
    _pump(app, 0.2)
    shot("02_point", "point")

    print("segmenting (real SAM 2 inference) ...", flush=True)
    t0 = time.time()
    worker = InferenceWorker(
        win._config.recommendation, sam_manager=win._ensure_sam_manager()
    )
    worker.set_task(win._frame_buffer.get(0), list(win._points))
    worker.finished.connect(win._on_segment_finished)
    worker.run()
    _pump(app, 0.3)
    print(f"  segmentation took {time.time() - t0:.1f}s, masks={len(win._masks)}", flush=True)
    shot("03_segment", "segment")

    print("propagating (real SAM 2 video tracking) ...", flush=True)
    t0 = time.time()
    win._run_propagation(1)
    grabbed = 0
    last = 0
    while win._propagation_worker is not None and win._propagation_worker.isRunning():
        app.processEvents()
        n = len(win._masks)
        if n != last and grabbed < 8 and time.time() - t0 > 0.8:
            grabbed += 1
            last = n
            shot(f"prop_{grabbed:02d}", "prop", masks=n)
        time.sleep(0.05)
    # Propagation schedules post-processing (edge smoothing + temporal vote) on
    # its completion handler; wait for that too, or every later grab carries the
    # "Post-processing" scrim.
    _pump(app, 0.4)
    _pump(
        app,
        until=lambda: win._post_worker is None or not win._post_worker.isRunning(),
        timeout=180,
    )
    _pump(app, 0.6)
    print(f"  propagation took {time.time() - t0:.1f}s, masks={len(win._masks)}", flush=True)
    shot("04_propagation_done", "done")

    total = win._video_reader.total_frames
    print("sweeping the finished matte ...", flush=True)
    steps = 14
    for i in range(steps):
        frame = round(i * (total - 1) / (steps - 1))
        win._set_frame(frame)
        _pump(app, 0.05)
        shot(f"05_sweep_{i:02d}", "sweep", frame=frame)

    print("switching to Cutout ...", flush=True)
    win._set_frame(round(total * 0.55))
    win._combo_preview.setCurrentIndex(1)
    _pump(app, 0.25)
    shot("06_cutout", "cutout")

    print("painting a brush correction ...", flush=True)
    win._combo_preview.setCurrentIndex(0)
    win._set_tool("brush")
    win._canvas.set_brush_size(34)
    canvas = win._canvas
    ix, iy = BRUSH_IMAGE_POINT
    wx = canvas._offset.x() + ix * canvas._scale
    wy = canvas._offset.y() + iy * canvas._scale
    canvas._brush_hover = (int(wx), int(wy))
    canvas.update()
    _pump(app, 0.2)
    shot("07_brush_before", "brush")
    for step in range(1, 6):
        canvas._apply_brush(int(wx + step * 9), int(wy + step * 5))
        win._on_canvas_mask_edited()
        _pump(app, 0.05)
        shot(f"07_brush_{step}", "brush")
    win._on_canvas_mask_edited()
    win._set_tool("points")

    print("capturing the export dialog ...", flush=True)
    dialog = ExportDialog(
        win,
        video_reader=win._video_reader,
        num_masks=len(win._masks),
        num_frames=total,
    )
    dialog.show()
    _pump(app, 0.4)
    dialog.grab().save(str(RAW_DIR / "08_export.png"))
    shots.append(Capture("08_export", RAW_DIR / "08_export.png", "export"))
    dialog.close()
    _pump(app, 0.1)
    print(f"  export dialog {dialog.width()}x{dialog.height()}", flush=True)

    # The demo session is throwaway: do not pop "save or discard?" on close.
    win._project_dirty = False
    win._auto_save_timer.stop()

    # Geometry the composer needs to place the cursor and captions over the
    # canvas rather than over the transport bar.
    from PySide6.QtCore import QPoint

    canvas = win._canvas
    origin = canvas.mapTo(win, QPoint(0, 0))
    prompt_window = [
        origin.x() + canvas._offset.x() + PROMPT_POINT[0] * canvas._scale,
        origin.y() + canvas._offset.y() + PROMPT_POINT[1] * canvas._scale,
    ]
    brush_window = [
        origin.x() + canvas._offset.x() + BRUSH_IMAGE_POINT[0] * canvas._scale,
        origin.y() + canvas._offset.y() + BRUSH_IMAGE_POINT[1] * canvas._scale,
    ]
    geometry = {
        "window": [win.width(), win.height()],
        "canvas": [origin.x(), origin.y(), canvas.width(), canvas.height()],
        "prompt": prompt_window,
        "brush": brush_window,
    }
    (BUILD_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"name": s.name, "kind": s.kind, "meta": s.meta}
                    for s in shots
                ],
                "geometry": geometry,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    win.close()
    print(f"captured {len(shots)} frames -> {RAW_DIR}", flush=True)
    return shots


# --------------------------------------------------------------------------- #
# Compose
# --------------------------------------------------------------------------- #
def _font(size: int, bold: bool = True):
    from PIL import ImageFont

    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        candidate = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def _cursor(img, x: float, y: float, scale: float = 1.0) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    s = 26 * scale
    points = [(x, y), (x, y + s), (x + s * 0.26, y + s * 0.76),
              (x + s * 0.44, y + s * 1.12), (x + s * 0.60, y + s * 1.05),
              (x + s * 0.43, y + s * 0.69), (x + s * 0.72, y + s * 0.66)]
    draw.polygon(points, fill=(255, 255, 255, 245), outline=(20, 20, 24, 255), width=2)
    draw.line(points + [points[0]], fill=(20, 20, 24, 255), width=2)


def _ripple(img, x: float, y: float, progress: float) -> None:
    from PIL import ImageDraw

    if not 0.0 <= progress <= 1.0:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    r = 10 + 42 * progress
    alpha = int(190 * (1 - progress))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 180, 240, alpha), width=3)


def _caption(img, canvas, number: str, text: str, alpha: float = 1.0) -> None:
    from PIL import ImageDraw

    if alpha <= 0.01:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    pad = 14
    font_num = _font(16, bold=True)
    font_text = _font(18, bold=True)
    box_h = 42
    box_w = 60 + draw.textlength(text, font=font_text) + pad
    # Top-left of the video canvas: on the sky/tree, not over the subject or the
    # transport bar.
    x = canvas[0] + 20
    y = canvas[1] + 18
    draw.rounded_rectangle(
        (x, y, x + box_w, y + box_h),
        radius=12,
        fill=(18, 18, 22, int(216 * alpha)),
        outline=(0, 180, 240, int(160 * alpha)),
        width=1,
    )
    draw.text((x + 12, y + 12), number, font=font_num,
              fill=(0, 180, 240, int(255 * alpha)))
    draw.text((x + 42, y + 11), text, font=font_text,
              fill=(235, 235, 240, int(255 * alpha)))


def _fit(img, width: int):
    from PIL import Image

    height = round(img.height * width / img.width)
    return img.convert("RGB").resize((width, height), Image.LANCZOS)


def _blend(a, b, t: float):
    from PIL import Image

    return Image.blend(a, b, _ease(t))


def compose() -> Path:
    from PIL import Image

    document = json.loads((BUILD_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest = document.get("frames", [])
    geometry = document.get("geometry", {})
    if not manifest:
        raise SystemExit("nothing captured - run `capture` first")

    def load(name: str):
        return _fit(Image.open(RAW_DIR / f"{name}.png"), GIF_WIDTH)

    by_kind: dict[str, list[str]] = {}
    for entry in manifest:
        by_kind.setdefault(entry["kind"], []).append(entry["name"])

    window_w, window_h = geometry.get("window", [GIF_WIDTH, GIF_WIDTH * 0.6])
    scale = GIF_WIDTH / window_w
    canvas = [v * scale for v in geometry.get("canvas", [0, 0, window_w, window_h])]
    click_x, click_y = (v * scale for v in geometry.get("prompt", [window_w * 0.6, window_h * 0.5]))
    brush_x, brush_y = (v * scale for v in geometry.get("brush", [window_w * 0.7, window_h * 0.7]))

    start = load("01_start")
    point = load("02_point")
    segment = load("03_segment")
    prop = [load(n) for n in by_kind.get("prop", [])] or [start]
    sweep = [load(n) for n in sorted(by_kind.get("sweep", []))]
    cutout = load("06_cutout")
    brush = [load(n) for n in sorted(by_kind.get("brush", []))] or [segment]
    export_full = Image.open(RAW_DIR / "08_export.png").convert("RGB")

    W, H = start.size
    # The export dialog is composited over a dimmed cutout frame.
    export_bg = cutout.point(lambda p: int(p * 0.35))
    dlg_w = int(W * 0.52)
    dlg = export_full.resize((dlg_w, round(export_full.height * dlg_w / export_full.width)), Image.LANCZOS)
    dlg_x = (W - dlg.width) // 2
    dlg_y = (H - dlg.height) // 2
    export = export_bg.copy()
    export.paste(dlg, (dlg_x, dlg_y))
    from PIL import ImageDraw

    ImageDraw.Draw(export).rounded_rectangle(
        (dlg_x, dlg_y, dlg_x + dlg.width, dlg_y + dlg.height),
        radius=10, outline=(70, 70, 80), width=2,
    )

    frames: list[Image.Image] = []

    def add(img, count: int) -> None:
        for _ in range(count):
            frames.append(img.copy())

    # 1 - idle, cursor glides from the bottom-right of the video onto the subject
    for i in range(8):
        frame = start.copy()
        t = _ease(i / 7)
        _cursor(
            frame,
            canvas[0] + canvas[2] * (0.94 - 0.42 * t),
            canvas[1] + canvas[3] * (1.06 - 0.46 * t),
            0.95,
        )
        _caption(frame, canvas, "1", "CLICK THE SUBJECT", alpha=min(1.0, i / 5))
        add(frame, 1)

    # 2 - click, prompt point lands
    point_frames = 4
    for i in range(point_frames):
        frame = point.copy()
        _cursor(frame, click_x, click_y, 0.95)
        _ripple(frame, click_x, click_y, i / (point_frames - 1))
        _caption(frame, canvas, "1", "CLICK THE SUBJECT")
        add(frame, 1)

    # 3 - mask appears
    for i in range(8):
        frame = _blend(point, segment, i / 7)
        _cursor(frame, click_x, click_y, 0.95)
        _caption(frame, canvas, "1", "CLICK THE SUBJECT")
        add(frame, 1)

    # 4 - propagate: real live captures while the tracker runs
    for i, img in enumerate(prop):
        frame = img.copy()
        _cursor(frame, canvas[0] + canvas[2] * 0.84, canvas[1] + canvas[3] * 0.94, 0.85)
        _caption(frame, canvas, "2", "PROPAGATE TO END", alpha=min(1.0, (i + 1) / 3))
        add(frame, 2)

    # 5 - sweep through the finished matte
    sweep_count = max(1, len(sweep))
    for i, img in enumerate(sweep):
        frame = img.copy()
        _caption(
            frame,
            canvas,
            "2",
            "PROPAGATE TO END",
            alpha=1.0 if i < sweep_count - 4 else (sweep_count - i) / 4,
        )
        add(frame, 1 if i < sweep_count - 4 else 2)

    # 6 - cutout: the real alpha matte
    for i in range(10):
        frame = _blend(sweep[-1] if sweep else segment, cutout, i / 9)
        _caption(frame, canvas, "3", "CHECK THE CUTOUT")
        add(frame, 1)
    for _ in range(4):
        frame = cutout.copy()
        _caption(frame, canvas, "3", "CHECK THE CUTOUT")
        add(frame, 1)

    # 7 - brush correction
    for i, img in enumerate(brush):
        frame = img.copy()
        t = i / max(1, len(brush) - 1)
        _cursor(frame, brush_x + 18 * t, brush_y + 12 * t, 0.95)
        _caption(frame, canvas, "4", "REFINE WITH THE BRUSH", alpha=min(1.0, (i + 1) / 3))
        add(frame, 2 if i else 4)

    # 8 - export dialog
    for i in range(5):
        frame = _blend(brush[-1], export, i / 4)
        _caption(frame, canvas, "5", "EXPORT", alpha=min(1.0, i / 3))
        add(frame, 1)
    for _ in range(10):
        frame = export.copy()
        _caption(frame, canvas, "5", "EXPORT")
        add(frame, 1)

    # 9 - seamless return to the first frame
    for i in range(7):
        frame = _blend(export, start, i / 6)
        add(frame, 1)

    if not frames:
        raise SystemExit("compose produced no frames")

    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = BUILD_DIR / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for i, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{i:04d}.png")

    ffmpeg = shutil.which("ffmpeg") or str(REPO_ROOT / "For-Agent" / "ffmpeg.exe")
    encoded = False
    if Path(ffmpeg).is_file() or shutil.which(ffmpeg):
        # palettegen + paletteuse with diff_mode=rectangle: only the changed
        # rectangle of each frame is stored, so the static UI stays cheap and
        # the video keeps a full dithered palette.
        filt = (
            "split[s0][s1];"
            "[s0]palettegen=max_colors=200:stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=floyd_steinberg:diff_mode=rectangle"
        )
        result = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-framerate", str(FPS),
                "-i", str(frame_dir / "frame_%04d.png"),
                "-vf", filt, "-loop", "0", str(OUTPUT_GIF),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        encoded = result.returncode == 0
        if not encoded:
            print("ffmpeg encode failed, falling back to Pillow:", result.stderr[-300:])

    if not encoded:
        # Fallback: one global palette (per-frame palettes shimmer and bloat).
        sample = frames[:: max(1, len(frames) // 12)]
        thumbs = [f.resize((W // 4, H // 4), Image.LANCZOS) for f in sample]
        sheet = Image.new("RGB", (W // 4, (H // 4) * len(thumbs)))
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, (0, i * (H // 4)))
        palette = sheet.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT)
        quantised = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
        quantised[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=quantised[1:],
            duration=round(1000 / FPS),
            loop=0,
            optimize=True,
            disposal=2,
        )

    print(
        f"wrote {OUTPUT_GIF} ({OUTPUT_GIF.stat().st_size / 1024 / 1024:.1f} MB, "
        f"{len(frames)} frames, {len(frames) / FPS:.1f}s)"
    )
    return OUTPUT_GIF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["capture", "compose", "all"], nargs="?", default="all")
    args = parser.parse_args()
    if args.stage in ("capture", "all"):
        capture()
    if args.stage in ("compose", "all"):
        compose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
