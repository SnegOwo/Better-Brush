"""Pipeline: simulate mode end to end, status bookkeeping, cancellation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from effectcraft_bridge.pipeline import JobCancelled, PipelineFailure, execute, run_simulate
from effectcraft_bridge.protocol import (
    PROTOCOL_VERSION,
    STATE_CANCELLED,
    STATE_DONE,
    STATE_ERROR,
    STATE_RUNNING,
    Job,
    cancel_path,
    job_path,
    read_status,
    write_json_atomic,
)

cv2 = pytest.importorskip("cv2")


def make_source(tmp_path: Path, frames: int = 6, size: tuple[int, int] = (80, 60)) -> Path:
    """A tiny real video (or a still, if no codec is available)."""
    w, h = size
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    if writer.isOpened():
        for i in range(frames):
            frame = np.full((h, w, 3), (i * 30) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        if path.is_file() and path.stat().st_size > 0:
            return path
    # Fallback: a still image, which the pipeline treats as a one-frame job.
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((h, w, 3), 120, dtype=np.uint8))
    return still


def make_job(tmp_path: Path, source: Path, **over) -> Job:
    data = {
        "protocol": PROTOCOL_VERSION,
        "id": "bb-pipe-1",
        "mode": "simulate",
        "source": str(source),
        "range": {"frame_in": 0, "frame_out": 5},
        "media": {"width": 80, "height": 60, "frame_rate": 24.0},
        "comp": {"name": "C", "width": 80, "height": 60, "frame_rate": 24.0,
                 "duration": 1.0, "time": 0.0},
        "source_layer": {"id": 1, "name": "Shot", "index": 1, "in": 0.0,
                         "out": 0.25, "start": 0.0, "stretch": 100},
        "target": {"matte_index": 1},
        "transform": {"anchor": [40.0, 30.0, 0.0], "position": [40.0, 30.0, 0.0],
                      "scale": [100.0, 100.0, 100.0], "rotation": 0.0},
        "options": {"preset": "anime", "quality": "balanced", "matting": False},
    }
    data.update(over)
    return Job.from_dict(data)


def write_job(root: Path, job: Job) -> Path:
    p = job_path(root, job.id)
    write_json_atomic(p, job.to_dict())
    return p


def test_simulate_produces_rgba_pngs(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))

    seen: list[tuple[str, int, int]] = []

    def progress(stage, current, total, message):
        seen.append((stage, current, total))

    result = run_simulate(job, job.png_dir(root), progress, lambda: False)

    frames = sorted(job.png_dir(root).glob("*.png"))
    assert frames, "no frames written"
    first = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    assert first.ndim == 3 and first.shape[2] == 4, "frame has no alpha channel"
    assert result["simulated"] is True
    assert result["engine"] == "simulate"
    assert result["frame_count"] >= 1
    # Stage names must stay in the set the panel renders labels for.
    assert {"extract", "analyze", "export"} <= {s for s, _, _ in seen}


def test_simulated_alpha_moves_between_frames(tmp_path: Path):
    """A static cut-out would look like a tracking failure in the timeline."""
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    run_simulate(job, job.png_dir(root), lambda *a: None, lambda: False)
    frames = sorted(job.png_dir(root).glob("*.png"))
    if len(frames) < 2:
        pytest.skip("source decoded to a single frame")
    a = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)[:, :, 3].astype(int)
    b = cv2.imread(str(frames[-1]), cv2.IMREAD_UNCHANGED)[:, :, 3].astype(int)
    assert not np.array_equal(a, b)


def test_execute_writes_terminal_status_with_commands(tmp_path: Path):
    from effectcraft_bridge.cli import _decorator

    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    write_job(root, job)

    status = execute(job, root, on_done=_decorator(root, job))

    assert status.state == STATE_DONE
    on_disk = read_status(root, job.id)
    assert on_disk is not None and on_disk.state == STATE_DONE
    assert on_disk.result is not None
    assert on_disk.result["commands"], "no EffectCraft commands attached"
    assert on_disk.result["source_index"] == on_disk.result["matte_index"] + 1
    # The import points at a file that really exists.
    first = on_disk.result["commands"][0]["params"]["paths"][0]
    assert Path(first).is_file()
    # The progress bar has a real total by the time the panel sees `done`.
    assert on_disk.total > 0 and on_disk.current == on_disk.total


def test_execute_reports_missing_source_as_error(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, tmp_path / "nope.mp4")

    status = execute(job, root)

    assert status.state == STATE_ERROR
    assert status.error is not None
    assert status.error["kind"] in {"unsupported", "internal"}
    assert "nope.mp4" in status.error["message"]
    on_disk = read_status(root, job.id)
    assert on_disk is not None and on_disk.state == STATE_ERROR


def test_cancel_flag_stops_and_is_cleared(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))

    # Drop the cancel flag before the job starts: it must stop immediately.
    cancel_path(root, job.id).write_text("1", encoding="utf-8")

    status = execute(job, root)

    assert status.state == STATE_CANCELLED
    assert not cancel_path(root, job.id).exists(), \
        "a stale cancel flag would kill the next job with the same id"


def test_stale_cancel_flag_is_cleared_even_on_success(tmp_path: Path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))

    # Simulate never consults the flag after the first probe; put one there
    # and confirm a finished job still clears it.
    cancel_path(root, job.id).write_text("1", encoding="utf-8")
    monkeypatch.setattr(
        "effectcraft_bridge.pipeline.cancel_requested",
        lambda *_: False,
    )
    status = execute(job, root)
    assert status.state == STATE_DONE
    assert not cancel_path(root, job.id).exists()


def test_run_simulate_raises_job_cancelled_when_flagged(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    with pytest.raises(JobCancelled):
        run_simulate(job, job.png_dir(root), lambda *a: None, lambda: True)


def test_pipeline_failure_carries_a_kind():
    err = PipelineFailure("bad", "unsupported")
    assert err.kind == "unsupported"
    assert str(err) == "bad"


def test_pending_job_detection(tmp_path: Path):
    from effectcraft_bridge.cli import pending_jobs

    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    write_job(root, job)
    assert [p.name for p in pending_jobs(root)] == [f"job-{job.id}.json"]

    execute(job, root)
    assert pending_jobs(root) == []


def test_bad_job_still_gets_a_terminal_status(tmp_path: Path):
    from effectcraft_bridge.cli import run_one

    root = tmp_path / "bridge"
    root.mkdir()
    bad = root / "job-bad.json"
    bad.write_text('{"protocol": 1, "id": "bad"}', encoding="utf-8")

    status = run_one(root, bad, None)

    assert status is not None and status.state == STATE_ERROR
    assert status.error is not None and status.error["kind"] == "bad_request"
    # Terminal, so the watcher will not retry it forever.
    assert read_status(root, "bad").state == STATE_ERROR


def test_running_state_is_visible_mid_job(tmp_path: Path, monkeypatch):
    """The panel polls mid-flight; a status file must exist before the end."""
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))

    import effectcraft_bridge.pipeline as pipeline_mod
    from effectcraft_bridge.protocol import write_status as real_write

    seen: list[str] = []

    def spy(root_, status):
        seen.append(status.state)
        return real_write(root_, status)

    monkeypatch.setattr(pipeline_mod, "write_status", spy)

    execute(job, root)

    assert STATE_RUNNING in seen
    assert seen[-1] == STATE_DONE


# --------------------------------------------------------------------- #
# Brush corrections and the post-processing toggle
# --------------------------------------------------------------------- #
def correction_over(**over):
    base = {
        "resume_dir": "unused-in-simulate",
        "frame": 0,
        "add": [[[0, 0], [79, 0], [79, 59], [0, 59]]],   # the whole frame
        "remove": [[[20, 20], [40, 20], [40, 40], [20, 40]]],
        "replace_item": 3,
        "mask_names": ["+ all", "- hole"],
    }
    base.update(over)
    return base


def test_simulate_applies_add_then_remove_polygons(tmp_path: Path):
    """Same layering as engine _apply_correction: add wins, remove trims last."""
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(
        root,
        make_source(tmp_path),
        correction=correction_over(),
    )

    run_simulate(job, job.png_dir(root), lambda *a: None, lambda: False)

    frames = sorted(job.png_dir(root).glob("*.png"))
    alpha = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)[:, :, 3]
    # The add polygon covers everything, so even where the simulated subject
    # was absent the alpha is fully in…
    assert int(alpha[5, 5]) == 255
    # …except where the remove polygon trims it afterwards.
    assert int(alpha[30, 30]) == 0


def test_post_stage_runs_when_the_toggle_is_on(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    assert job.post_process is True  # the panel's default

    seen: list[str] = []
    result = run_simulate(job, job.png_dir(root),
                          lambda stage, *_: seen.append(stage), lambda: False)

    assert result["post_processed"] is True
    assert result["post"]["frames"] == result["frame_count"]
    assert "post" in seen


def test_post_stage_is_skipped_when_the_toggle_is_off(tmp_path: Path):
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path),
                   options={"preset": "anime", "quality": "balanced",
                            "matting": False, "post_process": False})

    seen: list[str] = []
    result = run_simulate(job, job.png_dir(root),
                          lambda stage, *_: seen.append(stage), lambda: False)

    assert result["post_processed"] is False
    assert result["post"] is None
    assert "post" not in seen


def test_resume_dir_must_live_in_this_bridge(tmp_path: Path):
    """A job file must not aim the adapter at arbitrary filesystem folders."""
    from effectcraft_bridge.pipeline import _resume_dir_for

    root = tmp_path / "bridge"
    (root / "out").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    job = make_job(root, make_source(tmp_path), correction=correction_over(
        resume_dir=str(outside)))

    with pytest.raises(PipelineFailure, match="outside this bridge"):
        _resume_dir_for(job, root)

    gone = make_job(root, make_source(tmp_path), correction=correction_over(
        resume_dir=str(root / "out" / "vanished")))
    with pytest.raises(PipelineFailure, match="is gone"):
        _resume_dir_for(gone, root)

    kept = root / "out" / "bb-prev"
    kept.mkdir()
    ok = make_job(root, make_source(tmp_path), correction=correction_over(
        resume_dir=str(kept)))
    assert Path(_resume_dir_for(ok, root)) == kept.resolve()


def test_stroke_png_is_a_full_size_grayscale_mask(tmp_path: Path):
    from effectcraft_bridge.pipeline import _stroke_png

    job = make_job(tmp_path, make_source(tmp_path), correction=correction_over())
    payload = _stroke_png(job, "add")
    assert payload  # base64, not None

    import base64

    raw = base64.b64decode(payload)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (60, 80)          # job media height x width
    assert int(img[30, 40]) == 255        # inside the stroke
    assert _stroke_png(job, "remove") is not None
    assert _stroke_png(make_job(tmp_path, make_source(tmp_path)), "add") is None


# --------------------------------------------------------------------- #
# The result image the panel shows
# --------------------------------------------------------------------- #
def test_execute_writes_a_preview_the_panel_can_display(tmp_path: Path):
    from effectcraft_bridge.pipeline import PREVIEW_SIZE

    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))

    status = execute(job, root)

    assert status.state == STATE_DONE
    preview = status.result["preview"]
    assert preview and Path(preview).is_file()
    img = cv2.imread(preview, cv2.IMREAD_COLOR)
    # A *fixed* canvas: ScriptUI renders images at their natural size, so the
    # panel's layout depends on the adapter keeping this constant.
    assert (img.shape[1], img.shape[0]) == PREVIEW_SIZE
    # Not a black rectangle: the checkerboard behind the transparent alpha
    # must be visible somewhere.
    assert img.std() > 5.0


def test_preview_picks_the_correction_frame(tmp_path: Path):
    from effectcraft_bridge.pipeline import write_preview

    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path), correction=correction_over(frame=3))
    run_simulate(job, job.png_dir(root), lambda *a: None, lambda: False)

    path = write_preview(job.png_dir(root), job)

    assert path and Path(path).is_file()
    # Frame 3 of the simulated run differs from frame 0 — the preview must be
    # the frame the strokes were painted on, not simply the first file.
    first = cv2.imread(str(job.png_dir(root) / "00000.png"), cv2.IMREAD_UNCHANGED)
    picked = cv2.imread(str(job.png_dir(root) / "00003.png"), cv2.IMREAD_UNCHANGED)
    assert not np.array_equal(first[:, :, :3], picked[:, :, :3])
    assert Path(path).name == "preview.jpg"


def test_preview_failure_is_silent(tmp_path: Path):
    from effectcraft_bridge.pipeline import write_preview

    empty = tmp_path / "nothing"
    empty.mkdir()
    job = make_job(tmp_path, make_source(tmp_path))
    assert write_preview(empty, job) == ""
