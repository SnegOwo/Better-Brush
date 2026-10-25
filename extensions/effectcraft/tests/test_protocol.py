"""Wire protocol: round trips, rejections, atomic status writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effectcraft_bridge.protocol import (
    PROTOCOL_VERSION,
    STATE_ERROR,
    STATE_RUNNING,
    Job,
    ProtocolError,
    Status,
    cancel_path,
    cancel_requested,
    job_path,
    read_json,
    read_status,
    status_path,
    write_json_atomic,
    write_status,
)


def make_job_dict(**over):
    base = {
        "protocol": PROTOCOL_VERSION,
        "id": "bb-test-1",
        "created": "2026-10-25T14:30:12",
        "mode": "simulate",
        "source": "C:/footage/clip.mp4",
        "range": {"frame_in": 10, "frame_out": 29},
        "media": {"width": 640, "height": 360, "frame_rate": 30.0},
        "comp": {
            "name": "Comp",
            "width": 1920,
            "height": 1080,
            "frame_rate": 30.0,
            "duration": 10.0,
            "time": 2.5,
        },
        "source_layer": {
            "id": 42,
            "name": "Shot",
            "index": 3,
            "in": 1.0,
            "out": 2.0,
            "start": 1.0,
            "stretch": 100,
        },
        "target": {"matte_index": 3},
        "transform": {"position": [10.0, 20.0, 0.0]},
        "options": {"preset": "anime", "quality": "balanced", "matting": False},
    }
    base.update(over)
    return base


def test_job_round_trip():
    job = Job.from_dict(make_job_dict())
    assert job.id == "bb-test-1"
    assert job.frame_in == 10
    assert job.frame_out == 29
    assert job.frame_count == 20
    assert job.matte_index == 3
    assert job.width == 640

    again = Job.from_dict(job.to_dict())
    assert again == job


def test_media_size_falls_back_to_comp():
    """Solids and precomps have no media size; the matte still must line up."""
    data = make_job_dict()
    data["media"] = {"width": 0, "height": 0, "frame_rate": 30.0}
    job = Job.from_dict(data)
    assert (job.width, job.height) == (1920, 1080)


def test_rejects_bad_protocol_and_mode():
    with pytest.raises(ProtocolError, match="unsupported protocol"):
        Job.from_dict(make_job_dict(protocol=99))
    with pytest.raises(ProtocolError, match="mode"):
        Job.from_dict(make_job_dict(mode="magic"))


def test_rejects_inverted_range_and_missing_source():
    with pytest.raises(ProtocolError, match="frame_out"):
        Job.from_dict(make_job_dict(range={"frame_in": 5, "frame_out": 1}))
    bad = make_job_dict()
    del bad["source"]
    with pytest.raises(ProtocolError, match="source"):
        Job.from_dict(bad)


def test_rejects_non_numeric_frames():
    with pytest.raises(ProtocolError, match="frame_in"):
        Job.from_dict(make_job_dict(range={"frame_in": "soon", "frame_out": 4}))


def test_status_fraction():
    assert Status(id="x", state="running", current=5, total=10).fraction == 0.5
    # No total yet: indeterminate, which the panel renders as a low bar.
    assert Status(id="x", state="running", current=0, total=0).fraction == -1.0
    assert Status(id="x", state="done").fraction == 1.0
    assert Status(id="x", state="error").fraction == 0.0


def test_atomic_status_write_is_readable(tmp_path: Path):
    status = Status(id="bb-1", state=STATE_RUNNING, stage="extract",
                    current=1, total=4, message="reading")
    path = write_status(tmp_path, status)
    assert path == status_path(tmp_path, "bb-1")
    loaded = read_status(tmp_path, "bb-1")
    assert loaded is not None
    assert loaded.stage == "extract"
    # No temp file left behind by the replace.
    assert list(tmp_path.glob("*.tmp")) == []


def test_status_rejects_unknown_state(tmp_path: Path):
    write_json_atomic(status_path(tmp_path, "bb-2"), {"id": "bb-2", "state": "wat"})
    with pytest.raises(ProtocolError, match="unknown state"):
        read_status(tmp_path, "bb-2")


def test_read_json_tolerates_a_utf8_bom(tmp_path: Path):
    """PowerShell and some editors write a BOM; the panel may too."""
    p = tmp_path / "job-bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + b'{"protocol": 1, "id": "x", "source": "a.mp4"}')
    assert read_json(p)["id"] == "x"


def test_recorded_bridge_dir_tolerates_a_bom(tmp_path: Path):
    """install.ps1 used to write bridge.path.json with a BOM, which made the
    adapter silently ignore the recorded folder and watch a fallback one the
    panel never writes to."""
    from effectcraft_bridge.protocol import recorded_bridge_dir

    cfg = tmp_path / "bridge.path.json"
    cfg.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"bridge_dir": "C:/x/betterbrush-bridge"}).encode("utf-8"))
    assert recorded_bridge_dir(cfg) == Path("C:/x/betterbrush-bridge")


def test_recorded_bridge_dir_rejects_junk(tmp_path: Path):
    from effectcraft_bridge.protocol import recorded_bridge_dir

    cfg = tmp_path / "bridge.path.json"
    cfg.write_text("{not json", encoding="utf-8")
    assert recorded_bridge_dir(cfg) is None
    cfg.write_text("[1, 2, 3]", encoding="utf-8")
    assert recorded_bridge_dir(cfg) is None
    cfg.write_text('{"bridge_dir": ""}', encoding="utf-8")
    assert recorded_bridge_dir(cfg) is None
    assert recorded_bridge_dir(tmp_path / "missing.json") is None


def test_read_json_rejects_non_object(tmp_path: Path):
    p = tmp_path / "job-x.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ProtocolError, match="object"):
        read_json(p)
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProtocolError, match="JSON"):
        read_json(p)


def test_cancel_flag(tmp_path: Path):
    assert not cancel_requested(tmp_path, "bb-3")
    flag = cancel_path(tmp_path, "bb-3")
    flag.write_text("1", encoding="utf-8")
    assert cancel_requested(tmp_path, "bb-3")


def test_job_and_status_paths_do_not_collide(tmp_path: Path):
    j = job_path(tmp_path, "bb-4")
    s = status_path(tmp_path, "bb-4")
    assert j != s
    assert j.name == "job-bb-4.json"
    assert s.name == "status-bb-4.json"


def test_written_job_is_valid_json(tmp_path: Path):
    job = Job.from_dict(make_job_dict())
    p = tmp_path / "job-bb-test-1.json"
    write_json_atomic(p, job.to_dict())
    assert json.loads(p.read_text(encoding="utf-8"))["id"] == "bb-test-1"


def test_error_status_shape():
    s = Status(id="bb-5", state=STATE_ERROR,
               error={"kind": "unsupported", "message": "nope"})
    blob = s.to_dict()
    assert blob["error"]["kind"] == "unsupported"
    assert Status.from_dict(blob).error == {"kind": "unsupported", "message": "nope"}


# --------------------------------------------------------------------- #
# Options and brush corrections
# --------------------------------------------------------------------- #
def correction_blob(**over):
    base = {
        "resume_dir": "C:/bridge/out/bb-prev",
        "frame": 7,
        "add": [[[10, 10], [40, 12], [38, 40]]],
        "remove": [[[100, 100], [140, 105], [135, 150]]],
        "replace_item": 12,
        "mask_names": ["+ hair", "- sky"],
    }
    base.update(over)
    return base


def test_post_process_option_defaults_on_and_honours_false():
    assert Job.from_dict(make_job_dict()).post_process is True
    data = make_job_dict()
    data["options"] = dict(data["options"], post_process=False)
    assert Job.from_dict(data).post_process is False


def test_post_stage_is_advertised():
    from effectcraft_bridge.protocol import STAGES

    assert "post" in STAGES


def test_correction_round_trip():
    job = Job.from_dict(make_job_dict(correction=correction_blob()))
    assert job.correction is not None
    assert job.correction.resume_dir.endswith("bb-prev")
    assert job.correction.frame == 7
    # Polygons arrive as [x, y] lists and are normalised to tuples, so the
    # adapter can hand them to cv2.fillPoly without a further walk.
    assert job.correction.add[0][1] == (40.0, 12.0)
    assert job.correction.remove[0][0] == (100.0, 100.0)
    assert job.correction.replace_item == 12
    assert job.correction.mask_names == ["+ hair", "- sky"]
    assert Job.from_dict(job.to_dict()) == job


def test_correction_keeps_an_item_name_intact():
    """file.replaceFootage takes an id or a name; never coerce one into the other."""
    job = Job.from_dict(make_job_dict(correction=correction_blob(replace_item="00000.png")))
    assert job.correction.replace_item == "00000.png"
    whole = Job.from_dict(make_job_dict())
    assert whole.correction is None


def test_correction_without_strokes_is_rejected():
    with pytest.raises(ProtocolError, match="no strokes"):
        Job.from_dict(make_job_dict(correction=correction_blob(add=[], remove=[])))


def test_correction_needs_a_resume_dir():
    with pytest.raises(ProtocolError, match="resume_dir"):
        Job.from_dict(make_job_dict(correction=correction_blob(resume_dir="  ")))


def test_correction_rejects_degenerate_polygons():
    with pytest.raises(ProtocolError, match="at least 3 points"):
        Job.from_dict(make_job_dict(correction=correction_blob(add=[[[1, 2], [3, 4]]])))
    with pytest.raises(ProtocolError, match=r"\[x, y\]"):
        Job.from_dict(
            make_job_dict(
                correction=correction_blob(add=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
            )
        )
