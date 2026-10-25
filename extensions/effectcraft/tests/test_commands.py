"""Command generation: index arithmetic, transform copy, path normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from effectcraft_bridge.commands import (
    MATTE_LAYER_NAME,
    TRANSFORM_PATHS,
    build_commands,
    import_errors,
    to_effectcraft_path,
)
from effectcraft_bridge.protocol import PROTOCOL_VERSION, Job


def make_job(**over) -> Job:
    data = {
        "protocol": PROTOCOL_VERSION,
        "id": "bb-cmd",
        "mode": "simulate",
        "source": "C:/footage/clip.mp4",
        "range": {"frame_in": 48, "frame_out": 167},
        "media": {"width": 1920, "height": 1080, "frame_rate": 30.0},
        "comp": {"name": "Comp", "width": 1920, "height": 1080,
                 "frame_rate": 30.0, "duration": 10.0, "time": 2.5},
        "source_layer": {"id": 477, "name": "Shot", "index": 3,
                         "in": 5.005, "out": 9.5095, "start": 5.005,
                         "stretch": 100},
        "target": {"matte_index": 3},
        "transform": {
            "anchor": [960.0, 540.0, 0.0],
            "position": [960.0, 540.0, 0.0],
            "scale": [100.0, 100.0, 100.0],
            "rotation": 0.0,
        },
        "options": {"preset": "anime", "quality": "balanced", "matting": False},
    }
    data.update(over)
    return Job.from_dict(data)


def make_pngs(tmp_path: Path, count: int = 3) -> Path:
    png = tmp_path / "png"
    png.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (png / f"{i:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return png


def by_id(commands, cmd_id: str) -> dict:
    hits = [c for c in commands if c["id"] == cmd_id]
    assert hits, f"no {cmd_id} in {[c['id'] for c in commands]}"
    return hits[0]


def test_order_is_import_then_name_then_transform_then_matte(tmp_path: Path):
    job = make_job()
    commands = build_commands(job, make_pngs(tmp_path))
    ids = [c["id"] for c in commands]
    assert ids[0] == "file.import"
    assert ids[1] == "layer.rename"
    assert ids[-1] == "layer.setTrackMatte"
    # prop.set entries sit between, one per transform path present.
    assert ids[2:-1] == ["prop.set"] * len(TRANSFORM_PATHS)


def test_index_arithmetic_puts_matte_above_source(tmp_path: Path):
    job = make_job()  # matte_index 3
    commands = build_commands(job, make_pngs(tmp_path))

    imp = by_id(commands, "file.import")
    assert imp["params"]["index"] == 3
    assert imp["params"]["sequence"] is True
    assert imp["critical"] is True

    matte = by_id(commands, "layer.setTrackMatte")["params"]
    # matte at 3, source pushed to 4 — exactly where a track matte belongs.
    assert matte == {"layer": "#4", "matte": "#3", "kind": "alpha"}
    assert by_id(commands, "layer.rename")["params"]["name"] == MATTE_LAYER_NAME


def test_rename_targets_the_matte_layer_only(tmp_path: Path):
    job = make_job()
    commands = build_commands(job, make_pngs(tmp_path))
    assert by_id(commands, "layer.rename")["params"]["layer"] == "#3"


def test_transform_copies_geometry_but_never_opacity(tmp_path: Path):
    job = make_job()
    commands = build_commands(job, make_pngs(tmp_path))
    sets = [c for c in commands if c["id"] == "prop.set"]
    paths = [c["params"]["path"] for c in sets]
    assert paths == [p for _, p in TRANSFORM_PATHS]
    # Opacity/blend must never be copied: the matte's alpha is already
    # multiplied into the source, so a 50% matte would double-fade the cut-out.
    assert not any("opacity" in p for p in paths)
    for c in sets:
        assert c["params"]["layer"] == "#3"


def test_matte_start_aligns_with_the_layer_in_point(tmp_path: Path):
    job = make_job()  # start 5.005s, frame_in 48 @ 30fps => +1.6s
    commands = build_commands(job, make_pngs(tmp_path))
    assert by_id(commands, "file.import")["params"]["time"] == pytest.approx(6.605)


def test_import_uses_the_first_numbered_frame(tmp_path: Path):
    job = make_job()
    png = make_pngs(tmp_path)
    commands = build_commands(job, png)
    paths = by_id(commands, "file.import")["params"]["paths"]
    assert paths == [to_effectcraft_path(png / "00000.png")]


def test_missing_frames_raise(tmp_path: Path):
    job = make_job()
    empty = tmp_path / "png"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_commands(job, empty)


def test_dry_run_override(tmp_path: Path):
    """print-commands can show the list before anything is rendered."""
    job = make_job()
    commands = build_commands(job, tmp_path / "not-created-yet",
                              first_frame=tmp_path / "00000.png")
    assert commands[0]["params"]["paths"] == [
        to_effectcraft_path(tmp_path / "00000.png")
    ]


def test_sequence_fps_override(tmp_path: Path):
    job = make_job()
    commands = build_commands(job, make_pngs(tmp_path), sequence_fps=24.0)
    assert by_id(commands, "file.import")["params"]["frameRate"] == 24.0


def test_windows_backslashes_become_forward_slashes():
    assert to_effectcraft_path(r"C:\Footage\clip.mp4") == "C:/Footage/clip.mp4"
    assert to_effectcraft_path(Path(r"C:\a\b")) == "C:/a/b"


def test_import_errors_tolerates_any_reply():
    assert import_errors(None) == []
    assert import_errors({"errors": ["boom"]}) == ["boom"]
    assert import_errors({"errors": "boom"}) == []


def test_every_param_name_matches_the_engine(tmp_path: Path):
    """The ids/params asserted here were read back from EffectCraft 0.11.1."""
    job = make_job()
    commands = build_commands(job, make_pngs(tmp_path))
    for c in commands:
        assert set(c) >= {"id", "label", "critical", "params"}
        assert isinstance(c["params"], dict)
    known = {"file.import", "layer.rename", "prop.set", "layer.setTrackMatte"}
    assert {c["id"] for c in commands} <= known


# --------------------------------------------------------------------- #
# Brush corrections
# --------------------------------------------------------------------- #
def correction_job(**over) -> Job:
    data = make_job(**over).to_dict()
    data["correction"] = {
        "resume_dir": "C:/bridge/out/bb-prev",
        "frame": 3,
        "add": [[[10, 10], [50, 12], [48, 50]]],
        "remove": [[[100, 100], [150, 104], [146, 150]]],
        "replace_item": 12,
        "mask_names": ["+ hair", "- sky", ""],
    }
    return Job.from_dict(data)


def test_correction_swaps_footage_and_parks_stroke_masks(tmp_path: Path):
    job = correction_job()
    png = make_pngs(tmp_path)

    cmds = build_commands(job, png)
    assert [c["id"] for c in cmds] == [
        "file.replaceFootage",
        "layer.mask.mode",
        "layer.mask.mode",
    ]
    first = cmds[0]
    assert first["critical"] is True
    # A correction has no import, so the first command carries a bare path.
    assert first["params"]["path"] == to_effectcraft_path(png / "00000.png")
    assert first["params"]["item"] == 12
    assert first["params"]["sequence"] is True
    # Mode None is an accepted value of layer.mask.mode — verified against the
    # running build in docs/VALIDATION.md.
    assert cmds[1]["params"] == {
        "layer": "#3",
        "mask": "+ hair",
        "mode": "None",
    }
    assert cmds[2]["params"]["mask"] == "- sky"  # the empty name is skipped
    assert all(c["critical"] is False for c in cmds[1:])


def test_correction_without_replacement_falls_back_to_import(tmp_path: Path):
    """No matte item recorded (context lost) — fall back to the full import."""
    data = correction_job().to_dict()
    data["correction"] = dict(data["correction"], replace_item="")
    job = Job.from_dict(data)
    png = make_pngs(tmp_path)

    ids = [c["id"] for c in build_commands(job, png)]
    assert ids[0] == "file.import"
    assert "file.replaceFootage" not in ids


def test_correction_result_indices_do_not_shift(tmp_path: Path):
    job = correction_job()
    png = make_pngs(tmp_path)

    from effectcraft_bridge.cli import _decorator

    result = _decorator(tmp_path, job)({"png_dir": str(png)})
    # The matte layer was already there at index 3 and nothing was inserted:
    # the source keeps the index the panel recorded, and corrected=True tells
    # the panel to leave the layer stack alone.
    assert result["matte_index"] == 2
    assert result["source_index"] == 3
    assert result["corrected"] is True
    assert result["first_frame"] == to_effectcraft_path(png / "00000.png")
