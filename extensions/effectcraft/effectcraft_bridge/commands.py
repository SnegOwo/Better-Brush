"""Build the EffectCraft command list that lands a rendered matte in a comp.

The panel never decides *what* to run — it executes whatever the adapter put in
``status.result.commands``. That keeps one implementation, in a language we can
unit test, instead of a second copy in JavaScript that drifts.

Every id and param name below was read back from the running engine with
``effectcraft-cli commands --filter ...`` and executed end to end; see
``docs/VALIDATION.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import Job

#: Geometry only. The matte layer's opacity and blend mode must NOT be copied:
#: a track matte multiplies the matte's alpha into the layer below, so copying
#: opacity 50 would fade the cut-out twice.
TRANSFORM_PATHS: tuple[tuple[str, str], ...] = (
    ("anchor", "transform/anchor"),
    ("position", "transform/position"),
    ("scale", "transform/scale"),
    ("rotation", "transform/rotation"),
)

MATTE_LAYER_NAME = "BetterBrush Matte"
TRACK_MATTE_KIND = "alpha"


def to_effectcraft_path(path: str | Path) -> str:
    """EffectCraft accepts forward slashes everywhere; JS would need escaping.

    Normalising here means the panel can hand the string straight to
    ``app.run`` without touching it, and a Windows path with backslashes never
    becomes an accidental escape sequence on the way through JSON.
    """
    return str(path).replace("\\", "/")


def _first_frame(png_dir: Path) -> Path | None:
    if not png_dir.is_dir():
        return None
    for name in sorted(p.name for p in png_dir.glob("*.png")):
        return png_dir / name
    return None


def build_commands(
    job: Job,
    png_dir: Path,
    *,
    sequence_fps: float | None = None,
    first_frame: Path | None = None,
) -> list[dict[str, Any]]:
    """The commands to run, in order, inside one undo group.

    Index arithmetic — the one thing worth getting right:

    ``job.matte_index`` is the source layer's 1-based position. Importing the
    sequence *at* that index inserts the matte above the source and pushes the
    source down by one, so afterwards the matte sits at ``matte_index`` and the
    source at ``matte_index + 1``. That is also exactly where After Effects and
    EffectCraft expect a track matte: the layer above.

    A command marked ``critical`` stops the chain; the ones after it assume the
    indices above and would otherwise land on the wrong layer.

    ``first_frame`` overrides the file the import points at, so
    ``print-commands`` can show a dry run before anything has been rendered.
    """
    if first_frame is None:
        first_frame = _first_frame(png_dir)
    if first_frame is None:
        raise FileNotFoundError(f"no PNG frames in {png_dir}")

    if job.correction is not None and job.correction.replace_item:
        return _correction_commands(job, first_frame)

    fps = sequence_fps if sequence_fps and sequence_fps > 0 else (job.fps or 24.0)
    matte = job.matte_index
    source = matte + 1

    # The sequence's first frame plays where the source layer's own frame
    # ``frame_in`` plays, so the matte lines up with the layer's first visible
    # frame instead of with comp time 0. ``frame_in`` is counted in source
    # frames, at the media's frame rate, not the comp's.
    media_fps = job.fps if job.fps > 0 else fps
    matte_start = job.layer_start + (job.frame_in / media_fps)

    commands: list[dict[str, Any]] = [
        {
            "id": "file.import",
            "label": "Import the matte sequence",
            "critical": True,
            "params": {
                "paths": [to_effectcraft_path(first_frame)],
                "sequence": True,
                "frameRate": round(float(fps), 6),
                "addToComp": True,
                "time": round(float(matte_start), 6),
                "index": int(matte),
            },
        },
        {
            "id": "layer.rename",
            "label": "Name the matte layer",
            "critical": False,
            "params": {"layer": f"#{matte}", "name": MATTE_LAYER_NAME},
        },
    ]

    for key, path in TRANSFORM_PATHS:
        value = job.transform.get(key)
        if value is None:
            continue
        commands.append(
            {
                "id": "prop.set",
                "label": f"Match {key}",
                "critical": False,
                "params": {"layer": f"#{matte}", "path": path, "value": value},
            }
        )

    commands.append(
        {
            "id": "layer.setTrackMatte",
            "label": "Apply the track matte",
            "critical": True,
            "params": {
                "layer": f"#{source}",
                "matte": f"#{matte}",
                "kind": TRACK_MATTE_KIND,
            },
        }
    )
    return commands


def _correction_commands(job: Job, first_frame: Path) -> list[dict[str, Any]]:
    """Land a *corrected* render by swapping the matte's footage in place.

    A brush correction re-tracks the same layer, so the matte layer is already
    there: its transform is right, the source already wears it as a track
    matte, and it sits at the right place in the stack. Re-importing would add
    a second matte and shift every index; ``file.replaceFootage`` swaps the
    frames underneath the existing layer instead, which keeps all of that and
    makes the whole correction one small undo step.

    The stroke masks themselves are parked in mask mode ``None`` afterwards:
    they were drawn on the footage layer to *describe* the correction, and left
    in Add mode they would keep clipping the layer they were drawn on.
    """
    corr = job.correction
    assert corr is not None  # guarded by the caller
    commands: list[dict[str, Any]] = [
        {
            "id": "file.replaceFootage",
            "label": "Swap in the corrected matte frames",
            "critical": True,
            "params": {
                "path": to_effectcraft_path(first_frame),
                "item": corr.replace_item,
                "sequence": True,
            },
        }
    ]
    for name in corr.mask_names:
        if not name:
            continue
        commands.append(
            {
                "id": "layer.mask.mode",
                "label": f"Park the stroke mask \"{name}\"",
                "critical": False,
                "params": {
                    "layer": f"#{job.matte_index}",
                    "mask": name,
                    "mode": "None",
                },
            }
        )
    return commands


def import_errors(result: Any) -> list[str]:
    """Human-readable failures from ``file.import``, if it replied with any."""
    if not isinstance(result, dict):
        return []
    errors = result.get("errors")
    if not isinstance(errors, list):
        return []
    return [str(e) for e in errors]
