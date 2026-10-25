"""Run the adapter's generated command list against a real EffectCraft build.

Validation does exactly what the panel does — ``app.run(id, params)`` for each
generated command, inside one undo group — because that is the path the MVP
ships. ``effectcraft-cli run`` deliberately is *not* used: its own parameter
validator rejects the ``time``/``index`` arguments of ``file.import`` that
``app.run`` accepts and honours (see docs/VALIDATION.md).

    python scripts/validate_commands.py <bridge-dir> \\
        --cli <path\\to\\effectcraft-cli.exe> [--config-dir DIR]

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MVP = Path(__file__).resolve().parents[1]

JSX = r"""
var out = [];
function p(s) { out.push(String(s)); }
var cmds = %s;
var transform = %s;
var index = cmds[0].params.index;
var failures = [];

app.run("comp.new", {name: "BB Validate", width: 320, height: 180,
                     frameRate: 24, duration: 2});
/* Newest layers land on top, so: Below, then Source, then (index - 1) dummies
 * pushed above it — that leaves Source at exactly the job's matte index. */
app.run("layer.newSolid", {name: "Below", width: 320, height: 180,
                           color: "#101010"});
app.run("layer.newSolid", {name: "Source", width: 320, height: 180,
                           color: "#203040"});
for (var d = 1; d < index; d++) {
  app.run("layer.newSolid", {name: "Above " + d, width: 320, height: 180,
                             color: "#303030"});
}
/* Drop the source's opacity to 40 so "the matte must not inherit opacity"
 * is a check that can actually fail. */
app.run("prop.set", {layer: "#" + index, path: "transform/opacity", value: 40});
/* Put the job's *own* transform on the source first: the copy check is only
 * meaningful when the two layers start from something the commands moved. */
var transformPaths = {
  anchor: "transform/anchor",
  position: "transform/position",
  scale: "transform/scale",
  rotation: "transform/rotation"
};
for (var key in transformPaths) {
  if (transform.hasOwnProperty(key)) {
    app.run("prop.set", {layer: "#" + index, path: transformPaths[key],
                         value: transform[key]});
  }
}
p("seeded: layers=" + app.project.activeItem.numLayers +
  " matte_index=" + index +
  " pos=" + app.project.activeItem.layer(index).transform.position.value);

app.beginUndoGroup("Better Brush validation");
for (var i = 0; i < cmds.length; i++) {
  var c = cmds[i];
  try {
    var res = app.run(c.id, c.params);
    p("ran " + c.id + " -> " + JSON.stringify(res));
    if (c.id === "file.import" && res && res.errors && res.errors.length) {
      failures.push(c.id + ": " + res.errors.join("; "));
      if (c.critical) { break; }
    }
  } catch (e) {
    failures.push(c.id + " (" + c.label + "): " + e.message);
    p("FAILED " + c.id + ": " + e.message);
    if (c.critical) { break; }
  }
}
app.endUndoGroup();

var comp = app.project.activeItem;
for (var n = 1; n <= comp.numLayers; n++) {
  var L = comp.layer(n);
  var info = "L" + n + " name=" + L.name +
    " enabled=" + L.enabled +
    " start=" + L.startTime +
    " pos=" + L.transform.position.value +
    " anchor=" + L.transform.anchorPoint.value +
    " scale=" + L.transform.scale.value +
    " rot=" + L.transform.rotation.value +
    " op=" + L.transform.opacity.value;
  try { info += " trackMatte=" + JSON.stringify(L.trackMatte); } catch (e1) {}
  try { info += " tmt=" + L.trackMatteType; } catch (e2) {}
  try { info += " has=" + L.hasTrackMatte; } catch (e3) {}
  p(info);
}
p("failures=" + JSON.stringify(failures));
out.join("\n");
"""


def _value(line: str, key: str) -> str:
    """Read one ``key=value`` field out of a printed layer line.

    Vectors print as ``[320, 180, 0] anchor=`` (spaces inside the brackets);
    scalars run to the next space.
    """
    if key not in line:
        return ""
    tail = line.split(key, 1)[1]
    if tail.startswith("["):
        return tail.split("] ", 1)[0] + "]" if "] " in tail else tail
    return tail.split(" ", 1)[0]


#: The brush correction path: ``file.replaceFootage`` swaps the finished
#: frames underneath the matte layer that is already there, and
#: ``layer.mask.mode`` parks the stroke masks so they stop clipping the layer
#: they were drawn on. Both are new ids for the MVP, so both are executed
#: against a real build rather than assumed. The script keeps the layer count
#: in the reply: a correction must swap footage, never add a layer.
CORRECTION_JSX = r"""
var out = [];
function p(s) { out.push(String(s)); }
var files = %s;
var failures = [];

app.run("comp.new", {name: "BB Correction", width: 320, height: 180,
                     frameRate: 24, duration: 2});
var imp = app.run("file.import", {paths: [files.a + "/00000.png"],
                                  sequence: true});
p("import -> " + JSON.stringify(imp));
var itemId = (imp && imp.items && imp.items.length) ? imp.items[0] : null;
app.run("layer.addItem", {item: itemId, time: 0, index: 1});
var comp = app.project.activeItem;
var layer = comp.layer(1);
var item = layer.source;
p("imported item=" + item.name + " file=" +
  (item.file ? item.file.fsName : "?"));

app.run("layer.addMask", {layer: "#1", shape: "rect",
                          rect: [40, 40, 120, 80], mode: "Add"});
var mask = layer.mask(1);
var maskName = mask.name;
var modeBefore = mask.maskMode;
p("mask=" + maskName + " modeBefore=" + modeBefore);

try {
  app.run("layer.mask.mode", {layer: "#1", mask: maskName, mode: "None"});
} catch (e) { failures.push("layer.mask.mode: " + e.message); }
var modeAfter = mask.maskMode;
var noneValue = "unavailable";
try { noneValue = String(MaskMode.NONE); } catch (e2) {}
p("modeAfter=" + modeAfter + " MaskMode.NONE=" + noneValue);

var beforeFile = item.file ? item.file.fsName : "";
try {
  var res = app.run("file.replaceFootage",
                    {path: files.b + "/00000.png", item: item.id,
                     sequence: true});
  p("replace -> " + JSON.stringify(res));
  if (res && res.errors && res.errors.length) {
    failures.push("file.replaceFootage: " + res.errors.join("; "));
  }
} catch (e3) { failures.push("file.replaceFootage: " + e3.message); }
var afterFile = item.file ? item.file.fsName : "";
p("beforeFile=" + beforeFile);
p("afterFile=" + afterFile);
p("layerCount=" + comp.numLayers);
p("failures=" + JSON.stringify(failures));
out.join("\n");
"""


def _write_png_sequence(folder: Path, colour: tuple[int, int, int]) -> None:
    """Two numbered frames in *folder* — enough to prove a sequence swap."""
    import numpy as np

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - needs opencv
        raise SystemExit(f"opencv is required for --correction: {exc}")
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        frame = np.full((60, 80, 3), colour, np.uint8)
        cv2.imwrite(str(folder / f"{i:05d}.png"), frame)


def validate_correction(cli: str, env: dict[str, str]) -> dict[str, bool]:
    """Execute ``file.replaceFootage`` + ``layer.mask.mode`` on the real build."""
    work = Path(tempfile.mkdtemp(prefix="bb-ec-correct-"))
    _write_png_sequence(work / "run-a", (40, 60, 200))
    _write_png_sequence(work / "run-b", (200, 60, 40))
    files = {"a": str(work / "run-a"), "b": str(work / "run-b")}
    jsx = work / "correct.jsx"
    jsx.write_text(CORRECTION_JSX % json.dumps(files), encoding="utf-8")

    run = subprocess.run([cli, "script", str(jsx), "--empty", "--json"],
                         capture_output=True, text=True, env=env,
                         encoding="utf-8", errors="replace")
    try:
        reply = json.loads(run.stdout)
    except ValueError:
        print("unparseable reply:", run.stdout, run.stderr)
        return {"correction script ran": False}
    if not reply.get("ok"):
        print("correction script failed:", json.dumps(reply, indent=2))
        return {"correction script ran": False}

    print(reply["result"])
    lines = reply["result"].splitlines()
    failures = next((l for l in lines if l.startswith("failures=")),
                    "failures=[?]")
    mode_before = _value(next((l for l in lines if l.startswith("mask=")), ""),
                         "modeBefore=")
    mode_after = _value(next((l for l in lines if l.startswith("modeAfter=")), ""),
                        "modeAfter=")
    none_value = _value(
        next((l for l in lines if l.startswith("modeAfter=")), ""),
        "MaskMode.NONE=",
    )
    before = _value(next((l for l in lines if l.startswith("beforeFile=")), ""),
                    "beforeFile=")
    after = _value(next((l for l in lines if l.startswith("afterFile=")), ""),
                   "afterFile=")
    layers = _value(next((l for l in lines if l.startswith("layerCount=")), ""),
                    "layerCount=")
    b_dir = str(work / "run-b")

    checks = {
        "correction: no command failed": failures == "failures=[]",
        # mode None must really land, not merely be accepted: the mask goes
        # from Add to whatever the host calls None (MaskMode.NONE when the
        # enum is exposed), and never back to Add.
        "correction: mask mode changed": mode_before != mode_after
        and mode_after != "" and mode_after != mode_before,
        "correction: mask mode is None": none_value == "unavailable"
        or mode_after == none_value,
        # The footage swap must move the *item's* file to the new run…
        "correction: footage replaced": after != before and b_dir in after,
        # …and must not have stacked a second matte layer.
        "correction: no extra layer": layers == "1",
    }
    for name, passed in checks.items():
        print(("  ok   " if passed else "  FAIL ") + name)
    return checks



def generate_commands(bridge: Path) -> list[dict]:
    job = sorted(bridge.glob("job-*.json"))
    if not job:
        raise SystemExit(f"no job-*.json in {bridge}; run `sample-job` first")
    proc = subprocess.run(
        [sys.executable, str(MVP / "run_adapter.py"), "--bridge-dir", str(bridge),
         "print-commands", "--job", str(job[-1])],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(MVP.parents[1]),
    )
    if proc.returncode != 0:
        raise SystemExit(f"print-commands failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)["commands"]


def latest_transform(bridge: Path) -> dict:
    """The transform recorded in the newest job — the source is seeded with it.

    Without this the validation comp's source sits at its default centre while
    the commands copy the *job's* transform onto the matte, and "position
    copied" fails for a reason that has nothing to do with the command list.
    """
    jobs = sorted(bridge.glob("job-*.json"))
    if not jobs:
        return {}
    try:
        doc = json.loads(jobs[-1].read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    transform = doc.get("transform") or {}
    return transform if isinstance(transform, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bridge", type=Path, help="bridge folder holding a job-*.json")
    parser.add_argument("--cli", default=os.environ.get("EFFECTCRAFT_CLI"),
                        help="path to effectcraft-cli.exe (or $EFFECTCRAFT_CLI)")
    parser.add_argument("--config-dir", default=None,
                        help="EffectCraft config dir for the CLI (read/write)")
    parser.add_argument("--correction-only", action="store_true",
                        help="skip the fresh-run pass; only check the brush "
                             "correction ids (needs no job file)")
    args = parser.parse_args(argv)
    if not args.cli:
        raise SystemExit("pass --cli <effectcraft-cli.exe> or set $EFFECTCRAFT_CLI")

    env = dict(os.environ)
    if args.config_dir:
        env["EFFECTCRAFT_CONFIG_DIR"] = args.config_dir

    checks: dict[str, bool] = {}

    if not args.correction_only:
        commands = generate_commands(args.bridge)
        print("commands:", [c["id"] for c in commands])

        work = Path(tempfile.mkdtemp(prefix="bb-ec-validate-"))
        jsx = work / "validate.jsx"
        jsx.write_text(JSX % (json.dumps(commands),
                              json.dumps(latest_transform(args.bridge))),
                       encoding="utf-8")

        run = subprocess.run([args.cli, "script", str(jsx), "--empty", "--json"],
                             capture_output=True, text=True, env=env,
                             encoding="utf-8", errors="replace")
        try:
            reply = json.loads(run.stdout)
        except ValueError:
            print("unparseable reply:", run.stdout, run.stderr)
            return 1
        if not reply.get("ok"):
            print("script failed:", json.dumps(reply, indent=2))
            return 1
        print(reply["result"])

        lines = reply["result"].splitlines()
        matte_index = commands[0]["params"]["index"]
        matte = next((l for l in lines if l.startswith(f"L{matte_index} ")), "")
        source = next((l for l in lines if l.startswith(f"L{matte_index + 1} ")), "")
        below = next((l for l in lines if l.startswith(f"L{matte_index + 2} ")), "")
        failures = next((l for l in lines if l.startswith("failures=")),
                        "failures=[?]")

        checks = {
            "no command failed": failures == "failures=[]",
            "matte named": "name=BetterBrush Matte" in matte,
            "matte disabled": "enabled=false" in matte,
            "source kept": "enabled=true" in source and "name=Source" in source,
            "below untouched": "enabled=true" in below and "name=Below" in below,
            # The script object model reports the matte through hasTrackMatte +
            # trackMatteType; layer.trackMatte itself stays undefined.
            "source matted": "has=true" in source
            and _value(source, "tmt=") != _value(below, "tmt="),
            "matte not self-matted": "has=false" in matte,
            "position copied": _value(source, "pos=") == _value(matte, "pos="),
            "anchor copied": _value(source, "anchor=") == _value(matte, "anchor="),
            "scale copied": _value(source, "scale=") == _value(matte, "scale="),
            "opacity NOT copied": _value(matte, "op=").startswith("100")
            and _value(source, "op=").startswith("40"),
        }
        for name, passed in checks.items():
            print(("  ok   " if passed else "  FAIL ") + name)

    checks.update(validate_correction(args.cli, env))
    ok = all(checks.values())
    print("VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
