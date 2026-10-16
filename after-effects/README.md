# Better Brush — After Effects panel

A dockable After Effects panel with **one button**: select a footage layer, click
**Remove Background**, and the layer is rotoscoped in place.

The AI work runs in a local headless Better Brush engine (the same SAM 2 stack as
the desktop app); the panel is a thin client. Nothing is sent anywhere — the
engine listens on `127.0.0.1` only.

- **Platform:** Windows, After Effects 2022+ (CEP).
- **Scope:** one **character** per click, on a **footage layer**.
- Anything ambiguous — no character, or several — is handed off to the full app
  instead of being guessed at.

---

## Install

One command, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\after-effects\install.ps1
```

What it does:

1. Locates the Python interpreter that has Better Brush installed (or use
   `-Python "C:\path\to\python.exe"`).
2. Copies this folder to `%APPDATA%\Adobe\CEP\extensions\BetterBrush`.
3. Writes `config.json` (repo path, Python paths, port).
4. Enables CEP player-debug mode for the current user (these are per-user
   registry values — no admin rights, nothing system-wide).

Then **restart After Effects** and open **Window → Extensions → Better Brush**.
Drag it into a dock if you want it permanent.

To remove it:

```powershell
powershell -ExecutionPolicy Bypass -File .\after-effects\install.ps1 -Uninstall
```

---

## Use

1. Open a comp and select the **footage layer** you want cut out.
2. Click **Remove Background**.

The first click of a session starts the engine (a few seconds). The panel then
walks the pipeline with a live progress bar and an elapsed timer — **loading the
model → reading the source → finding the character → tracking through the clip →
refining the edges (when matting is on) → writing the alpha movie** — and
finishes on a preview of the key frame on a checkerboard so you can see the
cut-out before touching the timeline.

- **Result → Replace layer** (default) swaps the layer's source, keeping trims
  and effects.
- **Result → Add as new layer** places the cut-out above the original.

Both are undoable with <kbd>Ctrl</kbd>+<kbd>Z</kbd>.

### Correcting the mask with the brush

The panel keeps a cut-out for **every frame** of the result, so you can fix the
frame that is actually wrong instead of a single key frame:

1. Scrub the **Frame** slider to that frame.
2. Pick **Add** (green) to paint pixels in, or **Remove** (red) to paint them
   out, and set the brush size.
3. **Apply correction.** The clip is re-tracked from *that* frame with your
   strokes layered onto the mask, and the result layer is replaced.

**Clear strokes** wipes the overlay. Corrections are repeatable — paint, apply,
scrub elsewhere, paint again — and each pass builds on the last. The frame you
paint on is treated as ground truth, so even a patch SAM 2 does not believe
belongs to the subject survives there. Moving the Frame slider clears the current
strokes, because strokes belong to the frame they were painted on.

### When it hands off to the full app

| Situation | What happens |
| --- | --- |
| Several characters in frame | Message + the full app opens — pick subjects there |
| No clear character found | Message + the full app opens — paint a brush stroke there |
| Layer is a pre-comp, time-remapped, or has no file | The panel says why and points you at the full app |

**Open in app** always opens Better Brush **on the clip you selected**, ready to
work on, rather than on the welcome screen.

That is deliberate: the button is for the common case, and the full app is one
click away for everything else.

---

## How it works

```
After Effects
  └─ CEP panel (index.html + js/panel.js)      thin client, dockable
       │  ExtendScript (js/host.jsx)           selection + apply back
       │  HTTP on 127.0.0.1:8765
       └─ Better Brush engine (python -m src.engine)
            VideoReader → SAM 2 auto-subject → propagate → close gaps
                        → MatAnyone 2 (soft alpha) → ffmpeg (ProRes 4444)
```

- The engine is **out of process** on purpose: no torch inside After Effects,
  no AEGP, nothing to crash the host.
- The panel starts the engine on demand via `config.json` and polls `/health`.
  It also attaches to a job that is already running, so reopening the panel
  never hides progress.
- Cancelling is supported (`Cancel` while working).
- After tracking, **enclosed gaps are closed** before anything else runs. SAM 2
  reads dark hair, finger webbing and the space inside a curled hand as
  background, which composites as see-through patches; filling them is what
  makes hair read as solid.
- The masked frames are kept next to the result (packed and compressed), so the
  brush can show any frame and a correction does not have to re-read the source.

---

## Settings

- **Quality** — `Tiny` (SAM 2.1 Tiny — the fastest tier, and the roughest:
  expect thin gaps and speckle around edges; good for blocking in timing before
  a careful pass), `Fast` (SAM 2.1 Small), `Balanced` (SAM 2.1 Base+), `Best`
  (your fine-tuned checkpoint if one is installed, Base+ otherwise).
- **Result** — Replace layer / Add as new layer.
- **Automatic matting** — on by default. After the outline is tracked, the
  MatAnyone 2 network refines it into a soft alpha that keeps hair, motion blur
  and partial edges instead of a hard cut. It is a second model pass and adds
  time to every run; turn it off for a fast, hard-edged outline. If the matting
  weights aren't installed the switch is disabled and the run falls back to the
  outline automatically.

The line under **Quality** always names the checkpoint the current choice will
load (`Uses SAM 2.1 Base+`), and the finished card names the one that actually
ran, so the panel never quietly uses something other than what it says.

Both settings persist between sessions.

A run reports up to six steps — loading the model, reading the source, finding
the character, tracking through the clip, refining the edges, writing the alpha
movie — with a bar that fills per step and a running clock. The first job of a
session pays for loading the checkpoint; later jobs in the same session reuse it
and skip that step's wait, so running the same shot again with a different
setting is much quicker than the first time.

---

## Where things are written

| What | Where |
| --- | --- |
| Alpha movie (`alpha.mov`) | `%LOCALAPPDATA%\BetterBrush\AfterEffects\<timestamp>_<id>\` |
| Key-frame preview | `preview.png` beside it (shown in the panel) |
| Correction data | `seed.png`/`seed.json` plus `masks.npz` — what the brush re-tracks from |
| Source frames | `frames\` beside them (kept so the brush is instant; dropped on very long clips) |
| Engine log | `%LOCALAPPDATA%\BetterBrush\engine.log` |

Results deliberately live in application data rather than `%TEMP%`, which
Windows is free to clear out from under a project that links to the file.

If something fails internally, the panel prints the log path in the error card —
that file has the full trace.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Panel missing from the Extensions menu | Re-run `install.ps1`, restart AE. Check `%APPDATA%\Adobe\CEP\extensions\BetterBrush\CSXS\manifest.xml` exists. |
| "engine offline" never clears | Run install again so `config.json` has the right Python path. Test it by hand: `python -m src.engine --port 8765`. |
| "config.json is missing" | Re-run `install.ps1`; it writes `config.json` next to `index.html`. |
| Engine starts then dies | That Python can't import `src.engine`/`torch`. Install the app into it: `python -m pip install -e .`. |
| A job fails with an odd message | Open `%LOCALAPPDATA%\BetterBrush\engine.log`; the panel's error card shows the path. |
| A job seems to hang at 0% | Update the panel (`install.ps1`) — an unread output pipe used to block the engine mid-job. The log now records every step, so a genuine stall names the step it stalled in. |
| A run takes much longer than the full app | Check the **Quality** line: `Balanced` (Base+) is several times slower than `Tiny`, and a cold first run also pays for loading the checkpoint. Automatic matting adds another model pass — turn it off for speed. |
| Matting switch is greyed out | The MatAnyone 2 weights aren't in `checkpoints\matanyone2\`. Run the app's model setup; until then runs fall back to the outline. |
| "Apply correction" does nothing | Paint at least one stroke first; a correction with no strokes is ignored. Also update the panel (`install.ps1`) if it's an older build. |
| Holes in hair or between fingers | The panel closes enclosed gaps automatically; update the panel (`install.ps1`) if you still see them. If a gap is genuinely open to the background, paint it in with the brush. |
| The brush shows the mask, not the soft matte | Expected: the brush surface is the tracked mask, which is what you are editing. The layer itself still uses the matted alpha. |
| Nothing happens when a fine-tuned model is picked | The sidecar in `checkpoints\custom\` must name a config the engine ships (`sam2.1_hiera_b+.yaml` and friends). |
| Panel needs inspecting | `install.ps1 -DebugPort 8092`, then open `http://localhost:8092` in Chrome. |

---

## Files

| Path | Purpose |
| --- | --- |
| `CSXS/manifest.xml` | CEP manifest (host `AEFT`, panel definition) |
| `index.html`, `panel.css` | Panel markup + styling |
| `js/panel.js` | Panel logic (engine client + UI) |
| `js/host.jsx` | ExtendScript: selection, apply-result, process launching |
| `config.sample.json` | Shape of the `config.json` the installer writes |
| `install.ps1` | Per-user installer / uninstaller |

The engine itself lives in the repository under `src/engine/`.
