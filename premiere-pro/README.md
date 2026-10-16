# Better Brush — Premiere Pro panel

A dockable Premiere Pro panel with **one button**: select a clip on the timeline,
click **Remove Background**, and the clip is rotoscoped in place.

The AI work runs in a local headless Better Brush engine (the same SAM 2 stack as
the desktop app); the panel is a thin client. Nothing is sent anywhere — the
engine listens on `127.0.0.1` only.

- **Platform:** Windows, Premiere Pro 2022+ (CEP).
- **Scope:** one **character** per clip, on a **video clip**.
- Anything ambiguous — no character, or several — is handed off to the full app
  instead of being guessed at.

> This is the **After Effects panel ported to Premiere Pro**. The markup, styling,
> the engine client and the brush are the same files; the whole of the port lives
> in `js/host.jsx`, because Premiere's ExtendScript is not After Effects'. Each
> extension folder is self-contained — a CEP extension cannot share files with
> another one — so the panel files are copies rather than shared. Fixes that
> touch `panel.css` / `js/panel.js` belong in both folders.

---

## Install

One command, from the repository root. Premiere Pro panels live in the **shared**
CEP folder, so this installs for all users and needs an elevated PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1
```

What it does:

1. Locates the Python interpreter that has Better Brush installed (or use
   `-Python "C:\path\to\python.exe"`).
2. Copies this folder to
   `C:\Program Files\Common Files\Adobe\CEP\extensions\BetterBrushPPro`.
3. Writes `config.json` (repo path, Python paths, port).
4. Enables CEP player-debug mode for the current user (a per-user registry value
   — that part needs no admin, but writing to `Program Files` does).

No administrator rights? Install for this account only:

```powershell
powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1 -Scope User
```

That uses `%APPDATA%\Adobe\CEP\extensions\BetterBrushPPro` instead. CEP scans both
locations, so the installer removes any copy in the other one on the way — two
copies of the same extension would otherwise show up twice in the menu.

Then **restart Premiere Pro** and open **Window → Extensions → Better Brush**.
Drag it into a dock if you want it permanent.

The installer uses its own extensions folder (`BetterBrushPPro`), so it sits
alongside the After Effects panel (`BetterBrush`) without disturbing it. Both
panels drive the **same** local engine on port 8765 — install one, or both.

To remove it (from every folder it might be in):

```powershell
powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1 -Uninstall
```

---

## Use

1. Open a sequence and select the **video clip** you want cut out.
2. Click **Remove Background**.

The first click of a session starts the engine (a few seconds). The panel then
walks the pipeline with a live progress bar and an elapsed timer — **loading the
model → reading the source → finding the character → tracking through the clip →
refining the edges (when matting is on) → writing the alpha movie** — and
finishes on a preview of the key frame on a checkerboard so you can see the
cut-out before touching the timeline.

- **Replace** (default) repoints the clip's media at the cut-out. The edit is
  untouched: trims, speed changes, effects and keyframes on every clip cut from
  that source all stay exactly where they were, and only the pixels change.
- **Above** imports the cut-out as a new clip on the **video track directly
  above** and switches the original off, so you keep both and can still see what
  the original looked like.

Both are undoable with <kbd>Ctrl</kbd>+<kbd>Z</kbd>.

### What "Replace" actually does

Premiere's scripting has no `replaceSource`. The call used is
`ProjectItem.changeMediaPath`, which repoints a project item at new media. It is
the same idea — the timeline edit survives — with one consequence worth knowing:
the change applies to **every** clip in the project cut from that source, and to
any sequence that uses it, not just the clip you selected. If you want the
original kept intact as well, use **Above**, or duplicate the clip (and its
source) first.

Also note the panel needs the clip to be **on disk** — an offline or unlinked
clip is reported as such rather than half-processed.

### "Above" needs a free track

Premiere's scripting cannot create a video track. So **Above** uses the track
directly above the clip, and only if nothing of the original already occupies
that span:

| Situation | What happens |
| --- | --- |
| Track above is free | The cut-out lands there, starting at the same time; the original clip is switched off |
| Track above already has footage there | The panel says so and changes nothing — move the clip down a track, or use Replace |
| The clip is on the topmost track | The panel says so and changes nothing — add a track above, or use Replace |

Nothing is ever overwritten or rippled on the way.

### Correcting the mask with the brush

The panel keeps a cut-out for **every frame** of the result, so you can fix the
frame that is actually wrong instead of a single key frame:

1. Scrub the **Frame** slider (`‹` / `›` step one frame, `,` and `.` do the same).
2. Pick **Add** (green) to paint pixels in, or **Remove** (red) to paint them
   out, and set the brush size. The cursor ring wears the same colour, so the
   mode is obvious from the brush itself.
3. **Apply correction.** The clip is re-tracked from *that* frame with your
   strokes layered onto the mask, and the result is applied again.

**Zoom** before you paint on anything small, and **scroll** to move around. A
dock at 200 px shows one source pixel as a tenth of a screen pixel; at 8× the
same brush slider paints an eighth as wide a stroke on the artwork, which is what
makes a hair correction possible.

| Action | How |
| --- | --- |
| Zoom in / out | `Ctrl`+scroll over the preview about the pointer, or `+` / `-`, or the `−` / `+` buttons |
| Scroll / pan | Scroll wheel, or middle-drag, or hold <kbd>Space</kbd> and drag |
| Back to the whole frame | `Fit`, or <kbd>0</kbd> |
| Compare with the original | **Original** (or `C`), then again to go back |

Zoom survives a frame change — zoom in on an ear, scrub, and you are still on the
ear — and resets when a new result arrives.

**Undo** removes the last stroke (<kbd>Ctrl</kbd>+<kbd>Z</kbd>) and **Clear**
wipes the overlay; both are disabled until something is painted. Corrections are
repeatable — paint, apply, scrub elsewhere, paint again — and each pass builds on
the last. The frame you paint on is treated as ground truth: **Add** makes the
pixels you cover part of the subject for real, and even a patch SAM 2 does not
believe belongs to the subject survives there. Moving the Frame slider clears the
current strokes, because strokes belong to the frame they were painted on.

Keyboard, while a result is on screen: `A` / `R` tool, `[` / `]` brush size,
`,` / `.` frame, `+` / `-` / `0` zoom, `C` compare, <kbd>Ctrl</kbd>+<kbd>Z</kbd>
undo, <kbd>Ctrl</kbd>+<kbd>Enter</kbd> apply.

### When it hands off to the full app

| Situation | What happens |
| --- | --- |
| Several characters in frame | Message + the full app opens — pick subjects there |
| No clear character found | Message + the full app opens — paint a brush stroke there |
| Several clips selected | The panel asks for exactly one |
| Clip has no media of its own (colour matte / adjustment layer) | The panel says why and points you at the full app |
| Clip is offline, or not linked to a file | The panel says why |

**Open in app** always opens Better Brush **on the clip you selected**, ready to
work on, rather than on the welcome screen. It is launched detached and without
a console, so closing the app afterwards cannot disturb Premiere.

---

## How it works

```
Premiere Pro
  └─ CEP panel (index.html + js/panel.js)      thin client, dockable
       │  ExtendScript (js/host.jsx)           selection + apply back
       │  HTTP on 127.0.0.1:8765
       └─ Better Brush engine (python -m src.engine)
            VideoReader → SAM 2 auto-subject → propagate → close gaps
                        → MatAnyone 2 (soft alpha) → ffmpeg (ProRes 4444)
```

- The engine is **out of process** on purpose: no torch inside Premiere, no
  plugin, nothing to crash the host.
- The panel starts the engine on demand via `config.json` and polls `/health`.
  It also attaches to a job that is already running, so reopening the panel
  never hides progress.
- Cancelling is supported (`Cancel` while working).
- **The engine reads the clip's real frame size, rate and length from the media
  file**, not from the panel: Premiere's scripting API does not expose a clip's
  dimensions at all. The chip under the timeline name shows the *sequence's*
  frame size and rate, which is the closest the panel can get — if your sequence
  and footage disagree, the chip is the one that is wrong, and the run is not.
- Before tracking, the found character is **re-prompted once with a box around
  it**. A point prompt only segments the part it happens to land on, and hair is
  what it most often leaves behind; asking for the largest object inside a
  padded box is what recovers the rest of the silhouette. The box may only ever
  *grow* the subject — a candidate that wandered off or came back smaller is
  discarded and the original mask is kept.
- After tracking, **enclosed gaps are closed** before anything else runs. SAM 2
  reads dark hair, finger webbing and the space inside a curled hand as
  background, which composites as see-through patches; filling them is what
  makes hair read as solid.
- The alpha is **decontaminated** at the edges. A partially covered pixel is a
  blend of subject and background, so it carries background colour at nearly full
  strength; subtracting that is what stops a hair tip wearing a halo of sky.
- With matting on, the matte is **snapped back to the mask wherever the source is
  flat** (inside the mask only). Anime is flat colour inside hard lines, so the
  network's answer in the middle of a cel is a ghost of the background rather
  than detail. Real edges, and anything the network finds beyond the painted
  outline, are left to the matte.
- The masked frames are kept next to the result (packed and compressed), so the
  brush can show any frame and a correction does not have to re-read the source.
  Rendered frames are cached, so scrubbing back to a frame is instant.

---

## Settings

- **Quality** — `Tiny` (SAM 2.1 Tiny — the fastest tier, and the roughest:
  expect thin gaps and speckle around edges; good for blocking in timing before
  a careful pass), `Fast` (SAM 2.1 Small), `Balanced` (SAM 2.1 Base+), `Best`
  (your fine-tuned checkpoint if one is installed, Base+ otherwise).
- **Result** — Replace / Above.
- **Matting** — on by default. After the outline is tracked, the MatAnyone 2
  network refines it into a soft alpha that keeps hair, motion blur and partial
  edges instead of a hard cut. It is a second model pass and adds time to every
  run; turn it off for a fast, hard-edged outline. If the matting weights aren't
  installed the switch is disabled and the run falls back to the outline
  automatically. The matte is tuned for drawn animation: it may soften the
  outline's edge, but it cannot hollow it out, and in a flat colour region it
  follows the outline rather than leaving a ghost of the background.

The name beside **Quality** is the checkpoint the current choice will load, and
the finished card names the one that actually ran, so the panel never quietly
uses something other than what it says.

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
| Alpha movie (`alpha.mov`) | `%LOCALAPPDATA%\BetterBrush\Plugins\<timestamp>_<id>\` |
| Key-frame preview | `preview.png` beside it (shown in the panel) |
| Correction data | `seed.png`/`seed.json` plus `masks.npz` — what the brush re-tracks from |
| Source frames | `frames\` beside them (kept so the brush is instant; dropped on very long clips) |
| Engine log | `%LOCALAPPDATA%\BetterBrush\engine.log` |

Results deliberately live in application data rather than `%TEMP%`, which
Windows is free to clear out from under a project that links to the file. The
folder is shared with the After Effects panel: it is the engine's work area, and
each run gets its own subfolder.

If something fails internally, the panel prints the log path in the error card —
that file has the full trace.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Panel missing from the Extensions menu | Re-run `install.ps1` (elevated), restart Premiere. Check `C:\Program Files\Common Files\Adobe\CEP\extensions\BetterBrushPPro\CSXS\manifest.xml` exists. |
| The panel doesn't see the clip you selected | Click the clip once in the timeline so it highlights, and give the panel a second — it polls. If it still refuses, the panel now repeats what Premiere reported, which is the useful part. |
| The installer says it needs administrator rights | It installs for all users by default. Re-run it from an **Administrator** PowerShell, or add `-Scope User` to install for your account only. |
| The panel is listed twice | There is a copy in two CEP folders. Re-run `install.ps1` — it removes the copies it isn't using — and restart Premiere. |
| "engine offline" never clears | Run install again so `config.json` has the right Python path. Test it by hand: `python -m src.engine --port 8765`. |
| "config.json is missing" | Re-run `install.ps1`; it writes `config.json` next to `index.html`. |
| Engine starts then dies | That Python can't import `src.engine`/`torch`. Install the app into it: `python -m pip install -e .`. |
| "Several clips are selected" | Premiere hands the panel whatever is selected; deselect the others (or select just the one clip) and it resumes on the next poll. |
| "This clip isn't linked to a file on disk" | The clip is offline, or it is a synthetic (colour matte, adjustment, transparent video). Relink it, or use the full app. |
| "No empty video track above this clip" | Premiere's scripting cannot create tracks. Add a track above and try again, or switch **Result** to Replace. |
| "The clip moved or was renamed while the job ran" | The panel re-finds the clip by its track and start time; a move that changes either one during a run loses the thread. Select the clip and run again. |
| A job fails with an odd message | Open `%LOCALAPPDATA%\BetterBrush\engine.log`; the panel's error card shows the path. |
| A run takes much longer than the full app | Check the **Quality** line: `Balanced` (Base+) is several times slower than `Tiny`, and a cold first run also pays for loading the checkpoint. Matting adds another model pass — turn it off for speed. |
| Scrolling the preview does nothing | The wheel only scrolls and zooms once a result is on screen (there has to be something to move into). At fit there is nowhere to pan to: zoom in first. |
| The brush shows the mask, not the soft matte | Expected: the brush surface is the tracked mask, which is what you are editing. The clip still uses the matted alpha. |
| Matting switch is greyed out | The MatAnyone 2 weights aren't in `checkpoints\matanyone2\`. Run the app's model setup; until then runs fall back to the outline. |
| Panel needs inspecting | `install.ps1 -DebugPort 8092`, then open `http://localhost:8092` in Chrome. |

---

## Files

| Path | Purpose |
| --- | --- |
| `CSXS/manifest.xml` | CEP manifest (host `PPRO`, panel definition) |
| `index.html`, `panel.css` | Panel markup + styling (same as the After Effects panel) |
| `js/panel.js` | Panel logic (engine client + UI), same except for the wording |
| `js/host.jsx` | **The port**: Premiere selection, media replacement, process launching |
| `config.sample.json` | Shape of the `config.json` the installer writes |
| `install.ps1` | Installer / uninstaller (`-Scope System` by default, `-Scope User` for one account) |

The engine itself lives in the repository under `src/engine/`.

### Differences from the After Effects host, in one table

| Concern | After Effects | Premiere Pro |
| --- | --- | --- |
| Host id | `AEFT` | `PPRO` |
| What is selected | `CompItem.selectedLayers[0]` | `Sequence.getSelection()` (video items only) |
| Putting the alpha back | `app.project.importFile(ImportOptions)` + `Layer.replaceSource` | `app.project.importFiles` + `ProjectItem.changeMediaPath` |
| "Keep the original" mode | New layer above the source layer | Overwrite onto the free track above; the original is disabled |
| Clip handle for a correction | `compId` / `layerId` | Track index + clip start time (Premiere has no ids) |
| Clip dimensions | `FootageItem.width` / `.height` / `.frameRate` | Not exposed at all — the engine reads the media instead |
| Installer target | `%APPDATA%\...\CEP\extensions\BetterBrush` | `C:\Program Files\Common Files\Adobe\CEP\extensions\BetterBrushPPro` |
