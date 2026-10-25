# Validation — EffectCraft 0.11.1

Every API this MVP relies on was checked against a real EffectCraft build, not
assumed from After Effects documentation.

**Build:** `effectcraft-0.11.1-windows-x64-portable` (`portable.txt` present, so
settings live in `<install>\EffectCraftData`, not `%APPDATA%\EffectCraft`).
**Harness:** `effectcraft-cli.exe` — `commands --filter … --json` for signatures,
`script <file>.jsx --demo|--empty` for behaviour, `props`/`info` for read-back.

Probe scripts from this session live in `%TEMP%\bb-ec-probe\` (`probe.jsx` …
`probe7.jsx`, `bb-probe.ecproj`).

---

## 1. Engine commands (read back from the running engine)

| id | params actually accepted | note |
| --- | --- | --- |
| `file.import` | `paths, sequence, alphabetical, frameRate, importAs, layer, page, drag, addToComp, background` **+ `time`, `index`, `position` via `app.run`** | returns `{comps, errors, items, unlabeledAlpha, warnings}`; `items` are footage **item** ids, not layer ids |
| `layer.rename` | `layer, name` | `#n` references resolve |
| `prop.set` | `layer, path, value` | paths: `transform/anchor`, `transform/position`, `transform/scale`, `transform/rotation`, `transform/opacity` |
| `layer.setTrackMatte` | `layer, matte, kind` (`alpha`\|`alphaInverted`\|`luma`\|`lumaInverted`) | records the matte on the *matted* layer and disables the matte layer |
| `comp.new`, `layer.newSolid` | as documented | used to seed validation compositions |

### Engine quirk found during validation

`effectcraft-cli run|exec` rejects `file.import` with
`unknown parameter(s) index, time` — its own validator's accepted list does not
match the `doc:` string it prints, which advertises *"at `time?`, `index?`,
`position?` as in `layer.addItem`"*.

`app.run("file.import", {…, time, index})` — the path the panel uses — accepts
both **and honours them**. Proven with `probe7.jsx`: importing at
`index: 2, time: 1.5` into a 3-layer comp put the sequence at `L2` with
`startTime 1.5`; omitting them put it at `L1` with `start 0`.

Consequence: validation must go through `app.run` (a generated `.jsx` run by
`effectcraft-cli script`), which is what `scripts/validate_commands.py` does —
and, happily, exactly the path the panel takes in production.

---

## 2. Script object model (EffectCraft's own dialect)

| fact | how it was checked |
| --- | --- |
| `JSON`, `File`, `Folder`, `app.run`, `app.beginUndoGroup`/`endUndoGroup`, `app.scheduleTask`/`cancelTask`, `app.settings.saveSetting`/`getSetting`/`haveSetting`, `$.fileName` all exist | `probe.jsx`, `probe3.jsx` |
| `Folder.temp`, `Folder.current`, `Folder.myDocuments`, `Folder.appData` are **`null`/`undefined`** | `probe2.jsx` — this is why the bridge folder is derived from `$.fileName` |
| `"onTimer" in Window` is **`false`**, `interval` likewise | `probe6.jsx` — no native timer, hence `app.scheduleTask` + a manual Check-status button |
| `app.scheduleTask(code, delay, repeat)` returns a task id and does **not** run the code while the script is still executing | `probe6.jsx`: scheduled ok, `fired=0` after a 200 ms busy-wait → callbacks run after the script returns, i.e. in the normal event loop |
| File writes are gated by *Preferences ▸ Scripting & Expressions ▸ Allow Scripts to Write Files and Access Network*; `Folder(...).create()` and `File.open("w")` both throw a sentence naming the preference | `probe2.jsx` → the panel probes once at load and says so instead of showing a dead button |
| `app.preferences.getPrefAsBool("Main Pref Section", "Pref_SCRIPTING_FILE_NETWORK_SECURITY")` returns `false` and `havePref` returns `false` — the gate cannot be read programmatically | `probe5.jsx` → probe-by-attempted-write is the only reliable check |
| `FootageItem.file` is a **property** (`typeof object`, `null` for solids), not a method; compositions have no `file` at all | `probe2.jsx`, `probe3.jsx` |
| `layer.trackMatte` is **`undefined`** script-side; read `layer.hasTrackMatte` (bool) and `layer.trackMatteType` (enum int) instead. `info`/`props` report `trackMatte` as an object `{kind:"Alpha Matte", layer:<id>}` | `probe.jsx`, `probe4.jsx`, validation run below |
| `layer.transform.anchorPoint` is the script accessor for `transform/anchor` | `probe4.jsx`, `probe5.jsx` |
| A layer's `source` can be `null` (text/shape/null) and solids/precomps have no usable file | `probe.jsx` → the panel rejects them with a message |
| Three resolvers exist for globals (`$.global`, `window`, bare `this`); none is documented | panel publishes `BetterBrushPanel` through all three |

---

## 3. End-to-end command chain (the MVP's actual output)

`scripts/validate_commands.py` generates a `.jsx` that seeds a composition,
runs the adapter's command list inside one undo group with `app.run`, and reads
the layers back.

Run 1 — `matte_index = 1`:

```
commands: ['file.import', 'layer.rename', 'prop.set', 'prop.set', 'prop.set', 'prop.set', 'layer.setTrackMatte']
seeded: layers=2 matte_index=1
ran file.import -> {"comps":[],"errors":[],"items":[51],…}
L1 name=BetterBrush Matte enabled=false pos=160,90,0 scale=100,100,100 op=100 tmt=5012 has=false
L2 name=Source           enabled=true  pos=160,90,0 scale=100,100,100 op=40  tmt=5013 has=true
L3 name=Below            enabled=true  …
  ok   no command failed / matte named / matte disabled / source kept / below untouched
  ok   source matted / matte not self-matted
  ok   position copied / anchor copied / scale copied
  ok   opacity NOT copied
VALIDATION: PASS
```

Run 2 — `matte_index = 3` (two layers above the source), same checks:

```
L1 name=Above 2            enabled=true
L2 name=Above 1            enabled=true
L3 name=BetterBrush Matte  enabled=false      <- inserted at the source's index
L4 name=Source             enabled=true has=true   <- pushed down by one
L5 name=Below              enabled=true
VALIDATION: PASS
```

Both runs confirm:

- the matte lands **directly above** the source at every stack position;
- the engine **disables the matte layer** and marks the source `hasTrackMatte`;
- `anchor`/`position`/`scale` are copied, `opacity` is **not** (source set to
  40 before the run; matte still reads 100);
- an untouched layer above and below the pair are unaffected;
- a failing `file.import` (missing PNG, run deliberately) stops the chain: no
  rename, no `prop.set`, no track matte is applied to the wrong layers.

---

## 4. Automated checks

```powershell
python -m pytest extensions\effectcraft\tests -q
# ...............................................................    61 passed
```

Covered: job round-trip and every protocol rejection, atomic status writes
(no `.tmp` left behind), progress fractions including indeterminate, command
order/index arithmetic/transform policy/path normalisation, a real
`simulate` run (RGBA frames, moving alpha, terminal status carrying commands),
missing-source error, cancellation (`cancel-<id>` present → `cancelled`, flag
cleared afterwards), a stale flag cleared even on success, corrupt status not
blocking the watcher, and a malformed job still receiving a terminal status.
Since v0.2 also: the `Correction` document (round trip, empty-stroke,
degenerate-polygon and `resume_dir` rejections, item id vs name), the
correction command list (`file.replaceFootage` + `layer.mask.mode`, import
fallback when no item was recorded, index arithmetic that does not shift),
the adapter's `post` stage (on/off, on-disk rewrite), polygon application in
`simulate` order (add, then remove), the `resume_dir` path guard, the stroke
rasteriser, the post pass itself (flicker reduced, solid regions and RGB
untouched, cancellation), and the panel's result image (fixed 300 × 169
canvas, correction key frame selected, silent `""` on failure).

Adapter smoke checks performed manually:

| check | result |
| --- | --- |
| `doctor` | prints bridge candidates, ffmpeg, numpy, cv2, engine importability |
| `sample-job` → `once --mode simulate` | 12 PNGs with alpha, `status` `done`, commands attached |
| `once` again | `nothing pending` — terminal statuses are respected |
| `watch --interval 0.2` for 5 s | picked up a new job, did **not** re-run the finished one |
| `print-commands` | JSON command list, forward-slash paths, `index`/`time` correct |

---

## 5. The panel in a live EffectCraft session

Verified by installing into a portable 0.11.1 build (`EffectCraftData\Scripts\ScriptUI
Panels\BetterBrush.jsx`) and driving it from the real GUI with a real video layer.
Every line below was learned the hard way — a field failure, then the fix.

### Engine facts

| fact | consequence |
| --- | --- |
| A **panel** script sees `$.fileName` as a bare file name (`"BetterBrush.jsx"`, no directory), while a plain script gets a full path | the panel cannot walk up to its own folder. `install.ps1` now **stamps** the absolute bridge path into the copy it ships (`@@BB_BRIDGE@@` → path); the panel validates the stamp against `<bridge>/../Scripts/ScriptUI Panels/BetterBrush.jsx` before trusting it, and keeps the `$.fileName` walk as fallback |
| `Folder.temp`, `Folder.current`, `Folder.appData`, `Folder.desktop`, `Folder.commonAppData`, `Folder.system` are **all `null`/`undefined`** | there is no root from which to search for the settings folder — hence the stamp, not a scan |
| `prefs.get` is **schema-locked** (`unknown setting betterbrush.bridgeDir` for any key EffectCraft does not know) | a bridge path cannot be smuggled through `prefs.json` |
| file writes are gated by *Preferences ▸ Scripting & Expressions ▸ Allow Scripts to Write Files and Access Network*; with it off, `Folder.create()` / `File.open("w")` throw a sentence naming the preference | the panel probes once at load and shows the engine's own sentence (not a guess) |
| the gate's stored pref can read `true` on disk while a session reports `false` (`prefs.get` → `false`, `prefs.set` → writes, `get` still `false`) | headless sessions appear to force it off; only the panel's own write probe tells the truth in the GUI |
| `File` does **not** honour `seek(0, 2)` | appending with `"e"` + seek left overlapping fragments in `panel.log`; the log is now read-then-rewritten |
| `app.scheduleTask(code, delay, repeat)` returns a task id but **EffectCraft 0.11.1 never executes it** — its desktop session is built with `script_ui.dispatch` set and `script_ui.tasks = None` (`crates/host/src/lib.rs`), so `ScriptUi::run_tasks` returns `None`, the per-frame `run_script_tasks()` bails immediately, and no repaint is ever requested for the task. One-shot tasks don't run either (`repeat` never mattered). Upstream **PR #1867** (24 Oct 2026) replaced that line with `effectcraft_script::install(&mut s)`; verified present in the **v0.17.0** tag (§8) | auto-update therefore needs *two* pumps: one repeating task with a 4 s heartbeat (live on a fixed build) **plus** a `mousemove` listener on the window root that throttles `tick()` while the pointer is over the panel (the only pump an unfixed build has — events are the one thing that still reaches the script). `noteDeadTimer()` distinguishes the cases: it runs from an interaction, which proves a frame happened, and `run_script_tasks()` runs at the top of every frame *before* any handler — so a task that had not fired by then never will |
| the panel must publish its hook to **every** global (`$.global`, `window`, the bare global) and the task string must probe all three | testing one bare `BetterBrushPanel` evaluated to `false` forever: the job finished, the panel never noticed. Field symptom was a permanent "Watching …" line |
| `layer.index` is 1-based and the matte is inserted *at* the source's index, pushing the source to `index + 1` **after** the commands run | the pre-apply drift guard must check the source at its **recorded** index; checking `matte_index + 1` refused every job whose source was the bottom layer |

### Verified end to end (GUI, real footage)

```
panel writes job-bb-….json → adapter `watch --mode engine` → status running/done
→ panel auto-poll (no clicks) → one undo group →
L1 BetterBrush Matte (disabled, alpha matte)  L2 <source> (hasTrackMatte)
status: "Applied (simulated matte) — BetterBrush Matte is the track matte of layer 2."
```

`panel.log` (in the bridge folder) records, for any run: `panel loaded … bridge=…
writable=… hook@N`, `tick armed task=N (waiting for first tick)`,
`first tick (auto-update running)`, `submitted …`, every de-duplicated poll
failure (`no status file yet`, `status parse failed`, …), `terminal state=… cmds=…`
and `apply failed: …` — enough to name a cause without guessing.

### Fixed during field testing (all shipped in this folder)

1. `install.ps1` treated the **install root** as the settings folder → panel
   installed where EffectCraft never looks. Now targets `EffectCraftData`
   (beside `portable.txt`) and warns if `prefs.json` is missing there.
2. `bridge.path.json` was written **with a BOM** → the adapter's JSON read
   failed *silently* and it watched `%APPDATA%\EffectCraft` instead. Now
   `utf-8-sig` on read, BOM-free on write, with regression tests.
3. "Check status" replaced the real startup error with a generic
   "Waiting for file-write permission (see above)" that pointed at nothing.
   The panel now keeps one `state.blocked` sentence and echoes it verbatim.
4. Cancel dropped an already-finished result (it cleared `state.job` without
   looking first). Cancel now polls first and applies a terminal result.
5. Auto-poll scope bug (above) and the drift-guard index bug (above).

---

## 6. Not yet verified

- **Sequence import edge cases**: non-contiguous frame numbers, `00000.png`
  present but a gap later, sequences longer than the layer's out point.
- **Cancel mid-render** in the GUI (the flag is written and the adapter honours
  it — tested in pytest — but not clicked during a live engine job).
- **Panel layout at narrow widths** and undocked/floating palette behaviour.
- **A brush correction end to end in the GUI** (the two new command ids are
  validated live — see §8 — but no correction has yet been driven through a
  running engine job from the panel).
- **The post-processing pass, visually**: asserted by tests (flicker down,
  solid regions and RGB untouched) but not yet A/B'd on real footage.
- **The panel's result image**: `group.add("image", undefined, File)` is the
  documented ScriptUI form and every failure degrades to a caption, but the
  control has not yet been rendered by this EffectCraft build.
- **The hover pump's cost**: `mousemove` over the panel is a session hand-off
  per event (the *work* is throttled to `POLL_MS`, the event is not).
  Unmeasured across a long sweep; if it ever feels laggy, narrow the listener
  from the window root to the status block.
- **The 0.11.1 → 0.17.0 upgrade itself**: the panel installs, the bridge is
  stamped into the new portable folder, and a full engine job has been run
  there (§8 v0.3.2 and v0.4.0). Nothing else about 0.17.0 (prefs, workspaces,
  other panels) has been compared against 0.11.1.
- **The stroke workflow in the GUI**: adopting a mask painted with the
  freehand tool, parking it in mask mode `None` and sending it through a
  correction is unit-tested and syntax-checked, but the *painted* path has
  not yet been driven by hand in a live session.

---

## 7. `--mode engine` — the real Better Brush pipeline

Run against the same footage the panel had loaded (torch 2.11+cu128, CUDA on an
RTX 5060 Laptop, `checkpoints/sam2.1_hiera_base_plus.pt` already present):

```powershell
python extensions\effectcraft\run_adapter.py --bridge-dir <bridge> once --mode engine
# 2026-10-25 12:42:56 starting bb-engine-test (engine) …
# 2026-10-25 12:43:45 finished bb-engine-test: done: rendered 12 frames in 49.701s
```

| field | value |
| --- | --- |
| state / message | `done` / `rendered 12 frames in 49.701s` |
| `result.engine` | `betterbrush` |
| `result.model` | `sam2.1_hiera_base_plus` |
| output | 13 RGBA PNGs, 1920×1080, `…/out/<id>/png/%05d.png` |
| alpha | real segmentation, subject coverage 34.3 % → 62.7 % between first and last frame |
| commands | all 7, attached before the final `done` status |
| stderr | TFLite/XNNPACK face-landmarker warnings only — no errors |

FFmpeg was not needed: `run_job` returned a PNG sequence directly
(`output_kind == "png_sequence"`); the `alpha.mov` → PNG fallback stays in place
for builds that emit a movie.

Because the panel now writes `mode: "engine"`, a plain

```powershell
python extensions\effectcraft\run_adapter.py watch
```

runs the real model. `--mode simulate` remains the override for a cheap
plumbing check (no GPU, ~0.1 s for 12 frames).

Measured cost on this machine: **≈ 4 s/frame including model load** for 12
frames (50 s total); the propagation loop itself runs at ~9 it/s, so a 51-frame
layer lands in the 1–2 minute range — during which the panel shows the
`propagate` stage and a moving progress bar, and Cancel stays available.

---

## 8. v0.2 — quality, matting, post-processing, brush corrections

The panel now mirrors the After Effects plugin and the desktop app's dock:

| control | wire field | what it reaches |
| --- | --- | --- |
| Quality (Tiny/Fast/Balanced/Best) | `options.quality` | `JobRequest.quality` → `VALID_QUALITIES` (`tiny`/`fast`/`balanced`/`best`); default Balanced |
| Matting checkbox (on) | `options.matting` | `JobRequest.matting` → MatAnyone 2 |
| Post-processing checkbox (on) | `options.post_process` | the **adapter's own** stage `post` (`postprocess.py`): similarity-gated temporal smoothing + edge-band median cleanup of the finished alpha. The engine's mask-domain hole-filling has no `JobRequest` knob and stays unconditional |
| Brush section (list, +/−, Refresh, Apply corrections) | `job.correction` | polygons in layer pixel space → rasterised by the adapter into `correction_add`/`correction_remove` PNG masks with `resume_dir` + `correction_frame` |

### Correction flow

Masks are drawn on the footage layer with AE/EC's native Freehand/Pen tools
(the panel does not paint — native tools give native feedback), listed by the
panel, and marked `+` (paint in) or `−` (paint out) with the buttons or a name
prefix. **Unmarked masks are never sent**, so a mask the layer already had for
another purpose cannot silently join a correction. `Apply corrections` sends
the marked polygons plus the context saved after the last successful apply
(`brush-context.json` in the bridge: comp, layer id, work folder, matte item),
and the adapter answers with two commands instead of the seven-command import
chain: `file.replaceFootage` (frames swap in under the existing matte layer —
no index churn, one Ctrl+Z) and `layer.mask.mode {mode: "None"}` per stroke
mask, which parks the masks so they stop clipping the layer they were drawn
on.

### Live validation of the two new ids

`scripts/validate_commands.py --correction-only` builds two numbered PNG runs,
imports one, adds a mask, then executes exactly what the panel will run:

```
import -> {"comps":[],"errors":[],"items":[2], …}
imported item=[00000-00001].png file=C:\…\run-a\00000.png
mask=Mask 1 modeBefore=6813
modeAfter=6812 MaskMode.NONE=6812
replace -> null
beforeFile=C:\…\run-a\00000.png
afterFile=C:\…\run-b\00000.png
layerCount=1
failures=[]
  ok   correction: no command failed
  ok   correction: mask mode changed
  ok   correction: mask mode is None
  ok   correction: footage replaced
  ok   correction: no extra layer
VALIDATION: PASS
```

Facts learned doing it:

| fact | consequence |
| --- | --- |
| `MaskMode.NONE` is **6812** in this dialect (Add is 6813) | `mode: "None"` really lands — checked against the enum, not merely accepted |
| `file.replaceFootage {path, item: <id>, sequence: true}` re-points the *item's* file and adds **no layer** (`layerCount=1`) | a correction keeps the matte layer, its index and its track-matte wiring intact |
| `file.import {paths, time, index}` **does not add a layer to an empty comp** (returns `items:[2]`, `numLayers` stays 0) | the validation script seeds/uses `layer.addItem {item, time, index}` instead; worth remembering for any script that imports into a fresh comp |
| `layer.mask.mode` takes `mask` by **name** as well as index/uid | the panel can park a mask without knowing its index |
| the fresh-run validator seeded only the source's *opacity*, so "position copied" compared the job's transform against the comp's default centre and failed for a reason unrelated to the command list | the script now writes the job's own `transform` onto the source before running the commands, so the copy checks can actually fail (or pass) on merit |

The post pass (`postprocess.py`) and the correction round trip are covered by
the pytest suite above; what remains unverified in the GUI is listed in §6.

### v0.3 — the panel updates itself, and shows the result

Two things came out of the first field session with v0.2: the operator was
still pressing **Check status**, and there was no image to judge a render by.

**No button.** The panel runs a single repeating `app.scheduleTask(TASK_CODE,
TICK_MS, true)`; `tick()` does one unit of work and never re-arms (a repeating
task re-armed from inside itself would multiply tasks with every fire). A tick
that throws logs one de-duplicated line instead of disturbing the task. Idle
ticks are not wasted: they compare a cheap signature (comp name, selected layer
id, mask count and names) and rebuild the mask list only when it changes,
otherwise just refreshing the **Frame** line as the time indicator moves.
Rebuilding only on change is what keeps the listbox selection — and the `+`/`−`
marks — intact.

**Fallback / heartbeat.** Every control's handler starts with `wake()`, which
pumps a tick (and, if the stored task has gone quiet for `HEARTBEAT_MS` = 4 s,
cancels it and arms a fresh one). `armTick` logs `tick armed task=<id>` once per
panel load, `tick` logs `first tick (auto-update running)` once, and a repair
logs `tick repaired: …` once — so a dead build is diagnosable from `panel.log`
without flooding it. `say()` also drops consecutive duplicate lines now: the
tick re-sends the same stage text every 700 ms and `panel.log` is rewritten in
full on every line (v0.2 logged "Writing PNG frames" seven times in six
seconds).

**The image.** The adapter now writes `out/<id>/preview.jpg`
(`pipeline.write_preview`): a fixed **300 × 169** letterbox of the correction's
key frame (or frame 0), alpha composited over a checkerboard so a cut-out reads
as a cut-out. Fixed size because ScriptUI renders an image at its natural size
— the panel cannot scale it, so the layout must not depend on the source's
aspect. The panel swaps it into a fixed slot (an image control has no
"set source"), captions it `frames · model · quality`, and remembers it in
`brush-context.json` so a reopened panel still shows the last result. A failed
preview returns `""` and never fails a job; a failed image control degrades to
a caption reading `(preview unavailable: …)`.

Panel `v0.3.0`; 61 tests.

### v0.3.1 — the tick that never fired (field report)

Second field session: the status line froze at
`Job … queued — waiting for the adapter…` and no progress appeared, while
`adapter.log` showed the job running and finishing normally
(`14:11:26 starting … 14:12:03 finished … rendered 49 frames`).

**Measured cause.** `panel.log` is decisive:

| version | arming | lines after submit |
| --- | --- | --- |
| v0.2.0 | `scheduleTask(code, 700, true)` | stage lines every tick, plus `terminal state=done` |
| v0.3.0 | `scheduleTask(code, 700, false)` (one-shot chain) | **none** — `submitted` / `queued` and then nothing until the panel was reloaded |

So in this build `repeat = false` **never fires at all** — not "fires once",
never. A one-shot task whose whole job is to schedule the next one-shot task
therefore never starts the chain, and because `wake()` only pumped when
`state.task <= 0` (and the arm had *succeeded*, returning a valid id), the
opportunistic fallback did not cover it either.

**Fix (v0.3.1).**

1. Back to `scheduleTask(TASK_CODE, TICK_MS, true)` — the setting v0.1/v0.2
   proved — and `tick()` never re-arms, so a repeating task cannot multiply.
2. **Heartbeat**: `state.lastTick` (falling back to `state.armedAt`, so a task
   that was armed and *never fired* counts as stale too) versus
   `HEARTBEAT_MS = 4000`. `wake()` — which every control handler calls —
   cancels a stale task (`app.cancelTask`) and arms a fresh one, then pumps a
   tick, so a dead timer is repaired by the first touch of the panel.
3. **Diagnosis lines**, each logged once: `tick armed task=<id> (waiting for
   first tick)`, `first tick (auto-update running)`, `tick repaired: …`. The
   next session's `panel.log` says outright whether the auto-update is alive.
4. The task source falls back `p.tick ? p.tick() : (p.poll ? p.poll() : false)`
   and the hook is published to `$.global` / `window` / the bare global
   unconditionally (a hook left by a previous load is replaced, not kept);
   `start()` now runs *after* the publish, so the first scheduled task always
   has a hook to find.
5. A previous load's repeating task id is parked on `$.global` and cancelled
   when the panel re-arms — otherwise two tasks would tick one panel.

**Progress you can see.** While a job runs the status line now carries the
adapter's `current/total` when it reports one, and a separate **elapsed**
counter (`elapsed 34s` / `elapsed 1m 12s`) sits under it — deliberately on its
own control, because `say()`'s text is written to `panel.log` and a per-tick
counter would rewrite the whole log every 700 ms. `say()` also drops
consecutive duplicate lines now, so a stage that repeats does not flood the log
(the v0.2 log had seven identical `Writing PNG frames` lines in six seconds).

Panel `v0.3.1`; 61 tests.

> **Superseded:** the diagnosis above was wrong about the cause. `repeat` had
> nothing to do with it — no scheduled task ever ran on this build (v0.3.2
> below).

### v0.3.2 — root cause: 0.11.1 has no task runner at all

Third field session: the panel froze again, and the user pressed Cancel *after*
the adapter had already finished — only then did the result import.

**Measured cause — the log ordering is decisive** (`panel.log`, the 14:35 local
session):

```
13:35:29.796Z tick armed task=1 (waiting for first tick)   ← panel load
13:35:34.053Z tick repaired: no tick for 4000 ms           ← from a click
13:35:34.056Z first tick (auto-update running)             ← the same click, 3 ms later
13:35:49.102Z submitted bb-…-abf985
13:38:19.452Z terminal state=done cmds=7                   ← from the Cancel click
```

`first tick` lands *after* the repair line, so both came from `wake()`: the task
was **3.3 s overdue in that same frame and still did not run** —
`run_script_tasks()` executes at the top of `frame()`, before any handler. And
between 13:35:49 and 13:38:19 (the adapter finished at 13:36:42) there is not
one stage line, although `poll()` logs every stage change. So the task never ran
on this build or any earlier one: **every "update" in the v0.1–v0.3.1 logs was
the user's own click.** That also rewrites the conclusion above — "`repeat =
false` never fires" was never about `repeat`.

**Engine cause.** `crates/host/src/lib.rs` @ v0.11.1:

```rust
script: Some(effectcraft_script::runner),
script_ui: ScriptUi { dispatch: Some(effectcraft_script::dispatch_ui), ..Default::default() },
```

`dispatch` is installed (so the panel runs and clicks work) but `script_ui.tasks`
is left `None`. `ScriptUi::run_tasks` then returns `None`, so the GUI's
per-frame `run_script_tasks()` returns before touching the queue: no task is ever
executed and — because `next` is never reported — **no repaint is requested for
it either**, so the app repaints only when the user does something. Everything
observed follows, including v0.3.0's silent one-shot.

Upstream **PR #1867** ("Fix scheduled ScriptUI tasks in host sessions", merged
24 Oct 2026) replaces that line with `effectcraft_script::install(&mut s)`,
installing runner + dispatch + task runner together; its own validation says
*"A local v0.11.1 desktop build with this fix completed a large JSON import
after one Open click, without manual task pumping."* Verified present in the
**v0.17.0** tag, released 25 Oct 2026.

**Fix (v0.3.2) — works on both builds.**

1. **Hover pump** (`onPanelMove`): a `mousemove` listener on the window root —
   ScriptUI bubbles mouse events to their containers, so one listener covers
   every control — throttled to `POLL_MS`, and stood down whenever `lastTick`
   is fresh, so a build that *does* run timers never pays for it twice. On
   0.11.1 this is the only self-update that exists: events are the one thing
   that still reaches the script.
2. **`noteDeadTimer()`**, called from every interaction. An interaction proves a
   frame ran, and `run_script_tasks()` runs before any handler in it — so if
   `taskFired` is still false 5 s (`TIMER_PROBE_MS`) after arming, this build
   never executed the task. It logs which case it is — `never ran the scheduled
   task` / `ran Nx but never reached the panel hook` / `did not arm` — with
   `app.version`, and states the same fact in the status line while idle: the
   panel never looks frozen without saying why.
3. **The task counts its own executions** on `$.global.BetterBrushTaskRuns`
   before it looks for the hook, so "never ran" and "ran but lost the hook" are
   distinguishable from the panel side.
4. `tick(via)` records *who* pumped it — `first tick … via wake|hover|task` —
   and the first `task`-driven tick logs `engine timer running (scheduled task
   fired by EffectCraft)`: the one line that proves a fixed build is fixed.
5. Installed into the new **0.17.0** portable folder
   (`install.ps1 -SettingsDir …\effectcraft-0.17.0-windows-x64-portable\…\EffectCraftData`),
   which re-records `bridge.path.json` (the adapter follows it), writes
   `scriptui_panels.json` so the panel auto-opens, and now records the
   *detected* `effectcraft_ver` instead of a hard-coded `0.11.1`.

Panel `v0.3.2`; 61 tests.

### v0.4.0 — the brush, "Open in Better Brush", one file to install

**v0.3.2's fix, confirmed in the field** (0.17.0, the first session on a fixed
build). `panel.log`, from load to applied, with no clicks after submit:

```
15:03:01.444Z panel loaded v0.3.2 … hook@2
15:03:02.132Z engine timer running (scheduled task fired by EffectCraft)
15:03:02.137Z first tick (auto-update running) via task
15:03:17.730Z submitted bb-20261025-160317-4cf07a (Job)
15:03:33.098Z Loading the model — loading sam2.1_hiera_tiny
… stage lines every tick …
15:04:23.287Z terminal state=done cmds=7
15:04:23.307Z Applied — BetterBrush Matte is the track matte of layer 2.
```

The task fired 688 ms after load, the hover pump never had to do anything, and
every stage line landed on its own. 0.11.1 could not have produced this log.

**1. The brush became a brush.** The section used to be a mask browser: draw
masks by hand, find them in a list, mark each + or −. It is now the tool the
reference panels (After Effects / Premiere) are:

- **It opens when the matte lands** (`brushReady()`), on the layer the run
  belonged to rather than whatever is selected — `strokeLayer()` finds it by
  id, then by name, then falls back to the selection.
- **Masks painted afterwards are adopted as strokes.** Masks that already
  existed are remembered in `ignored` at arm time and can never become
  strokes; a stroke's intent is its own mask mode when the host gave it one
  (painting out gives Subtract), else the tool the brush is armed with.
- **Every stroke is parked in mask mode `None`** the moment it is adopted
  (`layer.mask.mode`), so it describes the correction without clipping the
  layer. The values were read back live: `MaskMode.NONE = 6812`,
  `Add = 6813`, `Subtract = 6814`.
- **+ Add / − Remove** arm the next stroke, or re-mark the selected one —
  the two-context shape the reference panels give a tool segment, with the
  status line always saying which happened.
- **Undo stroke / Clear** delete the masks (`layer.mask.remove`), each one an
  ordinary undoable engine step, so Ctrl+Z brings them back.
- **A correction consumes its strokes**: after the swap lands, `armStrokes()`
  re-baselines, so the next correction starts from an empty list instead of
  re-applying yesterday's paint.

ScriptUI has no canvas and cannot draw over the viewer, which is the one thing
the HTML panels do; the mask is the stroke medium here and everything about
masks is the panel's bookkeeping, not the user's.

**2. Open in Better Brush.** A panel cannot start a process — this build's
`system.callSystem` throws and `File.execute` only opens a file — so the
request is another bridge message: `app-open-<id>.json` → the adapter launches
`BetterBrush.exe <clip>` detached (its own process group, no console) →
`app-open-status-<id>.json`. `effectcraft_bridge/appbridge.py` finds the exe
from `app-path.json`, else a search of the usual install folders and the
repo's own builds, and **writes whatever it found back** — which is what makes
"point at it once" true. A missing exe answers with the sentence that tells
the user to paste the path into the panel's field and click Save (ScriptUI has
no file dialog: `File.openDialog` returns `null`).

**3. A one-file release.** `scripts/make_dist.ps1` builds
`dist\BetterBrush-EffectCraft-Setup-<ver>.cmd` — one double-clickable file
that carries the panel, the adapter and the installer as a base64 zip after a
marker, decodes itself into `%TEMP%` and runs the installer inside. That
installer (`scripts/dist-install.ps1`) finds EffectCraft (asking when it
cannot decide), copies the panel in with the bridge stamped, merges it into
`scriptui_panels.json`, installs the adapter to
`%LOCALAPPDATA%\BetterBrush\effectcraft-adapter`, and registers it in
`HKCU\…\Run` so it starts in the background at logon — and starts it now.

With `-BuildAdapter` the payload also carries `betterbrush-adapter.exe`, a
92 MB PyInstaller onefile of the adapter, so the target machine needs no
Python at all. **Testing that exe found a real bug**: a frozen adapter's
`__file__` lives in PyInstaller's extraction folder, so
`protocol.recorded_bridge_dir()` looked for `bridge.path.json` in a temp dir
and silently fell back to `%APPDATA%\EffectCraft`. `run_adapter.py` now
translates the recorded file into `$BETTERBRUSH_BRIDGE_DIR` when
`sys.frozen` is set; verified by running the packaged exe against a sandbox
bridge (recorded path → launch → answer → request consumed).

Panel `v0.4.0`; 79 tests.

### v0.4.1 — the adapter starts with EffectCraft, not with Windows

The ask: **no logon launch** — the adapter may only come up when EffectCraft
does.

**Three mechanisms that do not exist on this machine** (measured, not assumed):

| candidate | why it is out |
| --- | --- |
| Task Scheduler WMI process trigger | `MSFT_TaskWmiEventTrigger` is not in `ROOT/Microsoft/Windows/TaskScheduler` here, a task XML containing `<WmiTrigger>` is rejected as "unexpected node", and `Register-ScheduledTask` refuses the untyped CIM instance |
| Event-log trigger on process start | no EffectCraft log exists, `Microsoft-Windows-Process-Trace/Operational` is absent, and process creation (4688) only lands in the Security log behind an **admin** audit policy — this session is not elevated |
| A logon-time watcher process | the adapter itself would still be preceded by something that runs at logon, which is what was asked to stop |

**What ships: the panel starts the adapter.** The panel is opened *by*
EffectCraft, so it is the one thing that provably runs when EffectCraft does:

1. The adapter stamps `<bridge>/adapter.alive` with an epoch on every watch
   cycle **and on every progress report** — a render blocks the watch loop for
   the whole job, and a heartbeat that stopped with it would read as "no
   adapter" mid-job and start a second one. (`test_heartbeat.py`).
2. The installer writes `<bridge>/start-adapter.vbs` — wscript runs it with no
   window, where a `.cmd` would flash a console — plus `start-adapter.cmd` for
   a human to double-click. **No Run key**: anything v0.4.0 registered is
   removed on install and on uninstall.
3. The panel runs it once per session, either 8 s after it loads (grace for a
   onefile exe's boot) or when a submitted job has had no status for 12 s
   while the heartbeat is stale. It goes through `File.execute()` →
   `file.executeFile`, which needs the write permission the panel already
   requires.
4. `file.executeFile` replies `{needsConfirmation}` when **Warn User When
   Executing Files** is on (the engine default) — a script can never confirm
   for itself (`confirmed && !running_for_script()`, upstream #1246). The
   panel then says what the dialog is and how to stop seeing it, instead of
   failing quietly.

**Field failure: `'NoneType' object has no attribute 'write' (while tracking
the subject)`.** The first jobs started through `start-adapter.vbs` died in
`propagate`, while the identical job had completed an hour earlier — and
`adapter.log` looked perfectly healthy, because `print()` is a **silent
no-op** when `sys.stdout` is `None`.

Measured cause, reproduced by launching a probe exactly the way the `.vbs`
does (wscript, no console):

```
stdout=None
stderr=None
tqdm raised: AttributeError: 'NoneType' object has no attribute 'write'
```

Python hands a console-free start no streams; SAM 2's tqdm bar in
`propagate` writes to `sys.stderr` directly and took the job down at its
first progress bar. Running the same job in a console succeeded end to end,
which is what isolated the launch method as the only variable.

**Fix.** `give_the_process_streams()` in `run_adapter.py` runs before
anything is imported and points a `None` stream at the null device; a start
that *does* have a console keeps its own (both asserted in
`test_heartbeat.py`, including a real `tqdm` run). Re-verified through the
real path: restart via `start-adapter.vbs`, queue the same job, watch
`finished bb-repro-…: done: rendered 49 frames in 49.323s`.

**The same round closed the release's own gap**: the packaged
`betterbrush-adapter.exe` carries numpy and OpenCV but no torch, so it can
only ever run `--mode simulate`. `install.ps1` now records `repo_root` in
`bridge.path.json`, `run_adapter.py` puts it on `sys.path`, and the dist
installer prefers a Python that really imports `torch, cv2, numpy` over the
exe — printing which it picked. Verified from an installed copy outside the
checkout: `doctor` → `engine: importable (mode=engine available)`.

**Field failure: "always waiting for the adapter".** The panel sat at
`Job … queued — waiting for the adapter…` with `elapsed 0s` while
`adapter.log` said nothing at all. Two EffectCraft installs were live: the
portable **0.11.1** (still holding panel v0.3.1, stamped to its own bridge) and
the **0.17.0** the adapter was recorded against. The job file was written to
the 0.11.1 bridge; the watcher only ever looked at the recorded 0.17.0 one, so
nobody ever saw the job — and nothing was logged anywhere, because the panel
was asking a folder that had no reader.

**Fix.** `cmd_watch` / `cmd_once` now serve **every** bridge on the machine:
the recorded one, the standard candidate folders that exist, and portable
installs discovered once at startup by a bounded search of
Downloads/Desktop/Documents (`<install>\EffectCraftData\betterbrush-bridge`,
up to four levels). `--bridge-dir` / `$BETTERBRUSH_BRIDGE_DIR` still mean
exactly one folder, so tests and CI never touch a real install. Each served
bridge gets its own heartbeat, so a panel in either install sees a live
adapter and does not start a second one.

Verified on the stuck job itself: the adapter's startup line became
`watching 3 bridge(s)` (recorded 0.17.0, `%APPDATA%`, and the 0.11.1 install),
and the job that had waited 5 minutes finished —
`finished bb-20261025-190921-b0e1f5: done: rendered 49 frames in 18.637s`,
49 frames, 7 commands. The panel file was also reinstalled (v0.4.1 + its own
`start-adapter.vbs`) into the 0.11.1 install, so whichever build is opened
gets the current panel.

Panel `v0.4.1`; 94 tests.

### v0.5.0 — the brush paints on the frame

Two field reports about the same panel state: the result image appeared
**twice**, and painting did nothing.

**1. The doubled image.** `remove()` on an image control is a **no-op** in
this engine — verified: after `child.remove()` a group's `children` count is
unchanged, so the old `showPreview()` (remove, then add a new control) stacked
a second picture over the layout on every result. Fixed by creating the
control once and re-pointing it: `image.image = new File(...)`, which the
snapshot honours (`__uiSnapshot()`'s `src` follows the assignment), plus
`visible` for the empty state.

**2. Painting.** v0.4's "draw a mask with the mask tool and the panel adopts
it" assumed the wrong place: a user of the AE/Premiere panels paints on the
*picture*. The brush now does that:

* the result image is the canvas — this engine delivers `mousedown` /
  `mousemove` / `mouseup` to a widget with a listener, with coordinates
  **relative to that control** and `shiftKey` (`crates/script/src/scriptui.js`
  and the Rust `mouse_events`). A drag is a stroke; Shift swaps add/remove.
* the preview follows the **time indicator**, reading the rendered frame from
  `result.png_dir` — so what is painted is the frame that becomes the
  correction's key frame.
* a stroke becomes a polygon: the polyline offset by ±radius (a click becomes
  a 12-point ring) — the exact shape `JobRequest.correction_*` rasterises.
* brush size 4–160 px of the clip, Undo/Clear, and strokes are dropped when a
  matte lands (fresh run or correction).
* the correction re-uses the stored job document (`ctx.job`), so the source
  layer no longer has to stay selected while painting.

Verified without a mouse:

| check | result |
| --- | --- |
| polygon math (node) | band height exactly 2r, 12-point ring for a click, diagonal and degenerate strokes correct |
| the document the panel sends, through the real adapter | `print-commands` accepted it → `file.replaceFootage` |
| a full engine correction on the real clip | alpha in the painted **add** box 0.000 → **0.981**, in the **remove** box 0.604 → **0.083**, untouched region 1.000 → 1.000 |

Panel `v0.5.0`; 94 tests.

### v0.5.1 — the strokes are visible while painting

Without an overlay the brush is unusable: you cannot see what you are about to
apply. This engine paints a control's **`onDraw` list on top of its image**
when the handler calls `drawOSControl()` — without it the handler *replaces*
the control's paint (`custom && !os` in the ScriptUI view) and the frame goes
blank — so the preview image now has an `onDraw` that fills every stroke's
polygon in control coordinates: **add in green, remove in red, at 0.38
alpha**, including the stroke being dragged, so the ink appears as the mouse
moves.

Verified headlessly, with the same handler shape the panel uses:

```
draw = [ {op:"os"},
         {op:"fill", color:[0.16,0.84,0.36,0.38]},   <- add
         {op:"fill", color:[0.94,0.26,0.22,0.38]} ]  <- remove
```

the scaled band lands where the stroke is (a 100 px stroke at radius 12 in a
300×169 control draws at y 13.8–17.5), and clearing the strokes leaves only
`{op:"os"}` — the picture survives with no ink.

Panel `v0.5.1`; 94 tests.

### v0.6.0 — red means delete, and the folder is not an MVP any more

**Remove force.** A red stroke only trimmed the *seed* frame; re-propagation
crept back over a large object, and in the field the same arm had to be painted
six times. Two things now happen:

* `Correction.remove_force` (1-5, default 3) grows the painted region by
  `width * 0.25% * (force - 1)` — 19 px at 1920 for force 5 — and the grown
  region is what the engine receives as its seed trim;
* after the render, `erase_removed()` stamps that same region out of **every**
  finished frame, but only where the matte is still opaque: an object that has
  moved away leaves the background under it alone, and already-transparent
  pixels are not erased twice.

Measured on the real clip (49 frames, a 140x80 box painted at force 5):
`531,625` opaque pixels inside the box before -> **0** after, on all 49
frames; an untouched control area moved by 0.05% (re-track noise only).

**Folder.** `extensions/effectcraft-mvp` -> `extensions/effectcraft` — it is
not an MVP any more. Every path reference was updated, the installers were
re-run so `start-adapter.vbs` and `bridge.path.json` point at the new location,
and the adapter now runs from it.

**Panel polish**, because a brush that cannot be aimed at is not usable:

* a 1-px accent rule under the brand (ScriptUI has no borders; a one-line
  `onDraw` rect draws one);
* panels named **Result / Brush / Run** instead of Render / Brush / Job;
* the armed tool is shown by disabling its own button — the closest thing to the
  segmented control the AE panel has;
* a **Remove force** slider beside the brush size, with the same "3 / 5" readout;
* hints that say what to do ("Drag on the frame. + keeps pixels, − deletes
  them; hold Shift for the other. Armed: + keep.") and, once there are
  strokes, "2 keep · 1 delete — Apply fix & re-run, or keep painting."

Panel `v0.6.0`; 104 tests.
