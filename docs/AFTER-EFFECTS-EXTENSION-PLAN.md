# Better Brush for After Effects — integration plan

> **Status: v1 implemented** (Windows, AE 2022+). Shipping code lives in
> [`after-effects/`](../after-effects/README.md) (the CEP panel) and
> [`src/engine/`](../src/engine) (the headless engine). The rest of this
> document is the original plan, kept for rationale; §14 records the decisions
> and the deviations that were made while building it.
>
> **Note (1.0.0):** the Lite/Pro split and the Pro licensing gate described in
> §8 and §14 below were removed. Better Brush is now free and fully unlocked,
> and the source is published under FSL-1.1-MIT.

## 0 · The one-line pitch

A dockable After Effects panel where you select a clip on the timeline, click
**Remove Background**, and Better Brush's SAM 2 pipeline runs behind the scenes
and gives the result back as a new alpha layer — or replaces the layer in place.
A **Refine** button opens a small brush/point UI for corrections.

```
 ┌──────────── After Effects ────────────┐        ┌──────── Better Brush engine ────────┐
 │  selected layer ─┐                    │        │  torch + SAM 2 + matting (warm)     │
 │                  ▼                    │  HTTP  │                                     │
 │  [ BB panel ] ──ExtendScript/UXP──────┼───────►│  auto-subject → segment → track →   │
 │      ▲                                │  /WS   │  postprocess → soft matte → export  │
 │      └── progress / preview ◄─────────┼────────│                                     │
 └───────────────────────────────────────┘        └─────────────────────────────────────┘
```

The heavy AI work stays **out of process** — SAM 2 + PyTorch cannot live inside
After Effects safely. The panel is a thin client; the engine is the product.

---

## 1 · Scope

### In scope (v1)

- Background removal for a **selected timeline layer**.
- Two modes: **Replace layer** and **Add as new layer** (alpha).
- **One-click auto** (no prompt) using an automatic-subject heuristic.
- **Assisted refine**: brush add/remove + foreground/background points, per
  frame or keyframed, then re-run propagation.
- Anime/AMV preset (cel hold + line lock + tight QC) applied automatically.
- Progress, cancel, and a clear post-run QC summary (never silent).

### Explicitly out of scope (v1)

- AI object/logo removal (the `plates`/`removal` stack).
- Face-parts splitting, multi-object comps with per-object layers.
- 8K, audio recoding, colour-managed (ACES/linear) pipelines.
- Premiere / Resolve / Final Cut panels (later, same engine).

### Non-goals

- A native C++ AEGP plugin that loads torch. (Crash surface + ABI + sandbox.)
- Rewriting the desktop app. The engine is *shared*, not forked.

---

## 2 · Why a sidecar and not a "real" plugin

Three options were considered:

| Option | How it works | Verdict |
|---|---|---|
| **A · Local sidecar service** | Panel talks to a long-running local engine over localhost. Models stay warm. | **Chosen.** Best UX: first job pays model load, later jobs are fast. |
| **B · CLI subprocess per job** | Panel spawns a headless CLI per request. | Keep as **fallback / CI path**. Cold-starts torch every job (10–40 s), progress only via stdout. |
| **C · Project round-trip** | Panel writes a temp clip + `.bbproj`; user opens the desktop app, exports, panel re-imports. | Keep as the **power-user escape hatch** ("Open in Better Brush"). |
| **D · In-process AEGP + torch** | Native plugin loads CUDA/torch. | **Rejected.** AE would own a fragile GPU runtime; one crash takes the app with it. |

The engine is one process the panel manages: started on demand, idle-timeout
shut down, one instance per user (port lockfile), token-authenticated on
`127.0.0.1` only.

---

## 3 · Component breakdown

### 3.1 Engine core (headless, GUI-free) — new `src/engine/`

Today the orchestration lives in `src/ui/main_window.py` and the workers in
`src/processing/inference_worker.py` are `QThread` + `Signal` (GUI-coupled).
`src/export/render_queue.py` already proves the pattern we want: **stdlib-only
imports, heavy deps loaded lazily, no Qt**.

Plan: extract a Qt-free pipeline module with a narrow, testable API:

```
src/engine/
  pipeline.py     analyse → segment → propagate → postprocess → matte → export
  auto_subject.py no-prompt subject selection (new; see §4)
  qc_gate.py      run anime_qc / matte_qc and decide "auto-ok" vs "review"
  schemas.py      request/response/job dataclasses (stdlib only)
  server.py       localhost HTTP + WebSocket JSON-RPC host
  runner.py       job queue, cancellation, progress fan-out
```

Reuse existing, already-good modules unchanged: `video_reader`, `mask_selection`,
`temporal_smooth`, `edge_finish`, `matting`, `anime_qc`, `cel_hold`, `line_prior`,
`letterbox`, `exporter`, and the `profiler` for model selection. The extraction
is mechanical: move the *sequence* of calls out of the window into `pipeline.py`,
leaving the window as one caller among several.

### 3.2 AE panel — new `extensions/after-effects/`

```
extensions/after-effects/
  manifest.xml            (CEP)  and/or  manifest.json  (UXP)
  index.html  panel.css  panel.js
  host.jsx                ExtendScript: layer selection, render, replace/import
  icons/
```

Two panel technologies exist; the decision drives everything downstream:

| | **CEP** (ExtendScript + HTML) | **UXP** (modern JS) |
|---|---|---|
| AE support | CC 2018 → current | AE 2024+ (newer, still maturing) |
| Language | ES3-era ExtendScript | modern JS (no Node) |
| Packaging | folder / `.zxp` | `.ccx` |
| Scripting reach | full `app.project` API | same API, different binding |
| Risk | deprecated long-term | thin host API on some AE versions |

**Decision (locked):** target **After Effects 2024+** and build **UXP** only.
The host script stays small and API-shaped so a **CEP fallback** for older AE
(2020–2023) can wrap the same logic later if demand appears — but it is **not**
in v1 scope. UXP also fixes packaging (`.ccx`; see §10).

### 3.3 Diagnostics

A "Copy diagnostics" button that dumps panel version, engine version, AE
version, GPU, model tier, and last error into the clipboard. **No telemetry** —
consistent with the app's offline stance.

---

## 4 · The hard part: one-click means *auto-subject*

Background removal with no prompt has to decide **what the subject is**. Plan a
score-and-pick heuristic in `src/engine/auto_subject.py`:

1. **Propose** candidates: SAM 2 automatic mask generation (grid of points) at
   reduced resolution; also seed from the existing face detector when a face is
   found (`face_parts` already exists).
2. **Score** each candidate:
   - area (reject near-full-frame background and dust),
   - centrality (distance of centroid to frame centre),
   - **border penalty** (masks touching all four edges are usually background),
   - colour/edge coherence with the frame's line art (`line_prior`),
   - temporal stability (does it hold across sampled frames?).
3. **Pick** the winner; if a face seed exists, prefer the candidate containing
   it (anime protagonists are usually face-anchored).
4. **Confidence gate**: if the margin is thin or QC flags flicker, return
   `state = "needs_review"` instead of pretending. This keeps the product rule:
   *never change a shot silently.*

The output is a seed mask + a small confidence object, which the normal pipeline
(segment → propagate → postprocess) consumes exactly like a user prompt would.

---

## 5 · Job API (engine websocket protocol)

All messages are JSON. The panel opens one WebSocket to `127.0.0.1:<port>` and
authenticates with the token from the lockfile.

**Request — `submit_job`**
```json
{
  "id": "uuid",
  "input":  { "path": "C:/temp/bb/comp0001/layer_003.mov",
              "frame_in": 100, "frame_out": 220,
              "handle_before": 8, "handle_after": 8,
              "fps": 23.976, "width": 1920, "height": 1080 },
  "preset": "anime",                // anime | human | object | auto
  "quality": "balanced",            // fast(=proxy) | balanced | best
  "mode": "auto",                   // auto | refine
  "seed":  null,                    // optional mask/points from the panel
  "output": { "format": "prores4444",   // prores4444 | png_sequence | exr
              "premultiplied": false,   // straight alpha for AE by default
              "color_space": "sRGB",
              "background": null }      // optional solid fill instead of alpha
}
```

**Events (server → panel)**
```json
{"id":"uuid","type":"progress","stage":"propagate","current":42,"total":120,"eta_s":31}
{"id":"uuid","type":"preview","frame":118,"png_b64":"..."}   // throttled
{"id":"uuid","type":"done","result":{
    "output_path":"C:/temp/bb/comp0001/bb_out.mov",
    "frames":121, "fps":23.976,
    "qc":{"state":"ok","flicker":0.03,"edge_support":0.91},
    "confidence":0.94 }}
{"id":"uuid","type":"error","message":"CUDA out of memory","hint":"try quality=fast"}
```

**Other verbs:** `capabilities` (models, license state, max resolution),
`cancel`, `job_status`, `open_in_gui` (launch the desktop app on this clip),
`shutdown`.

---

## 6 · Timeline integration (the AE side)

### 6.1 Reading the selection

`host.jsx` reads the active comp and selected layers:
`app.project.activeItem.selectedLayers`, taking `startTime`, `inPoint`,
`outPoint`, `source`, and the source file path when the layer is straight
footage.

### 6.2 Getting frames out of AE

- **Straight footage layer, no time effects** → send the source file path +
  frame range directly. Cheapest, lossless enough.
- **Precomp / effects / time-remap / speed change** → AE must render it. Plan
  uses the render queue with a temporary **ProRes 422 HQ (or PNG sequence)**
  output template, written to a temp folder, then foreground/background send
  that file. Document the frame-handle maths carefully:
  - request `handle_before`/`handle_after` extra frames so tracking doesn't
    pop at the in/out;
  - map AE comp time ↔ source frames through `timeRemap`/`stretch`;
  - reject or warn on **time remap with hold keyframes** in v1.

### 6.3 Getting the result back in

- **Replace layer**: `layer.replaceSource(newFootage, false)` then preserve the
  layer's in/out, transforms, and effects; fix the source in/out to the rendered
  range. Keep the old layer as a disabled copy for one-click undo.
- **Add as new layer**: `comp.layers.add(newFootage)`, place directly above the
  source, set the same in/out, optional solo.

**Default (locked): Replace.** It is what "remove the background of *this clip*"
literally means. **Add as layer** stays one toggle away in the panel, and the
chosen mode is **persisted as a user setting** (alongside preset and quality) so
someone who prefers non-destructive compositing only picks it once.
- Import the output into the project bin under a folder **"Better Brush"**,
  named `<layer>_bb_v1`.
- **Alpha contract**: engine writes **straight (unpremultiplied)** alpha; AE
  imports MOV/PNG as straight by default. If the user's project interprets
  differently, offer a "premultiply toggle" and document the failure signature
  (dark/light fringing).
- **Colour contract v1**: sRGB / Rec.709, 8- or 16-bit. If the AE project's
  working space is ACES/linear, **warn and offer conversion** rather than
  guessing.
- **Frame-rate/PAR**: match comp fps and pixel aspect; warn on mismatch.

---

## 7 · The panel UI

Keep it small. One dock, four states.

```
 ┌─ Better Brush ───────────────────────────┐
 │ Source   Comp 1 · Layer "girl" · 100-220 │
 │ Preset   [ Anime ▾ ]   Quality [ Balanced ▾ ]
 │ Mode     (•) Replace layer  ( ) Add layer │
 │                                          │
 │            [  Remove Background  ]        │
 │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  propagating 42/120│
 │                                       [x] │
 │  after: ✓ QC ok · flicker 0.03 · [Refine] │
 └──────────────────────────────────────────┘
```

- **Idle**: source summary, preset, quality, replace/add toggle, primary button.
- **Running**: stage text, progress bar, ETA, cancel.
- **Needs review**: shows the auto-subject mask over one frame with a **Refine**
  button and a one-line reason ("auto-pick low confidence" / "flicker 0.11").
- **Refine mode**: frame scrubber, brush add/remove, FG/BG points,
  "keyframe this frame", "re-run from here". Mirrors the desktop app's core
  interactions, scaled down.
- Error states: no selection, unsupported colour space, engine not running,
  OOM (suggest `quality = fast`), license missing.

Reuse the existing dark theme tokens (`src/ui/theme.py`, docs-site CSS) so the
panel feels like Better Brush, not a generic Adobe panel.

---

## 8 · Licensing & editions (historical)

> **Superseded in 1.0.0.** The original plan below gated the panel behind a
> Pro licence. That split no longer exists: Better Brush is free, every feature
> is unlocked, and the source is published under FSL-1.1-MIT. The section is
> kept only to record the original design.

The original design reused the offline licensing module (since removed) and
made the AE extension a Pro feature, with this split:

| Capability | Lite (free) | Pro |
|---|---|---|
| Single-object background removal | ✅ (watermark optional) | ✅ |
| Multi-object / >1080p / no watermark | ❌ | ✅ |
| Batch all selected layers | ❌ | ✅ |
| Refine UI | ✅ | ✅ |

The panel asked the engine `capabilities` at connect, with the engine as the
single source of truth for licence state. The shipping 1.0.0 panel has no
licence check at all; it asks `/health` for readiness.

---

## 9 · Performance & hardware

- **Warm models**: model load (10–40 s) once per engine session; jobs after that
  are fast.
- **8 GB VRAM** (the user's RTX 5060 Laptop): default `quality=balanced` at
  1080p may need tiling; offer `quality=fast` = 720p proxy upscale. Reuse
  `profiler` tier selection + a render-time estimate (see roadmap idea #12).
- **GPU contention with AE**: both want CUDA. Mitigations: VRAM cap setting,
  `quality=fast` advisory when AE is actively rendering, and a "pause engine
  while AE renders" note in v1 docs.
- **Long clips**: stream frame-by-frame; chunk propagation (already how the app
  works) so memory stays flat.

---

## 10 · Packaging & distribution

- **Windows first** (matches the desktop app's primary platform); macOS later.
- Engine ships **with the existing Better Brush install**; the panel discovers it
  via a registry key / known path / lockfile — no second installer.
- **UXP**: package a `.ccx` (self-distributed, avoids Adobe Exchange review).
  **CEP fallback**: user-scope extension folder + `PlayerDebugMode`.
- **Signing**: Windows SmartScreen warning until signed; macOS notarisation
  later.
- **Version handshake**: panel and engine compare API version on connect; refuse
  politely with "update Better Brush" if incompatible.
- **Frame-handle + colour contracts** documented for each AE version tested.

---

## 11 · Milestones

Each milestone is independently demoable.

| # | Milestone | Est. | Acceptance |
|---|---|---|---|
| **M0** | Spike: `better-brush serve` + hello-world **UXP** panel that lists the selected layer and round-trips one frame | 1–2 wk | Panel shows the selected layer's name/range; engine echoes a PNG frame |
| **M1** | Headless engine: extract pipeline from `main_window`, `auto_subject`, PNG-alpha out, integration test on a tiny clip | 2–3 wk | `submit_job` on a test clip returns an alpha sequence; `pytest tests/engine` green, no Qt import |
| **M2** | One-click background removal end-to-end: render intermediates, progress/cancel, default **Replace** + Add-as-layer toggle, colour/alpha contract | 2–3 wk | Select 1080p layer → click → layer replaced with correct alpha; cancel works mid-job |
| **M3** | Refine UI: brush/points, keyframe a frame, re-run from frame | 2–3 wk | Paint a fix, re-run, result reflects it |
| **M4** | Anime polish: preset wiring, cel hold/line lock, QC gate → "needs review" | 1–2 wk | Low-confidence shot surfaces Review instead of silently replacing |
| **M5** | Packaging: `.ccx` (UXP) install, engine discovery, version handshake, diagnostics | 1–2 wk | Fresh machine: install app + panel, panel finds engine |
| **M6** | Hardening: AE version matrix, colour/premult edge cases, docs, optional macOS | 2–3 wk | Documented supported-AE matrix all pass |

Rough total: ~10–16 focused weeks. M0 proves the riskiest assumption (IPC +
timeline grab) fastest.

---

## 12 · Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| AE can't headlessly render arbitrary layers | Blocks "grab clip" for comp'd layers | Use render-queue temp output; document limits; v1 supports straight footage + precomps |
| AE-version support breadth | Panel won't load on AE < 2024 | Locked to 2024+ UXP; keep the host script API-shaped so a CEP fallback can be added later |
| Alpha/colour mismatch | Visible fringing or wrong gamma | Explicit straight-alpha + sRGB contract; premultiply toggle; golden-frame tests; warn on ACES |
| Auto-subject picks wrong subject | User distrust | Confidence gate + "needs review" state; one-click Refine |
| GPU contention / OOM with AE | Crashed renders | VRAM cap; `quality=fast` proxy; clear OOM hints |
| Time-remap / speed-remap frames | Misaligned output | Detect and warn; reject hold-keyframe remap in v1 |
| Engine cold start feels slow | First-run frustration | Warm engine on panel open; show "warming up" state |
| Bundled torch vs dev env drift | Load errors | Version handshake; single engine build from the app install |
| Adobe store review | Distribution delay | Self-distributed `.ccx` / CEP, no Exchange dependency |

---

## 13 · Testing strategy

- **Unit**: auto-subject scoring, schemas, colour/alpha conversion, frame-handle
  maths, capability gating.
- **Integration**: `tests/engine/` drives the headless pipeline on a tiny clip
  with a stub model — same "no GPU required" property the current suite has.
- **Golden frames**: alpha edge quality vs a baseline (flicker, edge support).
- **Panel**: mock engine server + assertions on layer-selection parsing and
  replace/add plumbing.
- **Manual AE matrix**: AE versions × straight/premult × colour spaces ×
  time-remap × proxies — recorded as the supported matrix in M6.
- Reuse the existing `pytest` harness; keep the suite headless and green.

---

## 14 · Decisions

**Locked**

1. **Platform: Windows, After Effects 2022+.** Built as a **CEP** panel, not UXP.
   Rationale in "Deviations" below.
2. **Default mode: Replace layer**, with **Add as new layer** one toggle away;
   the choice is **persisted** (panel `localStorage`).
3. **Licensing: none** — the panel asks `/health` for readiness. (A Pro gate
   existed in the original plan; it was removed in 1.0.0 when Better Brush
   became free.)
4. **Scope: characters only.** The one button removes the background behind a
   single character. Multi-object and anything ambiguous opens the full app.
5. **Cold start: acceptable.** The panel starts the engine on demand and polls
   `/health` for up to two minutes while showing "starting engine…".
6. **Panel placement: docked** in the standard AE workspace.

**Deviations from the original plan (and why)**

| Plan said | v1 does | Why |
| --- | --- | --- |
| UXP panel, AE 2024+ | **CEP**, AE 2022+ | The work needs the ExtendScript object model — `ImportOptions`, `Layer.replaceSource`, `FileSource.alphaMode`. The AE UXP API does not cover these yet, so a UXP panel could not finish the job. CEP also loads unsigned in player-debug mode, which makes beta installs trivial. |
| WebSocket job protocol (§5) | **Plain JSON over HTTP** on `127.0.0.1` | One job at a time, poll every 800 ms. Far less machinery, and `http.server` is already in the stdlib. |
| Auth token / port discovery | **Fixed port + loopback bind** | `127.0.0.1` is the security boundary; a token file added failure modes (CEP FS access) for no real gain. |
| `Refine` brush UI in the panel | **Hands off to the full app** | The panel is deliberately one button. A half-capable brush inside a 340 px CEP panel is worse than the real editor one click away. |
| Engine as `src/engine/{pipeline,qc_gate,runner}` | `src/engine/{schemas,model_setup,auto_subject,pipeline,jobs,server}__main__` | Same shape; `jobs.py` + `server.py` replaced `runner`/`qc_gate`. |

**Explicitly deferred (not in v1)**

- macOS, Premiere/Resolve/Final Cut panels.
- In-panel brush refine, preview scrubber, per-object layers.
- Multi-object auto-subject (single character only, by design).
- `.ccx` signing / Exchange distribution (installer copies into the user CEP
  folder and enables player-debug mode instead).

---

## 15 · What shipped (v1)

**New**

| Path | Purpose |
| --- | --- |
| `src/engine/__init__.py` | Version + API contract constants |
| `src/engine/schemas.py` | `JobRequest` / `JobResult` / progress / error kinds (stdlib only) |
| `src/engine/model_setup.py` | Deterministic model selection by total VRAM; `best` prefers an installed custom ("Lugnica SAM") model |
| `src/engine/auto_subject.py` | Character scoring + the "refuse when ambiguous" gate |
| `src/engine/pipeline.py` | Extract → auto-subject → propagate → stream ProRes 4444 (PNG fallback) |
| `src/engine/jobs.py` | Single-slot job registry, progress, cancel, error mapping |
| `src/engine/server.py`, `__main__.py` | `http.server` API on `127.0.0.1`; `python -m src.engine` |
| `after-effects/` | CEP panel: manifest, `index.html`, `panel.css`, `js/panel.js`, `js/host.jsx`, `install.ps1`, `README.md` |
| `tests/test_engine_*.py`, `tests/engine_helpers.py` | 51 tests (schemas, scoring, jobs, HTTP, end-to-end plumbing) |

**Modified**

- `src/models/sam2_manager.py` — added `predict_frame_multi` (one `set_image`,
  many probes) for the auto-subject grid.
- `pyproject.toml` — `better-brush-engine` console script.

**Reused as-is**
`video_reader`, `mask_selection`, `edge_finish`, `face_parts` (best-effort face
seed), `exporter` (`_resolve_ffmpeg`, `_compose_rgba`), `profiler`,
`checkpoint_manager`, `licensing`. The desktop app was **not** modified — the
engine shares code rather than forking it.

---

## 16 · Round 2 — after the first real test in After Effects

Three things came out of the first hands-on run. All three are fixed; the
reasoning is recorded here because two of them were **not** panel bugs.

### 16.1 "Invalid fine-tune model" on Balanced / Best

Root cause was one line in the profiler, and it hit the **desktop app too**:
`SAM2Profiler.recommend(force_model=ModelSize.CUSTOM)` raised
`KeyError(ModelSize.CUSTOM)`, because `_CONFIG_MAP` / `_CHECKPOINT_MAP` only
describe built-in downloads and a fine-tuned checkpoint is not one of them.
Picking "Anime AMV v1" in the app therefore did nothing (the `KeyError` died
inside a Qt slot), and in the panel it surfaced as a job error.

- `SAM2Profiler._refs_for()` now resolves `CUSTOM` from the sidecar
  (`checkpoints/custom/sam2_custom.json`) — the same source
  `CheckpointManager` already used for the weights — so every tier is
  addressable through one code path.
- **Policy change:** `resolve_model_name()` no longer upgrades `balanced` to the
  fine-tuned model. `balanced` means SAM 2.1 Base+ because that is what the
  control says; only `best` prefers the fine-tuned checkpoint. The panel now
  also *names* the resolved checkpoint (`Uses SAM 2.1 Base+`) and the finished
  card names the one that actually ran, so a label cannot drift from reality
  again.

### 16.2 "Errno 22 / Invalid argument" on Fast

`[Errno 22] Invalid argument` is what **Windows** raises when writing to a
subprocess pipe whose child has already exited — it is the symptom of ffmpeg
dying, not the cause. The old code caught that and threw ffmpeg's stderr away,
so the message named the symptom and nothing else.

Fixes, in order of usefulness:

1. The pipe-write failure path now **stops the child, reaps it, and folds
   ffmpeg's exit code and stderr into the message**. The next occurrence will
   say *why* ffmpeg refused, instead of "Invalid argument".
2. The export is validated before ffmpeg is launched: non-finite/zero fps falls
   back to 24, and a zero-sized source is refused with a plain explanation.
3. The frame size is checked against the `-s` ffmpeg was given, so a clip whose
   frames change size mid-stream fails with *"frame 42 is 1920x1088 but the movie
   was started as 1920x1080"* rather than a broken pipe.
4. An ffmpeg that dies mid-job is killed instead of being left holding the
   output file, and a cancelled job no longer leaks the child process.
5. Unexpected failures now carry the **step** they happened in
   (`… (while tracking the subject)`), which is the difference between a
   one-line report and a round trip.
6. The engine writes to `%LOCALAPPDATA%\BetterBrush\engine.log` (rotating,
   1 MB × 3). It runs under `pythonw.exe`, which has no console — until now a
   failed job left *no* trace anywhere.

`tests/test_engine_pipeline.py` covers the two diagnostic paths directly
(ffmpeg stderr surfaced; a mid-clip size change explained).

### 16.3 Panel UX

The first panel was a generic form: a `<dl>` of facts nobody needs, native
light-themed `<select>`s inside AE's dark chrome, a purple accent that matched
nothing else in the product, and no feedback beyond a text line. Rebuilt around
a single **hero state**:

- Palette taken from `src/ui/theme.py` (the desktop app's own theme), so the
  docked panel reads as the same product and sits comfortably in AE.
- Four states, each with one clear action: *select a layer* → *ready* →
  *working* (four-stage checklist, progress bar, elapsed timer, cancel) →
  *done* (key-frame preview on a checkerboard, frame count, and the model that
  ran).
- Segmented Quality / Result controls instead of dropdowns, with the resolved
  model named underneath.
- Failures get their own card, titled by error kind, and the hand-off cases
  ("no character", "several subjects") offer *Open full app*.
- It reattaches to a job that is already running, so reopening the panel never
  hides progress.

### 16.4 Found while reproducing

- **OpenCV 5** removed `CascadeClassifier` / `cv2.data.haarcascades`. The Haar
  stage of `face_parts` was throwing an `AttributeError` on every frame and
  quietly doing nothing. It is now probed once and cached; the remaining stages
  (mediapipe, skin-tone) still work, and `auto_subject` continues to function
  without a face seed. Mediapipe is not installed in the test environment, so
  in practice anime face seeding leans on the skin heuristic.
- `init_video` progress was reported as the `extract` stage, which made a
  stage-based UI jump *backwards* mid-job. It now reports `propagate`.

### 16.5 Found by rendering real results (panel preview and key frame)

Two defects only visible once a real clip had run end to end. Both were found
by looking at the panel's own output rather than at the code.

- **The preview (and the PNG-sequence fallback) wrote the wrong channels.**
  `_rgba()` builds RGBA; `cv2.imwrite` wants BGRA. The code reversed the *whole*
  array (`rgba[:, :, ::-1]`), which also moved alpha to the front, so the file's
  alpha channel held the red channel. Measured on a real result: 0% of pixels
  transparent, alpha mean 170.8. After the fix: 92% transparent, alpha mean 20.6
  (the subject's true 6–8% coverage plus feathering), with the colours right.
  A PNG-sequence result imported into AE was affected the same way. Fixed with
  `_write_rgba_png()`, which swaps only the three colour channels.
- **Auto-subject rotoscoped the *face*.** On the test clip the matte covered
  6.4% of the frame — Tanjiro's face — with the body left out. Probing the real
  candidate set explained it, and the numbers matter:
  - the **background** mask is what SAM 2 is most confident about
    (`sam_score` 0.99 against the character's 0.06), and it is the *better*
    mask on area, centring and solidity too, so it outscored the subject;
  - the **face** mask is the tidiest small mask and beat everything on score;
  - the **whole character** mask (area 0.44–0.49 on this close-up) scored third,
    and was excluded from promotion by a "character" band that stopped at 0.45.

  Three changes, all in `src/engine/auto_subject.py`:
  1. `_anchor_on_face()` — when a face is detected, only masks that contain it
     are considered. The background never does. If nothing contains it (a bad
     face hit) the full list is kept, so a wrong detector cannot refuse a clip.
  2. `_area_pref()`'s band now runs to 0.65, because a close-up character
     legitimately fills most of the frame.
  3. Nested masks count as *one* subject: `_prefer_whole()` promotes a part to
     the largest plausible mask containing it, and a candidate that sits inside
     the winner is no longer reported as a rival subject (previously the
     character's own face would have refused the job).

  Result on the same clip: coverage 0.064 → 0.543, the whole character cut out,
  and the multi-subject gate still refuses the two-subject dance clip ("More than
  one character is visible"), which is exactly the "opens the full app" path.

---

## 17 · Why this is the right next platform

- The value is already in the engine; the AE panel is a **thin** surface over it.
- `render_queue.py` proves the codebase can host a Qt-free, lazily-importing
  core — the engine extraction is a refactor, not a rewrite.
- It directly delivers roadmap items **#9 (NLE interchange)** and **#13 (Plugin
  SDK)**, and creates the shared engine that Premiere/Resolve panels can reuse.
- The one-button promise is honest only with a confidence gate — which the
  product already believes in ("never change a shot silently").

---

## 18 · Round 3 — the panel worked, then never finished

Reported after the second live test: *"I saw no error, but it takes so much time
even with my 5060 RTX, and I couldn't see the result because even after 10–15
minutes it was still working. The standalone app is way faster."*

### 18.1 The job was not slow — it was blocked, and the log was empty

The engine was still running, so it could be inspected while it was stuck.
`nvidia-smi` showed **0% GPU and 0 MiB allocated**: nothing was computing. The
job reported `propagate 0/49` — a **49-frame** clip, eight minutes in.

`py-spy dump` on the live process named the cause in one line:

```
Thread 24392 (idle)
    status_printer (tqdm\std.py:447)
    __init__ (tqdm\std.py:1101)
    load_video_frames_from_jpg_images (sam2\utils\misc.py:268)
    init_state (sam2\sam2_video_predictor.py:51)
    init_video (src\models\sam2_manager.py:604)
    _propagate (src\engine\pipeline.py:285)
```

SAM 2 draws a `tqdm` bar while it loads frames. The panel launches the engine
from After Effects with `start /b`, so the engine inherits an output pipe that
After Effects never drains; once the pipe's buffer filled, that write blocked
*forever*. The pythonw case is a red herring on its own — a direct
`pythonw -m run_job` finished in 79 s while the same job through the engine hung,
and the only difference was an unread pipe.

Two things made it undiagnosable:

- **`--quiet` was applied to the root logger**, so the file handler never saw
  INFO either. The engine log for that run contained no job at all. `--quiet`
  now silences only the console; the file always keeps INFO.
- **Nothing was written about the 20 s model load.** The panel sat on "Reading
  the source" with the bar at 0% for the whole cold start.

Fixes, in `src/engine/server.py`: `_detach_stdio()` points a non-console
stdout/stderr at the null device (and `dup2`s fds 1 and 2, so a C library cannot
block either), called *before* logging is configured because `StreamHandler`
captures `sys.stderr` when it is built. Running the engine in a terminal still
prints — the redirect only happens when the streams are not a console.

### 18.2 It was genuinely slower than it needed to be

Measured through the engine itself — piped, `--quiet`, `fast` (SAM 2.1 Small),
`TanjiroTest.mp4`, 120 frames at 1920×1080:

| Phase | First job | Same shot again |
| --- | --- | --- |
| Loading the model | 3.9 s | **0.0 s** (reused) |
| Reading the source | 8.4 s | 9.1 s |
| Finding the character | 7.2 s | 6.8 s |
| Tracking through the clip | 28.6 s | 29.3 s |
| Writing the alpha movie | 10.9 s | 8.4 s |
| **Total** | **63.5 s** | **53.9 s** |

Two changes:

- **The engine built a fresh `SAM2Manager` per job**, so every run paid to read
  the checkpoint and to build a new CUDA context. `_manager_for()` keeps one
  loaded model (matching checkpoint, device and offload flags) and the engine
  serves one job at a time, so a second run on the same shot skips that wait.
  The weights are also loaded **eagerly** in that step: previously SAM 2 loaded
  them lazily inside the first analysis call, which is why the old log read
  `phase model: 0.0s` and then charged the wait to "finding the character".
- **`Tiny` (SAM 2.1 Tiny) joins the quality tiers** as the cheapest option. The
  checkpoint was already on disk, so it needs no download. It is honestly a
  draft tier — on the same clip it cuts out the character with visible gaps and
  light speckle (coverage 0.29 against Small's 0.54) — and both the button's
  tooltip and the README say so.

Tracking, not the model, is the bulk of a run: ~29 s for 120 frames is SAM 2
propagating, and no tier changes that.

### 18.3 Progress that matches the work

- A new **`load`** stage is emitted before the model is built, and the panel
  lists it first, so the cold start is visible instead of looking frozen.
- A stage with no frame count sweeps instead of sitting at 0%.
- Each phase logs its own duration (`phase model: 3.9s`), so the next "why is
  this slow" answer takes one look at the log.

One panel bug fell out of adding the stage: `beginRun()` opened the checklist on
`extract`, and the monotonic stage index then marked `load` as already done.
Caught by rendering the panel and reading the DOM, not by the test suite.

---

## 19 · Round 4 — matting and a correction brush inside the panel

The panel was signed off as "works perfectly", and the two asks were: refine the
edges automatically, and let the user fix a wrong mask without leaving AE.

### 19.1 Automatic matting

The desktop app has always had a soft-alpha pass (`MattingRunner` /
MatAnyone 2, `src/processing/matting.py`) that turns the tracked outline into a
soft matte — hair, motion blur, partial coverage. The engine never called it;
its `_alpha_for` only ran `soft_coverage` on the binary mask.

- `JobRequest.matting` (default **off** on the wire, **on** in the panel) runs
  `MattingRunner.matte` after propagation, with the propagated mask as both the
  key-frame prompt and the per-frame clamp, so the learned matte can only soften
  the edge the animator agreed with.
- A new **`matte`** stage sits between `propagate` and `export` — the stages are
  still emitted in checklist order, and the panel grew a sixth row.
- Matting must never sink a job: a missing download or an out-of-memory card is
  caught, logged, and the run ships the outline instead (`result.matting=false`).
  The runner is unloaded in a `finally` — two networks on one card is the thing
  an 8 GB GPU cannot afford.
- `/health?full=1` reports `capabilities.matting`, so the switch greys out when
  the weights are absent instead of failing a run.

### 19.2 The correction brush

The self-contained version: correct on the key frame, re-track the clip.

- The engine writes `seed.png` + `seed.json` beside the alpha movie (the
  key-frame mask, its index and the subject stats). They cost nothing next to
  the movie and are what make a correction cheap.
- A correction is just another `POST /job`, carrying `resume_dir` plus
  `correction_add` / `correction_remove` (base64 PNG masks). No new route, no
  new state: `_resume_seed` loads the stored seed, ORs the add strokes, subtracts
  the remove strokes, and skips the subject search entirely.
- The panel overlays a canvas on the key-frame preview (pinned to the footage
  aspect ratio so pixels line up), paints with a green/red brush, and rasterises
  the strokes into the two masks on Apply.
- Applying targets the **result** layer, captured from `bbApplyResult`'s new
  `layerId`/`compId`, so corrections replace the current cut-out whichever
  `Result` mode produced it. The original footage is frozen in `state.source`
  before the first run, because after a replace `state.selection` describes the
  cut-out, not the source.

Verified end-to-end in a QtWebEngine probe that stubs the host and the HTTP
surface: at a 250 px and 300 px dock the page never exceeds the viewport
(`bodyScrollW == innerWidth`), the brush tools have no internal overflow, six
stage rows are present, and a painted stroke produces exactly one correction
`POST` with the right fields. Engine-side, 37 pipeline/schema tests pass,
including a correction that provably never calls `find_subject` again, and
matting that both applies (flat 0.5 alpha reaches the export) and falls back.

---

## 20 · Round 5 — making the brush actually work

Reported straight after Round 4 shipped: "I tried several times adding her hair
to the mask but it just doesn't work." The screenshot showed exactly that — the
tracked head with the dark patches of hair punched out of the matte.

Two separate defects were behind it, plus one automatic-quality gap.

### 20.1 Why the brush did nothing

- **The strokes were only a prompt.** The correction went back through SAM 2,
  which re-derived the mask from the painted seed. A patch it did not believe
  belonged to the subject — a disconnected clump of hair — was dropped again, so
  the result looked identical. The desktop app never had this problem because its
  canvas *is* the mask (`main_window._masks[idx] = mask`); SAM is only ever a
  convenience there.
- **Only the key frame could be painted.** The panel showed one frame, chosen by
  the engine, and any correction had to be made on it. If the bad frame was
  somewhere else in the clip, no amount of painting on the key frame would help.

### 20.2 What changed

- **The painted frame is authoritative.** After propagation the correction frame
  is set to exactly the corrected seed, so a patch SAM 2 disbelieves still
  survives where the user put it. Propagation carries it to the rest of the clip.
- **Any frame can be painted.** The masks are persisted once per job
  (`masks.npz`, bit-packed and compressed — a few MB for a 120-frame 1080p clip,
  never a float image per frame) and a new `GET /frame?dir=&index=` renders that
  frame's cut-out as a PNG. The panel grew a Frame scrubber that pulls the frame
  on demand, debounced, and the correction carries `correction_frame`.
- **Moving the slider clears the strokes**, because a stroke belongs to the frame
  it was painted on; the alternative is a stroke that silently lands somewhere
  the user never saw.
- A correction that would erase the subject entirely is refused with a readable
  message instead of producing an empty matte.

### 20.3 Automatic quality: close the gaps

`fill_mask_holes` (from the app's own post-processing) now runs on the seed,
before SAM is prompted, and again on every tracked frame before matting and
export. The app's default cap (~0.2 % of the frame) is smaller than the hair
defect, so the engine uses **1 %** — big enough for a clump of hair, still far
short of a genuine window behind the subject, and it still only fills regions
that are *enclosed* (anything touching the frame border is outside).

Order matters: the seed's gaps are closed *before* the user's strokes land, so a
"remove" that deliberately opens a gap is never quietly re-filled under them.

### 20.4 Speed

The frames are now kept beside the result, which does two things: the brush can
render any frame without touching the source, and a correction skips the
re-extraction pass entirely (about 9 s of the 120-frame test clip). Packs longer
than 1 800 frames drop the frames again and fall back to decoding the source on
demand, so disk does not grow without limit.

### 20.5 Open in app, on the clip

`bbOpenFullApp` now takes the selected file and hands it to the app's command
line; `core/app.py` accepts any supported media path as well as a `.bbproj`, so
the app opens straight on the footage instead of the welcome launcher. The panel
passes the *original* source, not the current selection — after a replace the
selection is the cut-out.

### 20.6 Verified

Engine: 47 pipeline/schema tests, including a correction that provably never
calls `find_subject` again, a correction whose painted block provably survives on
the frame it was painted on, hole filling that closes an enclosed gap and leaves
the outside alone, and `render_frame` returning a true-colour PNG of the right
size. Panel: the same QtWebEngine probe at 250 px and 300 px, extended to cover
the frame scrubber and the `/frame` fetch.

## 21 · Round 6 — the add brush, the flashing console, and the crash on close

Three reports from the first real session with Round 5: the delete brush works
but the add brush does not; a `cmd` window appears just before the rotoscope
finishes; and opening the full app, then closing it, takes After Effects down.

### 21.1 Why "add" looked like it did nothing

The clamp that keeps the learned matte inside the segmentation mask was
**one-sided**: `alpha = alpha × mask`. It can only ever *remove* coverage.

That is exactly right for the delete brush — the mask says background, so the
matte is forced to zero and the region disappears whatever the network thinks.
It is exactly wrong for the add brush. Painting hair in puts it *inside* the
mask, and the region is then handed back to MatAnyone 2, which reads it as
background and returns roughly zero there. `0 × 1 = 0`: the painted hair
composited as fully transparent and the correction looked like a no-op.

`_clamp_alpha_to_mask` now takes `solid_interior`. The mask owns its interior —
everything more than a short erosion inside it is opaque — and the network owns
only the boundary band, which is the part a soft matte is actually for. Thin
structures (hair strands, fingertips) fall outside the eroded core, so nothing
that needs real softness loses it.

The engine turns this on (`MattingSettings(clamp_solid_interior=True)`); the app
keeps the old one-sided behaviour, so its loose-mask workflow is untouched.

### 21.2 Automatic quality: ask for the hair

A point prompt segments the part it happens to land on, and the character's hair
is the usual casualty — it is a large, obvious region that nothing ever asked
SAM 2 for. `predict_frame_from_mask` grew `pad_ratio` and `grow`, and
`auto_subject.refine_mask` uses them on the winning frame: pad the box out
around what was found and keep the **largest** object inside it that still
covers the mask (IoU ranking would just hand back the incomplete input again).

Refinement may only grow the subject. The candidate is rejected if it is
implausible as a whole character, if it has wandered off the mask, or if it came
back smaller. Every failure mode returns `None` and the mask we already had is
kept, so this can never make a job worse than it was.

### 21.3 The console window mid-rotoscope

The engine runs under `pythonw.exe`, so it has no console to hand down — and on
Windows a console-subsystem child then allocates one of its own. The child in
question is ffmpeg, spawned for the export that runs right as the rotoscope
finishes. `src/utils/proc.py` exposes `no_window()` (`CREATE_NO_WINDOW`) and
every `Popen`/`run` in the engine, the exporter and the library recode path now
spreads it in.

### 21.4 The crash after closing the app

`bbOpenFullApp` launched `python.exe` — a console binary — from inside After
Effects' own process tree. It now launches `pythonw.exe`, detached, exactly like
the engine: no console at all, so there is nothing to close underneath AE.

The second half is `register_file_assoc`. It re-registered on every launch and
called `SHChangeNotify(SHCNE_ASSOCCHANGED)` each time, and that is a broadcast
to **every top-level window on the desktop** — After Effects included. Opening
Better Brush beside AE therefore poked AE's shell integration on every launch.
Registration now compares the values first and only writes (and only notifies)
when the association actually changed, which after the first run it never does.

### 21.5 Verified

- Engine: `_clamp_alpha_to_mask` is proven one-sided by default and
  interior-solid when asked, with the boundary band still the matte's;
  `_rank_candidates` proven to grow to the containing mask when asked and to keep
  its old IoU answer when not; `refine_mask` proven to grow, and to refuse a
  candidate that wandered off, shrank, covers the whole frame, or came back
  unchanged, plus every "no helper / old signature / raise" fallback.
- Panel: the QtWebEngine probe at 250 px still has no overflow, still paints and
  corrects frame 42, and now reports `bbOpenFullApp("…/pythonw.exe", …)`.

## 22 · Round 7 — zoom, anime matting, and doing less work

Round 6 works. This round is the seven-item list that followed it: two of them
are features in the panel, one is a quality pass on the matte, one is speed, and
three are designs that are written down rather than built.

### 22.1 Zooming the brush surface

A docked panel is about 200 px wide and the source is 1920: at fit, one source
pixel is a tenth of a screen pixel, so a hair correction is guesswork. The brush
stage now has zoom and pan.

The implementation is deliberately the smallest one that works. The image and
the stroke canvas are wrapped in `#brushLayer`, and zoom is a single CSS
transform on that layer — `translate(-x0·z, -y0·z) scale(z)`, with the visible
window given as the top-left corner `(x0, y0)` and a span of `1/z` in normalised
source coordinates. Three things fall out of that choice for free:

- The image and the strokes can never drift apart, because one transform moves
  both.
- `canvasPoint()` needs no change at all. It maps through
  `getBoundingClientRect()`, and a zoomed canvas simply reports a proportionally
  bigger box, so the arithmetic lands back on the same source pixel.
- Zoom is therefore also a **brush-size** control: the width in source pixels is
  `size × (canvas.width / rect.width)`, so at 8× the same slider paints an
  eighth as wide a stroke on the artwork. That is the whole point of zooming in
  for detail.

The brush footprint is drawn as a DOM circle rather than on the canvas — it is
`size` screen pixels by definition, so it costs nothing per mouse move and it
stays exactly round at every zoom. The wheel zooms about the cursor, the middle
button and space-drag pan, `Fit`/`0` returns to the whole frame, and panning is
clamped so the window cannot leave the artwork. Zoom deliberately survives a
frame change (you zoom on the ear, scrub, and stay on the ear) and resets on a
new result.

### 22.2 Matting that suits drawn animation

MatAnyone 2 is trained on photography, where subject and background mix
gradually. Anime is the opposite: flat colour regions bounded by a hard line. The
network's answer in the middle of a flat region is not detail, it is a ghost of
whatever was behind the character, and it composites as a milky patch.

`MattingSettings.flat_snap` measures how flat the *source* is (a Sobel gradient
at reduced resolution, so it costs a couple of milliseconds) and, wherever the
image is flat **and inside the animator's mask**, hands the pixel back to the
mask instead of the network. Two consequences are worth spelling out:

- It is restricted to the inside of the mask. Outside it the network is left
  alone, so the hair it finds beyond the painted outline still survives — the
  one behaviour that made the app's loose-mask workflow worth keeping.
- Real edges are untouched, because a drawn line is detail by construction. The
  boundary band keeps the matte's soft ramp.

The engine turns this on along with the interior clamp. The app turns it on for
`ContentProfile.ANIME` only, so live-action keeps the network's own answer.

The engine also stopped shipping the halo: export composed the original RGB
against the alpha ramp, so every partially covered pixel carried background
colour. It now runs `edge_finish.finish_edges` (decontamination included) on the
outline path and `decontaminate_edges` on the matted path.

### 22.3 Doing less work

Three changes, none of which touch quality:

- **Extraction** encodes each JPEG on a thread while the next frame is being
  decoded. `cv2.imwrite` releases the GIL, so this is free parallelism, and
  extraction is decode-then-encode — the encode used to sit on the critical
  path in full.
- **Export** prefetches frames the same way (`_decoded_frames`). The write to
  the ffmpeg pipe stays strictly in order; only the decode overlaps. Four frames
  of RAM buys the decode off the critical path for the whole stage.
- **The brush surface** no longer re-encodes a frame it has already rendered.
  `/frame` results are cached by `(mode, work dir, index, masks.npz mtime)`,
  which is self-invalidating: a correction rewrites `masks.npz`, so everything
  rendered before it becomes unreachable and is evicted. PNG compression is
  dropped to 1 on this path — the image only ever crosses the loopback, so
  trading a bigger buffer for a ~3× faster encode is a straight win.

On the panel side, the 1.5 s selection poll no longer re-renders when the
selection JSON has not changed. That was re-laying out the brush tools and
re-asserting the preview image while the user was painting on it.

### 22.4 A friendlier panel

Alongside zoom: step buttons on either side of the frame slider, `Undo` beside
`Clear`, Apply/Undo/Clear disabled until something is painted, and an
**Original** toggle that swaps the cut-out for the untouched source frame (a new
`&original=1` on `/frame`, served as JPEG) so an edge can be judged against what
was actually behind it. Keyboard: `A`/`R` tool, `[`/`]` brush size, `,`/`.`
frame, `+`/`-`/`0` zoom, `Ctrl+Z` undo, `Ctrl+Enter` apply, `C` compare.

Two real bugs went with it. The preview image was re-assigned on every render, so
a scrubbed frame snapped back to the key frame within 1.5 s; the panel now tracks
what the image is showing (`preview` / `frame` / `original`) and only `render()`
writes it in `preview`. And a slow `/frame` response could land after the user had
moved on, so responses are sequence-numbered and superseded ones are discarded.

### 22.5 Verified

- Engine: flat snapping proven to make a flat in-mask region opaque, to leave a
  drawn edge to the matte, to be unable to add coverage outside the mask, and to
  be a no-op without one; `_flat_weight` proven flat-vs-line; the frame cache
  proven to serve a repeat without re-reading the source **and** to invalidate
  when the mask set is rewritten; `render_source_frame` proven to return the
  untouched frame; `_decoded_frames` proven to stream in order.
- Panel: the QtWebEngine probe at 250 px drives the whole flow and reports no
  overflow (body 250, brush rows 187 = client width), zoom clicks `2×`, wheel
  zoom `1.4×`, a middle-drag pan moving the transform from `-26.152%` to the
  `-36.268%` clamp **without** painting a stroke, a 48 px ring, `]` growing the
  brush 48 → 60, `r`/`a` switching tool, `.` stepping 42 → 44, undo disabling
  Apply again, a compare fetch of `index=44&original=1`, and a correction posted
  at `correction_frame: 44`.

### 22.6 Planned, not built

Three design documents were written for the work that is bigger than a round:

- `docs/AMD-WINDOWS-SUPPORT-PLAN.md` — what breaks on an AMD-only machine today
  (the DirectML path hands `torch` a `"directml:0"` string it cannot resolve, and
  reports a 0 MB adapter as a usable GPU), and a staged plan for a real device
  capability layer.
- `docs/SAM2-ANIME-FINETUNE-PLAN.md` — fine-tuning SAM 2.1 on anime, plugging
  into the existing sidecar checkpoint path and the "Best" tier.
- `docs/ANIME-ROTOSCOPE-TOOLBOX.md` — the candidate techniques, including the
  observation that the engine still does not run the cel-hold, line-prior and
  temporal-smoothing passes the desktop app already has.

## 23 · Round 8 — scroll, colour, quieten, and port to Premiere Pro

The user's list, in order: make scrolling on the image possible and colour the
brush by mode; make the panel clearer and more minimalistic; and make a new
folder that ports the extension to Adobe Premiere Pro.

### 23.1 The wheel scrolls

Round 7's wheel zoomed. It was the wrong default: a trackpad's two-finger gesture
is a scroll, and a mouse wheel is a scroll in every editor the user has ever
used. The wheel now **scrolls** the brush surface and `Ctrl`+wheel (or `⌘`)
**zooms about the pointer**; `+` / `-` / `Fit` / `0` are unchanged as the stepped
zoom.

Scrolling is expressed in the same normalised-window terms as panning, so it
needs no new state: the wheel delta is converted to a fraction of the stage's
real width/height and added to `(x0, y0)`, which `applyView` then clamps to the
frame. Delta units differ per device (pixels, lines, pages), so they are
normalised to pixels first, and <kbd>Shift</kbd> swaps the axis for a wheel that
only reports up/down. At fit there is nowhere to scroll to, so the wheel is inert
until you zoom — which is also why scrolling cannot be mistaken for a brush
stroke.

### 23.2 The brush is coloured by mode

The ring used `mix-blend-mode: difference` over a white border, which is readable
on any artwork but says nothing about the tool. It now takes the tool's colour —
the same green/red the strokes already use — with a dark inner and outer hairline
so it still reads over light and dark frames. The selected Add/Remove button is
tinted to match, scoped to the brush segment so the other segments stay neutral.
`BRUSH_INK` is the single source for both.

### 23.3 Saying the same thing in fewer rows

The footer was five stacked groups and six lines of copy for three settings. It is
now two rows plus the button:

- **Quality** on one row, with the resolved checkpoint as a right-aligned short
  name (`Base+`) rather than a "Uses SAM 2.1 Base+" line beneath it. The guarantee
  that the panel never quietly loads something other than what it says survives,
  one word shorter; the full name is in the button's tooltip and in the finished
  card.
- **Result** and **Matting** share the second row; the "Result" heading and the
  toggle's second line are gone, and the matting note now appears only when there
  is something the user cannot see (the model isn't installed).
- The elapsed clock moved onto the progress line, the stage checklist and
  spacing tightened, and the hero card is smaller.

The panel's behaviour is untouched — this is spacing, copy and two tooltips.

### 23.4 Premiere Pro

`premiere-pro/` is a second CEP extension. The panel files are copies, because a
CEP extension must be self-contained inside its own folder and cannot share code
with another one; the whole of the port is `js/host.jsx`, plus the host id
(`PPRO`) in the manifest and the installer's extensions folder
(`BetterBrushPPro`).

Three things about Premiere drive the design:

- **Nothing has an id.** After Effects hands out `comp.id` and `layer.id`, and
  both survive `replaceSource`. Premiere's scripting exposes no equivalent, so the
  host remembers the resolved target in a module global (`bbTarget`) while the job
  runs and re-finds the clip afterwards **by the track it is on and the time it
  starts** — the two things a media replacement cannot move. The name is only a
  tiebreaker, because `changeMediaPath` can rename the project item a clip is
  built from.
- **There is no `replaceSource`.** The equivalent is
  `ProjectItem.changeMediaPath`, which repoints a project item at new media: every
  clip cut from it keeps its trims, speed and effects, and only the pixels change.
  That is the same promise the After Effects panel makes. The consequence — it
  applies project-wide, not just to the selected clip — is stated in the panel's
  README rather than hidden.
- **A clip has no dimensions.** There is no `width`, `height` or `frameRate` on a
  Premiere track item at all. The engine never needed them — it reads the real
  values from the media file with its own reader — so the panel sends the
  *sequence's* frame size and rate as a hint and the README says plainly that the
  chip is the thing that can be wrong, not the run.

**Above** is the one mode that could not be ported faithfully: Premiere's
scripting cannot create a video track. It places the cut-out on the video track
directly above the clip and switches the original off — but only after checking
that nothing of the original already occupies that span, and only if a track
exists above at all. Otherwise it changes nothing and says why. Silent
overwriting or rippling would have been worse than a refusal.

Everything else is deliberately unchanged: the same engine, the same one-button
flow, the same brush with zoom, scroll and frame-accurate corrections, the same
hand-off to the full app, and the same `pythonw`-detached launches (a console
child of the host is what took After Effects down in round 6, and Premiere shares
the process group just the same).

Because both panels drive one engine on port 8765, the engine's work folder was
renamed from `BetterBrush\AfterEffects` to `BetterBrush\Plugins` — it is the
engine's folder, not After Effects'.

### 23.5 Where the Premiere panel is installed

Premiere Pro is a channel of CEP that expects panels in the **shared** folder,
`C:\Program Files\Common Files\Adobe\CEP\extensions`, which is scanned for every
user and every Adobe host (the manifest's `HostList` is what keeps it out of After
Effects). The Premiere installer therefore defaults to that folder, and to a
preflight that says plainly what to do when PowerShell is not elevated rather than
failing with an access-denied from `Copy-Item` halfway through. `-Scope User`
installs for one account with no admin rights, as before.

CEP scans three extension roots — the app's own, the shared `Program Files` one,
and `%APPDATA%` — so an extension present in two of them is listed twice. The
installer removes the copies it is not installing into, in both scopes, on install
and on uninstall.

This is also what confirmed the target: Premiere Pro 2026 reports itself as
26.0 with `PlugPlug.dll` 12.0.1 on board, i.e. CEP 12, so the `CSXS.12` key is the
one its player-debug setting lives under, and the manifest's `[22.0,99.9]` host
range covers it. The installer now writes the debug flag for CSXS.9 through
CSXS.16 rather than stopping at 13.

### 23.6 Verified, and what is not

- The panel changes are covered by the QtWebEngine probe: at 200 px and 250 px
  nothing overflows the dock, the wheel scrolls the view without painting a
  stroke, `Ctrl`+wheel zooms, the `+`/`-`/`Fit` buttons still step the fixed
  levels, and the ring reports the tool's colour.
- `panel.js` and `host.jsx` are parsed by the same JavaScript parser the rest of
  the panel code goes through, and both installers are checked by the PowerShell
  parser.
- The Premiere install was rehearsed in both scopes and then run for real, and the
  installed files were confirmed byte-identical to the repository with the shared
  copy in place and the per-user copy gone.
- **The Premiere host has since run the whole path.** The first real test in
  Premiere Pro 26.0 found that `TrackItem.type` is a *number* (1 is video, 2 is
  audio) where `TrackItem.mediaType` is the string "Video"/"Audio"; the host read
  the number and compared it to "Video", so it rejected the clip that was plainly
  selected and every other one. Fixed, and the fix is now covered: `host.jsx` runs
  unmodified against a mock Premiere DOM (`TrackItem.type` as a number, a linked
  audio twin in the selection) and the old expression is kept in the test as the
  guard that proves what was wrong. The apply half has run for real too: a
  **Replace** reached `ProjectItem.changeMediaPath` on a real timeline and came
  back on the timeline, which is what turned up the missing audio in §23.8.
  **Above** (`Track.overwriteClip` and `clip.disabled`) is the one branch still
  written against the documented API and defended call by call, because it needs a
  free track above the clip to run at all.
- The selection path is also tested for a `getSelection()` that answers with a
  keyed collection rather than an array, for an audio-only selection, for two
  different clips (it asks for one), for an offline clip, and for the track index
  it remembers to re-find the clip when the result comes back.
- When a selection cannot be used, the panel now says what Premiere reported
  ("Premiere reports 2 selected: "shot01.mp4" (video), "shot01.mp4" (audio)"),
  and a host script that throws is reported as such instead of being shown as an
  empty selection.

### 23.7 The second real test: one clip read as two, and nothing to launch with

Two more faults, both found by the first run in Premiere Pro 26.0 rather than by
reading the documentation.

**A selected clip and its linked audio came back as two clips.** The first fix had
separated them by asking each item what it was, and 26.0 answers that question the
same way for both halves. So the question was dropped: the selection is now
resolved *against the timeline*. Every clip on every video track is compared with
the selected items by the span it occupies under its name, and a clip that sits on
a video track is the only thing that can be cut out. Each clip is therefore
considered once however many of its halves were selected, and what comes back is
the timeline's own clip object rather than the selection's copy of it — which is
also what `bbTrackIndexOf` and `bbFindClip` want. `clip.parentTrack.video` is
consulted first where a build offers it, then `type`, then `mediaType`, and an item
that matches nothing is still offered rather than thrown away, so an unfamiliar
build fails in the engine where the reason is visible instead of here where it is
not.

**The engine could not be started, because `system` does not exist in Premiere.**
`system.callSystem` — how the After Effects panel starts the engine and the full
app — is an After Effects addition to ExtendScript. Premiere has no `system`
object at all, and the host script therefore threw `ReferenceError: system is
undefined` on the first click. The panel now starts both programs itself with
`child_process`, through the Node.js runtime that CEP provides when the manifest
asks for it (`--enable-nodejs`, `--mixed-context`). They are spawned detached,
with `stdio: "ignore"` and `windowsHide`, so closing Premiere leaves the engine
running and no console window appears. The host script is still the fallback for a
host that has `callSystem` and no Node, and `bbRunDetached` now reports rather than
throws when it has neither.

The collection shape was hardened at the same time: `videoTracks` and `clips` are
read through one set of helpers that accept `numTracks`/`numItems`, a bare
`length`, or `getVideoTrackCount`/`getVideoTrackAtIndex`, because all three turn up
in the wild and a silent zero there would look exactly like this bug again.

Verified: `host.jsx` runs unmodified against the mock Premiere DOM — 40 checks,
covering the linked pair with both halves described as video, the audio half
described as audio, all four collection shapes, two different clips, an audio-only
clip, offline media, the remembered target, and both launch entry points in a host
with no `system` object. The launch itself was run for real outside CEP with the
same call the panel makes: `pythonw -m src.engine --port 8765 --quiet`, detached
and windowless from the repository root, and `/health` answered in six seconds.

### 23.8 The third real test: the replacement had no sound

The panel then worked end to end, and Premiere stopped it with a question: *"Le
fichier sélectionné ne contient pas audio média utilisé par les références d'élément
dans une ou plusieurs séquences. Ces audio références d'élément seront supprimées..."*
— the chosen file has none of the audio the timeline's references to that media
expect, and proceeding deletes those references. The alpha movie is picture only,
and `changeMediaPath` will not quietly swap a media file that had sound for one
that has none.

Accepting the warning is not a fix: it drops the audio either way. The cut-out
should carry the clip's own audio, since the engine read the whole source and the
result *is* that clip minus the background. So the job request gained
`keep_audio`, and the pipeline muxes the source's audio into the finished movie
after the export — a stream copy, so it costs a pass over the file rather than a
re-encode, windowed to the frames the job actually covered so a partial range
still lands in sync, and applied to the audio input alone so the video is copied
whole instead of being trimmed to the audio's length.

The mux is the application's own `_mux_audio`, lifted to module level so the
headless engine runs the same code: lossless when the codec fits the container,
re-encoded to AAC when it does not, and silent-if-it-cannot rather than a failed
render that was otherwise finished. The engine asks whether the source has audio
at all before doing that work (ffprobe beside ffmpeg, and "assume yes" when it
cannot tell), so a silent source is never copied for nothing. `keep_audio` is off
by default — a bare API request still returns picture only — and both panels set
it, because a replace that silences the clip is the wrong answer in either host.

Verified with real files: a two-second clip with sound muxed into a silent ProRes
4444 movie came back carrying the audio, with the video still 2.000 s and still
ProRes; windowing the mux to `start=1.0, duration=1.0` gave 1.000 s of audio
against the same 2.000 s of picture; and a source with no audio was reported as
having none, so nothing was copied.


