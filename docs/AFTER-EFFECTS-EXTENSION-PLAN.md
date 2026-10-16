# Better Brush for After Effects — integration plan

> **Status: v1 implemented** (Windows, AE 2022+). Shipping code lives in
> [`after-effects/`](../after-effects/README.md) (the CEP panel) and
> [`src/engine/`](../src/engine) (the headless engine). The rest of this
> document is the original plan, kept for rationale; §14 records the decisions
> and the deviations that were made while building it.

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

## 8 · Licensing & editions

Reuse the offline HMAC licensing already in `src/core/licensing.py`.
**Locked: the AE extension is a Pro feature.** Split:

| Capability | Lite (free) | Pro |
|---|---|---|
| Single-object background removal | ✅ (watermark optional) | ✅ |
| Multi-object / >1080p / no watermark | ❌ | ✅ |
| Batch all selected layers | ❌ | ✅ |
| Refine UI | ✅ | ✅ |

The panel asks the engine `capabilities` at connect; the engine is the single
source of truth for license state (the panel never decides). No accounts, no
server, no telemetry — same as the desktop app.

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
| **M5** | Packaging: `.ccx` (UXP) install, engine discovery, version handshake, **Pro** licensing gate, diagnostics | 1–2 wk | Fresh machine: install app + panel, panel finds engine, gates on licence |
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
3. **Licensing: Pro feature** — the panel checks `/health` and blocks the button
   when the engine reports `pro: false` (same client-side model as the desktop
   app).
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
