# Better Brush in EffectCraft — beta guide

> [!WARNING]
> **Beta, and source-only for now.** The panel and adapter work today, but they
> are young: expect rough edges, and please report them.
>
> * **Tested on `effectcraft-0.17.0-windows-x64-portable` only.** No other
>   EffectCraft build is supported — 0.11.1 and older cannot run panel timers at
>   all (that was fixed upstream in 0.17.0), so the panel there only refreshes
>   while your pointer is over it.
> * **Install from source** (this guide). The portable build and one-file setup
>   go to **Hugging Face between 2 and 8 November**.
> * **Bugs & oddities** → [GitHub issues](https://github.com/SnegOwo/Better-Brush/issues).
>   Attach `panel.log` and `adapter.log` if you can.

![Better Brush inside EffectCraft](../docs/images/effectcraft-beta.webp)

EffectCraft's own panel dock shows the matte result, the quality preset and the
brush. The matte below it is the real track matte applied to the footage layer —
one undo step.

---

## 1. What this is

EffectCraft has no built-in way to cut a subject out of a clip. Better Brush adds
two pieces:

| Piece | What it does |
|---|---|
| **`BetterBrush.jsx`** (ScriptUI panel) | talks to the user: picks the layer, shows the matte, hosts the brush, runs EffectCraft commands |
| **`run_adapter.py`** (background adapter) | renders the matte with the Better Brush engine (SAM 2 + MatAnyone 2), and answers "open this clip in the desktop app" |

They talk through **flat JSON files in a bridge folder**, so the panel never
blocks EffectCraft's UI thread and a crashed adapter leaves a log rather than a
hung window.

## 2. Requirements

- **EffectCraft 0.17.0**, Windows x64, **portable** layout (`EffectCraftData`
  sits next to `effectcraft.exe`). The installed layout is detected as well, but
  it is not what this beta is tested on.
- **Windows 10/11**, 64-bit.
- **Python 3.11+ with the Better Brush environment** for real mattes. The
  adapter imports the same engine as the app: `torch`, `opencv-python`, `numpy`
  and the SAM 2 checkpoints. A clone of this repository is the supported way to
  get that (see [Install](../README.md#install) in the main README); the
  standalone adapter planned for the November release runs the bridge but not
  the SAM 2 engine.
- **Approx. 2 GB of disk** for the first matte (SAM 2 checkpoint + frames).

## 3. Install (from source)

### The quick way

1. Clone this repository:

   ```powershell
   git clone https://github.com/SnegOwo/Better-Brush.git
   cd Better-Brush
   ```

2. Double-click **`extensions\effectcraft\install.cmd`**.

   It finds your EffectCraft install (portable `EffectCraftData`, else
   `%APPDATA%\EffectCraft`), copies the panel, creates the bridge folder and
   writes the adapter start file. If you have more than one EffectCraft install,
   pass the folder explicitly:

   ```powershell
   .\extensions\effectcraft\install.cmd -SettingsDir "D:\EffectCraft\EffectCraftData"
   ```

3. Start EffectCraft. The panel opens on its own (the installer adds it to
   `scriptui_panels.json`). If it does not, open **Window ▸ Better Brush**.

4. EffectCraft may ask *"A script wants to open a file"* the first time the
   panel wants to start the adapter. That is EffectCraft's own permission for
   starting a program — allow it (or turn off *Warn User When Executing Files*
   in Preferences ▸ Scripting & Expressions if you would rather not be asked).

### The manual way

If you prefer to know exactly what lands where:

```powershell
# 1. Copy the panel. A panel script cannot see its own folder, so the bridge
#    path is stamped into the copy it installs.
$ec      = "D:\EffectCraft\EffectCraftData"               # your EffectCraftData
$bridge  = "$ec\betterbrush-bridge"
$panel   = "$ec\Scripts\ScriptUI Panels\BetterBrush.jsx"
New-Item -ItemType Directory -Force -Path $bridge, (Split-Path $panel) | Out-Null

# 2. Stamp + copy
$text = Get-Content .\extensions\effectcraft\panel\BetterBrush.jsx -Raw
$text.Replace('@@BB_BRIDGE@@', ($bridge -replace '\\','/')) |
  Set-Content $panel -Encoding UTF8 -NoNewline

# 3. Remember the bridge for the adapter
'{ "bridge_dir": "' + ($bridge -replace '\\','/') + '" }' |
  Set-Content "$bridge\..\bridge.path.json" -Encoding UTF8
```

Then start the adapter yourself, next to EffectCraft:

```powershell
python .\extensions\effectcraft\run_adapter.py watch
```

…and enable **Preferences ▸ Scripting & Expressions ▸ Allow Scripts to Write
Files and Access Network**.

## 4. Check it works

With EffectCraft open and a project loaded:

1. The panel is titled **Better Brush v0.6.x** and the status line under the
   result image says *Ready — select a footage layer…*.
2. Select a **footage layer** — not a solid, text layer, null or precomp.
3. Click **Remove Background**. Progress lines appear on their own.
4. When it finishes, the layer gets a **BetterBrush Matte** above it and the
   result image appears in the panel.

If the status line never changes, see [Troubleshooting](#7-troubleshooting).

## 5. Fixing a shot with the brush

The first matte is never the last one. This is the loop:

1. **Scrub to the frame that is wrong.** The panel's image follows the time
   indicator and reads that frame from the rendered sequence, so you always
   paint on the frame that becomes the correction's key frame.
2. **Pick a tool** — **＋ Add** keeps pixels the matte cut away, **− Remove**
   deletes what should not be there. Hold **Shift** while dragging to use the
   other one for that stroke.
3. **Drag on the image.** Strokes are drawn over the frame as you paint: green
   for *keep*, red for *delete*, both at a soft opacity so the matte stays
   readable.
4. **Choose how hard red bites.** *Remove force* 1 means "exactly what I
   painted"; 5 grows the painted region and erases it from **every** frame the
   matte still covers, which is what you want for a big leftover object.
5. **Apply fix & re-run.** The engine re-tracks from this frame with your
   strokes and swaps the new frames into the existing matte layer — no second
   matte, no index shifts, **one Ctrl+Z** to undo.

| Control | Effect |
|---|---|
| **＋ Add / − Remove** | arms the tool; with a stroke selected, flips that stroke instead |
| **Shift + drag** | paints with the other tool once |
| **Brush** | brush size, in pixels of the clip (4–160) |
| **Remove force** | 1–5, how much margin a delete stroke takes and whether it is stamped on every frame |
| **Undo stroke / Clear** | drop the last stroke / all of them |
| **Refresh** | re-read the frame (rarely needed; the panel keeps itself current) |
| **Apply fix & re-run** | the correction |

Strokes live in the panel only: they are never written into the composition, so
there are no stray masks to clean up and nothing for EffectCraft's undo history
to argue with.

## 6. Opening a clip in the desktop app

**Open in Better Brush** asks the adapter to launch the desktop app on the clip
you are working on. The first time, Better Brush may not be found: paste the
full path to `BetterBrush.exe` into the field under the button and click
**Save** — the adapter remembers it.

## 7. Troubleshooting

| Symptom | What to do |
|---|---|
| *"Job … queued — waiting for the adapter…"* forever | Nothing is watching that bridge. Check `adapter.log`; start it with `<EffectCraftData>\betterbrush-bridge\start-adapter.cmd`, or `python extensions\effectcraft\run_adapter.py watch`. |
| *"Script file writes are blocked"* | Preferences ▸ Scripting & Expressions ▸ **Allow Scripts to Write Files and Access Network**, then reopen the panel. |
| *"The Better Brush engine is not importable from here"* | The adapter was started with a Python that does not have the app's environment. Start it from the repository, or install `torch`, `opencv-python`, `numpy` into that Python. |
| Panel never updates by itself | EffectCraft ≤ 0.11.x does not run scheduled panel tasks. Upgrade to 0.17.0. |
| *"EffectCraft is asking permission to start the adapter"* | Allow it once; turn off *Warn User When Executing Files* to stop being asked. |
| The panel is not in the Window menu | The installer landed in the wrong settings folder — re-run with `-SettingsDir "<EffectCraftData>"`. Two installs on one machine are fine: the adapter serves every bridge it finds. |

**Logs**

| File | Contents |
|---|---|
| `<EffectCraftData>\betterbrush-bridge\panel.log` | every panel action, submit, stage line and error |
| `<EffectCraftData>\betterbrush-bridge\adapter.log` | every job, its stages and its failure |
| `<EffectCraftData>\betterbrush-bridge\brush-context.json` | what the panel can still correct |

## 8. Beta notes

- One job at a time, one footage layer at a time.
- Solids, precomps, nulls, 3D layers and time-stretched layers are rejected with
  a sentence rather than a crash.
- EffectCraft 0.17.0 portable is the only build this beta has been tested on.
- The portable build and the one-file setup will be on **Hugging Face between
  2 and 8 November**; until then, installing from source is the only supported
  route.
- Found a bug? [Open an issue](https://github.com/SnegOwo/Better-Brush/issues)
  with the two logs above and, if possible, the clip's frame size and the
  quality preset you used.

## 9. Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File .\extensions\effectcraft\scripts\install.ps1 `
  -Uninstall -SettingsDir "<EffectCraftData>"
```

This removes the panel, the adapter start files and any old autostart entry. The
bridge folder is kept — it holds your logs and the last run the panel could
correct. Delete it by hand if you want it gone.

## 10. How the bridge works (for the curious)

```
BetterBrush.jsx ── job-<id>.json ──▶ run_adapter.py ──▶ src.engine (SAM 2 / MatAnyone 2)
      ▲                                   │
      └────── status-<id>.json ◀──────────┘
      │
      └─ app.run("file.import" / "layer.setTrackMatte" / "file.replaceFootage" …)
```

- The panel writes one job document and polls its status file; it never waits on
  a call.
- The adapter writes the *command list* the panel will run, so the import and
  matte logic exists once (in Python, tested), not twice (in JavaScript, drifting).
- A correction reuses the previous run's work folder — the engine re-tracks from
  the painted frame instead of searching for a subject again.
- `adapter.alive` is the heartbeat the panel reads before starting an adapter.

See [VALIDATION.md](VALIDATION.md) for the engineering log: what was measured,
which engine quirks were found, and why each decision was taken.
