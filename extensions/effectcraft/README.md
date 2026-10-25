# Better Brush for EffectCraft **Beta**

A ScriptUI panel and a background adapter that put **Remove Background** — and a
brush for fixing what it gets wrong — inside [EffectCraft](https://github.com/storytold/effectcraft).
Everything runs locally with the same SAM 2 stack as the desktop app and the
After Effects / Premiere Pro panels: no upload, no account, no watermark.

![Better Brush inside EffectCraft](../../docs/images/effectcraft-beta.webp)

---

## Status: beta, source-only for now

| | |
|---|---|
| **Stage** | Beta. It works and it is used daily — expect rough edges. |
| **Tested on** | **`effectcraft-0.17.0-windows-x64-portable` only.** Nothing else. |
| **Install** | From source, right now (this folder + `install.cmd`). |
| **Portable build & one-file setup** | On Hugging Face between **2 and 8 November**. |
| **Bugs** | Please open a [GitHub issue](https://github.com/SnegOwo/Better-Brush/issues) — include `adapter.log` and `panel.log`. |

EffectCraft **0.11.1 and older cannot run panel timers** (upstream fixed that in
0.17.0), so on those builds the panel only refreshes while your pointer is over
it and it says so in the status line. Use 0.17.0.

## Requirements

- **EffectCraft 0.17.0**, Windows x64 **portable** (the `EffectCraftData` folder
  next to `effectcraft.exe`). The installed (non-portable) layout is detected
  too, but it is not what this beta is tested on.
- **Windows 10/11**.
- **Python 3.11+ with the Better Brush environment** for real mattes — the
  adapter calls the same engine as the app, so `torch`, `cv2`, `numpy` and the
  SAM 2 checkpoints must be importable. A checkout of this repository is the
  supported way to get that.
- In EffectCraft: **Preferences ▸ Scripting & Expressions ▸ Allow Scripts to
  Write Files and Access Network** must be on. The panel refuses to run
  otherwise and says why.

## Install (from source)

1. Clone the repository and open PowerShell in it:

   ```powershell
   git clone https://github.com/SnegOwo/Better-Brush.git
   cd Better-Brush
   ```

2. Double-click **`extensions\effectcraft\install.cmd`** (or run the same
   command from a terminal):

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\extensions\effectcraft\scripts\install.ps1
   ```

   It auto-detects the EffectCraft settings folder. For a portable install that
   is `<folder with effectcraft.exe>\EffectCraftData`; pass
   `-SettingsDir "<path>"` if you have more than one install and want to be
   explicit. `install.cmd -SettingsDir "<path>"` forwards the argument.

3. Start EffectCraft. The panel opens by itself (the installer adds it to
   `scriptui_panels.json`), and it starts the adapter in the background the
   first time it needs one.

4. Turn on the scripting permission above if you have not, then restart
   EffectCraft.

**What the installer touches** — only two folders:

| Path | What it is |
|---|---|
| `<EffectCraftData>\Scripts\ScriptUI Panels\BetterBrush.jsx` | the panel (stamped with the bridge path, because a panel script cannot see its own folder) |
| `<EffectCraftData>\betterbrush-bridge\` | job/status files, logs, `start-adapter.vbs` / `.cmd` |

Nothing is registered at logon. The adapter is started by the panel when
EffectCraft opens, and it stops being needed when you stop using it.

## Use it

1. Select a **footage layer** in EffectCraft.
2. Click **Remove Background**. The adapter renders a matte sequence and the
   panel imports it above the layer as **BetterBrush Matte**, then applies it as
   a track matte — one Ctrl+Z takes the whole thing back.
3. Scrub to the frame that looks wrong. **The result image follows the time
   indicator**, so you are always painting on the frame you are about to fix.
4. Paint on that image:

   | | |
   |---|---|
   | **＋ Add** (green) | keeps pixels — paint back something the matte cut away |
   | **− Remove** (red) | deletes pixels — paint over what should not be there |
   | **Shift + drag** | uses the other tool for that stroke |
   | **Brush** slider | brush size in clip pixels |
   | **Remove force** 1–5 | how hard red bites: 1 is exactly what you painted, 5 grows a margin and erases it from **every** frame the matte still covers |
   | **Undo stroke / Clear** | drop the last stroke / all of them |
   | **Apply fix & re-run** | re-tracks from this frame with your strokes and swaps the new frames into the existing matte |

   Strokes are drawn **on top of the frame** as you paint, so nothing is a
   surprise. They never touch the composition — no stray masks, nothing to clean
   up afterwards.

5. **Open in Better Brush** hands the clip to the desktop app when a frame needs
   more than a brush stroke (object removal, matting, export).

## Where things live

| File | What it is |
|---|---|
| `<EffectCraftData>\betterbrush-bridge\panel.log` | what the panel did — start here when something looks frozen |
| `…\adapter.log` | what the adapter did — every job, stage and error |
| `…\brush-context.json` | the last run the panel can correct (work folder, matte item, frame count) |
| `…\adapter.alive` | heartbeat; the panel starts an adapter only when this is stale |

The adapter watches **every bridge it can find** — the recorded one, the usual
`%APPDATA%` candidates, and portable installs under Downloads/Desktop/Documents
— so two EffectCraft installs do not strand each other's jobs.

## Troubleshooting

**“Job … queued — waiting for the adapter…”** — nothing is watching that bridge.
Check `adapter.log`: if it is not moving, start the adapter yourself with
`<EffectCraftData>\betterbrush-bridge\start-adapter.cmd`, or run
`python extensions\effectcraft\run_adapter.py watch`. If the panel says the
adapter is not running and there is no start file, re-run the installer for that
EffectCraft folder.

**“Script file writes are blocked”** — Preferences ▸ Scripting & Expressions ▸
*Allow Scripts to Write Files and Access Network*, then reopen the panel.

**“The Better Brush engine is not importable from here”** — the adapter was
started without the Better Brush environment. Run it from the repository (the
installer's start file does), or install `torch`, `opencv-python` and `numpy`
into the Python that runs it.

**The panel never updates itself** — you are on EffectCraft 0.11.x or older.
Panel timers do not run there; the panel only refreshes while the pointer is
over it. Upgrade to 0.17.0.

**EffectCraft asks “A script wants to open a file”** — that is EffectCraft's own
permission for starting the adapter (`Warn User When Executing Files`). Allow it
once, or turn that setting off if you would rather not be asked.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File .\extensions\effectcraft\scripts\install.ps1 -Uninstall -SettingsDir "<EffectCraftData>"
```

Removes the panel, the adapter start files and any old autostart entry. The
bridge folder (and its logs) is left alone — it is your work.

## How it works

```
BetterBrush.jsx  ── job-<id>.json ──▶  run_adapter.py  ──▶  src.engine (SAM 2)
   panel          ◀─ status-<id>.json ──  adapter        ◀──  PNG matte
      │                                        │
      └── app.run(...) commands to import, track-matte, replace footage
```

The panel never decides *what* to run: the adapter writes the EffectCraft command
list, the panel executes it inside one undo group. Everything crossing the
boundary is a flat JSON file, so a frozen panel can never block the host and a
crashed adapter leaves a readable log instead of a hung UI.

## Limits (beta)

- One job at a time, one footage layer at a time.
- Solids, precomps, nulls, 3D and time-stretched layers are rejected with a
  sentence rather than a crash.
- Engine mode needs the Better Brush Python environment; the standalone adapter
  planned for the Hugging Face release runs the bridge and `--mode simulate`, not
  the SAM 2 engine.
- EffectCraft 0.17.0 only — see above.

## License

FSL-1.1-MIT, same as the rest of Better Brush — see the repository
[LICENSE](../../LICENSE) and [THIRD-PARTY-NOTICES.md](../../THIRD-PARTY-NOTICES.md)
for the model weights.
