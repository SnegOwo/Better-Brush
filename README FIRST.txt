================================================================================
 Better Brush v1.0.0 — Windows
 The AI rotoscoping studio for video editors.
 100% local: your footage never leaves your machine.
 100% free: every feature is unlocked — no key, no account, no watermark.
================================================================================

WHAT'S IN THIS PACKAGE
--------------------------------------------------------------------------------

  BetterBrush.exe               The portable desktop app. No installer — unzip
                                anywhere on disk and double-click it.
  _internal\                    App runtime (bundled — do not reorganise it).
  checkpoints\                  AI model weights, bundled offline:
                                  - SAM 2.1 Tiny / Small / Base+ (segmentation)
                                  - EfficientTAM S (small-GPU tracker)
                                  - MatAnyone 2 (soft-matte edges)
                                  - the anime AMV fine-tune (checkpoints\custom)
                                  - LaMa inpainting (GPU + CPU)
                                  - face-landmarker model
                                (MatAnyone 1 and MiniMax-Remover weights are not
                                 bundled: they download on first use of those
                                 features, into checkpoints\.)
  after-effects\                After Effects 2022+ panel incl. install.ps1
  premiere-pro\                 Premiere Pro 2022+ panel incl. install.ps1
  src\ + requirements.txt +     The engine source. The installers for both
  pyproject.toml                panels launch the AI engine straight from this
                                folder (unzip the package first, don't run
                                the installers from inside the zip).
  README FIRST.txt              This file.
  Install Panels.bat /          One-click panel install / removal
  Uninstall Panels.bat          (detailed below).
  README.md, CHANGELOG.md,      Full documentation, license and third-party
  LICENSE,                      attributions.
  THIRD-PARTY-NOTICES.md

QUICK START — DESKTOP APP
--------------------------------------------------------------------------------

 1. Unzip the whole folder somewhere on disk (or run it from a USB stick)
    and keep the folder layout intact — BetterBrush.exe must stay next to
    _internal\ and checkpoints\.
 2. Double-click BetterBrush.exe.
 3. Ctrl+O to open a clip (or just drag one in), click your subject once,
    press Propagate to End, refine with the brush (B) if it needs it, then
    Ctrl+E to export.

  - Works best with an NVIDIA GPU, but runs on CPU too (slower) and has an
    AMD/Intel DirectML tier.
  - Everything is unlocked and free: multi-object scenes, exports up to 8K,
    original audio, learned matting, AI object removal and no watermark.
  - Exports land in exports\ next to the app; logs in logs\.
  - Fully reversible: checkpoints, per-shot profiles and version history are
    saved with each project, so you can always roll back.
  - Press the play-button Help → Quick Tutorial in-app for a guided tour.

QUICK START — AFTER EFFECTS / PREMIERE PRO PANELS
--------------------------------------------------------------------------------

 The panels are thin clients: they rotoscope the selected layer/clip in place
 by driving the same local AI engine. They need Python on your machine
 (one-time setup, ~10 minutes):

  0. LAZY WAY: double-click "Install Everything.bat" instead of steps 1-4 —
     it finds or silently installs Python, creates an isolated environment
     (.bb-env), pulls the CUDA torch + requirements, and registers both
     panels automatically. Then continue at step 5.
  1. One character per click; multi-subject and oddball layers are handed to
     the full app (which is included here, no extra install needed).
  2. Install Python 3.13 or 3.12 or 3.11 from python.org (64-bit, tick
     "Add to PATH" during setup). Newer/older versions are untested.
  3. Open a terminal in THIS folder (the one holding this file) and run:

        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        python -m pip install -r requirements.txt

    The first line installs the CUDA build of PyTorch (~2.5 GB) — plain
    `pip install torch` is CPU-only, don't skip the index-url.

  4. Double-click "Install Panels.bat" — it installs both panels for your
     user (no administrator rights needed). After Effects panel:
     %APPDATA%\Adobe\CEP\extensions\BetterBrush, Premiere:
     the user CEP folder too.
  5. Restart After Effects / Premiere Pro and open
     Window → Extensions → Better Brush.
  6. Select a footage layer (AE) or a timeline clip (Premiere), click
     Remove Background — the layer is rotoscoped in place with a live
     progress bar. Paint a correction and re-apply if the mask drifts.

 No administrator rights: Install Panels.bat uses the per-user scope.
 To install for every user of the machine instead (elevated PowerShell):

    powershell -ExecutionPolicy Bypass -File .\after-effects\install.ps1 -Scope System
    powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1 -Scope System

 Already have Python with the dependencies in a virtual environment? Pass it
 to the installer instead of relying on PATH:

    powershell -ExecutionPolicy Bypass -File .\after-effects\install.ps1 -Python "C:\path\to\venv\python.exe"

 TIP — create config.json from the installer after unpacking to disk: the
 panels are installed into the Adobe CEP extensions folder, and the installer
 records where this package lives, so the panels can find the engine and the
 full app. If you move the package later, re-run the installers.

 Uninstall: double-click "Uninstall Panels.bat" (removes both panels from
 every CEP scope).

TROUBLESHOOTING QUICK LIST
--------------------------------------------------------------------------------

  - Panel missing from the Extensions menu: re-run Install Panels, restart
    the host (CEP reads the manifest once at startup).
  - Panel install says Python was not found: step 2 above.
  - Panel says "the engine did not start in time": re-run the installer so
    config.json gets the right Python path, or check
    %LOCALAPPDATA%\BetterBrush\engine.log for the real error.
  - The panels refuse a clip with several subjects or no clear character:
    they hand off to the full app on purpose — pick subjects there.
  - Full details, plus a one-command uninstaller, live in
    after-effects\README.md and premiere-pro\README.md.

TROUBLESHOOTING — APP
--------------------------------------------------------------------------------

  - First segmentation downloads the requested tier of SAM 2 unless it is
    already in checkpoints\ (tiny/small/base+ are bundled).
  - Matting is on by default and reads weights from checkpoints\matanyone2\
    (bundled).
  - If a job fails, check the status bar and logs\better-brush.log.

LICENSE
--------------------------------------------------------------------------------

 Better Brush is source-available under the Functional Source License 1.1,
 MIT Future License (FSL-1.1-MIT) — see LICENSE. You may use, modify and
 share it, including to produce monetised YouTube / TikTok / AMV videos; you
 may not resell it, sell its source, or ship it inside a competing or paid
 tool. The license becomes MIT on 2028-10-23.

 The bundled AI weights keep their own terms: MatAnyone 2 (NTU S-Lab License
 1.0) and the downloaded MiniMax-Remover weights (CC-BY-NC-4.0) are licensed
 for non-commercial use only. Full attributions: THIRD-PARTY-NOTICES.md.

 There is nothing to buy or activate in this version: every feature is
 unlocked, and no data is transmitted (no account, no server, no telemetry).

 © 2026 Sneg — Sneg HQ, a company of Lugunica · sneg.dev
================================================================================
