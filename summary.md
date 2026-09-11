# Better Brush — Product Summary

**Better Brush** is AI-powered rotoscoping desktop software built on SAM 2.
Import a video, prompt SAM 2 with include/exclude points, propagate the mask
across the timeline, refine with the brush, then export transparent or
chroma-keyed video.

**Developer:** Sneg — Sneg HQ, a company of Lugunica. https://sneg.dev
**Version:** 0.1.0 (beta) · **Platform:** Windows (PySide6 + PyTorch + SAM 2)
**License:** Proprietary — All Rights Reserved (see `LICENSE`, shipped inside
every build). Third-party components retain their own licenses.

## Editions

| Feature                                              | Lite (free) | Pro (€5, one-time) |
|------------------------------------------------------|:-----------:|:------------------:|
| Full masking / keyframing / propagation              | ✅ single obj | ✅ multi-object   |
| H.264 MP4 export with chroma-key backgrounds         | ✅ ≤1080p   | ✅ ≤8K             |
| ProRes 4444 / QuickTime Animation (MOV, alpha)       | ✅ ≤1080p   | ✅ ≤8K             |
| PNG alpha sequence export                            | ✅          | ✅                 |
| Editing **more than one object** in a project        | ❌ (1 object) | ✅               |
| Resize beyond 1920×1080                              | ❌          | ✅                 |
| Mux original audio into the export                   | ❌          | ✅                 |
| Library recoding (MP4 ↔ MOV ↔ WebM)                  | ❌          | ✅                 |
| Watermark on video exports                           | small       | none              |

The free Lite tier keeps all real rendering value: full SAM 2 masking,
propagation, every export codec (incl. alpha-capable MOV) up to 1080p —
so a free user can produce professional results. Pro adds multi-object
editing, 4K+, original audio, recoding and watermark-free exports.

## Licensing & Activation

- **Proprietary EULA** replaces the old BUSL-1.1. Free users get the Lite
  edition automatically; Pro is unlocked with an offline, HMAC-signed key.
- Key format: `BBP-<payload>-<signature>` (base32 payload + truncated
  HMAC-SHA256 signature under a secret baked into the build).
- **Fully offline** — no account, no internet, no telemetry. Only the payload
  signature is stored locally in `%APPDATA%\BetterBrush\activation.json`,
  never the raw key.
- Issue keys (distributor side, from the repo root):

  ```
  python tools/make_license.py "Buyer Name"                        # 1 permanent key
  python tools/make_license.py "Buyer Name" expires:2027-12-31     # timed key
  python tools/make_license.py "Buyer" --count 10                  # 10 keys at once
  ```

- Users activate via **Help → Activate Pro…** and paste the key. About shows
  the current edition. Keys are tied to the licensee name printed at issue
  time — trust-based (keys can technically be shared; there is no server).

## Enforcement (defense in depth)

- **UI gates:** the "+ object" button, audio checkbox and Library recode are
  disabled/labelled for Lite; 4K sources are clamped to 1920×1080.
- **Backend enforcement (`src/export/exporter.py` + `main_window.py`):** even
  if a Lite session requests 4K/audio, the exporter silently scales to ≤1080p
  and drops audio; the second-object prompt is rejected by `_add_object`.
- **Watermark:** Lite video exports carry a small bottom-corner "Better Brush"
  mark, composited over the subject (`_draw_watermark`).
  PNG exports remain watermark-free for free-tier users.
- **In-app Lite texts** (status-bar €5 pill, first-launch notice, About,
  export dialog note, comparison dialog) are Lite-only and disappear entirely
  once a Pro key is activated — no residual ads for paying users.

## Build

```
pyinstaller --clean --noconfirm better_brush.spec                 # docker build
pyinstaller --clean --noconfirm --distpath dist_public better_brush.spec   # public release
```

- The **working** developer build lives in `dist/` (do not rebuild/overwrite).
- The **public release** is a separate folder, `dist_public/BetterBrush/`
  (EXE + `_internal/` with ffmpeg.exe and the EULA).
- Runtime data (`checkpoints/`, `exports/`, logs) resolves to the folder next
  to the EXE; user preferences and the activation file live in
  `%APPDATA%\BetterBrush\`.
- Tests: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/ -q` → 50/50.

## Recent polish (this release)

- Alpha-capable codecs (ProRes/QuickTime MOV) available in Lite — the paywall
  now sits on multi-object editing, 4K+, audio, recoding and the watermark.
- MP4 background picker shows **solid chroma colors only** (no confusing
  "No background / black" option); defaults to Vivid Green for keying.
- First-launch Lite notice (WinRAR-style, one-time) + status-bar €5 pill +
  Help → **Pro vs Lite…** comparison + export-dialog/A-about notes. Pro
  activation removes every one of those texts.
- Library moved to **File → Media Library (Ctrl+L)**.
- Launch splash no longer stays on top of other windows.
- Right sidebar decluttered (AI Model + Hardware collapsed, clearer hints).
- Tutorial rewritten: AMV / Human / Object presets, Propagate to End,
  Shift+brush = erase, chroma-keying, Library/recode.
- **Hold Shift while brushing to erase** (persistent eraser mode still works).
- (Prior) fixed frozen-build stdio crash, propagated masks on multi-object
  scenes, chroma-key backgrounds, hardened audio muxing, animated splash.