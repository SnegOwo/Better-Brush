# Better Brush V3 — Clean Up with AI inpainting (0.3.0)

**Status:** implemented and shipped in the **BetterbrushDistV3** bundle.
Build on top of the V2 Clean Up heuristic pipeline (`old/betterBrushV2/
docs/subtitle-logo-removal.md` — read that first for the detection design).
Per-fix details: `docs/V3-SESSION-WORKLOG.md`.

## 1. What V3 adds on top of V2

V2's Clean Up removed subtitles/logos with OpenCV Telea + donor-row
transplants. V3 keeps that pipeline as the always-available fallback and
adds:

1. **LaMa AI inpainting engine (optional).** `checkpoints/lama_fp32.onnx`
   (big-lama, fp32, ~200 MB) is loaded lazily through `onnxruntime`
   (CPU EP). When present, `_repair_rows_donor(..., use_ai=True)` uses
   LaMa as the base fill for inpaint-mode plates; the donor-row blend and
   Telea remain the fallbacks. LaMa reproduces fine line-art texture on
   busy anime frames that Telea smears.
2. **Adaptive latency guard.** The engine is CPU-only; a first repair that
   takes longer than `_LAMA_BUDGET_S` (2 s — a GPU does it in ~0.1 s,
   a CPU-only machine in ~10 s) downgrades the whole session to the
   OpenCV donor/Telea path and logs a warning. Per-frame exports can
   never stall. The chosen engine is visible in **Clean Up → Check-up**:
   - `LaMa (AI, ONNX)` — runtime + model present
   - `OpenCV donor/Telea (AI engine not bundled)` — no onnxruntime
   - `OpenCV donor/Telea (LaMa model missing)` — no model file
   - `OpenCV donor/Telea (LaMa too slow on this machine)` — guard tripped
3. **Clean Up → Check-up dialog** (`CheckUpDialog`): plain-language status
   of the inpainting engine and every armed plate (mode, region,
   tracking, reference mask). Never runs a probe inference — it opens
   instantly even on CPU-only machines.
4. **Detection fixes** (all covered by `tests/test_plates.py`, 64 tests):
   - `detect_subtitle_band` rewritten: per-pixel temporal-residual density
     (not row-wise Sobel energy) + a wide-horizontal-run validation. A
     solid residual block (flickering captions) is detected; a rolling
     ball / sparkle or a solid bright scenery stripe is rejected.
   - `detect_logo_rect` auto-suppresses the caption band (wide ≥ ½ frame)
     so stroked text is never reported as a logo; narrow static bright
     areas (small corner emblems) stay logo candidates.
   - `_glyph_components` gained an enclosed-hole rule (the outline of a
     solid scenery block encloses a huge hole; glyph strokes never do)
     and a brightness-uniformity rule (a solid panel is one flat value;
     glyph strokes alternate ink and outline). Takes the grayscale image
     as an optional argument.
   - `build_subtitle_plate` regime decision by **plate quality + frame
     coverage**: a band whose per-frame glyph masks are almost always
     present (caption switching lines, burned-in text) is always-on →
     `inpaint` mode; a genuine flicker plate stays `subtitles`.
   - `detect_cleanup_regions` crops user-drawn rects to the frame
     intersection (`_clip_rect`) instead of shifting them — a box that
     sticks out of the frame no longer moves the cleaned area.
   - `_repair_rows_donor` donor alignment probes only *clean* columns
     (probing text pixels mis-aligns the transplant); the donor layer is
     blended ~20 % over the Telea base (a full-strength transplant is
     noisier than Telea on grainy backgrounds — measured on PSNR).
   - `sample_frames` deterministic stagger: the plain `linspace` grid
     phase-locked to periodic overlays (a subtitle ON only on even frames
     sampled 24 identical even frames → nothing detected). Each sample is
     now offset by `7·i` (odd step ⇒ parity alternates) — still
     deterministic, still ~uniform.
5. **Version / packaging.** `APP_VERSION` 0.1.0 → **0.3.0** (constants,
   pyproject, EXE version info). `better_brush.spec` bundles
   `checkpoints/lama_fp32.onnx` and collects `onnxruntime` (removed from
   the excludes list); both are optional at runtime.

## 2. Build

```
pyinstaller --clean --noconfirm --distpath BetterbrushDistV3 -y better_brush.spec
```

- `dist/BetterBrush` — old working developer build (do not overwrite).
- `old/dist_public/BetterBrush`, `old/dist_public_open/BetterBrush` — old
  public releases (0.1.x / 0.2.x, no LaMa model), archived under `old/`.
- `BetterbrushDistV3/BetterBrush` — **V3 release**: EXE + `_internal/`
  (ffmpeg.exe, EULA, `checkpoints/` with SAM 2 + face landmarker +
  `lama_fp32.onnx`).

## 3. Limits to communicate to the user

- LaMa on CPU is too slow for per-frame video export by design — the guard
  switches to OpenCV automatically; the check-up dialog always shows which
  engine is actually in use.
- Everything else (best-effort heuristics, guide-with-rectangle, moving
  foreground crossing a band, 4K handling) is unchanged from V2 — see
  §5 of `old/betterBrushV2/docs/subtitle-logo-removal.md`.
