# BetterBrush — Anime & AMV Engine

<span style="color:#4ade80">Line lock · cel hold · letterbox detection · anime QC.</span>

---

BetterBrush is tuned first for **anime and AMV**. Live-action segmentation and
cel-shaded line art are different problems, and four features exist specifically
for the second one. All four are automatic when your content preset is
**Anime / AMV** — there is nothing to configure to get the baseline benefit.

| Feature | Problem it solves | Where it runs |
|---|---|---|
| **Line lock** | Mask edges float off the pencil line, or follow background line art | Segmentation + post-process |
| **Cel hold** | The matte *boils* (twitches) on frames that are held drawings | Post-process |
| **Letterbox detection** | Baked-in black bars are dead space you roto by accident | On import |
| **Anime QC metrics** | Boiling, line integrity and background-line inclusion are invisible in generic QC | Library API |

---

## 1. Line lock (line-art prior)

Anime is drawn with line art: dark, thin, high-contrast contours. A general
segmenter treats those lines as ordinary edges, which causes the two classic
anime failures:

- **Background-line inclusion** — the character outline and the background's
  line art are the *same* black lines, so a mask grows along a fence, a desk edge
  or a background character and swallows it.
- **Edge wobble** — the correct anime matte edge *is* the pencil line. When the
  model's edge floats a pixel off it, the result reads as fuzzy no matter how
  high the IoU is.

Line lock extracts the line art and uses it twice:

1. **Candidate selection.** During segmentation the prompt-aware mask selector
   adds a penalty for candidates that swallow line-art components the user never
   pointed at. A greedy-but-higher-score candidate loses to a clean one.
2. **Edge snapping.** After refinement, the contour is pulled onto the nearest
   line (grown onto lines just outside it, trimmed where it strays more than a
   couple of pixels). Interiors are never touched, and the whole pass is skipped
   when the contour does not already overlap line art — so it is a no-op on
   photographic footage and soft mattes.

**Module:** `src/processing/line_prior.py`
**API:** `extract_lines`, `boundary_line_support`, `background_line_penalty`,
`snap_to_lines`, `snap_to_anime_lines`

---

## 2. Cel hold (anti-boiling)

Hand-drawn animation is not redrawn every frame. A *cel* is **held** — usually on
2s or 3s, sometimes for a dozen frames — so large stretches of an anime clip are
literally the same drawing. A frame-by-frame tracker does not know that and
re-solves every frame anyway, which is how you get **boiling**: the matte contour
twitches on frames that are pixel-identical to the one before, while the artwork
is stone still.

Cel hold finds held frames cheaply (a downscaled signature comparison) and
**copies the previous mask verbatim** across the hold. On a hold the matte is
then not merely stable — it is *identical*, so boiling is zero by construction.
The held frames also cost nothing to segment, which speeds long clips up.

**Module:** `src/processing/cel_hold.py`
**API:** `frames_match`, `find_holds`, `apply_cel_hold`, `hold_runs`,
`frame_signature`

> Applied only during post-processing, and only for the **Anime / AMV** preset.
> Held frames are detected from the source video, never from the mask, so a real
> camera pan or a cut is never mistaken for a hold.

---

## 3. Letterbox / pillarbox detection

Old anime releases — the mainstay of AMV editing — are routinely stored with
solid black bars baked into the frame: 4:3 masters pillarboxed into 16:9,
letterboxed film prints, or both. Those bars are pure background, but a
segmenter has no idea: they are dark, static and huge, so they are easy to
accidentally include in a matte and they waste GPU time on pixels nobody wants.

On import, BetterBrush samples the clip, detects the bars, and reports them in
the status bar (`black bars T{top}/B{bottom}/L{left}/R{right}`). Detection is
deliberately conservative:

- A uniformly dark frame (a night scene, a fade) is **not** a letterbox.
- The crop is capped so it can never eat more than 45% of an edge.
- Across several frames the **smallest** bar is used, so content is never cropped
  that any sampled frame kept.

The detection is informational — it never crops anything on its own — but the
crop box (`Letterbox`) is available to frames and masks via `crop_frame`,
`crop_mask` and `embed_mask`.

**Module:** `src/processing/letterbox.py`
**API:** `detect_letterbox`, `detect_letterbox_sequence`, `combine_letterboxes`,
`crop_frame`, `crop_mask`, `embed_mask`

---

## 4. Anime QC metrics

The generic **Matte QC Inspector** answers "is this matte roughly right?" — hair,
holes, spill and gross flicker. It does not know the two things that define a
*good anime* roto:

- **Boiling** — how much the contour twitches on frames that are held drawings.
  Measured only on **held pairs**, where the artwork is frozen and any change is
  wrong, so it does not confuse real motion with boiling.
- **Line integrity** — whether the contour rides the pencil line, and how much
  *background* line art the mask swallowed.

These are exposed as cheap, dependency-light functions so they can be shown in
the inspector, logged per shot, or used to drive automatic keyframe suggestions
(`suggest_keyframes` flags frames where the contour moves on a held cel — the
first frames a fixer should look at).

**Module:** `src/processing/anime_qc.py`
**API:** `contour_change`, `boiling_score`, `line_support`,
`background_line_ratio`, `suggest_keyframes`, `analyze_anime_matte`

```python
from src.processing.anime_qc import analyze_anime_matte

report = analyze_anime_matte(frame, mask, previous_mask=prev, previous_frame=prev_frame)
if report.warning:
    print(report.message)  # e.g. "contour moves on a held cel; background line art included"
```

---

## Turning it off

Line lock and cel hold are tied to the **Anime / AMV** content preset; switching
to **Human** or **Object** disables both, and the segmentation step then behaves
exactly as before. The letterbox scan runs on every import but only ever logs.

## See also

- [Complete Feature Reference](COMPLETE-FEATURES.md) — the whole tool, in depth.
- [Fine-tuning guide](FINE-TUNING.md) — train a model on *your* anime.
- [Build Lugnica SAM](LUGNICA-SAM.md) — the short step-by-step to your own model.
