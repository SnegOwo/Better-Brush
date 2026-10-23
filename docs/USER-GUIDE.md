<div align="center">

# Better Brush User Guide

<span style="color:#4a9eff">From raw clip to clean alpha matte.</span>

</div>

This guide is for editors and animators. For code architecture and processing internals, read the [Developer Guide](DEVELOPER-GUIDE.md) instead.

---

## 1. The short version

1. **Open a video** with **File → Open Video** (`Ctrl+O`) or drop it on the window.
2. Choose a **content preset**: Anime/AMV, Human, or Object.
3. Switch to the **Point** tool and click the subject. Right-click to exclude background.
4. Press **Segment Frame**.
5. Use **Propagate to End** to follow the subject across the shot.
6. Inspect the result with **Inspect → Matte QC Inspector** (`Ctrl+Shift+M`).
7. Press `Ctrl+E`, choose a format, and export.

<span style="color:#f59e0b">Tip:</span> The first segmentation downloads a SAM 2.1 checkpoint. The app works offline after the model is available.

---

## 2. The window at a glance

| Area | What it is for |
|---|---|
| <span style="color:#4a9eff">Top menu</span> | File, Export, Inspect, Edit, and Help actions |
| <span style="color:#4a9eff">Branding bar</span> | App name and the detected GPU/device |
| <span style="color:#4a9eff">Canvas</span> | Frame view, mask overlay, points, brush, zoom, and pan |
| <span style="color:#4a9eff">Right panel</span> | Content, model, objects, tools, segmentation, propagation, cuts, cleanup, export, and hardware. Per-shot edge and colour settings live in **Inspect → Shot Profiles** (§9), not here |
| <span style="color:#4a9eff">Transport</span> | Play, stop, timecode, frame counter, and preview mode |
| <span style="color:#4a9eff">Timeline</span> | Keyframes, masked frames, scene cuts, and export range |
| <span style="color:#4a9eff">Status bar</span> | Current object, frame range, unsaved state, and feedback |

### Preview modes

The **View** menu above the timeline switches the canvas between:

- **Overlay** — the current mask is tinted over the frame.
- **Cutout** — the subject is shown on a checkerboard, which is best for seeing edge quality.
- **Raw** — the source frame without the mask layer.

---

## 3. Menus and shortcuts

### File

| Action | Shortcut | Purpose |
|---|---:|---|
| New Project | `Ctrl+Shift+N` | Clear the current session |
| Open Project | `Ctrl+Shift+O` | Load a `.bbproj` file |
| Save Project | `Ctrl+S` | Save points, masks, keyframes, objects, and profiles |
| Save Project As | `Ctrl+Shift+S` | Save a new project file |
| Automatic Save | — | Periodically write a recovery copy |
| Open Video | `Ctrl+O` | Import footage |
| Media Library | `Ctrl+L` | Browse and recode previous exports |

### Inspect

| Action | Shortcut | Purpose |
|---|---:|---|
| Matte QC Inspector | `Ctrl+Shift+M` or `Ctrl+M` | Compare source, matte, QC evidence, and coach advice |
| Performance Monitor | `Ctrl+Shift+G` | See live GPU load, VRAM, temperature, and render estimate |
| Shot Profiles | `Ctrl+Shift+P` | Tune the current object/shot independently, including edge softness and the colour spill cleaner (§9) |
| Version History | `Ctrl+Shift+H` | Compare and restore checkpoints |
| Create Checkpoint | `Ctrl+Alt+S` | Save a named recovery point |

### Export and Edit

| Action | Shortcut | Purpose |
|---|---:|---|
| Export | `Ctrl+E` | Open the export dialog |
| Batch Render Queue | `Ctrl+Shift+Q` | Queue several saved projects |
| Undo | `Ctrl+Z` | Restore the previous mask/point state |
| Redo | `Ctrl+Y` or `Ctrl+Shift+Z` | Move forward through edits |

### Canvas shortcuts

| Key | Action |
|---|---|
| `B` | Switch between Point and Brush tools |
| `Space` or `P` | Play/pause |
| `Left` / `Right` | Previous/next frame |
| Mouse wheel | Zoom the canvas |
| `Shift` + brush | Erase instead of add |
| `[` / `]` | Smaller/larger brush |
| `Delete` | Remove the selected point |
| `I` / `O` / `X` | Set timeline in/out points or clear the range |

---

## 4. Choosing content and a model

The **Content** preset changes SAM 2 mask-token and IoU behavior:

- **Anime / AMV** — crisp line art and hard edges.
- **Human** — faces, hair, and skin detail.
- **Object** — a safe general-purpose starting point.

The **AI Model** section can stay on **Auto (recommended)**. The profiler chooses a model that leaves safety headroom instead of filling all available VRAM. You can force a specific tier when you understand the memory cost.

**SAM 3.1 (Object Multiplex)** is an additional choice for capable CUDA machines. It is optional: install the `sam3` extra, request access to the gated `facebook/sam3.1` checkpoint, and authenticate with `hf auth login`. SAM 2 remains the default and is not removed.

<div align="center"><span style="color:#4ade80">Recommendation:</span> start with Auto; change the model per shot only when the QC inspector shows a real problem.</div>

---

## 5. Point prompting and segmentation

1. Select the **Point** tool.
2. Left-click the subject to add an include point (green).
3. Right-click the background to add an exclude point (red).
4. Press **Segment Frame**.

The result is a binary matte. The app stores the mask and adds the frame to the keyframe set. A single frame is often enough to start propagation; a second correction frame is useful when the subject changes shape quickly.

### When the mask is wrong

- Add an exclude point on the background that is being included.
- Add an include point inside the missing part of the subject.
- Re-segment the current frame.
- Use the Brush for a small local correction rather than redoing the whole shot.

---

## 6. Brush correction

Press `B` to switch to the Brush tool.

- Paint to add matte pixels.
- Hold `Shift` while painting to erase.
- Use `[` and `]` to resize the brush.
- The **Eraser** checkbox is an alternative to holding `Shift`.
- A brush stroke creates one undo step when the stroke ends.

The Brush works on the current frame and is also used as a prompt for a SAM 2 refinement pass.

---

## 7. Propagation, keyframes, and scene cuts

- **◀ Prop** and **Prop ▶** step the mask one direction.
- **Propagate to End** tracks from the current keyframe toward the end of the shot.
- Keyframe diamonds show where an explicit mask exists.
- **Detect Cuts** finds shot boundaries; propagation stops at a cut.
- Shift-drag the timeline to create an export range.

A cut is useful even if you do not change shots: it gives the shot-profile system a clear processing boundary and prevents flicker from leaking between edits.

---

## 8. Post-processing and shot profiles

**Post-process masks** runs two related operations after propagation:

- edge cleanup, using the guided-filter refinement stage;
- temporal smoothing, using optical flow to align neighbouring masks.

The global **Edge smoothness** slider is a starting point. For mixed footage, open **Inspect → Shot Profiles** (`Ctrl+Shift+P`) and set values per object/shot:

| Setting | Effect |
|---|---|
| Model | Auto or a fixed SAM 2 tier |
| Edge smoothness | 0 is raw; higher values clean the boundary more |
| Temporal smoothing | Reduces frame-to-frame flicker |
| Detail preservation | Protects hair and fine line art from over-smoothing |
| Edge softness | Anti-aliases the exported matte and decontaminates its edge (see §9) |
| Spill cleaner | Removes background colour from the matte edge (see §9) |

Profiles persist inside the project file. They are the recommended way to handle anime, live action, and graphic overlays in the same edit.

The global **Edge smoothness** slider stays editable even when post-processing is off. Change it, then release the slider or press **Apply** to re-run the current shot's cleanup on existing masks. If the model takes the sky with the subject, add a red exclude point on the sky before re-segmenting; the prompt-aware mask selector will prefer a candidate that respects it.

---

## 9. Edge quality and colour spill

Two separate problems often get confused, and they have separate controls. Both live in **Inspect → Shot Profiles** (`Ctrl+Shift+P`) for the current shot, together with a small preview of the current frame.

### Edge softness — jagged edges

An exported matte is normally hard: every pixel is either fully opaque or fully gone. That makes fine hair and motion-blur edges look like a staircase, because a one-pixel strand is simply in or out.

**Edge softness** fixes this. It builds a coverage ramp from the distance to the matte boundary, then subtracts the background's contribution from the partly covered pixels. The result is a smooth edge whose colour is the subject's own rather than a blend of the subject and whatever was behind it.

| Value | Effect |
|---|---|
| 0 | Hard matte — the previous behaviour, useful for a matte-only workflow |
| 2 | **Default.** Right for most work |
| 3–4 | Very soft hair, heavy motion blur |

It only affects exported video. `0` restores the hard matte if you need one downstream.

### Spill cleaner — coloured fringe

A subject shot against a saturated background picks up a fringe of that colour along its silhouette.

1. Set the mode: **Off**, **Despill** (keep the coverage, remove the hue) or **Alpha key** (fade the fringe out of the matte).
2. Press **Pick**, then click the fringe on the preview. The swatch shows what you sampled.
3. Raise **Tolerance** until the fringe goes. Start low; a high tolerance starts eating the subject.
4. **Edge band radius** sets how far either side of the matte edge counts as fringe.

Two further controls matter when only part of the frame is affected:

- **Set Area** — drag a box to clean *only* inside it. Use it for blue sky on the right while blue eyes sit on the left. Inside the box the cleaner may also work over the whole area rather than only the edge, which is what lets a large sky patch that sits well inside the matte be cleared at all.
- **Protect** — drag a box to shield it and clean everywhere else. Use it when part of the subject is genuinely the spill colour, such as blue hair under a blue sky.

Drawing a box with either button replaces the other, so the two cannot contradict each other.

**Fix colour cast** is for a different situation: a strongly tinted light (a violet sky, a green screen) tints the *subject* too, and no amount of edge work recovers a blonde read as blue. Tick it and the cast is measured from your Area box — or from the background — and corrected before cleaning, which also leaves the edge cleaner far less to do.

- Only enable it when the tint is unwanted light. It will neutralise an **intentional** backdrop.
- It needs a **small** Area box over the backdrop. A box covering the whole frame, or a matte that covers nearly everything, gives it nothing useful to measure and it will do almost nothing.

### A note on what this cannot fix

The spill cleaner works by comparing colour. If two things are genuinely the same colour — a cyan eye against a cyan sky — no tolerance setting separates them, and widening it will take the eye with the sky. Use **Protect** for those cases.

Likewise, neither of these tools can repair a matte that has holes in it. If wispy hair is missing from the matte entirely, the fix is a better matte or hand touch-up, not a colour operation. The **Matte QC Inspector** (§11) is the place to check that first.

---

## 10. Clean Up: subtitles and logos

The **Clean Up** section has independent targets:

- **Remove Subtitles** — never uses the logo detector.
- **Remove Logo** — targets static corner/edge watermarks.
- **Remove Both** — runs both pipelines independently.

You can choose automatic detection or draw a rectangle around the overlay. A drawn region is useful for a faint watermark, a moving logo, or a clip the heuristic cannot classify confidently.

### Logo and watermark removal

The logo pipeline understands saturated marks and faint low-opacity white marks. It uses local contrast and temporal stability, expands the detected region slightly, and repairs the mask with a feathered inpaint so anti-aliased edges do not remain as a ghost.

- **Mask to End** pre-cleans the whole clip for smooth scrubbing.
- **Regenerate** tries another deterministic repair variation.
- **Clear Cleanup** removes all active cleanup regions.
- **Check-up** shows which regions are armed and which inpainting engine is active.

The subtitle path remains separate. Removing a logo must not widen or alter subtitle detection.

---

## 11. Matte QC Inspector

Open **Inspect → Matte QC Inspector** (`Ctrl+Shift+M`) on a frame with a mask.

The window shows:

- **Source frame** on the left.
- **Current matte + QC overlay** on the right.
- Color evidence for hair/detail, background spill, holes, and edge flicker.
- Raw-versus-current score values and changed-pixel count.
- **QC Coach**, a local analysis of frame statistics and the current report.

### Mirrored navigation

**Link views** is enabled by default. Zooming or dragging one pane mirrors the other. Turn it off to inspect two areas independently. **Reset view** returns both panes to 100%.

### QC Coach

The coach is local and does not send footage to a service. You can:

1. Select a common complaint or type your own, such as “the hair is too smooth”.
2. Press **Ask** for a direct explanation and setting recommendation.
3. Press **Save as custom tip** to store that answer locally.
4. Press **Apply suggestion to current shot** to apply the recommended profile values to the current shot.

The answer direction is explicit: “too smooth” means lower smoothing and higher detail preservation; “too sharp” means higher smoothing, lower detail preservation, and temporal cleanup.

---

## 12. Performance Monitor

Open **Inspect → Performance Monitor** (`Ctrl+Shift+G`).

The monitor samples live GPU data when the platform exposes it:

- GPU utilization
- used and total VRAM
- temperature and thermal pressure
- estimated render time
- model-fit safety with a reserve margin

<span style="color:#f59e0b">Safety note:</span> A high temperature or insufficient free VRAM can downgrade a model from safe to caution/unsafe. Close the monitor if you are exporting on a laptop and need the GPU for another task.

---

## 13. Export

Press `Ctrl+E`.

| Format | Use |
|---|---|
| PNG Sequence (RGBA) | Frame-by-frame alpha delivery to another compositor |
| ProRes 4444 (MOV) | Production alpha video |
| QuickTime Animation (MOV) | Large alpha files with predictable editing |
| H.264 (MP4) | Review and chroma-key workflows; no alpha |

Choose combined masks, one object, or separate files per object. The range options are **All Masked Frames**, **Timeline Selection**, or a manual frame range.

The export dialog estimates size before starting. Export writes to the selected destination and can apply the current cleanup regions to the source frames first.

**Edge softness and spill are applied at render time**, not baked into your saved masks. That means you can keep adjusting them and re-export without re-running any segmentation, and turning them off restores the original masks exactly. The same transforms apply to the Batch Render Queue, so a queued render always matches what you previewed. See §9.

If you export **All Masked Frames** over a range where some scenes have no mask, the export contains only the frames that do — check the timeline's masked-frames band first if the result is shorter than you expected.

---

## 14. Batch Render Queue

**Export → Batch Render Queue** (`Ctrl+Shift+Q`) stores jobs between sessions.

- Add the current saved project or browse for other `.bbproj` files.
- Save a named preset from the current export settings.
- Set a final destination.
- Start, pause, resume, cancel, retry, or remove jobs.
- Progress and errors are stored per job.
- A job interrupted by a restart returns to **queued** instead of being stranded in **running**.

A project must be saved before it can be queued. The queue renders the masks already stored in the project; it does not re-run segmentation automatically.

---

## 15. Version history and recovery

**Inspect → Version History** (`Ctrl+Shift+H`) lists verified checkpoints.

- Automatic checkpoints are created after edits when a project is dirty.
- Manual checkpoints can be named.
- Checkpoints are checksummed and verified before restore.
- Retention keeps a bounded number of versions.
- The dialog compares the selected mask with the adjacent checkpoint.
- Restore asks for confirmation and keeps the current project path.

Use checkpoints before experimenting with a model change, aggressive smoothing, or a large cleanup region.

---

## 16. Editions and local files

There is only one edition: **free, fully unlocked**. No Lite/Pro split, no
activation, no watermark — multi-object scenes, original audio and
higher-resolution exports are all included. The source is published under
FSL-1.1-MIT (see `LICENSE`).

Everything is local. The app stores projects where you save them, models under `checkpoints/`, and user preferences under the Better Brush application-data directory.

---

## 17. Troubleshooting

| Symptom | Try |
|---|---|
| Mask is too large | Add an exclude point just outside the subject, then re-segment |
| Hair disappears | Lower edge smoothness and raise detail preservation |
| Edge flickers | Enable temporal smoothing, then use a two-frame window |
| Background fringe remains | Use Cutout view, lower smoothness, and add an exclude point |
| Exported edges look jagged | Set **Edge softness** to 2 or higher (§9) |
| Spill cleaner seems to do nothing | Check the mode is not **Off**; the controls stay live, but Off disables the pass |
| The cleaner eats part of the subject | Lower **Tolerance**, or **Protect** a box over that detail (§9) |
| Fix colour cast did nothing | It needs a *small* Area box over the backdrop, not the whole frame (§9) |
| Fix colour cast ruined the backdrop | It is for unwanted light only — untick it if the colour is intentional |
| Colour is gone from a region | A **Protect** box may still be active, or an Area box may be too large (§9) |
| A faint logo remains | Draw a slightly larger region with **Remove Logo**, then **Mask to End** |
| Export is slow | Check the Performance Monitor, reduce resolution, or use a smaller model |
| Export window sits on "Preparing export" | It is decoding every masked frame before rendering; a long clip takes a moment on first run |
| GPU warning | Close other GPU-heavy apps and let VRAM/thermal headroom recover |
| Project will not open | Keep the `.bbproj` beside its video, or use **Locate Video** |
| A long session is risky | Create a named checkpoint before continuing |

---

## 18. A good production habit

<div align="center">

<span style="color:#4ade80">Import → prompt → segment → inspect → correct → checkpoint → export</span>

</div>

Keep the source clip untouched. Make corrections on masks, use a shot profile when the style changes, and let QC/history make the risky decisions recoverable.

<div align="center"><span style="color:#555560">End of user guide</span></div>
