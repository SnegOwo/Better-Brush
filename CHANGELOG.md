# Changelog

All notable changes to Better Brush are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/), versioning is
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Soft matte coverage and edge decontamination on video export.** The exported
  alpha was hard 0/255, which is what made edges read as a staircase: a one-pixel
  hair strand was either fully in or fully gone, and the compositor had no
  partial coverage to anti-alias against. Export now derives a coverage ramp from
  the signed distance to the matte contour (a contour, not a mask blur, so thin
  detail survives) and then un-premultiplies the edge pixels — subtracting the
  estimated background and dividing out the coverage — so a hair tip keeps the
  hair's own colour instead of a blend of hair and sky. Chroma-key exports get
  the same anti-aliased key edge. Measured on `HardSky-Light-VioletBetter05`:
  alpha went from 2 levels and **0 partial pixels** to 6 levels and **14,873**,
  and edge blue-shift from 20.8 to -0.8. Controlled by **Edge softness**
  (0-4 px, default 2) in the Shot Profile dialog; 0 restores the hard matte.
- **Colour controls moved out of the sidebar into the Shot Profile dialog**
  (Ctrl+Shift+P), where eye-softening, spill, cast and edge settings now live
  together. The panel embeds a live preview of the current frame, so the
  eyedropper and the Area/Protect boxes still work while the dialog is open — a
  modal dialog would otherwise have blocked the main canvas. The Segment sidebar
  section is back to just Mask inset.
- **Color Spill Cleaner** for removing background spill from matte edges, with
  an eyedropper that samples the fringe straight off the canvas, a Despill vs
  Alpha Key mode, tolerance and radius controls, and a live preview that needs
  no re-segmentation. Despill pulls the fringe toward its own luminance so
  brightness and shading survive; Alpha Key fades it out of the matte instead.
  Per-shot, and persisted in `.bbproj` alongside the other profile fields.
- **Area and Protect boxes for the spill cleaner.** Draw a box to clean *only*
  inside it, or to shield a box and clean everywhere else. This is the fix for
  the hard case where a subject is genuinely the same colour as the background
  it is shot against — blue hair under a blue sky. Inside a Clean box the
  cleaner may also work over the whole area rather than just the matte edge,
  which is what lets a large sky patch that sits well inside the matte be
  cleared at all.
- **Color cast fix** for shots lit by a strongly tinted light (violet sky, green
  screen), where the subject is tinted too and no amount of edge work recovers
  a blonde read as blue. Measures the cast from your Area box — or from the
  background — and applies a per-channel gain that returns it to neutral while
  preserving overall brightness and local shading. Opt-in, with a strength
  slider, and it runs before the edge cleaner so despill has far less to do.
- **Matte QC Inspector** with a source-frame view, current-matte/QC overlay, linked zoom/pan views, local QC Coach analysis, persistent complaint answers, and `Ctrl+Shift+M` access from the Inspect menu.
- **Faint white logo/watermark removal** with local-contrast detection, padded regions, and lower-residue inpainting. Subtitle cleanup remains a separate, unchanged pipeline.
- **Interactive documentation** covering the full user workflow, developer architecture/techniques, and a 20-idea roadmap with five highlighted priorities.
- **SAM 3.1 as an optional model choice** with lazy adapter support, gated Hugging Face checkpoint resolution, CUDA/VRAM-aware hardware recommendation, and all existing SAM 2 models kept intact.
- **Local public documentation site** in `docs-site/` with responsive navigation, search, shortcuts, FAQ, and public guides only.
- **GPU Performance Monitor** with optional live NVML telemetry, VRAM/thermal safety checks, render-time estimates, and likely-OOM warnings.
- **Batch Render Queue** for persistent multi-project jobs, saved presets, final destinations, progress, pause/resume, cancel, and retry.
- **Per-shot and per-object processing profiles** for model, edge smoothness, temporal smoothing, and detail preservation; profiles persist in `.bbproj` files.
- **Versioned Project History** with automatic/manual checkpoints, retention, checksums, visual comparison, and one-click restore.
- Atomic project writes and recoverable project payloads to reduce corruption risk during long edits.

### Fixed
- **The spill cleaner panel no longer looks dead on a fresh project.** The Pick,
  Set Area and Protect buttons and both sliders were greyed out whenever the
  mode was Off, so a new user saw a disabled panel and pressing Pick did nothing.
  The controls are now always live and switch the mode on for you; only the cast
  strength depends on its checkbox.
- **The spill eyedropper samples a patch, not a single pixel.** It takes the
  per-channel median of a 5x5 neighbourhood, so compression speckle and mosquito
  noise can no longer aim the cleaner at the wrong hue.
- **The cast fix tells you when it has nothing to work with.** Grey-world treats
  whatever it is handed as the neutral reference, so an Area box covering the
  whole frame — or a matte that covers nearly everything — normalises to the
  scene average and does almost nothing. Draw a small box over the backdrop; the
  dialog says as much.
- **The shot profile dialog no longer wipes spill settings.** It builds a fresh
  `ShotProfile` from its own widgets, so pressing Apply silently reset the spill
  fields. They are now read back from the embedded colour panel.
- **Fixed a crash in post-processing.** The per-frame profile resolver returned
  a 4-tuple while `PostProcessWorker` unpacked 5, so post-processing raised
  `ValueError` whenever it was enabled.
- **The spill cleaner actually removes spill now.** The colour match fell off
  linearly from the target hue, so a pixel only scored highly by sitting almost
  exactly on it — and Alpha Key drops a pixel only when the match beats 50%.
  Even at maximum settings the cleaner did very little. Matching now uses a
  plateau, so a wide tolerance clears a wide fringe instead of only its centre
  line: a slightly-off target hue went from a ~0.0 score to 0.81 at tolerance 30,
  and the fringe is essentially neutralized (blue chroma 185 → 4) while the
  subject interior stays byte-identical.
- **Post-processing no longer dies on a profile-resolver arity mismatch.** The
  per-frame profile contract is positional, so a resolver returning the wrong
  number of values was only detected when the grouped settings were destructured
  — after every frame had been smoothed, deep inside a worker thread, where the
  `ValueError` was logged and the user only saw "post-processing failed". The
  resolver is now normalised when the worker is constructed: a legacy 4-tuple is
  accepted and padded with a neutral `mask_inset` of 0, and any other arity raises
  immediately with a message naming the expected shape. The `finished` signal was
  already emitted on both the success and failure paths and the window was already
  re-enabled, so the failure always surfaced rather than hanging.
- **Prompt-aware matte selection and background trimming.** Point segmentation now evaluates SAM 2's mask candidates against positive/negative prompts, limits broad multi-point sky takeovers, and trims a connected negative-point background region before storing the mask.
- **Edge smoothness is now a real per-shot control.** The slider stays editable, updates the active shot profile, and re-applies the value to existing masks on release or the new Apply button.
- **Public documentation site polish.** Cards use the vitrine-style cursor spotlight/hover language, the character slot is ready for the future asset, and sidebar scroll tracking no longer selects the wrong section.
- **Export progress window no longer floats on top of everything.** The
  progress dialog had `WindowStaysOnTopHint`, so it stayed in the foreground
  (even above other applications) during and after an export. It is now a
  normal modal dialog.
- **Masks no longer come out cut / jagged (e.g. hair clipped).** SAM 2's mask
  post-processing — hole-filling and small-island removal — depends on its
  compiled `_C` extension. When that extension is missing (no CUDA toolkit on
  the machine), sam2 silently skipped post-processing on *every* frame, leaving
  masks with holes and rough thin detail (visible with the small model + the
  AMV/anime preset). A scipy-backed `get_connected_components` fallback is now
  installed, so post-processing runs and masks are smooth and complete.

### Changed
- **The soft matte is now snapped back to the mask in flat colour regions.** The
  matte network is trained on photography, where subject and background mix
  gradually; anime is flat colour inside hard lines, so its half-covered answer
  in the middle of a cel was a ghost of the background rather than detail. Where
  the source has no gradient and the mask says subject, the mask now wins — but
  only *inside* the mask, so hair the network finds beyond a loose outline still
  survives, and real drawn edges are still the matte's call. Applied when the
  content profile is **Anime / AMV**.
- **UI polish pass — hierarchy, keyboard access, and dead-control clarity.** No
  layout or workflow changes; every addition is either a theme rule or a small
  component fix.
  - **Keyboard focus is now visible.** Tab navigation previously changed nothing
    on screen, so keyboard users had no idea where they were. Focus rings are
    scoped to `:focus-visible`, so a mouse click does not leave one behind.
  - **Line edits, list/tree/table views, text edits, group boxes and tab bars
    were entirely unstyled** and fell back to platform widgets that clashed with
    the dark panel. `QLineEdit` is used in the Shot Profile dialog, so it was the
    most visible omission.
  - **Disabled combos, spin boxes and checkboxes now look disabled.** Only the
    push button had a disabled state, so greyed-out controls still looked
    interactive and invited clicks that did nothing — the same confusion that
    made the spill panel read as broken.
  - **One primary action per sidebar section.** Segment Frame and Propagate to
    End use a new, deliberately quiet `primary` button style. The loud `accent`
    fill was reserved for Export; filling several sidebar buttons with it turned
    the panel into a pile of blue.
  - **Collapsible section headers respond to hover**, and an open section carries
    an accent-tinted chevron and label, so expanded and collapsed are
    distinguishable at a glance rather than only by the arrow glyph.
  - Slider handles, combo/spin box arrows, menu corners and tooltips were
    refined for consistency.
- Added `scipy` as a runtime dependency (used by the SAM 2 post-processing
  fallback above).
- The quick tutorial and README now cover the QC, profile, monitor, queue, and history workflows.
- Grouped QC, performance, shot-profile, and recovery actions under the new Inspect menu with keyboard shortcuts.
- The tutorial is now interactive with step progress and “Try it now” actions; the completion checkbox was removed.
- Minor UI/UX polish: clearer section hover/collapse affordances, onboarding hints, and tooltips for primary editing/export actions.

### Notes
- For GPU inference, PyTorch must be the CUDA build. `pip install torch` on
  Windows yields a CPU-only wheel (`+cpu`); install from
  `https://download.pytorch.org/whl/cu128` instead.
