# Better Brush — Implemented Features

These five ideas are now implemented in the editor. They remain additive and keep the existing project and export formats backward-compatible. The next-level backlog lives in [docs/ROADMAP-IDEAS.md](docs/ROADMAP-IDEAS.md), with the five highest-impact ideas highlighted.

## 1. Matte QC Inspector

A zoomed edge inspector with mask overlays for hair, background spill, holes, and edge flicker is available from **Inspect → Matte QC Inspector** (`Ctrl+Shift+M`). The left pane shows the untouched source frame, the right pane shows the current matte and QC evidence, and the panes can be linked for mirrored zoom and panning. A local QC Coach answers typed complaints from a persistent tip database and can apply setting suggestions.

## 2. GPU Performance Monitor

The right-side **Hardware → Open Performance Monitor** section shows optional live GPU utilization, VRAM use, estimated render time, temperature, thermal pressure, and whether the selected model has safe headroom. A likely out-of-memory condition is called out before starting a risky render.

## 3. Batch Render Queue

**Export → Batch Render Queue** accepts multiple saved clips/projects, named presets, independent output destinations, progress, pause/resume, cancellation, retry, and restart recovery.

## 4. Per-shot Processing Profiles

**Inspect → Shot Profiles** (`Ctrl+Shift+P`) stores model, edge smoothness, temporal smoothing, and detail-preservation settings per object and scene-cut shot. The active profile is applied to segmentation, propagation, and post-processing and persists in `.bbproj` files.

## 5. Versioned Project History

**Inspect → Version History** (`Ctrl+Shift+H`) provides automatic and manual recoverable checkpoints, visual mask comparison, integrity checks, retention, and one-click restore. Project writes are atomic.
