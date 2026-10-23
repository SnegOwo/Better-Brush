<div align="center">

# Better Brush: 20 Ideas for the Next Level

<span style="color:#f59e0b">A ranked product backlog after the first five roadmap items shipped.</span>

</div>

The ideas below are intentionally concrete. Each one should improve either <span style="color:#4ade80">editor control</span>, <span style="color:#4a9eff">throughput</span>, or <span style="color:#c084fc">production reliability</span>.

---

## The five best picks — status in 1.0.0

<div align="center">

| Rank | Idea | Why it won | Status |
|---:|---|---|---|
| <span style="color:#4ade80">1</span> | <strong>Soft alpha matting</strong> | Fixed the hair/transparency gap at the export boundary | Shipped — MatAnyone 2, guided-filter edges, optical-flow smoothing |
| <span style="color:#4ade80">2</span> | <strong>GPU-accelerated propagation</strong> | Turned long clips into a normal production step | Shipped — CUDA / DirectML propagation |
| <span style="color:#4ade80">3</span> | <strong>Automatic shot-aware profiles</strong> | Removed repetitive tuning while keeping per-shot control | Shipped — scene cuts + Shot Profiles |
| <span style="color:#4ade80">4</span> | <strong>Settings-aware undo/experiments</strong> | Made experimentation safe and fast | Shipped — version history and checkpoints |
| <span style="color:#4ade80">5</span> | <strong>Pre-flight QC report</strong> | Catches bad masks and traces before a long render | Partial — Matte QC Inspector and QC Coach shipped; batch pre-flight report still open |

</div>

### Why these five came first

1. **Soft alpha matting** had the highest visible quality payoff: semi-transparent hair, motion blur and glass kept their information at the edge.
2. **GPU-accelerated propagation** attacked the largest wall-clock cost, making a full-length clip practical on the hardware the app already detects.
3. **Automatic shot-aware profiles** built on the scene-cut and profile model, turning a generic “anime preset” into a measured suggestion.
4. **Settings-aware undo/experiments** closed the loop with the QC Coach: try a suggested profile, compare it, and revert without saving first.
5. **Pre-flight QC report** turned the inspector into a delivery gate for the current shot; a batch report across a whole project is still open.

The remaining ideas below are still open unless marked otherwise.

---

## The complete 20-idea backlog

| # | Idea | User value | Effort | Shape |
|---:|---|---|---|---|
| 1 | <span style="color:#4ade80">Soft alpha matting</span> (shipped) | Preserve hair, motion blur, glass, and semi-transparent edges | High | Guided-filter alpha band or small matting model applied at export |
| 2 | <span style="color:#4ade80">GPU-accelerated propagation</span> (shipped) | Faster long-clip tracking | High | CUDA kernels or optimized ONNX path with a CPU fallback |
| 3 | <span style="color:#4ade80">Automatic shot-aware profiles</span> (shipped) | Less repetitive tuning | Medium | Measure shot statistics and suggest a bounded profile change |
| 4 | <span style="color:#4ade80">Settings-aware undo/experiments</span> (shipped) | Safe experimentation | Medium | Include profiles, cleanup seeds, and model selection in snapshots |
| 5 | <span style="color:#4ade80">Pre-flight QC report</span> (partial) | Catch delivery problems early | Medium | Batch report with pass/warn/fail per shot and export format |
| 6 | Trimap-assisted cleanup | Better logo/subtitle repair on textured footage | Medium | Let the user mark keep/remove boundaries around a guided region |
| 7 | Optical-flow stabilization for QC | Measure true flicker rather than camera motion | Medium | Use the existing flow passes to produce a temporal quality graph |
| 8 | Multi-source reference cleanup | Handle logos that change color or move | Medium | Store a user reference crop and match it per frame |
| 9 | NLE interchange | Open the same work in Premiere/Resolve/AE | High | EDL, XML, or frame-handle metadata plus alpha-aware import |
| 10 | Layered project export | Keep objects, masks, and profiles as separate layers | Medium | Package a project plus per-object PNG/EXR sequences |
| 11 | Color-managed export | Predictable results across apps | High | Display transforms, alpha premultiplication choices, and warnings |
| 12 | Render benchmark history | Make performance estimates trustworthy | Low | Store machine/profile/frame timings and show a p50/p95 estimate |
| 13 | Plugin SDK | Let studios add detectors and encoders | High | Small Python interface for masks, frames, progress, and cancellation |
| 14 | Model manager | Faster and safer checkpoint setup | Medium | Search local/approved model stores, verify hashes, show disk cost |
| 15 | Proxy editing | Keep long projects responsive | Medium | Half-resolution preview masks with full-resolution export |
| 16 | Automatic keyframe suggestions | Reduce repetitive correction clicks | Low | Flag low-confidence frames and offer one-click keyframes |
| 17 | Compare/restore experiments (shipped) | Make “try it” reversible | Low | Named experiment branches with visual diff and one-click restore |
| 18 | Localization | Serve more markets | Medium | String catalog and right-to-left layout pass |
| 19 | Accessibility pass | Make the editor usable in more studios | Low | Focus order, keyboard navigation, high-contrast mode, screen-reader labels |
| 20 | Distilled/fast models | Make weaker GPUs viable | High | Quantized SAM 2 variant with a quality/speed benchmark gate |

> **#9 + #13, v1 shipped:** the After Effects panel
> ([`after-effects/`](../after-effects/README.md)) is the first exercised instance
> of NLE interchange and of the plugin SDK: a shared, headless engine
> ([`src/engine/`](../src/engine)) that any Premiere/Resolve panel would reuse.
> The original plan and the deviations it took are recorded in
> [`AFTER-EFFECTS-EXTENSION-PLAN.md`](AFTER-EFFECTS-EXTENSION-PLAN.md).

---

## Suggested delivery slices

### Slice A — quality

- soft alpha matte behind an experimental flag;
- pre-flight QC report for the first 10 shots;
- export comparison preview for binary vs soft alpha.

### Slice B — speed

- profile the existing Farneback/decoder path;
- benchmark a CUDA propagation kernel;
- add proxy mode only after the fast path is measurable.

### Slice C — confidence

- profile suggestions become bounded edits, never silent changes;
- every suggestion records the measurements that produced it;
- experiment history keeps restore one click away.

### Slice D — interoperability

- export a small alpha-aware project manifest;
- document the frame-handle contract;
- add one reference NLE round trip before expanding the format list.

---

## Product guardrails

<div align="center">

<span style="color:#4ade80">Never trade control for a black box.</span><br/>
<span style="color:#f59e0b">Never change a shot silently.</span><br/>
<span style="color:#4a9eff">Every automatic decision must show why.</span>

</div>

That rule keeps new automation aligned with the existing QC Coach, shot profiles, and versioned history.

<div align="center"><span style="color:#555560">End of roadmap</span></div>
