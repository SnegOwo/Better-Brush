# AMD GPU support on Windows - plan

> status: proposed

This document plans how Better Brush should behave on a Windows machine whose
only usable GPU is an AMD Radeon (discrete or integrated), and what it would
take to actually run the subject, matting and removal networks on that GPU
rather than on the CPU. It starts from what the shipped code assumes today -
which is almost entirely "an NVIDIA card reachable through `torch.cuda`" - then
stages the work: first stop crashing and tell the truth (stage 1), then decide
whether DirectML or ROCm-on-Windows can carry the models at all (stage 2), then
refactor the scattered device strings into one capability layer (stage 3). It
ends with the decision table we will point users and support at, an honest
statement of where AMD-on-Windows is probably a dead end for SAM 2.1, and the
fallbacks we will offer instead. Effort estimates are in engineer-days; every
stage has an explicit acceptance test.

Nothing in this document is implemented yet. Where a fact depends on what AMD,
Microsoft or PyTorch ship in 2026 rather than on this repository, it is marked
as a claim to verify and is paired with the exact command that settles it.

---

## 1 · What the code assumes today (the CUDA audit)

The application has three device surfaces: the hardware detector/profiler that
chooses a device and a model tier, the model runtimes that move tensors onto it,
and the packaging that ships the `torch` build. All three lean on NVIDIA.

### 1.1 Detection and model selection

| File | Symbol | CUDA assumption |
| --- | --- | --- |
| `src/hardware/detector.py` | `Backend` (CUDA / DIRECTML / CPU) | Only three backends exist; no ROCm, no XPU |
| `src/hardware/detector.py` | `_detect_cuda_gpus` | `torch.cuda.is_available()`, `torch.cuda.get_device_name`, `get_device_capability`, `device_count` |
| `src/hardware/detector.py` | `_fallback_vram` | `torch.cuda.get_device_properties`, `torch.cuda.memory_reserved`, `torch.cuda.memory_allocated` |
| `src/hardware/detector.py` | `_detect_directml_gpus` | Probes `torch_directml` but records `total_vram_mb=0` and a synthetic name |
| `src/hardware/detector.py` | `_pick_best_backend` | Returns `Backend.DIRECTML, f"directml:{i}"` - a device string `torch` cannot resolve |
| `src/hardware/profiler.py` | `_pick_target_gpu`, `_pick_model` | Prefers CUDA; falls back to the first DirectML adapter; `total_vram_mb == 0` -> TINY |
| `src/hardware/profiler.py` | `_supports_amp` | `True` only when `backend is CUDA` and compute capability >= 8 (bf16 tensor cores) |
| `src/hardware/profiler.py` | `_build_user_message` | Says "Aucun GPU detecte - inference sur CPU" when there is no CUDA/VRAM |
| `src/hardware/profiler.py` | `sam3_1_eligible` | Requires a CUDA GPU with >= 15 GB |

Two things fall out of this table immediately, and both are bugs rather than
missing features:

1. **The DirectML path is wired but non-functional.** `_detect_directml_gpus`
   produces an adapter with `total_vram_mb=0`, and `_pick_best_backend` then
   hands out `recommended_device = "directml:<n>"`. That string is passed
   straight into `build_sam2(..., device=...)` and `build_sam2_video_predictor`,
   where `torch` will reject it - `torch_directml` registers its devices under
   `privateuseone` and expects a `torch.device` from `torch_directml.device()`,
   not a `"directml:n"` string. So a user who installs the `directml` extra
   today gets a stack trace from deep inside model loading. Verify with:
   `python -c "import torch_directml as d; print(d.device(), d.device_name(0))"`.
2. **A DML adapter with 0 MB reports as if it were a GPU.** Because
   `_pick_target_gpu` returns it, `_pick_model` takes the `total_vram_mb == 0`
   branch and hides it behind a "No GPU detected" log line, while the device
   string still points at DirectML. Detection and device selection disagree.

### 1.2 Model runtimes

| File | Symbol | CUDA assumption |
| --- | --- | --- |
| `src/models/sam2_manager.py` | `_enable_fast_cuda_paths` | `torch.backends.cudnn.benchmark`, `torch.backends.cuda.matmul.allow_tf32` |
| `src/models/sam2_manager.py` | `_speed_context` | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` when `device.startswith("cuda")` |
| `src/models/sam2_manager.py` | `unload` | `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()` gated on `Backend.CUDA` |
| `src/models/sam2_manager.py` | `load_image_predictor` / `load_video_predictor` | Requires a CUDA GPU for `ModelSize.SAM3_1`; otherwise `device=self.rec.device` |
| `src/processing/matting.py` | `MattingRunner.__init__` | `device: str = "cuda"` default; `_autocast_device` returns `"cuda"` or `"cpu"` |
| `src/processing/matting.py` | `_propagate` | `autocast_enabled = self.device.startswith("cuda")`, `dtype=torch.bfloat16` |
| `src/processing/matting.py` | `unload` | `torch.cuda.empty_cache()` gated on `device.startswith("cuda")` |
| `src/processing/removal.py` | `RemovalRunner` | `device: str = "cuda"`, `torch_dtype=torch.float16`, `is_available` requires CUDA |
| `src/processing/removal.py` | `_free_vram_mb` | `pynvml` + `nvmlDeviceGetMemoryInfo` |
| `src/processing/plates.py` | `_LAMA_KIND` | `"cuda" | "onnx"`; `torch.cuda.is_available()` gates the GPU LaMa path |
| `src/models/sam3_adapter.py` | `_default_device` | `"cuda" if torch.cuda.is_available() else "cpu"` |
| `src/models/sam2_compat.py` | `install_cc_fallback` | Patches the compiled `sam2._C` op `get_connected_components` with a scipy version |
| `src/ui/main_window.py` | `_on_run_matting` | `device="cuda"` hard-coded into `MattingWorker` |
| `src/ui/main_window.py` | `_on_remove_object` | Blocks unless `torch.cuda.is_available()`, then `device="cuda"` |
| `src/ui/main_window.py` | GPU telemetry | `pynvml` NVML utilisation/memory |

### 1.3 Vendored packages

The vendored networks assume CUDA in their own device helpers, so changing only
Better Brush's own code is not enough:

- `src/vendor/matanyone2/utils/device.py` and `src/vendor/matanyone/utils/device.py`:
  `get_default_device()` returns `cuda` when available else `cpu`; `safe_autocast`
  only handles `cuda` and `cpu` device types (MPS and anything else yields with
  autocast off, which happens to be graceful).
- `src/vendor/matanyone2/utils/get_default_model.py` and the MatAnyone twin:
  `else: ... .cuda().eval()` when no device is passed. Better Brush always passes
  a device, so this is latent, not active.
- `src/vendor/efficienttam/build_efficienttam.py`: `device="cuda"` defaults at
  lines 103 and 131, and a compute-capability check at line 138
  (`torch.cuda.get_device_properties(0).major < 8`).
- `src/vendor/efficienttam/utils/misc.py`: `compute_device=torch.device("cuda")`
  defaults in the video-predictor helpers, and a `torch.cuda.get_device_properties`
  probe for flash-attention selection.
- `src/vendor/efficienttam/modeling/position_encoding.py`: `.to("cuda")` when
  CUDA is available.
- `src/vendor/efficienttam/modeling/transformer.py`: `freqs_cis.to("cuda")`
  when CUDA is available.

### 1.4 Packaging

- `pyproject.toml` declares `directml = ["torch-directml>=0.2.0; sys_platform == 'win32'"]`
  as an optional extra, and `requirements.txt` carries the same hint as a comment.
- `better_brush.spec` `collect_all`s `torch`, `torchvision`, `sam2`, `cv2`, `pynvml`
  and friends, and links the CUDA `torch` DLLs into the onedir bundle. There is no
  notion of a CPU-only or ROCm bundle.
- The README install instructions explicitly tell users to install the CUDA wheel
  (`--index-url https://download.pytorch.org/whl/cu128`) because plain
  `pip install torch` is CPU-only on Windows.

**Conclusion of the audit.** The app has exactly one code path that is exercised
in production (NVIDIA CUDA) and one that is described but broken (DirectML). A
CPU path exists and is the current implicit fallback for everything else, and it
works because `sam2_compat.install_cc_fallback` restores the mask
post-processing that the missing compiled extension would have provided.

---

## 2 · What happens on an AMD card today (observed behaviour, by install)

| Install | Detector result | Device handed to models | What the user sees |
| --- | --- | --- | --- |
| CPU `torch` only | No GPUs at all | `"cpu"`, TINY | Works, very slow. Message: "Aucun GPU detecte - inference sur CPU (lente)" |
| `torch-directml` extra | One `DirectML Adapter 0`, `total_vram_mb=0` | `"directml:0"` | Model load raises a `torch` device error; the AE panel shows it as `internal` |
| NVIDIA CUDA | CUDA GPUs with real VRAM | `"cuda:0"` | Normal |
| ROCm `torch` (if the wheel exists) | `torch.cuda.is_available()` is True on ROCm, reported as CUDA | `"cuda:0"`, `Backend.CUDA` | Likely normal; `pynvml` fails and `_fallback_vram` is used |

The ROCm row is the important one and the least obvious: **ROCm presents to
PyTorch as `torch.cuda`.** `torch.version.hip` is set instead of `torch.version.cuda`,
device strings are still `cuda:N`, and most of the app's code will take the CUDA
branch unchanged. That is a gift: if a usable ROCm-on-Windows wheel ships, the
fastest path to support is to label it correctly and stop using NVML, not to
write a new backend.

---

## 3 · Stage 1 - stop crashing and tell the truth

Goal: on any AMD machine, the app either runs on CPU with a correct, specific
message, or refuses with a message that names the hardware and the reason. No
raw `torch` traceback ever reaches the user. This is the smallest useful
increment and it is worth shipping on its own.

### 3.1 Changes

1. **`src/hardware/detector.py` - recognise an AMD/NVIDIA/Intel adapter even
   without a usable backend.** Add a lightweight, dependency-free probe that
   reports the adapter name and VRAM so the profile can say *why* the GPU is
   unusable. Options, in order of preference:
   - `torch.cuda.is_available()` first (covers NVIDIA and ROCm);
   - a `wmi`/`Win32_VideoController` query for `Name` and `AdapterRAM` on Windows
     (no new hard dependency: guard the import and degrade to "unknown adapter");
   - a `torch_directml.device_count()` probe only to say "a DirectML adapter is
     present" rather than to choose it.
   Represent these as a new `Backend.DIRECTML_UNAVAILABLE`-style state, or more
   simply as a `GPUInfo` with `backend=Backend.DIRECTML` and a new
   `usable: bool` field. Recommend the latter: it keeps one GPU list.
2. **`_pick_best_backend` must never return a device string `torch` cannot
   resolve.** Until stage 2 lands, a DirectML adapter is never chosen as
   `recommended_device`; it is recorded but the backend resolves to CPU.
3. **`src/hardware/profiler.py` - `_build_user_message`** gets an AMD branch:
   "AMD Radeon detected. SAM 2 does not run on this GPU yet; processing on the
   CPU (slow). See docs/AMD-WINDOWS-SUPPORT-PLAN.md." The message is currently
   in French; either localise the whole function or keep its new branch
   consistent with whatever the UI actually shows. This is worth a decision, not
   an accident.
4. **`src/models/sam2_manager.py` - guard the load.** In `load_image_predictor`
   and `load_video_predictor`, resolve the device through one helper (stage 3,
   but the stub goes in here) and raise a typed, user-readable error if it is not
   a real `torch` device. Today a bad device string raises an opaque
   `RuntimeError` from inside `hydra`/`torch`.
5. **`src/engine/pipeline.py` - map that failure to `ERR_UNSUPPORTED`.** The
   engine already has a clean error contract (`src/engine/schemas.py`); a device
   that cannot be used should produce
   `RequestError(..., kind=ERR_UNSUPPORTED)` with the AMD message, so the AE
   panel shows a titled card and "Open full app" rather than "internal error".
6. **`src/engine/model_setup.describe()`** already reports `device` and `backend`.
   Add `gpu_kind` (nvidia / amd / intel / none) and `usable_gpu: bool` so the
   panel can grey out the expensive options.

### 3.2 Acceptance test

- Unit (`tests/test_hardware.py`, `tests/test_profiler_policy.py`): feed
  `SAM2Profiler.recommend()` a hand-built `HardwareProfile` with a DirectML
  `GPUInfo` and assert `recommendation.device == "cpu"`, `backend is Backend.CPU`,
  and the AMD message. The profiler takes a profile object, so this needs no GPU
  and no mocking of `torch`.
- Unit: `SAM2Manager` with a recommendation whose `device` is `"directml:0"`
  raises a `RuntimeError` whose message names Better Brush, not `torch`.
- Engine: `tests/test_engine_model_setup.py` gains a case asserting
  `describe()` reports `usable_gpu=False` for a DML-only profile and that a job
  refuses with `kind == "unsupported"`.
- Manual: on a real AMD laptop, `python -m src.engine --port 8765` then
  `curl http://127.0.0.1:8765/health?full=1` shows the AMD adapter and
  `device: "cpu"`, and a job completes (slowly) instead of erroring.

Estimated effort: 2-4 days.

---

## 4 · Stage 2 - can the models actually run on an AMD GPU on Windows?

This is where the honest answer matters more than the plan. There are two
candidate runtimes, and both have a specific, testable blocker.

### 4.1 DirectML (`torch-directml`)

What it is: a Windows-only PyTorch build whose tensors live on a Direct3D 12
device. It historically pins an old `torch` (verify the exact pin with
`pip index versions torch-directml` and `pip download torch-directml -d .` then
read its `Requires-Dist`), it exposes devices under `privateuseone`, and it does
not support `torch.autocast` or the full operator set.

What SAM 2.1 / MatAnyone 2 / EfficientTAM need that DirectML typically does not:

| Operator / feature | Where | DirectML risk |
| --- | --- | --- |
| `torch.autocast("cuda")` | `sam2_manager._speed_context`, `matting._propagate` | Unsupported; already disabled for non-CUDA, so this is fine |
| `F.scaled_dot_product_attention` (flash/mem-efficient) | Hiera and memory attention | Often unsupported or falls back to a slow math kernel |
| Dynamic-shape windowed attention (`window_partition`) | Hiera trunk | Mixed support; shape-specialised graphs can fail |
| `grid_sample` / `interpolate` with non-trivial modes | memory encoder, FPN neck | Partial; nearest/bilinear usually OK |
| `scatter`, `index_put`, `bincount`-style ops | EfficientTAM `misc`, amg | Frequently missing |
| The compiled `sam2._C` op `get_connected_components` | mask post-processing | Absent on any non-CUDA build; `sam2_compat` covers it |
| `torch.compile` | EfficientTAM configs | Disabled in `sam2_manager._ETAM_DISABLE_COMPILE` already |

**The concrete blocker** is not one op; it is that neither SAM 2 nor the
vendored EfficientTAM has been validated on DirectML, and the failure mode is
not "a clean unsupported-op error" but a long fallback chain or a wrong result.
The realistic DirectML deliverable is therefore **not** the video predictor.

Recommended mitigation, in order:

1. **Do not attempt full SAM 2.1 propagation on DirectML.** Treat it as
   unvalidated and keep CPU fallback (stage 1).
2. **If we want *any* GPU acceleration on AMD, target the image predictor
   only** via an ONNX Runtime DirectML export of the SAM 2.1 Tiny encoder +
   decoder. The image path is one `set_image` plus one `predict` per frame, has
   static shapes, and does not touch the memory bank. This gets
   `predict_frame` / `predict_frame_multi` (auto-subject) partly onto the GPU
   while propagation stays CPU. Verify feasibility first:
   `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
   must list `DmlExecutionProvider`.
3. **Prefer EfficientTAM on AMD if it exports.** It is a plain ViT encoder and
   its ONNX export is far more tractable than Hiera plus the memory attention.
   It is already vendored and already selected on weak GPUs.

### 4.2 ROCm on Windows (`torch` ROCm wheels)

What it is: AMD's ROCm stack, and a PyTorch wheel built against it. On Linux this
is mature. On Windows it has been in preview (HIP SDK, RDNA3 targets) and the
availability, the supported GPU list and the wheel URLs change between 2025 and
2026.

Claim to verify before planning around it: AMD's ROCm-on-Windows PyTorch wheels
(and their supported adapter list) as of this writing. Settle it with:
- `https://pytorch.org/get-started/locally/` - does the selector offer a ROCm
  option on Windows, and what is the index URL?
- AMD's ROCm-on-Windows / "TheRock" documentation for the current supported GPU
  list.
- On a candidate machine:
  `python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"`.

If a usable wheel exists, the work is mostly stage 3 plus one real fix:

- **NVML must be optional.** `_detect_cuda_gpus` calls `pynvml.nvmlInit()`
  first; on ROCm that fails and the code already falls back to
  `_fallback_vram` (via `torch.cuda.get_device_properties`), which is correct.
  Test it: the fallback path must not require `nvidia-ml-py` at all.
- **`_fallback_vram` uses the caching allocator** and reports
  `total - reserved - allocated`, which is a fine proxy on ROCm.
- **Flash attention / SDPA** may pick a different backend; SAM 2 selects it
  internally and falls back. Verify a real clip.
- **MatAnyone 2 and MiniMax-Remover** both use `float16`/`bfloat16`
  (`removal.py` hard-codes `torch.float16`). ROCm consumer cards handle fp16
  well, so this is probably fine, but it must be tested on the target card.

### 4.3 The SAM 2 custom-op question, stated plainly

SAM 2's only CUDA-specific *extension* surface we ship against is the compiled
`sam2._C` used by the video predictor's mask post-processing
(`get_connected_components`, and `mask_to_box` in the same extension).
`src/models/sam2_compat.py:install_cc_fallback` already replaces
`get_connected_components` with a scipy implementation on any build where the
extension is missing, and it patches the vendored EfficientTAM copy too. So:

- On DirectML or ROCm, `get_connected_components` is covered.
- `mask_to_box` is **not** currently patched. Confirm whether any code path we
  exercise calls it:
  `rg "mask_to_box" $(python -c "import sam2, os; print(os.path.dirname(sam2.__file__))")`.
  If it does, extend `sam2_compat` with a pure-torch `mask_to_box` before
  claiming non-CUDA support.

If the GPU vendor's ops are the blocker rather than SAM 2's own extension, the
mitigation is the same in both cases: run the parts that are portable (image
encoder, mask decoder) on the GPU through ONNX Runtime DML, and the parts that
are not (memory attention propagation) on the CPU, or move the whole job to the
cloud.

### 4.4 Acceptance test for stage 2

The stage is accepted only when one of the following is demonstrated on real
AMD hardware, and the result is recorded in this document:

- **DirectML route:** `predict_frame_multi` on an ONNX-DML SAM 2.1 Tiny returns
  masks whose IoU against the PyTorch CPU result is >= 0.98 on 20 frames, and the
  auto-subject path picks the same subject on a fixed test clip.
- **ROCm route:** `pytest tests/ -q` is green on the ROCm machine, plus one
  end-to-end job through `python -m src.engine` produces an alpha movie with
  `qc.state == "ok"` and a comparable `flicker` to the CUDA baseline.
- If neither holds: stage 2 is closed as "not reachable on Windows AMD", and the
  documented fallback is CPU masks plus `quality=tiny`.

Estimated effort: DirectML image-path spike 5-8 days (mostly ONNX export and
operator wrangling); ROCm is 1-3 days *if* the wheel exists, otherwise zero.

---

## 5 · Stage 3 - one device abstraction layer

The problem is not that CUDA is hard-coded once; it is that the device is a raw
string that five modules re-derive independently (`"cuda"` defaults in
`matting.py`, `removal.py`, `main_window.py`, `sam3_adapter.py`, and the
`recommended_device` string from the detector). Any AMD work multiplies that.

Proposal: a new `src/hardware/devices.py` with

```python
class DeviceKind(Enum):
    CPU = "cpu"
    CUDA = "cuda"        # NVIDIA, torch.version.cuda
    ROCM = "rocm"        # AMD, torch.version.hip - device string is still cuda:N
    DIRECTML = "directml"
    XPU = "xpu"          # Intel, reserved

@dataclass(frozen=True)
class Capabilities:
    kind: DeviceKind
    torch_device: str          # the only string ever handed to .to()/build_sam2
    name: str
    total_memory_mb: int
    free_memory_mb: int | None
    supports_autocast: bool
    amp_dtype: torch.dtype | None
    supports_bf16: bool
    supports_tf32: bool

def resolve_device(profile: HardwareProfile) -> Capabilities: ...
def memory_snapshot(caps: Capabilities) -> tuple[int, int]: ...  # (total, free)
def empty_cache(caps: Capabilities) -> None: ...
```

Rules that make this worth doing:

1. **`torch_device` is the single source of truth.** No other module builds a
   device string. `SAM2Recommendation.device` keeps its meaning but is populated
   from `Capabilities.torch_device`.
2. **ROCm is CUDA-shaped.** `torch.version.hip` set -> `kind = ROCM`,
   `torch_device = "cuda:N"`, `supports_tf32 = False` (MI-series differs), and
   `memory_snapshot` uses `torch.cuda.mem_get_info` / device properties, never
   NVML.
3. **DirectML is `privateuseone:N`** from `torch_directml.device(i)`, never a
   `"directml:n"` string (fixes the stage-1 audit finding for real).
4. **`_supports_amp` becomes `caps.supports_autocast`.** NVIDIA bf16 only when
   compute capability >= 8 (the current rule, now expressed once); ROCm fp16 via
   `torch.autocast("cuda", dtype=torch.float16)` where validated; DirectML none.
5. **Memory query** moves behind `memory_snapshot`/`empty_cache`, which replaces
   the `torch.cuda.*` calls in `detector._fallback_vram`, `sam2_manager.unload`,
   `matting.unload`, `removal._free_vram_mb`, and the NVML telemetry in
   `src/hardware/monitor.py` and `main_window.py`.
6. **The auto-model-size chooser** keeps `_VRAM_THRESHOLDS` unchanged but reads
   its VRAM from `Capabilities` instead of `GPUInfo.total_vram_mb` directly. A
   device whose `total_memory_mb` is unknown (DirectML, today) is treated as
   "small": TINY or EfficientTAM, the same conservative choice as now.

Do **not** thread `Capabilities` through every call site in one change. Start by
backing it with the existing `Backend`/`GPUInfo` objects so
`resolve_device(profile)` is a pure function of the profile (testable without a
GPU), then migrate one caller per commit: `pipeline._matte`, the two
`main_window` workers, then `sam2_manager`.

### 5.1 Acceptance test for stage 3

- A new `tests/test_devices.py` builds synthetic profiles (NVIDIA cc 8.6, NVIDIA
  cc 7.5, ROCm, DirectML, CPU-only) and asserts `resolve_device` returns the
  expected `DeviceKind`, `torch_device`, and `supports_autocast`.
- `rg '"cuda"' src/ --glob '!src/hardware/devices.py'` returns only test files
  and the vendored tree. That is the mechanical proof the abstraction took.
- `pytest tests/ -q` stays green on CPU (the suite is already GPU-free by
  design).

Estimated effort: 4-6 days including the migration and tests.

---

## 6 · The After Effects panel and the headless engine

The panel is a thin client over `src/engine/server.py`, so almost all of the
above is invisible to it as long as the engine degrades instead of crashing.

- **`GET /health?full=1`** already returns `capabilities.device`,
  `capabilities.backend`, and `capabilities.total_vram_mb` from
  `model_setup.describe()`. Add `gpu_kind` and `usable_gpu` (stage 1). The panel
  can then disable the Quality tier below `fast` or show a one-line warning
  ("Running on CPU - AMD GPU not supported yet") instead of silently taking 10x
  longer.
- **Job errors** must map to `ERR_UNSUPPORTED` (stage 1) so the panel shows a
  titled card with "Open full app" rather than the generic internal-error card.
- **Matting** in the engine is already wrapped so it "must never sink a job"
  (`pipeline._matte` catches everything and ships the outline). On AMD/CPU that
  is exactly the behaviour we want: the mask still comes out, the soft edge does
  not. Confirm the timeout is acceptable: MatAnyone 2 on CPU for a 120-frame
  1080p clip is likely tens of minutes, so the panel should offer to leave
  matting off when `usable_gpu` is false. Recommend: default `matting=false`
  when `aspect.usable_gpu` is false, with the switch still available.
- **The engine's device** is chosen by `model_setup.build_recommendation`, which
  calls `HardwareDetector().detect()`. Once stage 3 lands, this is one call to
  `resolve_device` and the panel needs no change.

---

## 7 · Packaging and distribution

This is the part that decides whether AMD support is a feature or a second
product.

| Approach | Wheel / bundle size | Pros | Cons |
| --- | --- | --- | --- |
| Ship CPU `torch` to everyone | ~800 MB (CPU torch) + ~200 MB app | One bundle, works everywhere, no CUDA DLLs | NVIDIA users get CPU unless they install the CUDA torch themselves |
| Current: CUDA `torch` in the bundle | ~3-4 GB | Fast out of the box for NVIDIA | Useless on AMD; cannot be re-pointed at ROCm |
| Separate AMD/ROCm bundle | ~3-4 GB per channel | Correct for the platform | Two builds, two CI pipelines, two SmartScreen journeys |
| CPU bundle + "bring your own torch" | smallest | One artifact; power users opt in to ROCm/DirectML | Non-technical users get CPU-only |

Recommendation, staged:

1. **Stage 1 has no packaging change.** The CPU fallback already works.
2. **Stage 2, if ROCm works: a second bundle** built in the same `better_brush.spec`
   with `torch` collected from the ROCm install. Guard it with a `--amd` build
   flag that swaps the `collect_all("torch")` source. The `.spec` is already
   parameterised enough to read an env var; the only new work is CI and a
   separate download page that names the supported GPUs.
3. **DirectML stays an extra, never a bundle.** `sam2` is installed from GitHub
   (`pyproject.toml` line 34, `requirements.txt` line 11) and builds its `_C`
   extension only when a CUDA toolkit is present; on DirectML/CPU the scipy
   fallback covers it. So the `sam2` package is available, but it is plain
   Python, and the cost is in `torch`, not `sam2`.

Open packaging risk: `nvidia-ml-py` is a hard dependency today
(`pyproject.toml`, `requirements.txt`). It is harmless on AMD (the import guard
in `_detect_cuda_gpus` catches the failure), but it should become optional in
stage 3 when NVML stops being the primary memory source.

---

## 8 · Testing without the hardware

The suite is already designed to run without a GPU; AMD support must preserve
that.

- **The profiler is pure.** `SAM2Profiler.recommend(profile, ...)` takes a
  `HardwareProfile`, and `tests/test_profiler_policy.py` already constructs
  profiles. Every AMD/ROCm/DirectML selection rule is a unit test with no
  hardware.
- **A fake device.** `tests/test_devices.py` (stage 3) is the home for this: a
  `FakeCapabilities` factory and monkeypatched `torch.version.hip` /
  `torch.cuda.is_available` / a stub `torch_directml` module. Do not reach for a
  real GPU in tests.
- **Headless GUI.** The GUI smoke test already runs offscreen - the documented
  trick is `QT_QPA_PLATFORM=offscreen` (see the QtWebEngine probe note in
  `docs/AFTER-EFFECTS-EXTENSION-PLAN.md` section 19.2 and `tests/test_gui_smoke.py`).
  Verify: `set QT_QPA_PLATFORM=offscreen && python -m pytest tests/test_gui_smoke.py -q`.
- **CI matrix.** One CPU runner is enough for correctness. Add a second runner
  that installs `torch-directml` on a Windows box *without* a DML GPU purely to
  prove the import path and the graceful failure, and a third (optional, GPU
  runners are expensive) that runs one clipped job on real ROCm hardware before
  a release. GitHub-hosted Windows runners have no GPU, so the DirectML runner
  tests detection and refusal, not inference.
- **A real-hardware checklist.** Because ROCm-on-Windows availability is the
  gating unknown, keep a short manual script (`tools/amd_probe.py`, not a test):
  print `torch.__version__`, `torch.version.hip`, `torch.cuda.is_available()`,
  `torch.cuda.get_device_name(0)`, `torch.cuda.mem_get_info()`, and the result of
  one `predict_frame_multi`. If that file is run on an AMD machine and the
  output is attached to an issue, we have the answer.

---

## 9 · Decision table

The table we will put in the docs and support macro. "Tell the user" is the
exact message the app should surface, not an internal note.

| Hardware | Backend | Expected speed (120-frame 1080p, Tiny) | Quality tier | Tell the user |
| --- | --- | --- | --- | --- |
| CPU only | `cpu` | ~10-30 min | TINY only | "No supported GPU found. Processing on the CPU - this is slow. A CUDA NVIDIA GPU is required for real-time work." |
| NVIDIA CUDA (>= 8 GB) | `cuda` | ~30-60 s | SMALL..BASE_PLUS | Normal; `_build_user_message` names the card and VRAM |
| NVIDIA CUDA (< 3.5 GB) | `cuda` | slow | EFFICIENTTAM | "Your NVIDIA GPU has little VRAM; using the lightweight EfficientTAM model. Anime line quality will be lower." |
| AMD Radeon, ROCm wheel installed (Windows, if it ships) | ROCM | 0.5-0.8x of CUDA | SMALL..BASE_PLUS | "Running on your AMD GPU via ROCm. Supported model: SAM 2.1 <tier>." |
| AMD Radeon, no ROCm (the common case) | `cpu` (stage 1) or DIRECTML (stage 2 image-only) | CPU: ~10-30 min; DML: auto-subject only is faster | TINY | "Your AMD GPU is not supported for tracking yet. Masks are computed on the CPU. See docs/AMD-WINDOWS-SUPPORT-PLAN.md." |
| Intel Arc / integrated | `cpu` | ~10-30 min | TINY | "Your Intel GPU is not supported yet. Processing on the CPU." |
| Apple Silicon (out of scope for Windows) | MPS, unimplemented | n/a | n/a | macOS is not a supported platform in v1 |

"Tiny" here is `ModelSize.TINY` (SAM 2.1 Tiny), not a synonym for "fast".

---

## 10 · Milestones

| # | Milestone | Effort | Acceptance |
| --- | --- | --- | --- |
| A1 | Detect an AMD adapter and report it; never select a DirectML device string | 2-4 d | Profiler unit tests pass; `/health?full=1` names the AMD card; model load raises a typed error, not a `torch` traceback |
| A2 | User-facing AMD/CPU message and engine `ERR_UNSUPPORTED` mapping | 1 d | A job on an AMD machine (or a forced-DML profile) fails with `kind=unsupported` and the AMD message |
| A3 | `src/hardware/devices.py` + `resolve_device()` + `tests/test_devices.py` | 4-6 d | No `"cuda"` string outside the abstraction and vendored code; suite green on CPU |
| A4 | ROCm probe: run `tools/amd_probe.py` on real hardware if a wheel exists | 1 d | Probe output attached; go/no-go recorded in this doc |
| A5 | DirectML image-path spike (ONNX Runtime DML, SAM 2.1 Tiny encoder+decoder) | 5-8 d | IoU >= 0.98 vs CPU on 20 frames, same auto-subject pick |
| A6 | (conditional, only if A4 passes) AMD/ROCm bundle and CI channel | 3-5 d | A fresh AMD machine installs the AMD bundle and completes one engine job |

A1+A2 are the "ship something honest" slice and should land regardless of the
later stages.

---

## 11 · What we are not doing

- **We are not making SAM 2.1 video propagation run on DirectML in v1.** The
  memory attention and Hiera paths are unvalidated on DML, and a half-working
  tracker is worse than an honest CPU fallback.
- **We are not writing a native D3D12/Vulkan compute backend.** Out of scope for
  a PySide6 product; if we ever need it, it is ONNX Runtime's job, not ours.
- **We are not supporting Intel Arc/XPU in this plan.** XPU is a separate
  `torch` flavour with its own operator gaps; it gets its own plan if demand
  appears.
- **We are not changing the model-size thresholds** (`_VRAM_THRESHOLDS`). The
  conservative ~35% headroom policy stays; AMD only changes how we query memory,
  not the policy.
- **We are not shipping a separate CUDA-optional installer** in stage 1. The
  existing CPU torch path is the fallback.

---

## 12 · Open questions / how to verify

1. **Does a Windows ROCm PyTorch wheel exist, and for which GPUs?** Verify with
   the official PyTorch install selector and AMD's ROCm-on-Windows docs; on a
   candidate machine run
   `python -c "import torch;print(torch.__version__, torch.version.hip)"`.
   Uncertain as of this writing; this is the single decision the whole plan
   hinges on.
2. **What `torch` version does `torch-directml` pin, and does the app's minimum
   (`torch>=2.3.0`) allow it?** `pip download torch-directml -d .` and read its
   metadata.
3. **Does anything we run call `sam2._C.mask_to_box`?** `rg mask_to_box` inside
   the installed `sam2` package. If yes, `sam2_compat` needs a second fallback.
4. **How slow is MatAnyone 2 on CPU for a realistic clip, and is the engine's
   matting timeout acceptable?** Measure with `--quiet` on the CPU path and
   compare to the CUDA numbers in `AFTER-EFFECTS-EXTENSION-PLAN.md` section 18.2.
5. **Does `pynvml` import cleanly on an AMD box, or does it raise at import?**
   The guard in `_detect_cuda_gpus` is inside the CUDA branch, so it should not
   be reached, but confirm on real hardware.
6. **Is the `_build_user_message` French a bug or intentional?** It is the only
   French string in an otherwise English product; decide before adding a branch
   to it.
