"""Subtitle / watermark cleanup plates.

Detects and removes always-on subtitles and corner logos from video frames:

- ``subtitles`` plates replace pixels that differ from a temporal percentile
  background (text flickers ON/OFF over a static plate).
- ``logo`` plates replace low-temporal-variance pixels (a static watermark),
  so moving foreground behind the logo is preserved.
- ``inpaint`` plates re-mask the overlay on *each* frame and fill it from that
  frame's own surroundings — used for always-on burned-in overlays over moving
  backgrounds where no temporal plate exists.  An optional LaMa AI engine
  (ONNX) reproduces fine line-art texture; when it is absent every path falls
  back to the built-in OpenCV donor-row / Telea repair.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional AI inpainting engine (LaMa).  Loaded lazily; when absent every
# pipeline path falls back to the OpenCV donor/Telea repair.
#   - GPU: big-lama TorchScript through the bundled CUDA torch — runs the
#     region at *native* resolution (~0.1 s per region on an RTX 3060).
#   - CPU: big-lama fp32 ONNX through onnxruntime (fixed 512×512 input).
_LAMA_INSTANCE: _LamaTorch | _LamaEngine | None = None
_LAMA_KIND = ""  # "cuda" | "onnx"
_LAMA_TRYED = False
_LAMA_SLOW = False
_LAMA_CALLS = 0  # inferences served by the current engine
_LAMA_SLOW_RUN = 0  # consecutive over-budget inferences
# Per-region inference budget: a first call slower than this disables the AI
# engine for the rest of the session (a CPU-only machine takes ~10 s per
# 512×512 region, which would stall frame-by-frame exports; a GPU does it
# in ~0.1 s).
_LAMA_BUDGET_S = 2.0
# The GPU engine is fast enough for per-frame use but a whole-band region can
# take a few seconds on a big frame — give it more headroom than the ONNX
# engine (which is a fallback, not a per-frame path).
_LAMA_BUDGET_CUDA_S = 8.0
# Long-side cap for GPU inference (a 2048×2048 fp32 region fits ~6-8 GB).
_LAMA_MAX_SIDE = 2048
# The first inference on a freshly loaded engine pays for kernel selection /
# cuDNN autotuning, so it is never allowed to trip the speed guard, and the
# engine is only retired after this many *consecutive* slow calls (one hiccup
# — another process touching the GPU, a page-fault storm — must not silently
# downgrade the whole session to the OpenCV repair).
_LAMA_WARMUP_CALLS = 1
_LAMA_SLOW_STREAK = 2
# Consecutive inference *errors* (a transient CUDA OOM, a bad frame) before the
# AI engine is retired.  One hiccup must not silently downgrade the whole
# session to the OpenCV repair.
_LAMA_FAIL_RUN = 0
_LAMA_FAIL_STREAK = 3


def _mask_variation(mask: np.ndarray, seed: int) -> np.ndarray:
    """Return a slightly perturbed copy of ``mask`` for a given ``seed``.

    The AI / donor repairs are otherwise fully deterministic, so "regenerate
    with another seed" would change nothing.  A one-pixel dilate / erode (or
    keep) reshapes the carved hole a little, which is enough for LaMa and the
    Telea fill to produce a visibly different texture while staying plausible.
    ``seed == 0`` is the canonical, reproducible carving.
    """
    if not seed:
        return mask
    rng = np.random.default_rng(int(seed))
    op = int(rng.integers(0, 3))
    if op == 1:
        return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if op == 2:
        return cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _find_lama_torch() -> str | None:
    """Locate the big-lama TorchScript archive (GPU engine)."""
    env = os.environ.get("BB_LAMA_TORCH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "checkpoints" / "big-lama.pt")
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "checkpoints" / "big-lama.pt")
    for root in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        candidates.append(root / "checkpoints" / "big-lama.pt")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_lama_model() -> str | None:
    """Locate ``lama_fp32.onnx`` next to the app / in checkpoints."""
    env = os.environ.get("BB_LAMA_MODEL")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "checkpoints" / "lama_fp32.onnx")
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "checkpoints" / "lama_fp32.onnx")
    for root in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        candidates.append(root / "checkpoints" / "lama_fp32.onnx")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def lama_available() -> bool:
    """True when any LaMa engine can actually run."""
    if _find_lama_torch() is not None:
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
    try:
        import onnxruntime  # noqa: F401

        return _find_lama_model() is not None
    except ImportError:
        return False


def inpaint_engine_status() -> str:
    """Short human-readable name of the inpainting engine in use.

    The AI engine is only *proven* after the first real repair; before that
    the status reports the engine that *will* be used (runtime + model
    present).  A first repair slower than :data:`_LAMA_BUDGET_S` downgrades
    the whole session to the OpenCV fallback.
    """
    if _LAMA_SLOW:
        return "OpenCV donor/Telea (LaMa too slow on this machine)"
    if _LAMA_INSTANCE is not None:
        on_gpu = _LAMA_KIND == "cuda" or (
            getattr(_LAMA_INSTANCE, "provider", "CPU") == "GPU"
        )
        return "LaMa (AI, GPU)" if on_gpu else "LaMa (AI, CPU)"
    try:
        import torch

        if torch.cuda.is_available() and _find_lama_torch() is not None:
            return "LaMa (AI, GPU)"
    except ImportError:
        pass
    if _find_lama_torch() is not None:
        return "LaMa (AI, CPU/Torch)"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "OpenCV donor/Telea (AI engine not bundled)"
    if _find_lama_model() is None:
        return "OpenCV donor/Telea (LaMa model missing)"
    return "LaMa (AI, CPU/ONNX)"


def _lama_repair(rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """Repair with LaMa if available, else ``None`` (caller falls back)."""
    global _LAMA_INSTANCE, _LAMA_TRYED, _LAMA_SLOW, _LAMA_KIND
    global _LAMA_CALLS, _LAMA_SLOW_RUN, _LAMA_FAIL_RUN
    if _LAMA_SLOW:
        return None
    if _LAMA_INSTANCE is None:
        if _LAMA_TRYED:
            return None
        _LAMA_TRYED = True
        # 1) GPU TorchScript (native resolution, fast)…
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if has_cuda and _find_lama_torch() is not None:
            try:
                _LAMA_INSTANCE = _LamaTorch()
                _LAMA_KIND = "cuda"
                logger.info("LaMa engine: TorchScript on %s", _LAMA_INSTANCE.device)
            except Exception as exc:  # noqa: BLE001 - optional engine
                logger.warning("LaMa (Torch) failed to load: %s", exc)
                _LAMA_INSTANCE = None
        # 2) …else the ONNX engine (CPU or CUDA EP when available).
        if _LAMA_INSTANCE is None and lama_available():
            try:
                _LAMA_INSTANCE = _LamaEngine()
                _LAMA_KIND = "onnx"
                logger.info(
                    "LaMa engine: ONNX (%s)",
                    getattr(_LAMA_INSTANCE, "provider", "CPU"),
                )
            except Exception as exc:  # noqa: BLE001 - optional engine
                logger.warning("LaMa (ONNX) failed to load: %s", exc)
                _LAMA_INSTANCE = None
        if _LAMA_INSTANCE is None:
            return None
        _warm_up_lama(_LAMA_INSTANCE)
    t0 = time.perf_counter()
    try:
        result = _LAMA_INSTANCE.inpaint(rgb, mask, seed)
    except Exception as exc:  # noqa: BLE001 - AI is optional, never crash
        # A single failure (transient OOM, one malformed frame) used to retire
        # the AI engine for the whole session.  Only give up after a *streak*
        # of consecutive errors, so one hiccup falls back for that frame only.
        _LAMA_FAIL_RUN += 1
        logger.warning(
            "LaMa repair failed (%d/%d) (%s); falling back to OpenCV",
            _LAMA_FAIL_RUN,
            _LAMA_FAIL_STREAK,
            exc,
        )
        if _LAMA_FAIL_RUN >= _LAMA_FAIL_STREAK:
            _LAMA_SLOW = True
            logger.warning(
                "LaMa failed %d times in a row; keeping the OpenCV repair "
                "for the rest of this session.",
                _LAMA_FAIL_RUN,
            )
        return None
    _LAMA_FAIL_RUN = 0
    dt = time.perf_counter() - t0
    budget = _LAMA_BUDGET_CUDA_S if _LAMA_KIND == "cuda" else _LAMA_BUDGET_S
    _LAMA_CALLS += 1
    if _LAMA_CALLS <= _LAMA_WARMUP_CALLS:
        # Warm-up call (kernel selection / cuDNN autotune): never judges.
        logger.info(
            "LaMa warm-up inference: %.2fs (%s engine)", dt, _LAMA_KIND or "?"
        )
        return result
    if dt > budget:
        _LAMA_SLOW_RUN += 1
        if _LAMA_SLOW_RUN >= _LAMA_SLOW_STREAK:
            _LAMA_SLOW = True
            logger.warning(
                "LaMa inference took too long for per-frame use "
                "(%.1fs > %.1fs, %d in a row); falling back to the OpenCV "
                "donor/Telea repair.",
                dt,
                budget,
                _LAMA_SLOW_RUN,
            )
    else:
        _LAMA_SLOW_RUN = 0
    return result


def _warm_up_lama(instance: object) -> None:
    """Run one throwaway inference so the first real frame is not penalised."""
    tiny_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    tiny_mask = np.full((64, 64), 255, dtype=np.uint8)
    try:
        instance.inpaint(tiny_rgb, tiny_mask, 0)
    except Exception as exc:  # noqa: BLE001 - warm-up is best effort
        logger.debug("LaMa warm-up failed (%s); continuing", exc)


class _LamaEngine:
    """Lazily-loaded LaMa (big-lama) ONNX inpainting engine.

    Prefers a GPU execution provider when the installed onnxruntime offers
    one (CUDA / TensorRT — the ``onnxruntime-gpu`` build) and falls back to
    CPU. The fp32 model runs at 512×512 either way.

    Model I/O conventions (verified against Carve's ONNX demo):
    - ``image`` input: float [0, 1] in CHW layout (NOT [-1, 1]).
    - ``mask`` input: float, 1.0 = hole, 0.0 = keep.
    - ``output``: already float 0..255 in HWC after transpose — cast to uint8.
    """

    #: Execution providers to try, best first. Only the ones the installed
    #: onnxruntime actually ships are used.
    _PROVIDER_PREFERENCE = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(self) -> None:
        import onnxruntime as ort

        model = _find_lama_model()
        if model is None:
            raise FileNotFoundError("lama_fp32.onnx not found")
        available = set(ort.get_available_providers())
        chosen = [
            p for p in self._PROVIDER_PREFERENCE if p in available
        ] or ["CPUExecutionProvider"]
        session = ort.InferenceSession(model, providers=chosen)
        self._inputs = [i.name for i in session.get_inputs()]
        self._output = session.get_outputs()[0].name
        self._sess = session
        active = session.get_providers()[0] if session.get_providers() else chosen[0]
        self.provider = (
            "GPU" if "CPU" not in active else "CPU"
        )
        logger.info("LaMa ONNX providers: %s", chosen)

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
        """Inpaint ``mask`` (255 = hole) over ``rgb`` (h, w, 3) → RGB copy."""
        if cv2.countNonZero(mask) == 0:
            return rgb.copy()  # nothing to fill — skip the 512×512 inference
        mask = _mask_variation(mask, seed)
        if cv2.countNonZero(mask) == 0:
            return rgb.copy()
        h, w = rgb.shape[:2]
        scale = 512.0 / max(h, w)
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        # Pad to the model's fixed 512×512 input (reflect keeps edge texture).
        if nh < 512 or nw < 512:
            img = cv2.copyMakeBorder(img, 0, 512 - nh, 0, 512 - nw, cv2.BORDER_REFLECT_101)
            m = cv2.copyMakeBorder(m, 0, 512 - nh, 0, 512 - nw, cv2.BORDER_CONSTANT, value=0)
        inp = img.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]  # (1, 3, H, W) float [0,1]
        msk = (m.astype(np.float32) / 255.0)[None, None]  # (1, 1, H, W) 1=hole
        feed = {"image": inp, "mask": msk}
        out = self._sess.run([self._output], feed)[0]
        out = np.clip(out[0], 0, 255).astype(np.uint8)  # (3, H, W)
        out = np.transpose(out, (1, 2, 0))
        out = out[:nh, :nw]
        if (h, w) != (nh, nw):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out


class _LamaTorch:
    """big-lama TorchScript inpainting engine (CUDA when available).

    The TorchScript export is fully convolutional, so the region is
    inpainted at its **native** resolution (padded to a multiple of 8) —
    no 512×512 downscale, which is what made the ONNX path look mushy on
    1080p content.  Model I/O: ``image`` float [0, 1] CHW, ``mask`` float
    1.0 = hole CHW; the output is [0, 1] (official jit export).
    """

    def __init__(self) -> None:
        import torch

        model = _find_lama_torch()
        if model is None:
            raise FileNotFoundError("big-lama.pt not found")
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(model, map_location=self.device)
        self._model.eval()
        # Figure out the call convention once (kwargs vs dict vs positional).
        # 64×64: the model's FFC levels need a reasonably sized input.
        dummy_img = torch.zeros(1, 3, 64, 64, device=self.device)
        dummy_msk = torch.zeros(1, 1, 64, 64, device=self.device)
        with torch.inference_mode():
            for convention in ("kwargs", "dict", "positional"):
                try:
                    out = self._forward(dummy_img, dummy_msk, convention)
                    if out.shape[0] == 1 and out.shape[1] == 3:
                        self._convention = convention
                        break
                except Exception as exc:  # noqa: BLE001 - probe each convention
                    logger.debug("LaMa call convention %s failed: %s", convention, exc)
            else:
                raise RuntimeError("big-lama.pt: no known call convention works")

    def _forward(self, t_img, t_msk, convention: str | None = None):
        conv = convention or self._convention
        if conv == "kwargs":
            return self._model(image=t_img, mask=t_msk)
        if conv == "dict":
            return self._model({"image": t_img, "mask": t_msk})
        return self._model(t_img, t_msk)

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
        """Inpaint ``mask`` (255 = hole) over ``rgb`` (h, w, 3) → RGB copy."""
        if cv2.countNonZero(mask) == 0:
            return rgb.copy()
        mask = _mask_variation(mask, seed)
        if cv2.countNonZero(mask) == 0:
            return rgb.copy()
        torch = self._torch
        h0, w0 = rgb.shape[:2]
        # Cap the long side for VRAM safety, then pad to a multiple of 8.
        scale = min(1.0, _LAMA_MAX_SIDE / max(h0, w0))
        if scale < 1.0:
            rgb = cv2.resize(
                rgb, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA
            )
            mask = cv2.resize(
                mask, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_NEAREST
            )
        h, w = rgb.shape[:2]
        # Quantise the working shape: TorchScript/cuDNN pay a one-off per-shape
        # cost (~2 s) for every *new* input size, and the per-frame mask bbox
        # jitters by a few pixels — without quantisation every frame pays it.
        # Rounding up to a 128-px ladder (edge padding, never rescaling) gives
        # the engine a handful of shapes, each warm after its first frame.
        q = 128
        qh = max(q, -(-h // q) * q)
        qw = max(q, -(-w // q) * q)
        ph, pw = qh - h, qw - w
        img = np.pad(
            rgb.astype(np.float32) / 255.0, ((0, ph), (0, pw), (0, 0)), mode="edge"
        )
        msk = np.pad(mask.astype(np.float32) / 255.0, ((0, ph), (0, pw)))
        t_img = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        t_msk = torch.from_numpy(msk)[None, None].to(self.device)
        with torch.inference_mode():
            out = self._forward(t_img, t_msk)
        out = out[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        if float(out.max()) <= 1.001:  # [0, 1] output (official jit export)
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)[:h, :w]
        if scale < 1.0:
            out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return out


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class DirtyPlate:
    """A prepared cleanup region used to patch frames."""

    mode: str  # "subtitles" | "logo" | "inpaint"
    region: tuple[int, int, int, int]  # (x, y, w, h)
    clean: np.ndarray  # temporal plate with the overlay removed (h, w, 3)
    var_map: np.ndarray  # temporal per-pixel variance (h, w)
    t_static: float = 10.0
    min_diff: float = 8.0
    kind: str = "text"  # "text" | "logo"
    track: bool = False
    ref_mask: np.ndarray | None = None
    # Temporally stable caption-ink reference (same size as ``region``): the
    # pixels that look like caption ink (thin bright strokes with a dark
    # outline) in most sampled frames.  Applying it as a gate keeps the
    # per-frame repair off moving line art, which looks ink-like but is not
    # present in the same place on every frame.
    ink_ref: np.ndarray | None = None
    _bands: list = field(default_factory=list, repr=False, compare=False)


@dataclass
class CleanupResult:
    subtitle_plate: DirtyPlate | None
    logo_plate: DirtyPlate | None
    elapsed_ms: int


# ── Video sampling ───────────────────────────────────────────────────────────


def sample_frames(video_path: str, n: int = 24) -> list[np.ndarray]:
    """Sample ``n`` frames spaced uniformly across the video (RGB uint8)."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            total = max(len(frames), 1)
            frames.clear()
        idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
        # Deterministic stagger: the exact linspace grid phase-locks to a
        # periodic overlay pattern (e.g. a subtitle that is ON only on even
        # frames → every sampled frame would be identical).  Offsetting each
        # sample by its index keeps the sampling deterministic while
        # spreading it over the whole period.
        if n > 3 and total >= 2 * n:
            offsets = np.arange(n) * 7  # odd step → parity alternates
            idxs = (idxs + offsets) % max(total, 1)
            idxs.sort()
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"could not read any frames from {video_path}")
    # Fall back to sequential if seeking failed and we got very few.
    if len(frames) < 2:
        cap = cv2.VideoCapture(str(video_path))
        try:
            while len(frames) < n:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    return frames[:n] if len(frames) > n else frames


# ── Subtitle detection ───────────────────────────────────────────────────────


def _row_energy(gray: np.ndarray, sigma: int = 9) -> np.ndarray:
    """Per-row "text energy": smoothed density of thin bright strokes."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = (mag > 90).astype(np.float32)
    counts = edge.sum(axis=1)
    k = max(3, int(sigma) | 1)
    counts = cv2.GaussianBlur(counts.reshape(-1, 1), (k, 1), 0).ravel()
    return counts


def _caption_ink_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Thin bright strokes ringed by dark ink — the caption signature.

    Burned-in subtitles are white glyphs with a dark outline.  Unlike the
    temporal passes this needs **no static background**: it works on footage
    where every pixel moves (the TestSequence03 failure mode).  Wide bright
    fills (a shirt, a wall) are rejected by the stroke-width and dark-halo
    tests, and scenery outlines are rejected later by
    :func:`_glyph_components`.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = (gray >= 195).astype(np.uint8)
    if not bright.any():
        return None
    # Some very dark ink within a small neighbourhood (the glyph outline).
    dark_near = cv2.erode(gray, np.ones((5, 5), np.uint8)) < 90
    # Thin strokes only: distance to the background inside the bright mask.
    dist = cv2.distanceTransform(bright, cv2.DIST_L2, 3)
    thin = dist <= 4.0
    mask = ((bright > 0) & thin & dark_near).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def _caption_blocks(ink: np.ndarray) -> np.ndarray:
    """Keep only ink strokes that cluster into caption-like text blocks.

    Scattered anime line art (rim light, specular highlights) passes the
    thin-bright-stroke test but never forms a *word*: a short, wide
    horizontal block of sparse strokes.  Bridging small horizontal gaps and
    keeping only such blocks removes the line art while the caption — many
    parallel strokes in a line — survives intact.
    """
    h, w = ink.shape[:2]
    words = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(words, 8)
    if n <= 1:
        return np.zeros_like(ink)
    keep = np.zeros_like(ink)
    max_h = max(40, int(0.9 * h))  # a caption line inside this region
    for i in range(1, n):
        bx, by, bw_, bh_, _area = (int(v) for v in stats[i])
        if bh_ < 8 or bh_ > max_h:
            continue  # too thin to be a text line, or taller than a caption
        if bw_ < 40 or bw_ > 0.98 * w:
            continue  # a speck, or something spanning the whole width
        sub = labels[by : by + bh_, bx : bx + bw_] == i
        strokes = sub & (ink[by : by + bh_, bx : bx + bw_] > 0)
        ink_px = int(strokes.sum())
        fill = ink_px / max(1, bw_ * bh_)
        if fill > 0.6:
            continue  # a solid shape (billboard, glare), not glyph strokes
        if fill < 0.02:
            continue  # a lone line-art stroke, not a word
        region = keep[by : by + bh_, bx : bx + bw_]
        region[strokes] = 255
    return keep


def _overlay_mask_for_frame(
    region_rgb: np.ndarray, glyph_only: bool = True
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Per-frame overlay mask for the repair, plus the repair bbox.

    Uses the deviation/glyph mask (glyphs deviate sharply from a local
    background estimate), which measured ~95 % caption-ink removal on real
    footage vs ~74 % for the thin-bright-stroke signature — the stroke
    signature traces anime line art as well as glyphs, so it smears scenery.

    The second element is the bbox (x0, y0, x1, y1) the caller should repair,
    so the engine runs on a small patch instead of the whole band.  ``None``
    means "use the whole region".
    """
    mask = _text_mask_per_frame(region_rgb, glyph_only=glyph_only)
    if mask is None or cv2.countNonZero(mask) == 0:
        return mask, None
    ys, xs = np.nonzero(mask)
    h, w = region_rgb.shape[:2]
    pad = 10
    bbox = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + 1 + pad),
        min(h, int(ys.max()) + 1 + pad),
    )
    return mask, bbox


def _row_density(mask: np.ndarray, sigma: int = 7) -> np.ndarray:
    """Per-row density of candidate pixels, smoothed over neighbouring rows."""
    counts = (mask > 0).astype(np.float32).sum(axis=1)
    k = max(3, int(sigma) | 1)
    return cv2.GaussianBlur(counts.reshape(-1, 1), (k, 1), 0).ravel()


def detect_subtitle_band(frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """Locate the subtitle band (x, y, w, h) in the sampled frames.

    Three complementary signals are combined, strongest first:

    - **Caption ink** (``_caption_ink_mask``): thin bright strokes with a
      dark outline, vote-aggregated over frames.  Needs **no static
      background**, so it is the only pass that survives footage where every
      pixel moves (hard-cut anime with burned-in subtitles).
    - **Static text**: the median frame's stroked line-art via the glyph
      filter (rejects solid scenery / 1 px bezel lines).
    - **Temporal residual** (``|frame − median|``): text that alternates
      ON/OFF over a static background.

    Every candidate is capped to a plausible caption size, so a scene-wide
    motion response can never arm a full-frame region.
    """
    if len(frames) < 4:
        return None
    step = max(1, len(frames) // 24)
    sel = frames[::step][:24]
    if len(sel) < 3:
        sel = frames
    H, W = sel[0].shape[:2]
    gray = np.mean(np.stack(sel), axis=3).astype(np.float32)  # (n, H, W)
    med = np.median(gray, axis=0)
    resid = np.mean(np.abs(gray - med), axis=0)  # (H, W)

    # 0) Caption ink *blocks*: thin bright strokes with a dark outline that
    #    cluster into word-shaped blocks (scenery line art does not).  Row
    #    densities are averaged over frames, so a caption whose text changes
    #    every shot — and footage that hard-cuts — is still found by its
    #    stable row band.
    block_scores: list[np.ndarray] = []
    block_union = np.zeros((H, W), np.uint8)
    for fr in sel:
        ink = _caption_ink_mask(fr)
        if ink is None:
            continue
        blocks = _caption_blocks(ink)
        if cv2.countNonZero(blocks) == 0:
            continue
        block_scores.append(_row_density(blocks))
        block_union |= blocks
    ink_score = (
        np.mean(block_scores, axis=0).astype(np.float32)
        if block_scores
        else np.zeros(H, dtype=np.float32)
    )
    ink_agg = (
        _glyph_components(block_union, gray=med) if cv2.countNonZero(block_union) > 0 else None
    )

    # 1) Flicker / alternating text → high temporal residual, text-shaped.
    #    A solid residual *block* (the whole text area lights up) is the
    #    signal — counting only Sobel edges would see the two block borders
    #    and reject a 16-row caption that spans a few hot rows only.
    resid_text = (resid > 25.0).astype(np.uint8) * 255  # per-pixel candidates
    resid_score = _row_density(resid_text)
    # On hard-cut / whole-frame-motion footage nearly every pixel lights up,
    # so the residual says nothing about subtitles — trusting it arms a
    # scenery band in the middle of the frame (the TestSequence03 failure).
    resid_usable = float((resid > 25.0).mean()) < 0.5
    # 2) Always-on stroked text → glyph mask on the median frame.
    med_rgb = np.dstack([med.astype(np.uint8)] * 3)
    glyph = _text_mask_per_frame(med_rgb, glyph_only=True)
    static_score = (
        np.zeros(H, dtype=np.float32)
        if glyph is None
        else (cv2.reduce((glyph > 0).astype(np.float32), 1, cv2.REDUCE_SUM)).ravel()
    )
    # 2) Always-on stroked text → glyph mask on the median frame.
    med_rgb = np.dstack([med.astype(np.uint8)] * 3)
    glyph = _text_mask_per_frame(med_rgb, glyph_only=True)
    static_score = (
        np.zeros(H, dtype=np.float32)
        if glyph is None
        else (cv2.reduce((glyph > 0).astype(np.float32), 1, cv2.REDUCE_SUM)).ravel()
    )

    def _longest_run(row: np.ndarray) -> int:
        """Longest run of ``True`` in a 1-D bool array."""
        best = run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    def _best_band(
        row_scores: np.ndarray, mono: np.ndarray, label: str
    ) -> tuple[int, int, int, int] | None:
        """Rows with consistently high text energy → one horizontal band.

        The band must contain a *wide horizontal run of text pixels* (after
        closing small letter gaps): real captions span a good part of the
        frame width, while isolated moving blobs (a rolling ball, a sparkle)
        stay small and are rejected here.  A candidate taller than a
        plausible caption is trimmed to its densest window instead of being
        rejected (a scene-wide motion response must never arm the whole
        frame).
        """
        thr = max(8.0, 0.4 * float(row_scores.max()))
        hot = row_scores > thr
        # Merge tiny gaps (< 4 rows) so a two-line caption is one band.
        y = 0
        found: tuple[int, int] | None = None
        while y < H:
            if not hot[y]:
                y += 1
                continue
            y0 = y
            while y < H and (hot[y] or (y - y0) < 4 and y0 + 1 < H):
                y += 1
            # require the peaks to actually be spread over ≥ 8 rows
            if sum(row_scores[y0:y] > thr) >= 8 and (y - y0) >= 8:
                found = (y0, y)
                break
        if found is None:
            return None
        y0, y1 = found
        # Caption-size cap: trim an oversized run to its densest window
        # (max ~1/3 of the frame height, and never more than ~1/8 of the
        # frame *area* — a real caption band is a small strip).
        max_h = max(24, int(0.32 * H))
        max_h = min(max_h, max(16, int(0.125 * H * W / max(1, W))))
        if (y1 - y0) > max_h:
            best = (float("-inf"), y0)
            for win in range(y0, max(y0 + 1, y1 - max_h + 1), 2):
                s = float(row_scores[win : win + max_h].sum())
                if s > best[0]:
                    best = (s, win)
            logger.info(
                "Subtitle band (%s) trimmed %d→%d rows around y=%d",
                label, y1 - y0, max_h, best[1],
            )
            y0, y1 = best[1], best[1] + max_h
        # Columns of the text inside the winning rows.
        region = mono[y0:y1]
        ys, xs = np.nonzero(region)
        if ys.size == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        # Tighten the x-extent to columns that carry a real vertical run of
        # overlay pixels: glyph strokes span many rows, while stray line art
        # contributes 1-2 px per column and would stretch the band to the
        # whole frame width.
        col_counts = (region > 0).sum(axis=0)
        keep_cols = col_counts >= max(2, int(0.06 * (y1 - y0)))
        if keep_cols.any():
            kept = np.flatnonzero(keep_cols)
            x0, x1 = int(kept.min()), int(kept.max()) + 1
        w = x1 - x0
        if w < max(0.18 * W, 40):
            return None
        # Horizontal continuity: close letter gaps (≤ 8 px) and require the
        # median row to carry a long text run — a wide *scatter* of unrelated
        # motion blobs is not a caption.
        band_mono = (region > 0).astype(np.uint8)
        closed = cv2.morphologyEx(band_mono, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
        runs = [_longest_run(row > 0) for row in closed]
        need = max(0.20 * W, 40)
        if float(np.median(runs)) < need:
            logger.debug("subtitle band rejected (%s): median run < %.0f px", label, need)
            return None
        x_pad = 6
        y_pad = max(3, (y1 - y0) // 8)
        # Clamp to the frame *before* returning.  Callers index the score
        # arrays with ``band[1] : band[1] + band[3]``; an un-clamped
        # ``y0 - y_pad`` (negative for a caption hugging the top edge) made
        # NumPy wrap to the end of the array and score the wrong rows, which
        # promoted the wrong band.  Mirrors ``_clamp`` below.
        bx = max(1, min(x0 - x_pad, W - 1))
        by = max(1, min(y0 - y_pad, H - 1))
        bw = max(8, min(w + 2 * x_pad, W - bx))
        bh = max(8, min((y1 - y0) + 2 * y_pad, H - by))
        return (bx, by, bw, bh)

    mono = glyph if glyph is not None else (resid_text > 0).astype(np.uint8) * 255
    resid_mono = cv2.morphologyEx(resid_text, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))

    def _pos_weight(band: tuple[int, int, int, int]) -> float:
        """Caption-position prior: burned-in captions live at an edge.

        Ripless subtitle files put the text in the lower third (occasionally
        the top for signage/honorifics).  A text-shaped response in the
        middle of the frame is far more likely to be scenery than a caption,
        so it needs a much stronger signal to win.
        """
        _x, y, _w, h = band
        cy = (y + h / 2.0) / max(1.0, float(H))
        if cy >= 0.62:
            return 1.0
        if cy <= 0.20:
            return 0.9
        return 0.35

    candidates = []
    if ink_agg is not None:
        band = _best_band(ink_score, ink_agg, "ink")
        if band:
            # Most specific signal: prefer it unless another pass is far
            # stronger (×1.5 weight).
            candidates.append(
                (
                    "ink",
                    band,
                    1.5 * _pos_weight(band) * float(ink_score[band[1] : band[1] + band[3]].sum()),
                )
            )
    band = _best_band(static_score, mono, "static")
    if band:
        candidates.append(
            (
                "static",
                band,
                _pos_weight(band) * float(static_score[band[1] : band[1] + band[3]].sum()),
            )
        )
    if resid_usable:
        band = _best_band(resid_score, resid_mono, "resid")
        if band:
            candidates.append(
                (
                    "resid",
                    band,
                    _pos_weight(band) * float(resid_score[band[1] : band[1] + band[3]].sum()),
                )
            )

    def _clamp(band: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = band
        x = max(1, min(x, W - 1))
        y = max(1, min(y, H - 1))
        w = max(8, min(w, W - x))
        h = max(8, min(h, H - y))
        return (x, y, w, h)

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2], reverse=True)
    return _clamp(candidates[0][1])


# ── Logo detection ───────────────────────────────────────────────────────────


def _logo_blob_from_image(
    image_rgb: np.ndarray,
    var_map: np.ndarray | None = None,
    static_limit: float = 18.0,
) -> np.ndarray:
    """Return a logo/overlay mask including faint low-opacity white marks.

    A logo is not necessarily saturated or pure white: a 20–40 % white
    watermark still lifts local luminance and has almost no temporal variance.
    Strong color/brightness evidence is combined with a conservative
    positive local-contrast gate so those faint marks are not left behind.
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1]
    sigma = max(3, (min(h, w) // 12) | 1)
    local_bg = cv2.GaussianBlur(value, (0, 0), sigmaX=float(sigma))
    deviation = value - local_bg
    positive = np.clip(deviation, 0.0, None)
    p95 = float(np.percentile(positive, 95)) if positive.size else 0.0
    noise = float(np.percentile(np.abs(deviation), 60)) if deviation.size else 0.0
    faint_threshold = max(5.0, min(24.0, max(p95 * 0.40, noise * 1.5)))
    strong = (value > 180.0) | (saturation > 100)
    faint = (value > 95.0) & (deviation >= faint_threshold)
    mask = (strong | faint).astype(np.uint8) * 255
    if var_map is not None and var_map.shape == mask.shape:
        mask[var_map > max(4.0, float(static_limit))] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return mask


def _static_bright_blobs(
    frames: list[np.ndarray], subtitle_rect: tuple[int, int, int, int] | None = None
) -> list[tuple[int, int, int, int, float]]:
    """Connected blobs that are temporally static AND bright/saturated.

    Returns ``(x, y, w, h, score)`` sorted by descending score.  The score
    rewards a blob that is square-ish (not a wide subtitle band), reasonably
    sized, and near a frame corner (a logo's favourite spot) — while still
    accepting mid-edge / centre logos.
    """
    step = max(1, len(frames) // 24)
    sel = frames[::step][:24]
    if len(sel) < 3:
        sel = frames
    H, W = sel[0].shape[:2]
    stack = np.stack(sel).astype(np.float32)
    var_map = np.mean(np.var(stack, axis=0), axis=2)  # (H, W)
    med = np.median(stack, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1]
    local_bg = cv2.GaussianBlur(v, (0, 0), sigmaX=max(3.0, min(H, W) / 24.0))
    deviation = v - local_bg
    positive = np.clip(deviation, 0.0, None)
    p95 = float(np.percentile(positive, 95))
    noise = float(np.percentile(np.abs(deviation), 60))
    faint_threshold = max(5.0, min(24.0, max(p95 * 0.40, noise * 1.5)))
    bright = ((v > 170.0) | (s > 100.0) | ((v > 95.0) & (deviation >= faint_threshold))).astype(np.uint8) * 255
    static = (var_map < 8.0).astype(np.uint8) * 255
    cand = cv2.bitwise_and(bright, static)
    if subtitle_rect is not None:
        sx, sy, sw, sh = subtitle_rect
        cand[max(0, sy - 4) : sy + sh + 4, max(0, sx - 4) : sx + sw + 4] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    blobs: list[tuple[int, int, int, int, float]] = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, k]) for k in range(5))
        if area < 12 or w >= W * 0.8 or h >= H * 0.5:
            continue
        ratio = w / max(1, h)
        # A logo is roughly square / emblem-shaped; a wide flat band is a
        # subtitle line, not a watermark.
        if ratio > 3.0 or ratio < 0.25:
            continue
        # Corner bonus: distance to the nearest corner (normalised).
        dx = min(x, W - (x + w)) / max(1.0, W)
        dy = min(y, H - (y + h)) / max(1.0, H)
        corner = 1.0 - (dx + dy) / 2.0
        density = area / max(1, w * h)
        score = corner * (0.5 + 0.5 * density) * np.sqrt(area)
        blobs.append((x, y, w, h, float(score)))
    blobs.sort(key=lambda b: b[4], reverse=True)
    return blobs


def detect_logo_rect(
    frames: list[np.ndarray],
    subtitle_rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Locate a corner / edge watermark (x, y, w, h), or ``None``.

    Uses temporal stillness + brightness, then prefers square-ish blobs near
    frame corners.  Subtitle bands are excluded (wide flat shape) so stroked
    text alone is never reported as a logo: when the caller does not pass a
    band, it is auto-detected here (logo-only mode) and suppressed.
    """
    if subtitle_rect is None:
        # Logo-only mode: auto-suppress a detected caption band so text is
        # never reported as a logo.  Only *wide* bands (≥ half the frame)
        # count as captions — narrow static bright areas (a small corner
        # emblem, a gradient patch) are watermark candidates, not text, and
        # must stay available to the logo detector.
        try:
            band = detect_subtitle_band(frames)
        except Exception:  # noqa: BLE001 - suppression is best-effort
            band = None
        if band is not None and band[2] >= 0.5 * frames[0].shape[1]:
            subtitle_rect = band
    blobs = _static_bright_blobs(frames, subtitle_rect=subtitle_rect)
    if not blobs:
        return None
    x, y, w, h, _ = blobs[0]
    pad = max(4, round(0.08 * max(w, h)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frames[0].shape[1], x + w + pad)
    y1 = min(frames[0].shape[0], y + h + pad)
    return (x0, y0, x1 - x0, y1 - y0)


# ── Plate builders ───────────────────────────────────────────────────────────


def build_subtitle_plate(
    frames: list[np.ndarray],
    rect: tuple[int, int, int, int],
    force_per_frame: bool = False,
) -> DirtyPlate:
    """Clean plate for the subtitle band.

    ``force_per_frame`` (guided subtitle box): always use the per-frame
    inpaint regime so a burned-in caption over a moving background is removed
    even though no temporal plate exists.

    Otherwise the regime is chosen automatically: if the band has text-free
    moments/percentiles a temporal ``"subtitles"`` plate is built (clean =
    low percentile background); if text is present in *every* frame the plate
    switches to ``"inpaint"`` (per-frame glyph masking + donor fill).
    """
    x, y, w, h = (int(v) for v in rect)
    stack = np.stack([f[y : y + h, x : x + w].astype(np.float32) for f in frames], axis=0)
    clean = np.percentile(stack, 10, axis=0).astype(np.uint8)
    var_map = np.mean(np.var(stack, axis=0), axis=2)  # (h, w)

    # Plate quality: if the low-percentile clean plate still carries crisp
    # glyph strokes, the text is burned into *every* frame — no temporal
    # statistic contains a text-free background, so the flicker patch would
    # do nothing.  A caption that *switches* between lines still leaves the
    # p10 plate text-free (each line is absent half the time), so the band
    # is only the flicker regime when most sampled frames are text-free.
    def _has_text(img: np.ndarray) -> int:
        g = _text_mask_per_frame(img, glyph_only=True)
        return 0 if g is None else cv2.countNonZero(g)

    strokes = _has_text(clean)
    gate = max(24, int(0.008 * w * h))
    text_free = sum(1 for fr in frames if _has_text(fr[y : y + h, x : x + w]) < gate)
    always_on = strokes >= gate or text_free < len(frames) // 2

    if force_per_frame or always_on:
        mode = "inpaint"
        track = True
    else:
        mode = "subtitles"
        track = False

    logger.info(
        "Subtitle plate: region=(%d,%d,%d,%d) mode=%s track=%s strokes=%d",
        x, y, w, h, mode, track, strokes,
    )

    # Temporally stable caption ink: vote per-frame caption *block* masks over
    # the sampled frames.  Pixels that look like caption strokes in most
    # frames survive — a caption whose wording changes each shot keeps the
    # same rows but not the same pixels, so a temporally thin reference is
    # only available for a genuinely static caption (in which case the gate
    # is a big precision win on busy footage).
    votes = np.zeros((h, w), np.float32)
    for fr in frames:
        ink = _caption_ink_mask(fr[y : y + h, x : x + w])
        if ink is not None:
            ink = _caption_blocks(ink)
            if cv2.countNonZero(ink) > 0:
                votes += (ink > 0)
    ink_ref: np.ndarray | None = None
    agg = (votes >= 0.5 * len(frames)).astype(np.uint8) * 255
    if cv2.countNonZero(agg) > 0:
        agg = _glyph_components(agg, gray=np.mean(stack, axis=0).astype(np.uint8))
        if agg is not None and cv2.countNonZero(agg) > 0:
            # Pad so a caption that shifts by a few px is still covered.
            ink_ref = cv2.dilate(agg, np.ones((13, 13), np.uint8))
            logger.info(
                "Caption ink reference: %d px (%.2f%% of the band)",
                cv2.countNonZero(agg), 100.0 * cv2.countNonZero(agg) / max(1, w * h),
            )

    return DirtyPlate(
        mode=mode,
        region=(x, y, w, h),
        clean=clean,
        var_map=var_map,
        t_static=10.0,
        min_diff=8.0,
        kind="text",
        track=track,
        ink_ref=ink_ref,
    )


def build_logo_plate(
    frames: list[np.ndarray],
    rect: tuple[int, int, int, int],
    per_frame: bool = False,
) -> DirtyPlate:
    """Clean plate for a static logo region.

    ``per_frame`` (guided logo box): return the plate in ``"inpaint"`` mode
    so each frame re-detects the logo blob from that frame and inpaints it
    away from its own surroundings — works even when the corner background
    moves (which defeats the low-variance static gate).

    Otherwise the plate is the temporal median with the logo blob inpainted;
    apply-time gates on low temporal variance to keep foreground intact.
    """
    x, y, w, h = (int(v) for v in rect)
    stack = np.stack([f[y : y + h, x : x + w].astype(np.float32) for f in frames], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)

    var_map = np.mean(np.var(stack.astype(np.float32), axis=0), axis=2)  # (h, w)

    blob = _logo_blob_from_image(median, var_map=var_map if not per_frame else None)

    if cv2.countNonZero(blob) > 0:
        clean = cv2.inpaint(median, blob, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        logger.info("Logo plate: inpainted blob (%.2f%% coverage)", float(blob.mean()) * 100)
    else:
        clean = median
        logger.info("Logo plate: median used directly (no blob found)")

    mode = "inpaint" if per_frame else "logo"
    kind = "logo"
    # For the per-frame regime the median blob becomes a reference mask:
    # OR-ed into every frame's logo mask so a *semi-transparent* logo gets
    # complete coverage even when a single frame's contrast is too low to
    # isolate it (otherwise only part of the logo is wiped → "reduced
    # opacity" look).
    ref_mask = None
    if per_frame and cv2.countNonZero(blob) > 0:
        ref_mask = cv2.dilate(blob, np.ones((7, 7), np.uint8))
    return DirtyPlate(
        mode=mode,
        region=(x, y, w, h),
        clean=clean,
        var_map=var_map,
        t_static=10.0,
        min_diff=3.0,
        kind=kind,
        ref_mask=ref_mask,
    )


# ── Masking ──────────────────────────────────────────────────────────────────


def _spatial_bg_estimate(gray: np.ndarray, sigma: int) -> np.ndarray:
    """Local background of a region: heavy Gaussian so thin glyph strokes
    disappear while broad scenery / flat anime cells survive."""
    sigma = max(3, int(sigma))
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)


def _text_mask_per_frame(region_rgb: np.ndarray, glyph_only: bool = True) -> np.ndarray | None:
    """Mask of subtitle glyphs inside one frame's region crop.

    Text strokes deviate sharply from the *local* background of the same
    frame (either brighter — white glyphs — or darker — dark outline/ink).
    Because the estimate is spatial, it works regardless of how much the
    background moves between frames.

    ``glyph_only`` additionally removes giant / solid connected components so
    busy anime scenery is never mistaken for text (and never smeared by the
    inpaint).
    """
    h, w = region_rgb.shape[:2]
    gray = np.mean(region_rgb, axis=2).astype(np.uint8)
    # Subtitle strokes are thin (~10–30 px), so the background estimate must
    # stay tight.  A region that grows (tracked band unions two caption
    # lines) must not inflate sigma — that would bleed the strokes into the
    # background and suppress the glyph mask.
    sigma = max(6, (min(h, w) // 8) | 1)
    sigma = min(sigma, 21)
    bg = _spatial_bg_estimate(gray, sigma)
    dev = gray.astype(np.int16) - bg.astype(np.int16)
    absdev = np.abs(dev)
    th = max(16, int(np.percentile(absdev, 90) * 0.6))
    if th <= 0:
        return None
    mask = (absdev > th).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    if cv2.countNonZero(mask) == 0:
        return None
    if glyph_only:
        mask = _glyph_components(mask, gray=gray)
        if mask is None:
            return None
    # Grow slightly so glyph anti-aliasing borders are fully covered.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _glyph_components(mask: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray | None:
    """Keep only text-like connected components (not huge scenery blobs).

    Thresholding against the local background also flags bright anime art
    (glowing eyes, sparkles, rim light).  Glyphs are *strokes*: thin, wide,
    and above all **edge-dense** — the mask pixels are packed with ink
    borders.  Solid scenery (a bright backing panel, a particle burst) is
    smooth: its interior has next to no edges even when it is as wide as a
    text line, so it is rejected — this is what stops the inpaint from
    destroying the clip.  A whole merged multi-line subtitle survives
    because every pixel sits on a stroke edge.

    ``gray`` (optional) is the grayscale image the mask was derived from;
    when given, brightness-uniform components (a solid scenery panel is one
    flat value, while glyph strokes alternate bright ink and dark outline)
    are rejected as well.
    """
    h, w = mask.shape[:2]
    area_band = h * w
    # Local gradient of the *mask* itself: ink borders (alternating
    # bright stroke / dark outline) produce strong interior edges, while a
    # solid scenery fill produces edges only at its boundary.
    gx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        bx, by, cr_w, cr_h, area = (int(stats[i, k]) for k in range(5))
        if area < 6:
            continue
        px = labels == i
        interior = mag[px]
        # Edge density across the masked pixels: text strokes are mostly
        # edge, solid scenery is mostly flat.
        edge_frac = float((interior > 400).mean())
        # A 1-length bezel / grid line that spans the whole band is scenery.
        if cr_w > 0.9 * w and cr_h <= 6:
            continue
        # A component that dominates the band AND is internally smooth is
        # scenery (full-height panel, backing card), not subtitles.
        if (
            cr_h > max(12, h * 0.4)
            and area > area_band * 0.15
            and edge_frac < 0.3
        ):
            continue
        # A smaller component that covers most of its box with no interior
        # edges is scenery too (a solid 200-value backing panel fills > 90 %
        # of its box with a nearly-uniform fill, while glyph strokes never
        # reach that kind of coverage).
        fill = area / max(1, cr_w * cr_h)
        if area > max(220, area_band * 0.05) and fill > 0.82 and edge_frac < 0.3:
            continue
        # Brightness-uniformity check: a solid scenery panel is one flat
        # value; glyph strokes alternate bright ink and dark outline.  This
        # catches the top half of a panel that the box-fill rule above
        # misses (its area can fall under the band-coverage gate).
        if gray is not None and area > 200 and fill > 0.7:
            vals = gray[px].astype(np.float32)
            spread = float(vals.max() - vals.min())
            if spread < 30:
                continue
        # The outline of a solid scenery block (bright backing panel) is
        # edge-dense *and* survives the smooth-interior rules above — but it
        # encloses a huge unmasked hole.  Real glyph strokes never enclose
        # far more background than their own ink, so wide components with a
        # big enclosed hole are scenery.
        if cr_w > 0.5 * w and cr_h < h:
            holes = _enclosed_hole_area(px[by : by + cr_h, bx : bx + cr_w])
            if holes > 0.5 * area:
                continue
        keep[px] = 255
    if cv2.countNonZero(keep) == 0:
        return None
    return keep


def _enclosed_hole_area(comp: np.ndarray) -> int:
    """Number of background pixels fully enclosed by a binary component."""
    ch, cw = comp.shape
    if ch < 3 or cw < 3:
        return 0
    pad = np.zeros((ch + 2, cw + 2), np.uint8)
    pad[1:-1, 1:-1] = (comp > 0).astype(np.uint8)
    ff = pad.copy()
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    mask2 = np.zeros((ch + 4, cw + 4), np.uint8)
    try:
        cv2.floodFill(ff, mask2, (0, 0), 1, flags=flags)
    except cv2.error:  # pragma: no cover - defensive
        return 0
    # After the border flood fill, every remaining 0 in ``ff`` is a hole
    # (background enclosed by the component ink).
    return int(np.count_nonzero(ff[1:-1, 1:-1] == 0))


def _logo_mask_per_frame(
    region_rgb: np.ndarray, ref_mask: np.ndarray | None = None
) -> np.ndarray | None:
    """Mask of a static logo blob inside one frame's region crop.

    Corner watermarks are bright / saturated against the local corner
    background; detected per frame so it works even when the corner itself
    moves or the region was drawn by the user over a busy spot.

    ``ref_mask`` (the median-built blob shape) is OR-ed in so semi-
    transparent logos get *complete* coverage on every frame instead of a
    partial wipe that looks like reduced opacity.
    """
    h, w = region_rgb.shape[:2]
    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1]
    sigma = max(7, (min(h, w) // 6) | 1)
    v_bg = _spatial_bg_estimate(v, sigma)
    deviation = v - v_bg
    positive = np.clip(deviation, 0.0, None)
    p95 = float(np.percentile(positive, 95)) if positive.size else 0.0
    faint_threshold = max(5.0, min(20.0, p95 * 0.40))
    mask = (
        (v > 170.0)
        | (s > 100.0)
        | ((v > 90.0) & (deviation >= faint_threshold))
    ).astype(np.uint8) * 255
    if ref_mask is not None:
        mask = cv2.bitwise_or(mask, ref_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    mask = cv2.dilate(mask, k3, iterations=1)
    if cv2.countNonZero(mask) < 6:
        return None
    return mask


def _mask_band(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (x, y, w, h) bounding box of the non-zero mask pixels."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _feather_alpha(mask: np.ndarray, spread: int = 6) -> np.ndarray:
    """Soft 0..255 alpha map around a binary repair mask.

    Returns 255 inside the (dilated) mask footprint and fades linearly to 0
    over ``spread`` pixels outward.  Compositing the repair with this alpha
    makes the inpainted region melt into the frame instead of ending on a
    hard line where the mask (or the rectangle the user drew) stops.
    """
    k3 = np.ones((3, 3), np.uint8)
    solid = cv2.dilate(mask, k3, iterations=2)
    if spread <= 0:
        return solid.astype(np.uint8)
    distance = cv2.distanceTransform((solid == 0).astype(np.uint8), cv2.DIST_L2, 3)
    ramp = np.clip(spread - distance, 0, spread).astype(np.float32) / spread
    alpha = np.maximum(solid.astype(np.float32) / 255.0, ramp)
    return (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)


def _repair_rows_donor(
    rgb: np.ndarray,
    mask: np.ndarray,
    donor_limit: int = 12,
    use_ai: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Texture-preserving repair using clean donor rows of the same frame.

    Instead of only smearing together the surrounding pixels (Telea),
    every masked row is re-filled from the *nearest clean row above/below*
    in the same frame — a real texture patch with matching line art.  A
    row that has no good donor alignment falls back to the Telea repair.

    ``seed`` adds mild, reproducible randomness (Telea radius, donor blend
    ratio, donor probe columns) so "regenerate with another seed" yields a
    visibly different grain while staying reasonable.  ``seed == 0`` keeps
    the canonical deterministic output.
    """
    _h, w = rgb.shape[:2]
    out = rgb.copy()
    rng = np.random.default_rng(int(seed)) if seed else None
    row_hits = mask.sum(axis=1)
    text_rows = np.where(row_hits > 0)[0]
    clean_rows = np.where(row_hits == 0)[0]
    # Telea radius grows with the largest mask blob so large logo regions
    # are fully filled; the fill only ever touches masked pixels, so a
    # bigger radius never smears unmasked scenery.
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    max_diam = 0
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_WIDTH] + stats[1:, cv2.CC_STAT_HEIGHT]
        max_diam = int(sizes.max())
    radius = int(np.clip(max_diam * 0.35, 4, 24))
    if rng is not None:
        radius = int(np.clip(radius + int(rng.integers(-1, 2)), 3, 26))
    telea_base = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
    base = None
    if use_ai:
        try:
            base = _lama_repair(rgb, mask, seed)
        except Exception as exc:  # noqa: BLE001 - AI is optional, never crash
            logger.warning("LaMa repair failed, using OpenCV repair: %s", exc)
            base = None
    if base is None:
        base = telea_base

    if use_ai and base is not telea_base:
        # LaMa regenerated the *whole* region — its unmasked pixels drift
        # slightly from the source (color/contrast wobble), which is what
        # made old results look like a visible box.  Keep the original
        # everywhere except a dilated, feathered mask so the repair melts
        # into the frame.
        k3 = np.ones((3, 3), np.uint8)
        keep = cv2.dilate(mask, k3, iterations=2)
        alpha = cv2.GaussianBlur(keep, (0, 0), 2.0).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        blend = (
            rgb.astype(np.float32) * (1.0 - alpha) + base.astype(np.float32) * alpha
        )
        return np.clip(blend, 0, 255).astype(np.uint8)

    # Donor-row refinement on top of the (Telea) base.  Donor rows carry real
    # texture but also the donor's grain — a full-strength transplant is
    # *noisier* than Telea's smooth fill on grainy backgrounds.  Blending a
    # modest donor layer into the base keeps the texture structure while the
    # base suppresses the grain (measured: beats plain Telea on PSNR).
    donor_layer = telea_base.copy()
    for y in text_rows:
        cols = np.nonzero(mask[y] > 0)[0]
        if cols.size < 2:
            continue
        up = clean_rows[clean_rows < y]
        down = clean_rows[clean_rows > y]
        if up.size and down.size:
            donor = (
                up[-1]
                if (y - up[-1]) <= (down[0] - y)
                else down[0]
            )
        elif up.size:
            donor = up[-1]
        elif down.size:
            donor = down[0]
        else:
            donor = None
        if donor is None or abs(donor - y) > donor_limit:
            continue
        # Candidate texture patch from the donor row.  Alignment probes must
        # NOT touch masked (text) columns of the target row — those pixels
        # are ink, not texture — so score the *clean* columns only.
        target_ctx = rgb[y, :].astype(np.float32)
        donor_ctx = rgb[donor, :].astype(np.float32)
        clean_cols = np.nonzero(mask[y] == 0)[0]
        if rng is not None and clean_cols.size >= 12:
            pick = np.sort(rng.choice(clean_cols.size, 12, replace=False))
            probes = clean_cols[pick]
        else:
            probes = (
                clean_cols[np.linspace(0, clean_cols.size - 1, 12).astype(int)]
                if clean_cols.size >= 12
                else clean_cols
            )
        best_shift = 0
        best_score = -1.0
        for shift in range(-6, 7):
            score = 0.0
            cnt = 0
            for c in probes:
                pc = int(c) + shift
                if pc < 0 or pc >= w:
                    continue
                # Mean absolute difference across the RGB channels.
                score += float(np.mean(np.abs(target_ctx[c] - donor_ctx[pc])))
                cnt += 1
            if cnt == 0:
                continue
            score = -score / cnt  # smaller difference = better
            if score > best_score:
                best_score = score
                best_shift = shift
        # Transplant the aligned donor texture only where the mask is set.
        for c in cols:
            pc = c + best_shift
            if 0 <= pc < w:
                donor_layer[y, c] = rgb[donor, pc]
    # Blend: ~20 % donor texture over the Telea base — see the comment above.
    # A variable blend ratio (per seed) is what visibly changes the grain when
    # the user hits "Regenerate".
    if rng is not None:
        donor_ratio = float(np.clip(0.12 + 0.16 * rng.random(), 0.05, 0.45))
    else:
        donor_ratio = 0.2
    out = (
        (1.0 - donor_ratio) * telea_base.astype(np.float32)
        + donor_ratio * donor_layer.astype(np.float32)
    ).astype(np.uint8)
    # 1-ring feather at mask edge so the transplants blend with Telea fill.
    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) ^ mask
    if cv2.countNonZero(ring) > 0:
        blend = cv2.GaussianBlur(out, (3, 3), 0)
        out[ring > 0] = blend[ring > 0]
    return out


def _track_text_band(
    frame_rgb: np.ndarray, plate: DirtyPlate
) -> tuple[int, int, int, int]:
    """Follow a moving / resizing subtitle band on the next frame.

    The glyph mask is re-detected in a *window* around the previous band.
    The band keeps the size of the union of the previous band and the newly
    detected text (so a faint second subtitle line stays covered when the two
    lines are not always detected together) while its position chases the
    detected text centre.  Chasing is bounded per frame — the band can travel
    a limited distance and can never stop covering the detected glyphs — so a
    genuinely drifting / resizing caption is followed without the band ever
    jumping to stray bright scenery.

    Growth / travel stays bounded: size is clamped to 0.5x–1.5x of the
    *original* detected band, so a stray bright blob can never expand the band
    to the whole frame and smear unrelated scenery.

    A temporally stable caption ink reference means the caption is *static*
    (its pixels sit in the same place on most sampled frames).  Chasing would
    still allow the band to be pulled by scenery inside the search window on
    the frames where the glyphs fall below the local threshold — the pull is
    bounded per frame but *cumulative*, so over a whole bake the band slowly
    wanders off the caption and the mask "moves" away from the subtitles.
    With a stable reference the band stays pinned to the plate region.
    """
    if plate.ink_ref is not None:
        return plate.region
    fh, fw = frame_rgb.shape[:2]
    px, py, pw, ph = plate._bands[-1] if plate._bands else plate.region
    _, _, ow, oh = plate.region
    # Search window: the text can drift or resize, but only modestly.
    margin_x = max(12, min(90, ow // 5))
    margin_y = max(12, min(60, oh // 2))
    y0, y1 = max(0, py - margin_y), min(fh, py + ph + margin_y)
    x0, x1 = max(0, px - margin_x), min(fw, px + pw + margin_x)
    crop = frame_rgb[y0:y1, x0:x1]
    mask = _text_mask_per_frame(crop)
    if mask is None:
        plate._bands.append((px, py, pw, ph))
        return px, py, pw, ph
    bb = _mask_band(mask)
    if bb is None:
        plate._bands.append((px, py, pw, ph))
        return px, py, pw, ph
    bx0 = bb[0] + x0
    by0 = bb[1] + y0
    bw, bh = bb[2], bb[3]
    # Hull of the previous band and the detected text → size to keep.
    hx0, hy0 = min(px, bx0), min(py, by0)
    hx1, hy1 = max(px + pw, bx0 + bw), max(py + ph, by0 + bh)
    nw = int(np.clip(hx1 - hx0, max(20, int(ow * 0.5)), int(ow * 1.5)))
    nh = int(np.clip(hy1 - hy0, max(12, int(oh * 0.5)), int(oh * 1.5)))
    # Chasing fences: the band may only travel a limited distance from the
    # previous band per frame — small enough that stray detected scenery can
    # never drag the band away, large enough that a genuinely drifting
    # caption is still followed.
    travel_x = max(6, min(33, margin_x // 4))
    travel_y = max(6, min(33, margin_y // 4))
    # The chase target is the density-weighted centroid of the mask, not the
    # bbox centre: subtitles are dense strokes while stray scenery is sparse,
    # so the centroid hugs the real caption instead of drifting to a couple
    # of scene edges on the flanks of the search window.
    dens = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    # ``dens`` is indexed [row=y, col=x].  Summing over the columns (axis=1)
    # yields the row / y-distribution; summing over the rows (axis=0) yields
    # the column / x-distribution.  The two were assigned to the wrong axes,
    # so the weighted "centroid" was transposed — a caption that merely drifted
    # *down* dragged the tracked band *sideways*.
    row_w = (dens.sum(axis=1) * np.arange(dens.shape[0])).sum()
    col_w = (dens.sum(axis=0) * np.arange(dens.shape[1])).sum()
    tot = float(dens.sum())
    if tot > 0:
        cx = int(col_w / tot) + x0
        cy = int(row_w / tot) + y0
    else:
        cx, cy = bx0 + bw // 2, by0 + bh // 2
    nx0 = int(np.clip(cx - nw / 2.0, px - travel_x, px + travel_x))
    ny0 = int(np.clip(cy - nh / 2.0, py - travel_y, py + travel_y))
    # Frame safety.
    nx1 = min(nx0 + nw, fw)
    nx0 = max(0, nx1 - nw)
    ny1 = min(ny0 + nh, fh)
    ny0 = max(0, ny1 - nh)
    best = (nx0, ny0, int(nx1 - nx0), int(ny1 - ny0))
    plate._bands.append(best)
    return best


def _shift_mask(
    mask: np.ndarray, dx: int, dy: int, shape: tuple[int, int]
) -> np.ndarray:
    """Translate a mask by (dx, dy) into a canvas of ``shape`` (h, w)."""
    if dx == 0 and dy == 0 and mask.shape == shape:
        return mask
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    x0, y0 = max(0, dx), max(0, dy)
    sx0, sy0 = max(0, -dx), max(0, -dy)
    cw = min(mw - sx0, w - x0)
    ch = min(mh - sy0, h - y0)
    if cw > 0 and ch > 0:
        out[y0 : y0 + ch, x0 : x0 + cw] = mask[sy0 : sy0 + ch, sx0 : sx0 + cw]
    return out


def _drop_speckles(mask: np.ndarray, min_area: int = 24) -> np.ndarray:
    """Remove tiny overlay blobs (line-art specks) from a mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if int(stats[i, 4]) >= min_area:
            keep[labels == i] = 255
    return keep if cv2.countNonZero(keep) > 0 else mask


def apply_clean(
    frame_rgb: np.ndarray,
    plates: list[DirtyPlate],
    seed: int = 0,
) -> np.ndarray:
    """Patch a frame using the clean plates.

    - ``"subtitles"`` plates replace every pixel that differs from the clean
      percentile background (text flickers ON/OFF over a static background).
    - ``"logo"`` plates only replace pixels with low temporal variance (the
      static watermark), so moving foreground behind the logo is left intact.
    - ``"inpaint"`` plates re-mask the overlay on *this* frame and inpaint it
      away from this frame's surroundings — used for always-on burned-in
      overlays over moving backgrounds, where no temporal plate exists.  When
      the plate follows a band (``track=True``) the region is re-located each
      frame, so subtitles that move or change size are still removed.

    ``seed`` feeds reproducible variation into the repair (and into a
    different "Regenerate another seed" pass); ``seed == 0`` is canonical.
    The repair is always composed back with a feathered alpha so it fades
    into the frame instead of stopping abruptly at the mask or drawn-box edge.
    """
    out = frame_rgb.copy()
    for plate in plates:
        x, y, w, h = plate.region

        if plate.mode == "inpaint":
            rx, ry, rw, rh = x, y, w, h
            if plate.track and plate.kind == "text":
                rx, ry, rw, rh = _track_text_band(frame_rgb, plate)
            fh, fw = frame_rgb.shape[:2]
            if plate.kind == "text":
                # Context margin: hand the repair real texture around the
                # band (LaMa fills far better with surrounding line art).
                # The bottom margin is generous — the mask is extended past
                # the box to swallow the glyph outline, so the repair hole
                # must keep clean footage *below* it or the fill comes out
                # as a blurred dark smear.
                my = max(10, rh // 2, 26)
                mx = max(10, rw // 8)
                cx0, cy0 = max(0, rx - mx), max(0, ry - my)
                cx1, cy1 = min(fw, rx + rw + mx), min(fh, ry + rh + my)
            else:
                cx0, cy0, cx1, cy1 = rx, ry, rx + rw, ry + rh
            region = out[cy0:cy1, cx0:cx1]
            if plate.kind == "logo":
                mask = _logo_mask_per_frame(region, plate.ref_mask)
                repair_bbox = None
            else:
                mask, repair_bbox = _overlay_mask_for_frame(region, glyph_only=True)
            # Stable caption coverage: the per-frame glyph mask adapts its
            # threshold to *this* frame's scenery, so on a noisy frame the
            # strokes can fall below it and the text would reappear while the
            # very same static caption was removed on a quieter frame.  The
            # plate's temporally stable ink reference (frame-independent) is
            # overlaid so a static caption is fully covered on every frame —
            # as a fallback when detection comes up empty, and OR-ed in when
            # detection only caught part of the strokes.
            if plate.kind == "text" and plate.ink_ref is not None:
                # ink_ref lives in the plate's own region; the crop is offset
                # by the tracked band's context margin.  Anchor to the plate
                # origin so a drifted tracked band can never move the mask
                # away from the (static) caption.
                ref = _shift_mask(
                    plate.ink_ref,
                    int(x) - int(cx0),
                    int(y) - int(cy0),
                    (region.shape[0], region.shape[1]),
                )
                if mask is None or cv2.countNonZero(mask) == 0:
                    mask = ref
                else:
                    # Temporal gate: only repair where caption ink is *stably*
                    # present.  Keeps a static caption over busy scenery
                    # precise.  The gate must not shave glyph parts that sit
                    # just OUTSIDE the stable-ink reference: the ink votes
                    # miss anti-aliased / thin-stroke borders (tops of
                    # letters, stroke ends), and an exact AND used to cut
                    # those rows out of the mask — the leftover slices then
                    # reappeared as a blurred outline after the repair.  The
                    # mask is therefore kept wherever it is *near* the
                    # reference (±10 px); only responses far from any stable
                    # ink (this frame's scenery false positives) are dropped.
                    near = cv2.dilate(ref, np.ones((21, 21), np.uint8))
                    mask = cv2.bitwise_and(mask, near)
                    mask = cv2.bitwise_or(mask, ref)
                # The per-frame bbox may not cover the reference's extension
                # or the box-bottom fill — derive it from the final mask.
                repair_bbox = None
            elif mask is None or cv2.countNonZero(mask) == 0:
                continue
            mask = _drop_speckles(mask)
            if cv2.countNonZero(mask) == 0:
                continue
            # Cover anti-aliased glyph edges completely — a 1-px fringe of
            # half-covered outline is what reads as "something was here".
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
            # A glyph's dark bottom outline / drop shadow just below the detected
            # strokes often falls under the adaptive threshold and survives
            # as a thin dark line.  Always cover a few rows under the glyphs.
            # The repair feather + Gaussian alpha fades over ~4-12 px, so any
            # outline left *just outside* the mask comes back as a blurred
            # dark smear — for a static caption the mask is pushed well past
            # the drawn box bottom (16 px) so the whole outline and the
            # entire feather ramp sit over real, unmasked background.
            ys, xs = np.nonzero(mask)
            if len(xs) > 0:
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                orig_bottom = int(ys.max())
                pad_end = min(mask.shape[0], orig_bottom + 7)
                if pad_end > orig_bottom + 1:
                    mask[orig_bottom + 1 : pad_end, x0:x1] = 255
                static = plate.ink_ref is not None or not plate.track
                if static:
                    # Push the mask past the drawn box bottom.  Use the
                    # *effective* region actually cropped (``ry``/``rh``) and
                    # the crop offset ``cy0``: for a tracked caption those
                    # follow the moving band, while the fixed ``plate.region``
                    # y mixed coordinate origins and, once the band drifted,
                    # produced a band_bottom above the glyphs so the outline
                    # below them was never covered.
                    band_bottom = min(
                        mask.shape[0], (int(ry) - int(cy0)) + int(rh) + 16
                    )
                else:
                    band_bottom = min(mask.shape[0], orig_bottom + 16)
                cur = int(np.nonzero(mask)[0].max())
                if band_bottom > cur + 1:
                    mask[cur + 1 : band_bottom, x0:x1] = 255
                # The bottom extension exceeds the per-glyph bbox handed back
                # by the detector — derive the repair region from the final
                # mask, otherwise the added rows are never repaired.
                repair_bbox = None
            if repair_bbox is not None:
                bx0, by0, bx1, by1 = repair_bbox
            else:
                ys, xs = np.nonzero(mask)
                bx0, by0 = int(xs.min()), int(ys.min())
                bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
            m_y = max(16, (by1 - by0) // 2)
            m_x = max(16, (bx1 - bx0) // 8)
            py0 = max(0, by0 - m_y)
            py1 = min(region.shape[0], by1 + m_y)
            px0 = max(0, bx0 - m_x)
            px1 = min(region.shape[1], bx1 + m_x)
            patch = region[py0:py1, px0:px1]
            pmask = mask[py0:py1, px0:px1]
            if patch.size == 0 or cv2.countNonZero(pmask) == 0:
                continue
            src_patch = patch.copy()
            repaired = _repair_rows_donor(patch, pmask, seed=seed)
            # Composite the repair back with a soft outer feather.  Inside the
            # mask the repaired pixels win; a few pixels away the original
            # frame wins, so the fill dissolves into the footage instead of
            # ending on a hard cut at the mask / box edge.
            spread = int(np.clip(min(py1 - py0, px1 - px0) / 20.0, 4, 12))
            alpha = _feather_alpha(pmask, spread=spread).astype(np.float32)[..., None] / 255.0
            blended = (
                src_patch.astype(np.float32) * (1.0 - alpha)
                + repaired.astype(np.float32) * alpha
            )
            out[cy0 + py0 : cy0 + py1, cx0 + px0 : cx0 + px1] = np.clip(
                blended, 0, 255
            ).astype(np.uint8)
            continue

        if plate.mode == "subtitles":
            region = out[y : y + h, x : x + w]
            diff = np.mean(
                np.abs(region.astype(np.int16) - plate.clean.astype(np.int16)),
                axis=2,
            )
            change = diff > plate.min_diff
            if not change.any():
                continue
            out[y : y + h, x : x + w] = np.where(
                change[:, :, None], plate.clean, region
            )
            continue

        if plate.mode == "logo":
            region = out[y : y + h, x : x + w]
            static = plate.var_map < plate.t_static
            diff = np.mean(
                np.abs(region.astype(np.int16) - plate.clean.astype(np.int16)),
                axis=2,
            )
            change = diff > max(2.0, plate.min_diff)
            wipe = ((static & change).astype(np.uint8)) * 255
            if cv2.countNonZero(wipe) > 0:
                wipe = cv2.dilate(
                    wipe,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                out[y : y + h, x : x + w] = np.where(
                    (wipe > 0)[:, :, None], plate.clean, region
                )
            continue

    return out


# ── Full pipeline ────────────────────────────────────────────────────────────


def _clip_rect(
    rect: tuple[int, int, int, int], frame_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Intersect a user-drawn rect with the frame (no size-preserving clamp).

    A box that sticks out of the frame is *cropped*, not shifted: the built
    plate must stay inside both the frame and the user's drawn region.
    """
    fh, fw = frame_shape[:2]
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    return (
        x0,
        y0,
        max(4, x1 - x0),
        max(4, y1 - y0),
    )


def detect_cleanup_regions(
    video_path: str,
    targets: str = "both",
    progress_cb=None,
    subtitle_rect: tuple[int, int, int, int] | None = None,
    logo_rect: tuple[int, int, int, int] | None = None,
) -> CleanupResult:
    """Run the full detection pipeline over a video file.

    ``targets`` in {"both", "subtitles", "logo"}.  A user-drawn
    ``subtitle_rect`` / ``logo_rect`` disables auto-detection for that target
    and builds the plate directly (per-frame regime, since the user knows the
    overlay is there even when the heuristic could not find it).
    """
    import time

    t0 = time.perf_counter()
    if progress_cb:
        progress_cb("Sampling frames", 0, 2)
    frames = sample_frames(video_path, n=24)
    if progress_cb:
        progress_cb("Detecting overlays", 1, 2)

    subtitle_plate: DirtyPlate | None = None
    logo_plate: DirtyPlate | None = None

    if targets in ("both", "subtitles"):
        if subtitle_rect:
            rect = _clip_rect(subtitle_rect, frames[0].shape[:2])
            plate = build_subtitle_plate(frames, rect, force_per_frame=True)
            subtitle_plate = plate
        else:
            rect = detect_subtitle_band(frames)
            if rect is not None:
                fh, fw = frames[0].shape[:2]
                if rect[2] * rect[3] > 0.35 * fw * fh:
                    # Safety net: a caption band is a small strip.  Never arm
                    # a scene-wide region — that would inpaint the whole
                    # frame and destroy the shot.
                    logger.warning("Ignoring oversized subtitle band %s", rect)
                    rect = None
                else:
                    subtitle_plate = build_subtitle_plate(frames, rect)
            if rect is None:
                logger.info("No subtitle band detected automatically")

    if targets in ("both", "logo"):
        if logo_rect:
            rect = _clip_rect(logo_rect, frames[0].shape[:2])
            logo_plate = build_logo_plate(frames, rect, per_frame=True)
        else:
            rect = detect_logo_rect(frames)
            if rect is not None:
                logo_plate = build_logo_plate(frames, rect)
            else:
                logger.info("No logo detected automatically")

    if progress_cb:
        progress_cb("Done", 2, 2)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return CleanupResult(subtitle_plate=subtitle_plate, logo_plate=logo_plate, elapsed_ms=elapsed_ms)


def clean_video_frames(
    video_path: str,
    plates: list[DirtyPlate],
    seed: int = 0,
    start: int = 0,
    end: int | None = None,
    on_frame: Callable[[int, np.ndarray], None] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """Apply cleanup plates to *every* frame of a video, sequentially.

    Used by the "Mask to End" flow: pre-inpainting all frames up-front means
    scrubbing the timeline plays back already-cleaned frames instead of
    re-running the repair live (which is chunky and jittery).

    Runs in the calling thread with its *own* ``cv2.VideoCapture`` (never the
    one the UI reads from).  Each cleaned frame is handed to ``on_frame``
    immediately, so a caller can cache it with a bounded LRU — no memory
    explosion on long videos.  Returns the number of frames processed.
    """
    # A fresh bake always starts tracking from the plate region (the tracker
    # is sequential and stateful in ``_bands``; without a reset a repeated /
    # regenerated bake would inherit drift accumulated by the previous run).
    for plate in plates:
        plate._bands.clear()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # CAP_PROP_FRAME_COUNT is unreliable for some containers / live streams
        # and comes back as 0.  Treating that as "no frames" made the bake
        # silently process nothing; ``None`` now means "stream to EOF" so the
        # full video is still cleaned.
        stop = reported if reported > 0 else None
        start = max(0, int(start))
        if end is not None:
            stop = int(end) if stop is None else min(stop, int(end))
        if stop is not None and stop <= start:
            return 0
        # Fast-forward to ``start`` without decoding every frame.
        for _ in range(start):
            if cancel_event is not None and cancel_event.is_set():
                return 0
            if not cap.grab():
                break
        idx = start
        work = (stop - start) if stop is not None else 0
        while stop is None or idx < stop:
            if cancel_event is not None and cancel_event.is_set():
                break
            ok, bgr = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                cleaned = apply_clean(frame, plates, seed=seed)
            except Exception as exc:  # noqa: BLE001 - a bad frame never aborts the bake
                logger.warning("Cleanup bake skipped frame %d: %s", idx, exc)
                cleaned = frame
            if on_frame is not None:
                on_frame(idx, cleaned)
            idx += 1
            if progress_cb is not None:
                progress_cb("Inpainting frames", idx - start, work)
    finally:
        cap.release()
    return idx - start
