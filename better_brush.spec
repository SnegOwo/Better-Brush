"""PyInstaller spec for Better Brush (Windows onedir bundle).

Entry point: src/main.py (launches src.core.app:run).
Bundle contains PySide6, PyTorch, SAM 2, OpenCV, the ffmpeg.exe binary and
the assets/ folder — nothing else needs to be downloaded by end users.

Build with:
    pyinstaller --clean better_brush.spec
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

PROJECT_ROOT = Path(SPECPATH).resolve()  # noqa: F821 -- injected by PyInstaller

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []

# Heavy native packages whose data + DLLs + hidden imports must be shipped.
for _pkg in ("sam2", "torch", "torchvision", "cv2", "pynvml", "scipy"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# AMV features (Better Brush V2): scene-cut detection via PySceneDetect and
# medial face landmarking. Both are optional at runtime (native OpenCV
# fallbacks exist), so a failed collection never aborts the build.
for _pkg in ("scenedetect", "mediapipe", "sam3", "huggingface_hub", "iopath", "timm"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:  # noqa: BLE001 - optional deps, keep the build going
        pass

# V3 engines: AI object removal (MiniMax-Remover via diffusers) and learned
# matting (MatAnyone / MatAnyone 2 via imageio).  Both optional at runtime
# with friendly errors, so a failed collection never aborts the build.
for _pkg in ("diffusers", "accelerate", "imageio", "imageio_ffmpeg", "omegaconf"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:  # noqa: BLE001 - optional deps, keep the build going
        pass

# Vendored model packages ship inside src/vendor; make sure PyInstaller
# follows them (including the Hydra-free YAML configs loaded from disk).
hiddenimports += [
    "src.vendor.efficienttam",
    "src.vendor.efficienttam.build_efficienttam",
    "src.vendor.efficienttam.efficienttam_video_predictor",
    "src.vendor.efficienttam.efficienttam_image_predictor",
    "src.vendor.minimax",
    "src.vendor.minimax.transformer",
    "src.vendor.matanyone",
    "src.vendor.matanyone2",
]
datas.append((str(PROJECT_ROOT / "src" / "vendor"), "src/vendor"))

# A local dev run leaves __pycache__ inside src/vendor.  Every vendored package
# ships its .py sources, so the stale bytecode is dropped here (Python simply
# recompiles); this keeps the bundle deterministic and free of build machine
# leftovers.  Scoped to src/vendor so packages that genuinely ship only .pyc
# are left untouched.
datas = [
    (src, dest)
    for src, dest in datas
    if not (
        dest.replace("\\", "/").startswith("src/vendor")
        and (
            "__pycache__" in dest.replace("\\", "/").split("/")
            or dest.endswith(".pyc")
        )
    )
]

# Dev-only torch payloads: C++ headers and unit-test data are not needed.
datas = [
    (src, dest)
    for src, dest in datas
    if not dest.replace("\\", "/").startswith(("torch/include", "torch/testing"))
]

# SAM 2 pulls in Hydra/OmegaConf at runtime for its config system.
for _pkg in ("hydra", "omegaconf"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# App assets (logo + icons).
datas.append((str(PROJECT_ROOT / "assets"), "assets"))

# Face-landmarker model for one-click mouth/eye isolation.
_face_model = PROJECT_ROOT / "checkpoints" / "face_landmarker.task"
if _face_model.is_file():
    datas.append((str(_face_model), "checkpoints"))

# LaMa inpainting engine (subtitle/logo Clean Up, AI texture repair).  The
# model is large (~200 MB); it ships with the bundle so the AI path works
# out of the box, and the app falls back to OpenCV donor/Telea when absent.
_lama_model = PROJECT_ROOT / "checkpoints" / "lama_fp32.onnx"
if _lama_model.is_file():
    datas.append((str(_lama_model), "checkpoints"))

# onnxruntime powers the LaMa engine.  Optional at runtime (OpenCV fallback
# exists), so a failed collection never aborts the build — but it is part of
# the standard bundle.
try:
    _d, _b, _h = collect_all("onnxruntime")
    datas += _d
    binaries += _b
    hiddenimports += _h
except Exception:  # noqa: BLE001 - optional dep, keep the build going
    pass

# End-user license terms travel with the product (shown next to the EXE).
datas.append((str(PROJECT_ROOT / "LICENSE"), "."))

# Third-party attributions (bundled models, FFmpeg, Qt, …) travel with it too.
_third_party = PROJECT_ROOT / "THIRD-PARTY-NOTICES.md"
if _third_party.is_file():
    datas.append((str(_third_party), "."))

# Self-contained FFmpeg: sits next to BetterBrush.exe on disk so export
# works without the user installing anything.
datas.append((str(PROJECT_ROOT / "For-Agent" / "ffmpeg.exe"), "."))

_version_info = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=(1, 0, 0, 0),
        prodvers=(1, 0, 0, 0),
        mask=0x3F,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0),
    ),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    "040904B0",
                    [
                        StringStruct("CompanyName", "Sneg HQ, a company of Lugunica"),
                        StringStruct("FileDescription", "Better Brush — AI rotoscoping"),
                        StringStruct("FileVersion", "1.0.0"),
                        StringStruct("InternalName", "BetterBrush"),
                        StringStruct(
                            "LegalCopyright",
                            "© 2026 Sneg — Sneg HQ, a company of Lugunica",
                        ),
                        StringStruct("OriginalFilename", "BetterBrush.exe"),
                        StringStruct("ProductName", "Better Brush"),
                        StringStruct("ProductVersion", "1.0.0"),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct("Translation", [1033, 1200])]),
    ],
)

_a = Analysis(  # noqa: F821 -- injected global by PyInstaller
    ["src/main.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "pandas",
        "pygame",
        "numba",
        "llvmlite",
        "imageio_v2",
        "pyarrow",
        "moviepy",
        "av",
        "grpc",
        "cryptography",
        "torchaudio",
        "supervision",
        "modelscope",
        "openai_whisper",
        "whisper",
    ],
    noarchive=False,
)
_pyz = PYZ(_a.pure)  # noqa: F821 -- injected global by PyInstaller

_exe = EXE(  # noqa: F821 -- injected global by PyInstaller
    _pyz,
    _a.scripts,
    [],
    exclude_binaries=True,
    name="BetterBrush",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "assets" / "icons" / "BetterBrush.ico"),
    version=_version_info,
)

_coll = COLLECT(  # noqa: F821 -- injected global by PyInstaller
    _exe,
    _a.binaries,
    _a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BetterBrush",
)