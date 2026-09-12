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
for _pkg in ("sam2", "torch", "torchvision", "cv2", "pynvml"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

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

# End-user license terms travel with the product (shown next to the EXE).
datas.append((str(PROJECT_ROOT / "LICENSE"), "."))

# Self-contained FFmpeg: sits next to BetterBrush.exe on disk so export
# works without the user installing anything.
datas.append((str(PROJECT_ROOT / "For-Agent" / "ffmpeg.exe"), "."))

_version_info = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=(0, 1, 0, 0),
        prodvers=(0, 1, 0, 0),
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
                        StringStruct("FileVersion", "0.1.0"),
                        StringStruct("InternalName", "BetterBrush"),
                        StringStruct(
                            "LegalCopyright",
                            "© 2026 Sneg — Sneg HQ, a company of Lugunica",
                        ),
                        StringStruct("OriginalFilename", "BetterBrush.exe"),
                        StringStruct("ProductName", "Better Brush"),
                        StringStruct("ProductVersion", "0.1.0"),
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
        "scipy",
        "pygame",
        "numba",
        "llvmlite",
        "imageio",
        "imageio_ffmpeg",
        "pyarrow",
        "moviepy",
        "av",
        "onnxruntime",
        "grpc",
        "cryptography",
        "torchaudio",
        "diffusers",
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