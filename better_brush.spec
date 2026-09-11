"""PyInstaller spec for Better Brush (Windows onedir bundle).

Entry point: src/main.py (launches src.core.app:run).
Bundle contains PySide6, PyTorch, SAM 2, OpenCV and the assets/ folder.

Build with:
    pyinstaller better_brush.spec

Do NOT run from inside the bundle; checkpoint/export dirs are resolved
relative to the project root at build time in src/core/constants.py.
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = Path(__file__).resolve().parent

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []

# Heavy native packages whose data + DLLs + hidden imports must be shipped.
for _pkg in ("sam2", "torch", "torchvision", "cv2", "pynvml"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# SAM 2 pulls in Hydra/OmegaConf at runtime for its config system.
for _pkg in ("hydra", "omegaconf"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# App assets (logo / icons) are resolved by src/ui/assets.py.
datas.append((str(PROJECT_ROOT / "assets"), "assets"))

_a = Analysis(
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
    ],
    noarchive=False,
)
_pyz = PYZ(_a.pure)

_exe = EXE(
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
    icon=None,
)

_coll = COLLECT(
    _exe,
    _a.binaries,
    _a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BetterBrush",
)