"""Best-effort, per-user ``.bbproj`` file association (no admin rights).

Registers everything under ``HKCU\\Software\\Classes`` so Windows Explorer
shows the product icon on ``.bbproj`` project files and a double-click opens
them in Better Brush. Per-user keys apply immediately and never need a UAC
prompt; each launch re-registers so the icon path stays valid if the app is
moved to a new folder.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_EXT = ".bbproj"
_PROGID = "BetterBrush.bbproj"
_PROGDESC = "Better Brush Project"
_HKCU_CLASSES = r"Software\Classes"

# SHCNE_ASSOCCHANGED | SHCNF_IDLIST — tells Explorer to redraw icons.
_SHCNE_ASSOCCHANGED = 0x08000000
_SHCNF_IDLIST = 0x0000


def _app_executable() -> Path | None:
    """Path of the running app executable (None in a source-tree/dev run)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable)
    return None


def register() -> bool:
    """Register the file association. Returns True on success (best-effort)."""
    import winreg

    from src.ui.assets import bbproj_ico_path

    icon = bbproj_ico_path()
    if not icon.is_file():
        logger.debug("bbproj icon missing (%s); skipping association", icon)
        return False
    try:
        root = winreg.HKEY_CURRENT_USER

        def _set(subkey: str, name: str, value: str) -> None:
            with winreg.CreateKeyEx(root, subkey, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

        # Extension -> ProgID + open-with entry.
        _set(_HKCU_CLASSES + "\\" + _EXT, "", _PROGID)
        _set(_HKCU_CLASSES + "\\" + _EXT + "\\OpenWithProgids", _PROGID, "")
        # ProgID: friendly name + icon.
        _set(_HKCU_CLASSES + "\\" + _PROGID, "", _PROGDESC)
        _set(_HKCU_CLASSES + "\\" + _PROGID + "\\DefaultIcon", "", f'"{icon}"')
        # Double-click action (only when running from a real .exe).
        exe = _app_executable()
        if exe is not None:
            _set(
                _HKCU_CLASSES + "\\" + _PROGID + "\\shell\\open\\command",
                "",
                f'"{exe}" "%1"',
            )
        _notify_assoc_changed()
        logger.debug("registered .bbproj association (icon: %s)", icon)
        return True
    except Exception:
        logger.debug("file association registration failed", exc_info=True)
        return False


def _notify_assoc_changed() -> None:
    try:
        ctypes.windll.shell32.SHChangeNotify(
            _SHCNE_ASSOCCHANGED, _SHCNF_IDLIST, None, None
        )
    except Exception:
        logger.debug("SHChangeNotify failed", exc_info=True)
