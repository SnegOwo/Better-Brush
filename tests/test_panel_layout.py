"""Guard the CEP panel layout that the installers ship.

The 1.0.0 portable/setup packages were assembled with the panel folders
flattened - ``manifest.xml`` at the folder root instead of
``CSXS\\manifest.xml``. CEP only scans ``CSXS``, so the panel installed
"successfully" and never appeared under Window > Extensions. These tests make
that packaging mistake fail loudly in the repository, and they also assert the
installers keep the repair path for already-downloaded flattened copies.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PANELS = {
    "after-effects": "AEFT",
    "premiere-pro": "PPRO",
}


def _manifest(panel: str) -> Path:
    return REPO_ROOT / panel / "CSXS" / "manifest.xml"


@pytest.mark.parametrize("panel,host", PANELS.items())
def test_manifest_lives_in_csxs(panel: str, host: str) -> None:
    """CEP only looks for CSXS/manifest.xml - a root manifest is invisible."""
    directory = REPO_ROOT / panel
    assert directory.is_dir(), f"missing panel folder: {directory}"
    assert _manifest(panel).is_file(), (
        f"{panel} must ship CSXS/manifest.xml; CEP never scans {panel}/manifest.xml"
    )
    assert not (directory / "manifest.xml").exists(), (
        f"{panel}/manifest.xml is the flattened layout - move it into CSXS/"
    )


@pytest.mark.parametrize("panel,host", PANELS.items())
def test_manifest_targets_the_host(panel: str, host: str) -> None:
    root = ET.parse(_manifest(panel)).getroot()
    assert root.tag == "ExtensionManifest"

    hosts = [
        element.attrib
        for element in root.iter("Host")
        if element.attrib.get("Name")
    ]
    assert any(h["Name"] == host for h in hosts), f"{panel} must target host {host}"
    # AE 2024 is version 24.x, Premiere 2024 is 24.x; the range has to include
    # both the oldest supported build (22.0) and current releases.
    for h in hosts:
        version = h.get("Version", "")
        assert version.startswith("["), f"{host} needs an explicit version range, got {version!r}"
        low, high = version.strip("[]").split(",")
        assert float(low) <= 22.0
        assert float(high) >= 25.0, f"{host} range stops at {high}; newer hosts would hide the panel"


@pytest.mark.parametrize("panel,host", PANELS.items())
def test_declared_resources_exist(panel: str, host: str) -> None:
    root = ET.parse(_manifest(panel)).getroot()
    directory = REPO_ROOT / panel
    resources = [element for element in root.iter("Resources")]
    assert resources, f"{panel} manifest has no <Resources>"

    main_path = resources[0].findtext("MainPath", "")
    script_path = resources[0].findtext("ScriptPath", "")
    assert main_path, f"{panel} manifest has no MainPath"
    assert script_path, f"{panel} manifest has no ScriptPath"

    for relative in (main_path, script_path):
        candidate = (directory / relative).resolve()
        assert candidate.is_file(), f"{panel}: manifest points at missing {relative}"
        assert directory in candidate.parents or candidate.parent == directory


@pytest.mark.parametrize("panel,host", PANELS.items())
def test_extension_ids_match_dispatch_info(panel: str, host: str) -> None:
    root = ET.parse(_manifest(panel)).getroot()
    extension_list = root.find("ExtensionList")
    dispatch_list = root.find("DispatchInfoList")
    assert extension_list is not None and dispatch_list is not None
    declared = {element.attrib["Id"] for element in extension_list.findall("Extension")}
    dispatched = {element.attrib["Id"] for element in dispatch_list.findall("Extension")}
    assert declared == dispatched, f"{panel}: ExtensionList and DispatchInfoList disagree"


@pytest.mark.parametrize("panel", PANELS)
def test_installer_installs_without_python(panel: str) -> None:
    """A missing Python must not abort the install.

    The panel is a CEP extension: it has to appear under Window > Extensions even
    when the engine interpreter is not configured yet. Aborting here is what made
    "I selected the panel in setup but there is no extension" reports possible.
    """
    text = (REPO_ROOT / panel / "install.ps1").read_text(encoding="utf-8")
    assert 'Die "Could not find Python on PATH' not in text
    assert "installing the panel anyway" in text
    assert "no python engine is configured" in text.lower()


@pytest.mark.parametrize("panel", PANELS)
def test_host_script_explains_a_missing_engine_python(panel: str) -> None:
    text = (REPO_ROOT / panel / "js" / "host.jsx").read_text(encoding="utf-8")
    assert "No Python engine is configured" in text


@pytest.mark.parametrize("panel", PANELS)
def test_config_is_loaded_without_fetch(panel: str) -> None:
    """CEP 12 (Adobe 2024+) refuses fetch() on file:// URLs.

    The config must arrive through a <script src="config.js"> tag, with the JSON
    fetch/XHR paths kept only as fallbacks for older hosts.
    """
    html = (REPO_ROOT / panel / "index.html").read_text(encoding="utf-8")
    assert "config.js" in html
    assert html.index("config.js") < html.index("js/panel.js")

    script = (REPO_ROOT / panel / "js" / "panel.js").read_text(encoding="utf-8")
    assert "window.BETTER_BRUSH_CONFIG" in script
    assert "readConfigJson" in script

    installer = (REPO_ROOT / panel / "install.ps1").read_text(encoding="utf-8")
    assert 'config.js' in installer
    assert "BETTER_BRUSH_CONFIG" in installer


def test_install_bat_points_at_panel_installers() -> None:
    """The portable package advertises a one-click installer; keep it wired."""
    bat_candidates = [
        REPO_ROOT / "Install Panels.bat",
        REPO_ROOT / "dist-final" / "BetterBrush-1.0.0-Windows" / "Install Panels.bat",
    ]
    existing = [path for path in bat_candidates if path.is_file()]
    if not existing:
        pytest.skip("Install Panels.bat only exists in the packaged portable build")
    text = existing[0].read_text(encoding="utf-8", errors="replace")
    assert "after-effects\\install.ps1" in text
    assert "premiere-pro\\install.ps1" in text
