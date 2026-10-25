<#
.SYNOPSIS
  Install the Better Brush panel into EffectCraft.

.DESCRIPTION
  Copies panel\BetterBrush.jsx into <EffectCraft settings>\Scripts\ScriptUI Panels\,
  creates the bridge folder <EffectCraft settings>\betterbrush-bridge\, and records
  that path in bridge.path.json next to this script so the Python adapter finds it
  without guessing.

  Only files inside the settings folder and this MVP folder are touched. Nothing
  else in the repository is modified.

.PARAMETER SettingsDir
  EffectCraft settings folder. Auto-detected when omitted: a portable install's
  EffectCraftData next to effectcraft.exe, else %APPDATA%\EffectCraft.

.PARAMETER PanelHost
  Where to copy the panel. Defaults to <SettingsDir>\Scripts\ScriptUI Panels.
#>
[CmdletBinding()]
param(
    [string]$SettingsDir = "",
    [string]$PanelHost = ""
)

$ErrorActionPreference = "Stop"

$mvp      = Split-Path -Parent $PSScriptRoot          # extensions\effectcraft
$panelSrc = Join-Path $psscriptRoot "..\panel\BetterBrush.jsx"
$bridgeName = "betterbrush-bridge"

function Find-EffectCraftSettings {
    # Portable layout (see the portable.txt file itself): it sits *next to*
    # effectcraft.exe, and settings live in the EffectCraftData folder beside
    # it. So the settings folder is EffectCraftData, never its parent — an
    # earlier version of this script returned the install root, which put the
    # panel where EffectCraft never looks.
    $roots = @(
        (Join-Path $env:LOCALAPPDATA "EffectCraft"),
        (Join-Path $env:APPDATA "EffectCraft"),
        (Join-Path $env:USERPROFILE "Downloads"),
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:USERPROFILE "Documents")
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $data = Get-ChildItem -Path $root -Directory -Recurse -Depth 4 -Filter "EffectCraftData" -ErrorAction SilentlyContinue
        foreach ($d in $data) {
            $installDir = Split-Path -Parent $d.FullName
            $isPortable = Test-Path (Join-Path $installDir "portable.txt")
            $hasPrefs   = Test-Path (Join-Path $d.FullName "prefs.json")
            if ($isPortable -or $hasPrefs) { return $d.FullName }
        }
    }
    # Non-portable: %APPDATA%\EffectCraft.
    $roaming = Join-Path $env:APPDATA "EffectCraft"
    if (Test-Path $roaming) { return $roaming }
    return $null
}

if (-not $SettingsDir) {
    $SettingsDir = Find-EffectCraftSettings
    if (-not $SettingsDir) {
        Write-Error "Could not find the EffectCraft settings folder. Pass -SettingsDir <path>."
    }
}
$SettingsDir = [System.IO.Path]::GetFullPath($SettingsDir)
Write-Host "EffectCraft settings: $SettingsDir"

# Sanity check: the settings folder always holds prefs.json. Landing here means
# detection picked a folder EffectCraft never reads (an install root, say) and
# the panel would be invisible in the Window menu.
if (-not (Test-Path (Join-Path $SettingsDir "prefs.json"))) {
    Write-Warning ("No prefs.json in $SettingsDir - this does not look like the " +
        "EffectCraft settings folder. Pass -SettingsDir <path> to override.")
}

# The bridge folder: <settings>\betterbrush-bridge. Computed before the panel
# copy because the installer stamps this exact path into the panel file.
$bridge = Join-Path $SettingsDir $bridgeName
New-Item -ItemType Directory -Force -Path $bridge | Out-Null
Write-Host "Bridge folder:        $bridge"

if (-not $PanelHost) { $PanelHost = Join-Path $SettingsDir "Scripts\ScriptUI Panels" }
New-Item -ItemType Directory -Force -Path $PanelHost | Out-Null

# Back up a panel file that is already there before it is replaced: a user may
# have their own BetterBrush.jsx from an earlier install or from the repo.
$panelDst = Join-Path $PanelHost "BetterBrush.jsx"
$backupRoot = Join-Path $mvp "backups"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $backupRoot $stamp
if (Test-Path $panelDst) {
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    Copy-Item -Force $panelDst (Join-Path $backupDir "BetterBrush.jsx")
    Write-Host "Backed up existing panel to $backupDir"
}

# Copy the panel with the bridge path stamped in. A *panel* script sees
# $.fileName as a bare file name (EffectCraft hands it "BetterBrush.jsx", no
# directory), so the panel cannot walk up to its own folder — the installer
# writes the answer into the copy it ships, and the panel validates the stamp
# against the panel file's own location before trusting it.
# Written with UTF8Encoding($false): PS 5.1's -Encoding UTF8 would add a BOM.
$panelText = [System.IO.File]::ReadAllText($panelSrc)
if (-not $panelText.Contains("@@BB_BRIDGE@@")) {
    Write-Warning "@@BB_BRIDGE@@ placeholder missing from the panel source - the panel will fall back to path walking."
}
$bridgeSlash = $bridge -replace "\\", "/"
[System.IO.File]::WriteAllText($panelDst, $panelText.Replace("@@BB_BRIDGE@@", $bridgeSlash),
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Panel installed:      $panelDst"
Write-Host "Panel stamped with:   $bridgeSlash"

# The version of the install we just stamped into, read from the folder it
# lives in (`effectcraft-0.17.0-windows-x64-portable` -> 0.17.0). Provenance
# for bridge.path.json only: an unrecognised layout stays "unknown" instead of
# claiming a version this script never saw.
$installName = Split-Path -Leaf (Split-Path -Parent $SettingsDir)
$ecVer = if ($installName -match "(\d+\.\d+\.\d+)") { $Matches[1] } else { "unknown" }

# Record the resolved path for the adapter (protocol.recorded_bridge_dir()).
# The checkout this adapter lives in, recorded for the copy: run_adapter.py
# puts it on sys.path so `src.engine` imports and jobs run in engine mode
# instead of simulate. install.ps1 is the one place that knows it.
$repoGuess = Split-Path -Parent (Split-Path -Parent $mvp)
$repoRoot = if (Test-Path (Join-Path $repoGuess "src")) {
    ($repoGuess -replace "\\", "/")
} else { "" }

$record = @{
    bridge_dir      = ($bridge -replace "\\", "/")
    settings_dir    = ($SettingsDir -replace "\\", "/")
    panel           = ($panelDst -replace "\\", "/")
    installed       = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    effectcraft_ver = $ecVer
    repo_root       = $repoRoot
} | ConvertTo-Json
# Back up bridge.path.json if one is already recorded, then overwrite it.
# UTF8Encoding($false): PS 5.1's -Encoding UTF8 adds a BOM, which a strict
# JSON reader rejects (the adapter tolerates it, other tools may not).
$recordPath = Join-Path $mvp "bridge.path.json"
if (Test-Path $recordPath) {
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    Copy-Item -Force $recordPath (Join-Path $backupDir "bridge.path.json")
}
[System.IO.File]::WriteAllText($recordPath, $record,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Recorded:             $recordPath"

# The file the *panel* runs when EffectCraft opens. Nothing is registered at
# logon (an earlier version did, and it was removed): the panel reads
# <bridge>\adapter.alive and executes this only when no adapter is watching.
# .vbs is what the panel executes - wscript starts it with no window at all,
# where a .cmd would flash a console.
$adapterPy = Join-Path $mvp "run_adapter.py"
$pythonExe = $null
foreach ($name in @("pythonw.exe", "python.exe")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $pythonExe = $cmd.Source; break }
}
if ($pythonExe) {
    $launcher = "`"$pythonExe`" `"$adapterPy`" watch"
    $startVbs = Join-Path $bridge "start-adapter.vbs"
    # VBS escapes a quote by doubling it, and the launcher is full of them -
    # built with single-quoted PowerShell strings so no escape is ambiguous,
    # and hoisted into a variable: inside @() PowerShell parses the commas.
    $vbsRun = 'sh.Run "' + $launcher.Replace('"', '""') + '", 0, False'
    $vbs = @(
        "' Written by install.ps1. The panel runs this file when EffectCraft",
        "' opens - nothing starts at logon.",
        'Set sh = CreateObject("WScript.Shell")',
        $vbsRun
    ) -join "`r`n"
    [System.IO.File]::WriteAllText($startVbs, $vbs + "`r`n",
        (New-Object System.Text.UTF8Encoding($false)))
    $startCmd = Join-Path $bridge "start-adapter.cmd"
    @("@echo off", "rem The panel runs start-adapter.vbs when EffectCraft opens.",
      "start `"`" /b $launcher") | Set-Content -Path $startCmd -Encoding Default
    Write-Host "Start files:          $startVbs"
} else {
    Write-Warning "No python on PATH - no start-adapter files were written."
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. EffectCraft > Preferences > Scripting & Expressions >"
Write-Host "     allow 'Allow Scripts to Write Files and Access Network'."
Write-Host "  2. Restart EffectCraft; the panel appears under Window > Better Brush"
Write-Host "     (or View > Workspace, depending on the build), and it starts the"
Write-Host "     adapter itself when EffectCraft opens."
