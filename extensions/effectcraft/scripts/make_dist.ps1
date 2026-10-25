<#
.SYNOPSIS
  Build the one-file Better Brush installer for EffectCraft.

.DESCRIPTION
  Produces a single double-clickable file

      dist\BetterBrush-EffectCraft-Setup-<version>.cmd

  that carries everything it needs: the panel, the Python adapter (and,
  when present, the standalone adapter.exe so no Python install is
  required), and the installer itself. Running it finds EffectCraft, copies
  the panel in, and writes the start file the *panel* runs when EffectCraft
  opens - nothing is registered at logon.

  The file is a command script with the payload zip appended as base64
  after a marker: it decodes itself into %TEMP%, expands it, and runs the
  installer inside. One file, no unzip step, no prerequisites beyond
  Windows.

.PARAMETER OutDir
  Where the setup file goes. Defaults to <mvp>\dist.

.PARAMETER BuildAdapter
  Build dist\betterbrush-adapter.exe with PyInstaller first, so the setup
  needs no Python at all on the target machine.

.PARAMETER AdapterExe
  Use this adapter executable instead of the one in dist\.
#>
[CmdletBinding()]
param(
    [string]$OutDir = "",
    [switch]$BuildAdapter,
    [string]$AdapterExe = ""
)

$ErrorActionPreference = "Stop"

$mvp      = Split-Path -Parent $PSScriptRoot
$panelSrc = Join-Path $mvp "panel\BetterBrush.jsx"
$adapterSrc = Join-Path $mvp "effectcraft_bridge"
$distInstall = Join-Path $PSScriptRoot "dist-install.ps1"

$panelText = [System.IO.File]::ReadAllText($panelSrc)
if ($panelText -notmatch 'var VERSION = "([\d\.]+)"') {
    throw "could not read the panel version from $panelSrc"
}
$version = $Matches[1]

if (-not $OutDir) { $OutDir = Join-Path $mvp "dist" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ------------------------------------------------------------------ #
# The adapter executable (optional, but it removes the Python requirement)
# ------------------------------------------------------------------ #
$defaultExe = Join-Path $mvp "dist\betterbrush-adapter.exe"
if ($BuildAdapter -and -not $AdapterExe) {
    Write-Host "Building the standalone adapter (PyInstaller)..."
    Push-Location $mvp
    try {
        python -m PyInstaller --noconfirm --clean --onefile `
            --name betterbrush-adapter --hidden-import cv2 --hidden-import numpy `
            run_adapter.py | Out-Null
    } finally {
        Pop-Location
    }
}
if (-not $AdapterExe -and (Test-Path $defaultExe)) { $AdapterExe = $defaultExe }
if ($AdapterExe) {
    if (-not (Test-Path $AdapterExe)) { throw "no such adapter executable: $AdapterExe" }
    Write-Host "Standalone adapter:    $AdapterExe ($([math]::Round((Get-Item $AdapterExe).Length/1MB,1)) MB)"
} else {
    Write-Host "No standalone adapter: the setup will need Python 3 (with numpy) at install time."
}

# ------------------------------------------------------------------ #
# Stage the payload
# ------------------------------------------------------------------ #
$stage = Join-Path $env:TEMP "bb-dist-$version"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path "$stage\panel", "$stage\adapter\effectcraft_bridge" | Out-Null

Copy-Item -Force $panelSrc "$stage\panel\BetterBrush.jsx"
Copy-Item -Force (Join-Path $mvp "run_adapter.py") "$stage\adapter\run_adapter.py"
Get-ChildItem -Path $adapterSrc -Filter "*.py" | ForEach-Object {
    Copy-Item -Force $_.FullName "$stage\adapter\effectcraft_bridge\$($_.Name)"
}
if ($AdapterExe) {
    Copy-Item -Force $AdapterExe "$stage\adapter\betterbrush-adapter.exe"
}
Copy-Item -Force $distInstall "$stage\install.ps1"

@'
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
'@ | Set-Content "$stage\install.cmd" -Encoding Ascii

@'
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Uninstall
'@ | Set-Content "$stage\uninstall.cmd" -Encoding Ascii

@'
Better Brush for EffectCraft
============================

  install.cmd       install the panel and the background adapter
  uninstall.cmd     remove them again

The adapter runs from %LOCALAPPDATA%\BetterBrush\effectcraft-adapter and
logs to <EffectCraft settings>\betterbrush-bridge\adapter.log.
'@ | Set-Content "$stage\README.txt" -Encoding Ascii

$zip = Join-Path $env:TEMP "bb-dist-$version.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal
$bytes = [System.IO.File]::ReadAllBytes($zip)
$payload = [Convert]::ToBase64String($bytes)
Write-Host "Payload:               $([math]::Round($bytes.Length/1MB,1)) MB compressed"

# ------------------------------------------------------------------ #
# The single file: a command script with the payload appended
# ------------------------------------------------------------------ #
$header = @"
@echo off
title Better Brush for EffectCraft
echo.
echo  Better Brush for EffectCraft $version - setup
echo  ---------------------------------------------
echo.
echo  Unpacking...
powershell -NoProfile -ExecutionPolicy Bypass -Command "`$t=[IO.File]::ReadAllText('%~f0');`$m='-----BEGIN '+'PAYLOAD-----';`$i=`$t.IndexOf(`$m);if(`$i -lt 0){Write-Host 'payload missing';exit 1};`$b=`$t.Substring(`$i+`$m.Length) -replace '\s','';`$w=Join-Path `$env:TEMP 'bb-ec-setup';New-Item -ItemType Directory -Force -Path `$w|Out-Null;[IO.File]::WriteAllBytes((Join-Path `$w 'payload.zip'),[Convert]::FromBase64String(`$b));Expand-Archive -Force (Join-Path `$w 'payload.zip') `$w;& (Join-Path `$w 'install.cmd')"
echo.
pause
"@

$out = Join-Path $OutDir "BetterBrush-EffectCraft-Setup-$version.cmd"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$writer = [System.IO.StreamWriter]::new($out, $false, [System.Text.Encoding]::ASCII)
try {
    $writer.Write($header)
    $writer.WriteLine()
    $writer.WriteLine("-----BEGIN PAYLOAD-----")
    $writer.Write($payload)
    $writer.WriteLine()
} finally {
    $writer.Dispose()
}

Write-Host ""
Write-Host "Setup file:            $out ($([math]::Round((Get-Item $out).Length/1MB,1)) MB)"
Write-Host "  double-click it to install; the panel starts the adapter when"
Write-Host "  EffectCraft opens - nothing is registered at logon."
