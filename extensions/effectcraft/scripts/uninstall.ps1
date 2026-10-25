<#
.SYNOPSIS
  Remove the Better Brush panel from EffectCraft.

  Deletes only the two artifacts this MVP creates: the panel copy under
  <settings>\Scripts\ScriptUI Panels\BetterBrush.jsx and the bridge folder
  <settings>\betterbrush-bridge\. bridge.path.json in the MVP folder is left
  in place unless -RemoveRecord is given, so the install stays re-runnable.
#>
[CmdletBinding()]
param(
    [string]$SettingsDir = "",
    [switch]$RemoveBridge,
    [switch]$RemoveRecord
)

$ErrorActionPreference = "Stop"
$mvp = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $mvp "bridge.path.json"

if (-not $SettingsDir -and (Test-Path $recordPath)) {
    $record = Get-Content $recordPath -Raw | ConvertFrom-Json
    if ($record.settings_dir) { $SettingsDir = $record.settings_dir }
    if ($record.bridge_dir -and -not $RemoveBridge) { $RemoveBridge = $true }
}

if (-not $SettingsDir) {
    $roaming = Join-Path $env:APPDATA "EffectCraft"
    if (Test-Path $roaming) { $SettingsDir = $roaming }
}

if (-not $SettingsDir) {
    Write-Error "Could not determine the EffectCraft settings folder. Pass -SettingsDir <path>."
}

$panel = Join-Path $SettingsDir "Scripts\ScriptUI Panels\BetterBrush.jsx"
if (Test-Path $panel) {
    Remove-Item -Force $panel
    Write-Host "Removed panel: $panel"
} else {
    Write-Host "Panel not present: $panel"
}

if ($RemoveBridge) {
    $bridge = Join-Path $SettingsDir "betterbrush-bridge"
    if (Test-Path $bridge) {
        Remove-Item -Recurse -Force $bridge
        Write-Host "Removed bridge: $bridge"
    } else {
        Write-Host "Bridge not present: $bridge"
    }
}

if ($RemoveRecord -and (Test-Path $recordPath)) {
    Remove-Item -Force $recordPath
    Write-Host "Removed record: $recordPath"
}

Write-Host "Done. No other files were touched."
