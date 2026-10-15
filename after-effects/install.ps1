<#
.SYNOPSIS
  Installs (or removes) the Better Brush After Effects panel for the current
  Windows user.

.DESCRIPTION
  Copies this folder into the per-user CEP extensions directory, enables CEP
  player-debug mode so the unsigned panel loads, and writes config.json with the
  Python interpreter and repository path the panel should launch.

  Nothing is installed system-wide; everything lives under %APPDATA%.

.PARAMETER Python
  Path to the python.exe that has Better Brush's dependencies installed.
  Auto-detected from PATH when omitted.

.PARAMETER RepoRoot
  The Better Brush repository root. Defaults to this script's parent folder.

.PARAMETER Port
  Engine port. Keep the default unless something else already uses it.

.PARAMETER DebugPort
  When > 0, writes a CEP .debug file so the panel can be inspected in Chrome at
  localhost:<DebugPort>.

.PARAMETER Uninstall
  Remove the installed panel instead.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\after-effects\install.ps1
#>
[CmdletBinding()]
param(
  [string]$Python = "",
  [string]$RepoRoot = "",
  [int]$Port = 8765,
  [int]$DebugPort = 0,
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "  $msg" -ForegroundColor Gray }
function Good($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExtName = "BetterBrush"
$ExtDir = Join-Path $env:APPDATA "Adobe\CEP\extensions\$ExtName"

if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $ScriptDir }

Write-Host ""
Write-Host "Better Brush - After Effects panel installer" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------------------------- #
# Uninstall
# --------------------------------------------------------------------- #
if ($Uninstall) {
  if (Test-Path $ExtDir) {
    Remove-Item -Recurse -Force $ExtDir
    Good "Removed $ExtDir"
  } else {
    Info "Nothing installed at $ExtDir"
  }
  Write-Host "Done. Restart After Effects." -ForegroundColor Cyan
  exit 0
}

# --------------------------------------------------------------------- #
# Locate Python
# --------------------------------------------------------------------- #
function Resolve-Python([string]$explicit) {
  if ($explicit) {
    if (-not (Test-Path $explicit)) { Die "Python not found: $explicit" }
    return (Resolve-Path $explicit).Path
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $cmd = Get-Command python3 -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  Die "Could not find Python on PATH. Pass -Python 'C:\path\to\python.exe'."
}

$pythonExe = Resolve-Python $Python
Info "Python: $pythonExe"

$pythonwExe = ""
$candidate = Join-Path (Split-Path -Parent $pythonExe) "pythonw.exe"
if (Test-Path $candidate) { $pythonwExe = $candidate } else { $pythonwExe = $pythonExe }
Info "Windowless launcher: $pythonwExe"

# --------------------------------------------------------------------- #
# Sanity-check the interpreter actually has Better Brush's runtime
# --------------------------------------------------------------------- #
Push-Location $RepoRoot
try {
  & $pythonExe -c "import src.engine, torch" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Warn "This Python could not import 'src.engine' and 'torch' from $RepoRoot."
    Warn "The panel will install, but the engine may fail to start."
    Warn "Install the app first:  python -m pip install -e ."
  } else {
    Good "Runtime check passed."
  }
} finally {
  Pop-Location
}

# --------------------------------------------------------------------- #
# Copy the extension
# --------------------------------------------------------------------- #
if (Test-Path $ExtDir) { Remove-Item -Recurse -Force $ExtDir }
New-Item -ItemType Directory -Force -Path $ExtDir | Out-Null

Copy-Item -Path (Join-Path $ScriptDir "*") -Destination $ExtDir -Recurse -Force
Remove-Item -Force (Join-Path $ExtDir "install.ps1") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $ExtDir "config.sample.json") -ErrorAction SilentlyContinue
Good "Installed panel to $ExtDir"

# --------------------------------------------------------------------- #
# Write config.json
# --------------------------------------------------------------------- #
$config = [ordered]@{
  projectRoot = $RepoRoot
  pythonPath  = $pythonExe
  pythonwPath = $pythonwExe
  port        = $Port
}
$configPath = Join-Path $ExtDir "config.json"
$configJson = $config | ConvertTo-Json
[System.IO.File]::WriteAllText(
  $configPath,
  $configJson,
  (New-Object System.Text.UTF8Encoding($false))
)
Good "Wrote config.json"

# --------------------------------------------------------------------- #
# Enable CEP player-debug mode (unsigned extensions) for this user
# --------------------------------------------------------------------- #
foreach ($version in 9..13) {
  $key = "HKCU:\Software\Adobe\CSXS.$version"
  New-Item -Path $key -Force | Out-Null
  Set-ItemProperty -Path $key -Name "PlayerDebugMode" -Value "1" -Type String
}
Good "Enabled CEP player-debug mode (CSXS.9 - CSXS.13)"

# --------------------------------------------------------------------- #
# Optional remote debugging for the panel
# --------------------------------------------------------------------- #
$debugFile = Join-Path $ExtDir ".debug"
if ($DebugPort -gt 0) {
  @(
    "<?xml version=""1.0"" encoding=""UTF-8""?>",
    "<ExtensionList>",
    "  <Extension Id=""com.snegdev.betterbrush.panel"">",
    "    <HostList>",
    "      <Host Name=""AEFT"" Port=""$DebugPort""/>",
    "    </HostList>",
    "  </Extension>",
    "</ExtensionList>"
  ) | Set-Content -Encoding UTF8 $debugFile
  Good "Panel debugging enabled on http://localhost:$DebugPort"
} elseif (Test-Path $debugFile) {
  Remove-Item -Force $debugFile
}

# --------------------------------------------------------------------- #
Write-Host ""
Write-Host "Next steps" -ForegroundColor Cyan
Info "1. Restart After Effects."
Info "2. Window > Extensions > Better Brush  (dock the panel where you like)."
Info "3. Select a footage layer and click Remove Background."
Write-Host ""
Info "The panel starts the engine on demand; you normally never launch it by hand."
Info "To run it manually:"
Info "  cd `"$RepoRoot`""
Info "  `"$pythonExe`" -m src.engine --port $Port"
Write-Host ""
