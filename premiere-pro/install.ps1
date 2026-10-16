<#
.SYNOPSIS
  Installs (or removes) the Better Brush Premiere Pro panel for Windows.

.DESCRIPTION
  Copies this folder into a CEP extensions directory, enables CEP player-debug
  mode so the unsigned panel loads, and writes config.json with the Python
  interpreter and repository path the panel should launch.

  Two scopes are supported:

    System (default)  C:\Program Files\Common Files\Adobe\CEP\extensions
                      Visible to every user. Needs an elevated PowerShell.
    User              %APPDATA%\Adobe\CEP\extensions
                      Just this user. No administrator rights needed.

  CEP scans every scope, so an install in one removes any copy left in the
  others - two copies of the same extension would show up twice in the menu.

  The After Effects panel is a separate extension with its own bundle id and its
  own folder (BetterBrush); this one installs to BetterBrushPPro. Both talk to
  the same local engine, so install whichever you need - or both.

.PARAMETER Python
  Path to the python.exe that has Better Brush's dependencies installed.
  Auto-detected from PATH when omitted.

.PARAMETER RepoRoot
  The Better Brush repository root. Defaults to this script's parent folder.

.PARAMETER Port
  Engine port. Keep the default unless something else already uses it.

.PARAMETER Scope
  System (default) or User. See the description above.

.PARAMETER DebugPort
  When > 0, writes a CEP .debug file so the panel can be inspected in Chrome at
  localhost:<DebugPort>.

.PARAMETER Uninstall
  Remove the installed panel (from every scope).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\premiere-pro\install.ps1 -Scope User
#>
[CmdletBinding()]
param(
  [string]$Python = "",
  [string]$RepoRoot = "",
  [int]$Port = 8765,
  [ValidateSet("System", "User")]
  [string]$Scope = "System",
  [int]$DebugPort = 0,
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "  $msg" -ForegroundColor Gray }
function Good($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExtName = "BetterBrushPPro"

# The three CEP extension roots. The x86 one is legacy; it is only ever cleaned
# up, never installed into.
$UserRoot = Join-Path $env:APPDATA "Adobe\CEP\extensions"
$SystemRoot = Join-Path ${env:ProgramFiles} "Common Files\Adobe\CEP\extensions"
$x86Root = ""
if (${env:ProgramFiles(x86)}) {
  $x86Root = Join-Path ${env:ProgramFiles(x86)} "Common Files\Adobe\CEP\extensions"
}

$UserDir = Join-Path $UserRoot $ExtName
$SystemDir = Join-Path $SystemRoot $ExtName
$x86Dir = ""
if ($x86Root) { $x86Dir = Join-Path $x86Root $ExtName }

$ExtDir = if ($Scope -eq "System") { $SystemDir } else { $UserDir }

function Test-Admin {
  try {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  } catch {
    return $false
  }
}

# Any copy outside the scope we are installing into would make the panel appear
# twice, so it goes.
function Remove-OtherCopies([string]$keep) {
  foreach ($dir in @($UserDir, $SystemDir, $x86Dir)) {
    if (-not $dir -or $dir -eq $keep) { continue }
    if (Test-Path $dir) {
      try {
        Remove-Item -Recurse -Force $dir
        Good "Removed the copy in $dir"
      } catch {
        Warn "Could not remove the copy in $dir (needs administrator rights): $($_.Exception.Message)"
      }
    }
  }
}

if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $ScriptDir }

Write-Host ""
Write-Host "Better Brush - Premiere Pro panel installer" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------------------------- #
# Uninstall
# --------------------------------------------------------------------- #
if ($Uninstall) {
  $found = $false
  foreach ($dir in @($UserDir, $SystemDir, $x86Dir)) {
    if (-not $dir -or -not (Test-Path $dir)) { continue }
    $found = $true
    try {
      Remove-Item -Recurse -Force $dir
      Good "Removed $dir"
    } catch {
      Warn "Could not remove $dir (needs administrator rights): $($_.Exception.Message)"
    }
  }
  if (-not $found) { Info "Nothing installed in any CEP extensions folder" }
  Write-Host "Done. Restart Premiere Pro." -ForegroundColor Cyan
  exit 0
}

# --------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------- #
if ($Scope -eq "System" -and -not (Test-Admin)) {
  Die @"
Installing for all users writes to
  $SystemRoot
which needs administrator rights.

  Open PowerShell as Administrator (or use Start-Process -Verb RunAs), then:
    powershell -ExecutionPolicy Bypass -File "$ScriptDir\install.ps1"

  To install for this user only instead - no admin needed:
    powershell -ExecutionPolicy Bypass -File "$ScriptDir\install.ps1" -Scope User
"@
}
Info "Scope: $Scope  ($ExtDir)"

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
foreach ($dir in @($UserDir, $SystemDir, $x86Dir)) {
  if (-not $dir -or -not (Test-Path $dir)) { continue }
  try { Remove-Item -Recurse -Force $dir } catch {
    Die "Could not replace `"$dir`": $($_.Exception.Message)"
  }
}
New-Item -ItemType Directory -Force -Path $ExtDir | Out-Null

Copy-Item -Path (Join-Path $ScriptDir "*") -Destination $ExtDir -Recurse -Force
Remove-Item -Force (Join-Path $ExtDir "install.ps1") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $ExtDir "config.sample.json") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $ExtDir "README.md") -ErrorAction SilentlyContinue
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
foreach ($version in 9..16) {
  $key = "HKCU:\Software\Adobe\CSXS.$version"
  New-Item -Path $key -Force | Out-Null
  Set-ItemProperty -Path $key -Name "PlayerDebugMode" -Value "1" -Type String
}
Good "Enabled CEP player-debug mode (CSXS.9 - CSXS.16)"

# --------------------------------------------------------------------- #
# Optional remote debugging for the panel
# --------------------------------------------------------------------- #
$debugFile = Join-Path $ExtDir ".debug"
if ($DebugPort -gt 0) {
  @(
    "<?xml version=""1.0"" encoding=""UTF-8""?>",
    "<ExtensionList>",
    "  <Extension Id=""com.snegdev.betterbrush.premiere.panel"">",
    "    <HostList>",
    "      <Host Name=""PPRO"" Port=""$DebugPort""/>",
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
Info "1. Restart Premiere Pro."
Info "2. Window > Extensions > Better Brush  (dock the panel where you like)."
Info "3. Select a clip on the timeline and click Remove Background."
Write-Host ""
Info "The panel starts the engine on demand; you normally never launch it by hand."
Info "To run it manually:"
Info "  cd `"$RepoRoot`""
Info "  `"$pythonExe`" -m src.engine --port $Port"
Write-Host ""
