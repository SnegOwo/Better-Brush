<#
.SYNOPSIS
  Install Better Brush into EffectCraft - panel, bridge and background adapter.

.DESCRIPTION
  This is the installer that ships inside the one-file setup
  (see scripts\make_dist.ps1). It does four things, in order:

    1. finds the EffectCraft settings folder (portable install beside
       effectcraft.exe, else %APPDATA%\EffectCraft) - asking when it cannot
       decide;
    2. copies the panel into <settings>\Scripts\ScriptUI Panels\ with the
       bridge path stamped in, and adds it to scriptui_panels.json so it
       opens on launch;
    3. installs the adapter into %LOCALAPPDATA%\BetterBrush\effectcraft-adapter
       and points it at that bridge;
    4. writes <bridge>\start-adapter.cmd, the file the *panel* runs when
       EffectCraft opens, and starts the adapter now.

  Nothing is registered at logon: the adapter comes up when EffectCraft does
  and stays up until it is stopped. Nothing outside those two folders is
  touched.

.PARAMETER SettingsDir
  EffectCraft settings folder. Auto-detected when omitted.

.PARAMETER AdapterDir
  Where the adapter is installed. Defaults to
  %LOCALAPPDATA%\BetterBrush\effectcraft-adapter.

.PARAMETER NoStart
  Install everything but do not start the adapter now.

.PARAMETER Uninstall
  Remove the panel, the adapter and start-adapter.cmd. The bridge folder
  (and its logs) is left alone - it is the user's work.
#>
[CmdletBinding()]
param(
    [string]$SettingsDir = "",
    [string]$AdapterDir = "",
    [string]$RepoRoot = "",
    [switch]$NoStart,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$panelSrc = Join-Path $here "panel\BetterBrush.jsx"
$adapterSrc = Join-Path $here "adapter"
$bridgeName = "betterbrush-bridge"
$runValueName = "BetterBrush EffectCraft Adapter"

if (-not $AdapterDir) {
    $AdapterDir = Join-Path $env:LOCALAPPDATA "BetterBrush\effectcraft-adapter"
}

# A checkout, when this install knows one (-RepoRoot, or the environment).
# It is what makes engine mode possible for an adapter installed outside the
# repo: run_adapter.py puts it on sys.path, so `src.engine` imports.
if (-not $RepoRoot -and $env:BETTERBRUSH_ROOT) { $RepoRoot = $env:BETTERBRUSH_ROOT }
if ($RepoRoot -and -not (Test-Path (Join-Path $RepoRoot "src"))) {
    Write-Warning "-RepoRoot $RepoRoot has no src\ folder - ignoring it."
    $RepoRoot = ""
}
$repoRoot = if ($RepoRoot) { ($RepoRoot -replace "\\", "/") } else { "" }

# ------------------------------------------------------------------ #
# Finding EffectCraft
# ------------------------------------------------------------------ #
function Find-EffectCraftSettings {
    # A portable install keeps its settings in EffectCraftData *next to*
    # effectcraft.exe; the installed app keeps them in %APPDATA%\EffectCraft.
    $roots = @(
        (Join-Path $env:LOCALAPPDATA "EffectCraft"),
        (Join-Path $env:APPDATA "EffectCraft"),
        (Join-Path $env:USERPROFILE "Downloads"),
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:USERPROFILE "Documents")
    )
    $found = @()
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $data = Get-ChildItem -Path $root -Directory -Recurse -Depth 4 `
            -Filter "EffectCraftData" -ErrorAction SilentlyContinue
        foreach ($d in $data) {
            $installDir = Split-Path -Parent $d.FullName
            $isPortable = Test-Path (Join-Path $installDir "portable.txt")
            $hasPrefs   = Test-Path (Join-Path $d.FullName "prefs.json")
            if ($isPortable -or $hasPrefs) { $found += $d.FullName }
        }
    }
    $roaming = Join-Path $env:APPDATA "EffectCraft"
    if ((Test-Path $roaming) -and ($found -notcontains $roaming)) { $found += $roaming }
    return $found
}

function Resolve-SettingsDir {
    if ($SettingsDir) { return [System.IO.Path]::GetFullPath($SettingsDir) }

    $candidates = @(Find-EffectCraftSettings)
    if ($candidates.Count -eq 1) { return $candidates[0] }
    if ($candidates.Count -gt 1) {
        Write-Host ""
        Write-Host "More than one EffectCraft install was found:"
        for ($i = 0; $i -lt $candidates.Count; $i++) {
            Write-Host ("  [{0}] {1}" -f ($i + 1), $candidates[$i])
        }
        $answer = Read-Host "Which one should Better Brush install into? [1]"
        if (-not $answer) { $answer = "1" }
        $index = 0
        if ([int]::TryParse($answer, [ref]$index) -and $index -ge 1 -and $index -le $candidates.Count) {
            return $candidates[$index - 1]
        }
        return $candidates[0]
    }

    Write-Host ""
    Write-Host "Could not find an EffectCraft install automatically."
    Write-Host "Paste the folder that contains effectcraft.exe (portable), or"
    Write-Host "the EffectCraft settings folder itself:"
    $typed = Read-Host "Path"
    if (-not $typed) { throw "No path given - nothing was installed." }
    $typed = $typed.Trim('"')
    if (Test-Path (Join-Path $typed "effectcraft.exe")) {
        $data = Join-Path $typed "EffectCraftData"
        if (-not (Test-Path $data)) { New-Item -ItemType Directory -Force -Path $data | Out-Null }
        return $data
    }
    return [System.IO.Path]::GetFullPath($typed)
}

# ------------------------------------------------------------------ #
# Removing
# ------------------------------------------------------------------ #
function Uninstall {
    Write-Host "Uninstalling Better Brush from EffectCraft..."

    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    if (Get-ItemProperty -Path $runKey -Name $runValueName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty -Path $runKey -Name $runValueName -Force
        Write-Host "  old logon entry removed (v0.4.0 registered one)"
    }
    if ($SettingsDir) {
        foreach ($name in @("start-adapter.vbs", "start-adapter.cmd")) {
            $startFile = Join-Path $SettingsDir "$bridgeName\$name"
            if (Test-Path $startFile) {
                Remove-Item -Force $startFile
                Write-Host "  $name removed"
            }
        }
    }
    Get-Process -Name "betterbrush-adapter" -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  stopped the running adapter"
    }

    if ($SettingsDir) {
        $panel = Join-Path $SettingsDir "Scripts\ScriptUI Panels\BetterBrush.jsx"
        if (Test-Path $panel) {
            Remove-Item -Force $panel
            Write-Host "  panel removed: $panel"
        }
    }
    if (Test-Path $AdapterDir) {
        Remove-Item -Recurse -Force $AdapterDir
        Write-Host "  adapter removed: $AdapterDir"
    }
    Write-Host "Done. The bridge folder and its logs were left in place."
}

if ($Uninstall) {
    if (-not $SettingsDir) { $SettingsDir = Find-EffectCraftSettings | Select-Object -First 1 }
    Uninstall
    return
}

# ------------------------------------------------------------------ #
# Installing
# ------------------------------------------------------------------ #
$SettingsDir = Resolve-SettingsDir
Write-Host "EffectCraft settings: $SettingsDir"

$bridge = Join-Path $SettingsDir $bridgeName
New-Item -ItemType Directory -Force -Path $bridge | Out-Null
Write-Host "Bridge folder:        $bridge"

# --- the panel, with the bridge path stamped in --------------------- #
# A *panel* script sees $.fileName as a bare file name, so it cannot walk
# up to its own folder: the installer writes the answer into the copy it
# ships, and the panel validates the stamp against its own location.
$panelHost = Join-Path $SettingsDir "Scripts\ScriptUI Panels"
New-Item -ItemType Directory -Force -Path $panelHost | Out-Null
$panelDst = Join-Path $panelHost "BetterBrush.jsx"
$bridgeSlash = $bridge -replace "\\", "/"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($panelDst,
    ([System.IO.File]::ReadAllText($panelSrc)).Replace("@@BB_BRIDGE@@", $bridgeSlash),
    $utf8)
Write-Host "Panel installed:      $panelDst"

# --- open it on launch ---------------------------------------------- #
$panelsFile = Join-Path $SettingsDir "scriptui_panels.json"
$open = @("BetterBrush.jsx")
if (Test-Path $panelsFile) {
    try {
        $existing = Get-Content $panelsFile -Raw | ConvertFrom-Json
        if ($existing.open) {
            $open = @($existing.open) + @($open) | Select-Object -Unique
        }
    } catch {
        # An unreadable list is replaced, not preserved: a panel that never
        # opens is worse than a list we could not read.
    }
}
[System.IO.File]::WriteAllText($panelsFile,
    (@{ open = @($open) } | ConvertTo-Json), $utf8)
Write-Host "Auto-open:            $panelsFile"

# --- the adapter ----------------------------------------------------- #
New-Item -ItemType Directory -Force -Path $AdapterDir | Out-Null
Copy-Item -Force (Join-Path $adapterSrc "run_adapter.py") $AdapterDir
Copy-Item -Force -Recurse (Join-Path $adapterSrc "effectcraft_bridge") $AdapterDir

$exe = Join-Path $AdapterDir "betterbrush-adapter.exe"
if (Test-Path (Join-Path $adapterSrc "betterbrush-adapter.exe")) {
    Copy-Item -Force (Join-Path $adapterSrc "betterbrush-adapter.exe") $AdapterDir
    Write-Host "Adapter (standalone): $exe"
}

# Point the installed copy at this bridge: protocol.recorded_bridge_dir()
# reads bridge.path.json from the folder above its own package.
$record = @{
    bridge_dir      = $bridgeSlash
    settings_dir    = ($SettingsDir -replace "\\", "/")
    panel           = ($panelDst -replace "\\", "/")
    installed       = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    via             = "dist-installer"
    # A checkout, when this install knows one: it is what makes engine mode
    # (SAM 2, torch) possible for an adapter installed outside the repo.
    # run_adapter.py puts it on sys.path; absence means simulate only.
    repo_root       = $repoRoot
} | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $AdapterDir "bridge.path.json"), $record, $utf8)

# --- the command that starts it --------------------------------------- #
# Preference order: a Python that really has the engine (torch + opencv)
# first, because the *engine* mode only exists there - the standalone exe
# carries numpy and OpenCV but no torch, so it can only run --mode simulate.
# The exe beats a Python without torch only because it needs no install at
# all. Nothing usable at all is a warning, not a silent half-install.
$pythonw = $null
foreach ($name in @("pythonw.exe", "python.exe")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $pythonw = $cmd.Source; break }
}
if (-not $pythonw) {
    $py = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($py) { $pythonw = $py.Source }
}

$hasEngine = $false
$probe = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($probe) {
    # Importing torch is the cheapest reliable proof, and this costs a few
    # seconds once - at install time, in the installer's own console.
    & $probe.Source -c "import torch, cv2, numpy" 2>$null | Out-Null
    $hasEngine = ($LASTEXITCODE -eq 0)
}

$launcher = $null
if ($pythonw -and $hasEngine) {
    $pyArgs = if ($pythonw -like "*py.exe") { " -3w" } else { "" }
    $launcher = "`"$pythonw`"$pyArgs `"$AdapterDir\run_adapter.py`" watch"
    if ($repoRoot) {
        Write-Host "Adapter:             $pythonw (full engine - SAM 2, torch)"
    } else {
        Write-Host "Adapter:             $pythonw (engine code present, but no"
        Write-Host "                     checkout was recorded - jobs will run"
        Write-Host "                     in simulate mode until BETTERBRUSH_ROOT"
        Write-Host "                     points at one)"
    }
} elseif (Test-Path $exe) {
    $launcher = "`"$exe`" watch"
    Write-Host "Adapter:             the standalone exe (simulate mode only -"
    Write-Host "                     it carries no torch)"
} elseif ($pythonw) {
    $pyArgs = if ($pythonw -like "*py.exe") { " -3w" } else { "" }
    $launcher = "`"$pythonw`"$pyArgs `"$AdapterDir\run_adapter.py`" watch"
    Write-Host "Adapter:             $pythonw (simulate mode only - torch is"
    Write-Host "                     not importable from it)"
}

$startNow = $null
if ($launcher) {
    # No logon entry: the panel runs these files when EffectCraft opens (it
    # reads <bridge>\adapter.alive first and only starts what is missing).
    # Anything an earlier version registered at logon is cleaned up here.
    $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    if (Get-ItemProperty -Path $runKey -Name $runValueName -ErrorAction SilentlyContinue) {
        Remove-ItemProperty -Path $runKey -Name $runValueName -Force
        Write-Host "Removed the old logon entry (v0.4.0 registered one)."
    }
    # start-adapter.vbs is what the panel executes: wscript starts it with no
    # window at all, where a .cmd would flash a console. start-adapter.cmd is
    # the same command for a human to double-click.
    $startVbs = Join-Path $bridge "start-adapter.vbs"
    # VBS escapes a quote by doubling it, and the launcher is full of them -
    # built with single-quoted PowerShell strings so no escape is ambiguous,
    # and hoisted into a variable: inside @() PowerShell parses the commas.
    $vbsRun = 'sh.Run "' + $launcher.Replace('"', '""') + '", 0, False'
    $vbs = @(
        "' Written by the Better Brush installer. The panel runs this file",
        "' when EffectCraft opens - nothing starts at logon.",
        'Set sh = CreateObject("WScript.Shell")',
        $vbsRun
    ) -join "`r`n"
    [System.IO.File]::WriteAllText($startVbs, $vbs + "`r`n",
        (New-Object System.Text.UTF8Encoding($false)))
    $startCmd = Join-Path $bridge "start-adapter.cmd"
    @(
        "@echo off",
        "rem Written by the Better Brush installer for EffectCraft.",
        "rem The panel runs start-adapter.vbs when EffectCraft opens.",
        "start `"`" /b $launcher"
    ) | Set-Content -Path $startCmd -Encoding Default
    Write-Host "Start files:          $startVbs"
    Write-Host "  starts with:        $launcher"
    if (-not $NoStart) { $startNow = $startVbs }
} else {
    Write-Warning ("No Python and no standalone adapter were found - the " +
        "adapter is installed but nothing will start it. Install Python 3 " +
        "(with numpy) or re-run the setup with the standalone adapter.")
}

Write-Host ""
if ($startNow) {
    Write-Host "Starting the adapter in the background..."
    Start-Process -FilePath $startNow -WindowStyle Hidden
    Write-Host "It watches $bridge and logs to $bridge\adapter.log"
}
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start EffectCraft - the Better Brush panel opens by itself."
Write-Host "  2. Preferences > Scripting & Expressions > allow"
Write-Host "     'Allow Scripts to Write Files and Access Network'."
Write-Host "  3. The panel starts the adapter itself when EffectCraft opens."
Write-Host "     EffectCraft asks for permission the first time unless"
Write-Host "     'Warn User When Executing Files' is turned off."
Write-Host "  4. Select a footage layer and click Remove Background."
