@echo off
setlocal
title Better Brush for EffectCraft - installer
echo.
echo  Better Brush for EffectCraft (beta)
echo  ===================================
echo.
echo  This installs the panel into EffectCraft and writes the bridge folder.
echo  Nothing is registered at logon - the panel starts the adapter itself.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
if errorlevel 1 (
  echo.
  echo  The installer did not finish. If EffectCraft was not found, re-run with
  echo  the folder that holds effectcraft.exe:
  echo.
  echo      install.cmd -SettingsDir "D:\Apps\effectcraft-0.17.0-windows-x64-portable\EffectCraftData"
  echo.
)
pause
