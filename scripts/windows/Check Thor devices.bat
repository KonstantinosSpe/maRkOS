@echo off
title Check Thor devices
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Thor.ps1" -CheckOnly
echo.
pause
