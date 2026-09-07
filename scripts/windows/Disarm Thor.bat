@echo off
title Disarm Thor
call "%~dp0_common.bat"
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/disarm.sh
echo.
pause
