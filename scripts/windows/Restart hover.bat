@echo off
title Restart hover window
call "%~dp0_common.bat"
echo Closing the hover window and opening a fresh one. The bridge is not touched.
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/stop_hover.sh
start "" wsl.exe -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/start_hover.sh
timeout /t 3 > nul
