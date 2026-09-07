@echo off
title Open Thor RViz
call "%~dp0_common.bat"
rem RViz with the Thor model; with the bridge running it shows the arm's confirmed position.
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/rviz.sh
