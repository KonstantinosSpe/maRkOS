@echo off
title Calibrate desk
call "%~dp0_common.bat"
echo The tape calibration: place the bottle on each tape spot, press c, type its u v.
echo The camera is used, so close the hover window first. Do not move the laptop while you do this.
echo.
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/calibrate_desk.sh
