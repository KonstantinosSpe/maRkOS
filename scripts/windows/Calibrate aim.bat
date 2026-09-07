@echo off
title Calibrate aim
call "%~dp0_common.bat"
echo Measures where the gripper really goes, using the bottle as the ruler. The arm moves to a few poses.
echo Needs: Start Thor.bat already run, the hover window open with g OFF, the area around the arm clear.
echo Afterwards, "Calibrate aim.bat --apply" writes the result into the settings.
echo.
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/aim_probe.sh %*
