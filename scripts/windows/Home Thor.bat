@echo off
title Home Thor
call "%~dp0_common.bat"
echo This HOMES the arm: it disarms, then moves B, D and E to their home sensors. The arm moves for up to 40 seconds.
set /p ans=The area around the arm is clear and your hand is by the power switch? (y/N):
if /i not "%ans%"=="y" goto end
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/home.sh
:end
echo.
pause
