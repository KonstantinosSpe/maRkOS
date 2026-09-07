@echo off
title Arm Thor
call "%~dp0_common.bat"
echo This lets the arm move. It only moves while the hover window has g on and the bottle is recognised.
set /p ans=Arm it now, with your hand by the power switch? (y/N):
if /i not "%ans%"=="y" goto end
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/arm.sh
:end
echo.
pause
