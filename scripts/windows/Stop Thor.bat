@echo off
title Stop Thor
call "%~dp0_common.bat"
echo Disarming, then closing the hover window and the bridge...
wsl -d %THOR_DISTRO% --cd "%THOR_REPO%" -- bash scripts/wsl/stop_all.sh
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='usbipd.exe'\" | Where-Object { $_.CommandLine -like '*--auto-attach*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo The USB auto-attach was stopped too, so the devices will detach when WSL shuts down.
echo.
pause
