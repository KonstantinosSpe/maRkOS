# Starts the Thor arm project after a reboot.
#   1. finds the camera and the Arduino by their USB ids, attaches them to WSL and keeps them attached if they drop out
#   2. waits until WSL can see both and the serial port answers steadily
#   3. asks you to confirm the arm is clear, then starts the bridge, which HOMES the arm (it moves)
#   4. once homing is complete, starts the hover window
# It never arms the arm. Use "Arm Thor.bat" for that, then press g in the hover window.
#
# The USB ids are those of the author's laptop webcam and of the Arduino's CH340 serial chip. Find yours with `usbipd list`
# and pass them in (-CameraId 1234:abcd), or edit the defaults below. usbipd-win must be installed, and each device shared
# once from an administrator PowerShell:  usbipd bind --busid <id>
param(
    [switch]$CheckOnly,                    # attach and check the devices, but do not start the bridge or the hover window
    [string]$CameraId = "30c9:0035",
    [string]$ArduinoId = "1a86:7523",
    [string]$Distro = $(if ($env:THOR_DISTRO) { $env:THOR_DISTRO } else { "Ubuntu" }),
    [string]$SerialPort = "/dev/ttyUSB0"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BridgeLog = Join-Path $Repo "data\logs\bridge.log"
$Wanted = @(
    @{ Name = "camera";  Id = $CameraId },
    @{ Name = "Arduino"; Id = $ArduinoId }
)

function Get-Usb($id) {
    foreach ($line in (usbipd list 2>$null)) {
        if ($line -match "^\s*(\d+-\d+)\s+$([regex]::Escape($id))\s+.*?\s{2,}(Not shared|Shared|Attached)\s*$") {
            return @{ BusId = $Matches[1]; State = $Matches[2] }
        }
    }
    return $null
}

function Get-Helper($busId) {
    Get-CimInstance Win32_Process -Filter "Name='usbipd.exe'" |
        Where-Object { $_.CommandLine -like "*--auto-attach*" -and $_.CommandLine -like ("*--busid " + $busId + " *") }
}

Write-Host ""
Write-Host "== 1. USB devices =="
foreach ($d in $Wanted) {
    $usb = $null
    for ($i = 0; $i -lt 60 -and -not $usb; $i++) {
        $usb = Get-Usb $d.Id
        if (-not $usb) {
            if ($i -eq 0) {
                $extra = if ($d.Name -eq "Arduino") { " and that the arm's power is on." } else { "." }
                Write-Host ("Windows can't see the {0} yet. Check its USB cable{1}" -f $d.Name, $extra)
            }
            Start-Sleep -Seconds 1
        }
    }
    if (-not $usb) { Write-Host ("The {0} never appeared. Nothing was started." -f $d.Name) -ForegroundColor Red; exit 1 }
    if ($usb.State -eq "Not shared") {
        Write-Host ("The {0} (bus {1}) has never been shared. One time only, in an administrator PowerShell: usbipd bind --busid {1}" -f $d.Name, $usb.BusId) -ForegroundColor Red
        exit 1
    }
    # A helper that is already keeping this device attached is left alone: stopping it drops the device for a moment,
    # which is exactly when the bridge would be opening the port.
    if (Get-Helper $usb.BusId) {
        Write-Host ("{0}: bus {1}, already being kept attached." -f $d.Name, $usb.BusId)
    } else {
        Start-Process -FilePath "usbipd" -ArgumentList @("attach", "--wsl", "--busid", $usb.BusId, "--auto-attach") -WindowStyle Hidden
        Write-Host ("{0}: bus {1}, attaching (and re-attaching if it drops out)." -f $d.Name, $usb.BusId)
    }
}

Write-Host ""
Write-Host "== 2. Waiting for WSL to see them, and for the serial port to answer steadily =="
$stable = 0
for ($i = 0; $i -lt 60 -and $stable -lt 3; $i++) {
    $r = wsl -d $Distro -- bash -c "stty -F $SerialPort > /dev/null 2>&1 && test -e /dev/video0 && echo ready" 2>$null
    if ("$r".Trim() -eq "ready") { $stable++ } else { $stable = 0 }
    Start-Sleep -Seconds 1
}
if ($stable -lt 3) { Write-Host "The Arduino's serial port ($SerialPort) or the camera (/dev/video0) never became usable in WSL. Nothing was started." -ForegroundColor Red; exit 1 }
Write-Host "Both are there and steady: $SerialPort (the Arduino) and /dev/video0 (the camera)."
if ($CheckOnly) { Write-Host "Check only: nothing else was started."; exit 0 }

Write-Host ""
Write-Host "== 3. The bridge =="
Write-Host "Starting it HOMES the arm: B, then D, then E. The arm moves (D can take up to 40 seconds)."
Read-Host "Motor power on, the area around the arm clear, your hand by the power switch? Press Enter to start (Ctrl+C to cancel)"
New-Item -ItemType Directory -Force (Split-Path $BridgeLog) | Out-Null
Set-Content -Path $BridgeLog -Value "" -NoNewline
Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--cd", $Repo, "--", "bash", "scripts/wsl/start_bridge.sh")

$homed = $false
for ($i = 0; $i -lt 75 -and -not $homed; $i++) {
    Start-Sleep -Seconds 2
    $text = if (Test-Path $BridgeLog) { Get-Content $BridgeLog -Raw -ErrorAction SilentlyContinue } else { "" }
    if ($text -match "Homing complete") { $homed = $true }
    elseif ($text -match "Could not open|Traceback") {
        Write-Host "The bridge could not start. Its window shows why. Run this again; if it fails twice, check the Arduino's USB cable. The hover window was not started." -ForegroundColor Red
        exit 1
    }
}
if (-not $homed) { Write-Host "No 'Homing complete' after 150 seconds. Look at the bridge window. The hover window was not started." -ForegroundColor Red; exit 1 }
Write-Host "Homing complete."

Write-Host ""
Write-Host "== 4. The hover window =="
Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--cd", $Repo, "--", "bash", "scripts/wsl/start_hover.sh")

Write-Host ""
Write-Host "Running. The arm is NOT armed yet, and g is off in the hover window."
Write-Host "When the hover window shows the bottle recognised and steady: run 'Arm Thor.bat', then press g in it."
Write-Host "To stop the arm at once: 'Disarm Thor.bat' (or press g again). To close everything: 'Stop Thor.bat'."
