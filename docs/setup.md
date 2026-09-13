# Setup

The project was developed on Windows 10 with WSL2 (Ubuntu 26.04, ROS 2 Lyrical, Python 3.14). The Arduino and the webcam are plugged into
Windows and handed to WSL, where ROS 2 and the vision code run. Nothing is tied to those exact versions except the desktop teleop app,
which needs Python 3.10 or 3.11 because of MediaPipe 0.10.9.

## 1. Development environment only (no arm, no ROS)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,headless]"
pytest
```

## 2. The robot stack on Windows + WSL2

**Windows.** Install [usbipd-win](https://github.com/dorssel/usbipd-win) and a WSL2 Ubuntu (`wsl --install -d Ubuntu`). Find the USB ids of
your camera and Arduino with `usbipd list`. Share each **once**, in an administrator PowerShell:

```powershell
usbipd bind --busid <bus id of the camera>
usbipd bind --busid <bus id of the Arduino>
```

**WSL.** Install ROS 2 for your Ubuntu release from the [official instructions](https://docs.ros.org/) (the scripts default to
`/opt/ros/lyrical`; set `MARKOS_ROS_DISTRO` to change that), plus `colcon` and `xacro`. Then, from the repository:

```bash
scripts/wsl/setup_python.sh            # a virtual environment with OpenCV, ultralytics and this repository (editable)
scripts/wsl/download_models.sh         # the YOLO segmentation weights into data/models
scripts/wsl/build_ros.sh               # colcon build of ros2/ into ~/markos_ws
```

Every script finds the repository from its own location, so it can be cloned anywhere, including a Windows folder (WSL sees it under
`/mnt/c/...`). Per-machine settings go in `scripts/wsl/local.sh` (git-ignored), which `env.sh` reads:

```bash
# scripts/wsl/local.sh
export MARKOS_VENV=$HOME/my_venv
export THOR_SERIAL_PORT=/dev/ttyACM0
```

| Variable | Default | Meaning |
|---|---|---|
| `MARKOS_DATA_DIR` | `<repo>/data` | Calibration, enrolled bottles, star templates, models, logs |
| `MARKOS_ROS_WS` | `~/markos_ws` | Where `build_ros.sh` builds |
| `MARKOS_VENV` | `~/markos_venv` | Python environment the vision tools run in |
| `MARKOS_ROS_DISTRO` | `lyrical` | Which `/opt/ros/<distro>` to source |
| `THOR_SERIAL_PORT` | `/dev/ttyUSB0` | The Arduino inside WSL |
| `MARKOS_MOCK` | unset | `1` makes `start_bridge.sh` use the firmware simulator |

**Windows launchers.** Double-click the files in `scripts\windows\`. Check the two USB ids at the top of `Start-Thor.ps1` against
`usbipd list` (they are the author's laptop webcam and a CH340 serial chip), or pass `-CameraId` and `-ArduinoId`. A WSL distribution with
another name than `Ubuntu` is selected with the `THOR_DISTRO` environment variable.

## 3. First run, in order

1. **Star templates** (once): `python -m markos_vision.apps.select_stars`. Drag a box around each red star and press `c`; press `f` a few times
   with both stars detected to record focal-length samples (you type the measured distance to each star).
2. **Enrol the bottle**: `python -m markos_vision.apps.find_bottle`, press `a` with the bottle outlined, and give its name, real height, cap
   height and foot diameter.
3. **Check the directions** with the bridge running and homed (`MARKOS_MOCK=1` first to see the flow):
   `python -m markos_vision.apps.direction_test`.
4. **Calibrate the desk**: [calibration.md](calibration.md).
5. `Start Thor.bat`, then [operating.md](operating.md).

## 4. The desktop teleop app (Windows, no ROS)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements\teleop.txt
.venv\Scripts\python teleop\mark1os.py
```

Flash `firmware/thor_teleop_firmware/thor_teleop_firmware.ino` to the Arduino Mega first (Arduino IDE, no extra libraries beyond `Servo`).
The app and the ROS bridge cannot run at the same time: both need the serial port.
