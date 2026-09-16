#!/bin/bash
# Shared environment for the WSL scripts: source this file, do not run it.
#
# Everything is found from where this file lives, so the repository can be cloned anywhere. To change a default without
# editing the repository, put the variable in scripts/wsl/local.sh (git-ignored); it is read at the end of this file.
#
#   MARKOS_REPO       the repository root (derived)
#   MARKOS_DATA_DIR   calibration, enrolled bottles, star templates, models, logs   (default: <repo>/data)
#   MARKOS_ROS_WS     colcon workspace the ROS 2 packages are built into           (default: ~/markos_ws)
#   MARKOS_VENV       Python virtual environment with OpenCV, ultralytics, ...     (default: ~/markos_venv)
#   MARKOS_ROS_DISTRO the ROS 2 distribution installed under /opt/ros              (default: lyrical)
#   THOR_SERIAL_PORT  the Arduino's serial device inside WSL                        (default: /dev/ttyUSB0)

MARKOS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "$MARKOS_REPO/scripts/wsl/local.sh" ]; then
  # shellcheck disable=SC1091
  source "$MARKOS_REPO/scripts/wsl/local.sh"
fi

export MARKOS_REPO
export MARKOS_DATA_DIR="${MARKOS_DATA_DIR:-$MARKOS_REPO/data}"
export MARKOS_ROS_WS="${MARKOS_ROS_WS:-$HOME/markos_ws}"
export MARKOS_VENV="${MARKOS_VENV:-$HOME/markos_venv}"
export MARKOS_ROS_DISTRO="${MARKOS_ROS_DISTRO:-lyrical}"
export THOR_SERIAL_PORT="${THOR_SERIAL_PORT:-/dev/ttyUSB0}"
MARKOS_PYTHON="${MARKOS_PYTHON:-$MARKOS_VENV/bin/python3}"

# ROS 2, then the workspace the packages were built into (if it has been built)
# shellcheck disable=SC1090
source "/opt/ros/$MARKOS_ROS_DISTRO/setup.bash"
if [ -f "$MARKOS_ROS_WS/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$MARKOS_ROS_WS/install/setup.bash"
fi

# The source tree comes first, so an old installed copy of a package can never shadow the code being edited.
export PYTHONPATH="$MARKOS_REPO/vision:$MARKOS_REPO/ros2/markos${PYTHONPATH:+:$PYTHONPATH}"
