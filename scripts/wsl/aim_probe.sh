#!/bin/bash
# Measures where the gripper really goes, using the bottle as the ruler. See vision/markos_vision/apps/pose_probe.py.
#   scripts/wsl/aim_probe.sh            run the measurement
#   scripts/wsl/aim_probe.sh --apply    write the last fit into the hover settings and the calibration
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$MARKOS_PYTHON" -m markos_vision.apps.pose_probe "$@"
read -r -p "Press Enter to close this window. " _
