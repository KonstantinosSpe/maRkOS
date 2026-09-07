#!/bin/bash
# RViz with the Thor model. With the bridge running it shows the arm's confirmed position (and the hover targets).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export QT_QPA_PLATFORM=xcb
ros2 launch thor_urdf display.launch.py
