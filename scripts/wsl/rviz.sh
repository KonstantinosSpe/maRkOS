#!/bin/bash
# RViz with the Thor model.
#   scripts/wsl/rviz.sh             follows the bridge's /joint_states and shows the hover targets (use this with the bridge running)
#   scripts/wsl/rviz.sh --sliders   adds a joint slider window, to look at the model with no bridge (the sliders would fight a bridge)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export QT_QPA_PLATFORM=xcb
if [ "$1" = "--sliders" ]; then
  ros2 launch thor_urdf display.launch.py
else
  ros2 launch thor_urdf follow_palm.launch.py
fi
