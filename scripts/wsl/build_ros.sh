#!/bin/bash
# Builds the ROS 2 packages in ros2/ into a colcon workspace outside the repository (default ~/markos_ws).
# --symlink-install: the installed packages point at the sources here, so edits take effect without rebuilding.
# Extra arguments go to colcon, e.g.  scripts/wsl/build_ros.sh --packages-select markos
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$MARKOS_ROS_WS"
cd "$MARKOS_ROS_WS" || exit 1
colcon build --base-paths "$MARKOS_REPO/ros2" --symlink-install "$@"
echo
echo "Built into $MARKOS_ROS_WS. New terminals pick it up through scripts/wsl/env.sh."
