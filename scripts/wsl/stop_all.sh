#!/bin/bash
# Disarms, then closes the hover window and the bridge.
bash "$(dirname "${BASH_SOURCE[0]}")/disarm.sh"
pkill -f '[m]arkos_vision.apps.hover'
pkill -f '[t]hor_bridge --ros-args'
pkill -f 'ros2 run markos [t]hor_bridge'
sleep 2
echo "still running: $(pgrep -fa '[m]arkos_vision.apps.hover|[t]hor_bridge' | wc -l) processes"
