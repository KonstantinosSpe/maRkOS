#!/bin/bash
# Stops the arm from moving at once; the motors keep holding where it is.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
timeout 20 ros2 param set /thor_bridge armed false 2>&1 | tail -1
echo "armed now: $(timeout 10 ros2 param get /thor_bridge armed 2>&1 | tail -1)"
