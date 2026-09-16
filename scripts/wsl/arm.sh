#!/bin/bash
# Lets the bridge move the arm. It only moves while the hover window has g on and the bottle is recognised.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
timeout 20 ros2 param set /thor_bridge armed true 2>&1 | tail -1
echo "armed now: $(timeout 10 ros2 param get /thor_bridge armed 2>&1 | tail -1)"
