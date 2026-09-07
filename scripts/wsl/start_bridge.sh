#!/bin/bash
# The bridge to the real arm. Homes B, then D, then E on start, so the arm MOVES. Close this window to stop it.
#   MARKOS_MOCK=1 scripts/wsl/start_bridge.sh     talks to a simulated firmware instead: nothing moves
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LOG="$MARKOS_DATA_DIR/logs/bridge.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

if [ -n "$MARKOS_MOCK" ]; then
  echo "Thor bridge, MOCK mode: a simulated firmware, the arm is not involved. Close this window to stop it."
  ARGS=(-p mock:=true)
else
  echo "Thor bridge on $THOR_SERIAL_PORT. It homes the arm now; keep clear. Close this window to stop the bridge."
  ARGS=(-p "port:=$THOR_SERIAL_PORT")
fi
ros2 run markos thor_bridge --ros-args "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "The bridge stopped."
read -r -p "Press Enter to close this window. " _
