#!/bin/bash
# The tape calibration tool. It needs the camera, so the hover window has to be closed.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export QT_QPA_PLATFORM=xcb
if pgrep -f '[m]arkos_vision.apps.hover' > /dev/null; then
  echo "The hover window is open and is using the camera. Close it first (press q in it), then run this again."
  read -r -p "Press Enter to close this window. " _
  exit 1
fi
"$MARKOS_PYTHON" -m markos_vision.apps.calibrate_desk
read -r -p "Press Enter to close this window. " _
