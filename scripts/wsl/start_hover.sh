#!/bin/bash
# The hover window: watches the camera and, once the bridge is armed and you press g in it, sends the arm over the bottle.
# Extra arguments go to the program, e.g. --simulate 30,335 for a made-up bottle with no camera.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export QT_QPA_PLATFORM=xcb
"$MARKOS_PYTHON" -m markos_vision.apps.hover "$@"
echo "The hover window closed."
read -r -p "Press Enter to close this window. " _
