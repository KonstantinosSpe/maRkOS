#!/bin/bash
# Re-homes the arm through the running bridge: it disarms itself, then moves B, D and E to their home sensors.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
timeout 30 ros2 service call /home std_srvs/srv/Trigger "{}" 2>&1 | tail -3
echo "The bridge window says 'Homing complete' when it has finished (up to about 40 seconds). It is NOT armed afterwards."
