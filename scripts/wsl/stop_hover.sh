#!/bin/bash
# Closes the hover window only (the bridge keeps running). Its saved tuning is re-read the next time it starts.
pkill -f '[m]arkos_vision.apps.hover'
sleep 1
echo "hover window closed: $(pgrep -fa '[m]arkos_vision.apps.hover' | wc -l) still running"
