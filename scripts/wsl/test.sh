#!/bin/bash
# Runs the test suite in the robot environment (real ROS 2, the vision virtual environment), so the bridge test runs too.
# Arguments go to pytest, e.g.  scripts/wsl/test.sh -k diameter -q
# The tests use a temporary data folder, never the calibration in data/.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MARKOS_REPO" || exit 1
"$MARKOS_PYTHON" -m pytest "$@"
