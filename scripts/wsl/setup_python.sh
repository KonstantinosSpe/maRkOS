#!/bin/bash
# Creates the Python environment for the vision tools and installs the repository into it (editable).
#   scripts/wsl/setup_python.sh            OpenCV + ultralytics (CPU PyTorch): everything the hover window needs
#   scripts/wsl/setup_python.sh --no-yolo  only NumPy, OpenCV and pySerial: enough for the calibration tools and the tests
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ ! -x "$MARKOS_PYTHON" ]; then
  python3 -m venv "$MARKOS_VENV" || exit 1
fi
"$MARKOS_PYTHON" -m pip install --upgrade pip

EXTRAS="[gui]"
if [ "$1" != "--no-yolo" ]; then
  # the CPU build of PyTorch is far smaller than the default one and is all a 640x480 camera needs
  "$MARKOS_PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || exit 1
  EXTRAS="[yolo]"
fi
"$MARKOS_PYTHON" -m pip install -e "$MARKOS_REPO${EXTRAS}" || exit 1
echo "Python environment ready: $MARKOS_VENV"
