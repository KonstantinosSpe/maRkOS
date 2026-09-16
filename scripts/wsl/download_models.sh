#!/bin/bash
# Fetches the model weights into data/models. They are not committed (see NOTICE.md for their licenses).
#   scripts/wsl/download_models.sh              the YOLO11 segmentation model the bottle finder uses
#   scripts/wsl/download_models.sh --with-hands also the MediaPipe hand model (only the hand-following demo needs it)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODELS="$MARKOS_DATA_DIR/models"
mkdir -p "$MODELS"
cd "$MODELS" || exit 1

if [ ! -f yolo11n-seg.pt ]; then
  # ultralytics downloads a named model into the current directory
  "$MARKOS_PYTHON" -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')" || exit 1
fi
echo "yolo11n-seg.pt: $(du -h yolo11n-seg.pt | cut -f1)"

if [ "$1" = "--with-hands" ] && [ ! -f hand_landmarker.task ]; then
  curl -fL -o hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task || exit 1
  echo "hand_landmarker.task: $(du -h hand_landmarker.task | cut -f1)"
fi
