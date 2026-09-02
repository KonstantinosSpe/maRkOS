"""Diagnostic for the colour/contour star detector: saves the raw frame, the colour masks and an annotated frame.

Part of the first (colour + contour shape) approach to finding the stars, replaced by template matching in
``markos_vision.perception.star_tracking``.
"""

import os
import time

import cv2
import numpy as np

from markos_vision.config import STAR_DEBUG_DIR as OUT_DIR
from markos_vision.config import STAR_REFERENCE_CONTOUR as REF_PATH

LOWER_RED_1 = np.array([0, 40, 90])
UPPER_RED_1 = np.array([15, 200, 255])
LOWER_RED_2 = np.array([165, 40, 90])
UPPER_RED_2 = np.array([179, 200, 255])
MIN_AREA = 50
SHAPE_MATCH_MAX = 0.3

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("FAILED to open camera")
    raise SystemExit(1)

# warm up (auto-exposure/white-balance settle)
ok, frame = False, None
for i in range(25):
    ok, frame = cap.read()
    time.sleep(0.03)

if not ok or frame is None:
    print("FAILED to read frame")
    raise SystemExit(1)

cv2.imwrite(f"{OUT_DIR}/raw.png", frame)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_raw = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
cv2.imwrite(f"{OUT_DIR}/mask_raw.png", mask_raw)

mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
cv2.imwrite(f"{OUT_DIR}/mask_opened.png", mask)

all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in all_contours if cv2.contourArea(c) >= MIN_AREA]

reference = np.load(REF_PATH) if os.path.exists(REF_PATH) else None
print(f"Reference loaded: {reference is not None} ({REF_PATH})")
print(f"{len(all_contours)} raw contours, {len(contours)} pass MIN_AREA={MIN_AREA}\n")

annotated = frame.copy()
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] == 0:
        continue
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    score = cv2.matchShapes(c, reference, cv2.CONTOURS_MATCH_I1, 0.0) if reference is not None else -1
    x, y, w, h = cv2.boundingRect(c)
    roi_hsv = hsv[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    pix = roi_hsv[roi_mask > 0].astype(np.float32)
    if len(pix):
        hmean, smean, vmean = pix.mean(axis=0)
        hstd, sstd, vstd = pix.std(axis=0)
    else:
        hmean = smean = vmean = hstd = sstd = vstd = -1
    perim = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else -1
    print(f"[{i}] area={area:.0f} bbox=({x},{y},{w},{h}) centroid=({cx:.0f},{cy:.0f}) "
          f"circularity={circularity:.3f} matchShapes={score:.3f}")
    print(f"     H={hmean:.1f}+-{hstd:.1f}  S={smean:.1f}+-{sstd:.1f}  V={vmean:.1f}+-{vstd:.1f}  npix={len(pix)}")

    is_star = reference is not None and score <= SHAPE_MATCH_MAX
    color = (0, 255, 0) if is_star else (0, 0, 255)
    cv2.drawContours(annotated, [c], -1, color, 2)
    cv2.putText(annotated, f"{i}:{score:.2f}", (int(cx) - 15, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(f"{OUT_DIR}/annotated.png", annotated)
print(f"\nSaved raw.png, mask_raw.png, mask_opened.png, annotated.png -> {OUT_DIR}/")
cap.release()
