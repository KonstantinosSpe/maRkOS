"""Live check of the colour/contour star detector (first approach; see ``diagnose_stars`` and ``capture_star_shape``)."""

import os

import cv2
import numpy as np

from markos_vision.config import STAR_REFERENCE_CONTOUR as REF_PATH

REAL_DISTANCE_CM = 7.9
SHAPE_MATCH_MAX = 0.3  # lower = stricter; matchShapes score, ~0 = identical

LOWER_RED_1 = np.array([0, 50, 225])
UPPER_RED_1 = np.array([10, 140, 255])
LOWER_RED_2 = np.array([170, 50, 225])
UPPER_RED_2 = np.array([179, 140, 255])

MIN_AREA = 50


def find_contours(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_AREA]


def main():
    reference = None
    if os.path.exists(REF_PATH):
        reference = np.load(REF_PATH)
        print(f"Loaded reference shape ({len(reference)} points) from {REF_PATH}")
    else:
        print(f"No reference at {REF_PATH} yet -- run capture_star_shape.py first. "
              "Falling back to color-only matching for now.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            contours = find_contours(frame)

            accepted = []
            for c in contours:
                score = cv2.matchShapes(c, reference, cv2.CONTOURS_MATCH_I1, 0.0) if reference is not None else 0.0
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                is_star = reference is None or score <= SHAPE_MATCH_MAX
                color = (0, 255, 0) if is_star else (0, 0, 255)
                cv2.drawContours(frame, [c], -1, color, 2)
                cv2.putText(frame, f"{score:.2f}", (int(cx) - 10, int(cy) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                if is_star:
                    accepted.append((cx, cy))

            if len(accepted) >= 2:
                (x1, y1), (x2, y2) = accepted[0], accepted[1]
                pixel_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                px_per_cm = pixel_dist / REAL_DISTANCE_CM
                status = f"{len(accepted)} confirmed star(s)  dist={pixel_dist:.1f}px  {px_per_cm:.1f}px/cm"
            else:
                status = f"{len(accepted)} confirmed star(s), {len(contours)} color candidate(s)"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
