"""Records the outline of one clean red star as a reference contour (first, colour-based star detector).

Hold a single star steady and it saves itself. Replaced by template matching (``markos_vision.apps.select_stars``).
"""

import time

import cv2
import numpy as np

from markos_vision.config import STAR_REFERENCE_CONTOUR as OUT_PATH
from markos_vision.config import ensure_parent

LOWER_RED_1 = np.array([0, 50, 225])
UPPER_RED_1 = np.array([10, 140, 255])
LOWER_RED_2 = np.array([170, 50, 225])
UPPER_RED_2 = np.array([179, 140, 255])

MIN_AREA = 50

HOLD_SECONDS = 1.2
POS_TOLERANCE_PX = 8
AREA_TOLERANCE_FRAC = 0.15


def find_contours(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_AREA]


def centroid(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return

    print("Point at ONE clean, isolated star and hold it steady -- it saves itself")
    print("automatically once it's been stable for a second or so. 'q' to cancel.")

    hold_start = None
    ref_pos = None
    ref_area = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            contours = find_contours(frame)
            contours.sort(key=cv2.contourArea, reverse=True)

            stable_frac = 0.0
            if len(contours) == 1:
                c = contours[0]
                pos = centroid(c)
                area = cv2.contourArea(c)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

                if pos is not None:
                    if (ref_pos is not None
                            and abs(pos[0] - ref_pos[0]) <= POS_TOLERANCE_PX
                            and abs(pos[1] - ref_pos[1]) <= POS_TOLERANCE_PX
                            and abs(area - ref_area) <= AREA_TOLERANCE_FRAC * ref_area):
                        # still within tolerance of when the hold started
                        pass
                    else:
                        hold_start = time.time()
                        ref_pos, ref_area = pos, area

                    elapsed = time.time() - hold_start
                    stable_frac = min(elapsed / HOLD_SECONDS, 1.0)
                    if elapsed >= HOLD_SECONDS:
                        np.save(ensure_parent(OUT_PATH), c)
                        print(f"Saved reference contour ({len(c)} points, area={area:.0f}px) -> {OUT_PATH}")
                        break
            else:
                hold_start = None
                ref_pos = ref_area = None
                cv2.putText(frame, f"{len(contours)} blob(s) in view -- show just one",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

            if len(contours) == 1:
                cv2.putText(frame, f"holding... {stable_frac*100:.0f}%",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
