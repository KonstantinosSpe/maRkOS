"""Prints where the palm is, in centimetres, in the frame defined by the two red stars on the robot's base.

First step of the hand-following demo (``follow_palm_ros.py`` adds the arm). Needs the star templates
(``markos_vision.apps.select_stars``) and the MediaPipe hand model (``scripts/download_models.sh``).
"""

import sys

import cv2
import mediapipe as mp

from markos_vision.config import HAND_LANDMARKER as MODEL_PATH
from markos_vision.perception.camera import open_camera
from markos_vision.perception.star_tracking import StarTracker, load_templates, star_frame, star_side_order, to_local_cm

PALM_LANDMARK = 9  # middle finger MCP -- same stable "hand center" point gesture_math.py's hand_scale() uses


def main():
    templates = load_templates()
    if templates is None:
        print("No saved star templates found -- run markos_vision.apps.select_stars first, "
              "box both stars, and press 'c' to lock + save them.")
        sys.exit(1)
    tracker = StarTracker(templates)

    base_opts = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)

    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return

    print(f"Loaded {len(templates)} star template(s). Tracking palm (landmark {PALM_LANDMARK}).")
    print("Offsets are in cm, relative to the midpoint between the two stars.")
    print("Press 'q' to quit.")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            display = frame.copy()
            h_img, w_img = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = tracker.match(gray)
            order = star_side_order(detections)
            for (cx, cy, score, (x, y, w, h), live) in detections:
                color = (0, 255, 0) if live else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            frame_geom = star_frame(detections, order) if len(detections) >= 2 else None
            if frame_geom is not None:
                ox, oy = frame_geom[0]
                cv2.circle(display, (int(ox), int(oy)), 5, (0, 255, 255), -1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000 / 30))
            frame_idx += 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status = "no hand"
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                for point in lm:
                    cv2.circle(display, (int(point.x * w_img), int(point.y * h_img)), 2, (200, 200, 0), -1)
                px, py = lm[PALM_LANDMARK].x * w_img, lm[PALM_LANDMARK].y * h_img
                cv2.circle(display, (int(px), int(py)), 6, (255, 0, 255), -1)

                if frame_geom is not None:
                    dx_cm, dy_cm = to_local_cm((px, py), frame_geom)
                    status = f"palm offset: x={dx_cm:+.1f}cm  y={dy_cm:+.1f}cm"
                    ox, oy = frame_geom[0]
                    cv2.line(display, (int(ox), int(oy)), (int(px), int(py)), (255, 0, 255), 1)
                else:
                    status = "hand detected, need both stars for offset"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("follow palm", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
