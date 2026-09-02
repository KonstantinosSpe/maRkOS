"""One-time setup of the two red stars on the robot's base.

Drag a box around each star in the camera picture and press ``c``: the image patches are saved as templates that
``perception.star_tracking`` matches from then on. ``f`` adds a focal-length sample (you type the measured distance to each
star), ``s`` saves a debug snapshot, ``r`` clears the templates, ``x`` clears the focal-length samples, ``q`` quits.

Run: python -m markos_vision.apps.select_stars
"""

import os

import cv2
import numpy as np

from markos_vision.config import STAR_DEBUG_DIR as DEBUG_DIR
from markos_vision.config import ensure_parent
from markos_vision.perception.camera import open_camera
from markos_vision.perception.star_tracking import (
    FOCAL_SAMPLES_PATH,
    REAL_DISTANCE_CM,
    TEMPLATES_PATH,
    StarTracker,
    load_templates,
    star_side_order,
)

boxes = []
drag_start = None
drag_end = None
dragging = False


def on_mouse(event, x, y, flags, param):
    global drag_start, drag_end, dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start = (x, y)
        drag_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        drag_end = (x, y)
        x0, y0 = drag_start
        x1, y1 = drag_end
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
            print(f"Box added: {boxes[-1]}  ({len(boxes)} total)")
        drag_start = drag_end = None


def save_templates(templates):
    data = {f"t{i}": t for i, t in enumerate(templates)}
    np.savez(ensure_parent(TEMPLATES_PATH), count=len(templates), **data)
    print(f"Saved {len(templates)} template(s) -> {TEMPLATES_PATH}")


def save_focal_samples(estimates):
    np.savez(ensure_parent(FOCAL_SAMPLES_PATH), estimates=np.array(estimates, dtype=np.float64))


def load_focal_samples():
    if not os.path.exists(FOCAL_SAMPLES_PATH):
        return []
    return list(np.load(FOCAL_SAMPLES_PATH)["estimates"])


def make_templates(frame):
    if not boxes:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    templates = []
    for (x, y, w, h) in boxes:
        t = gray[y:y + h, x:x + w].copy()
        if t.size == 0:
            continue
        templates.append(t)
    if not templates:
        return None
    save_templates(templates)
    return templates


def dump_debug(frame, tracker, detections):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    annotated = frame.copy()
    print(f"\n--- debug snapshot: {len(detections)}/{len(tracker.templates)} template(s) matched "
          f"(min_separation={tracker.min_separation}px) ---")
    for i, (cx, cy, score, (x, y, w, h), live) in enumerate(detections):
        color = (0, 255, 0) if live else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"{score:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"  [{i}] score={score:.3f} bbox=({x},{y},{w},{h}) {'live' if live else 'HELD (stale)'}")
    cv2.imwrite(f"{DEBUG_DIR}/live_raw.png", frame)
    cv2.imwrite(f"{DEBUG_DIR}/live_annotated.png", annotated)
    print(f"Saved -> {DEBUG_DIR}/live_raw.png, live_annotated.png\n")


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return

    win = "drag boxes | c=calibrate  f=add dist sample  x=clear samples  r=reset  s=debug dump  q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    templates = load_templates()
    tracker = StarTracker(templates) if templates is not None else None
    focal_samples = load_focal_samples()
    focal_length_px = float(np.mean(focal_samples)) if focal_samples else None
    if tracker is not None:
        print(f"Loaded {len(templates)} saved template(s) from {TEMPLATES_PATH} -- detecting live already.")
    else:
        print("Drag a box tightly around each star with the mouse.")
    if focal_samples:
        std = float(np.std(focal_samples)) if len(focal_samples) > 1 else 0.0
        print(f"Loaded {len(focal_samples)} focal-length sample(s) -> "
              f"estimate {focal_length_px:.1f}px (std={std:.1f}px)")
    print("Press 'c' once you've boxed them to lock in templates + start detecting live.")
    print("Press 'f' (with both stars detected) to add a camera-distance sample -- you'll be")
    print("  asked in this terminal for the measured distance to EACH star separately (L/R")
    print("  tags shown on screen match the prompts). Each press ADDS a sample and re-averages;")
    print("  do it a few times at different distances for a steadier estimate.")
    print("Press 'x' to clear all focal-length samples and start over.")
    print("Press 's' any time after calibrating to save a debug snapshot to disk.")
    print("Press 'r' to clear boxes, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            display = frame.copy()
            detections = []
            pixel_dist = None

            if tracker is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = tracker.match(gray)
                # label by actual on-screen x position (L=leftmost, R=rightmost), not
                # detection/template order -- that order has no relation to screen side
                order = star_side_order(detections)
                side_label = {idx: ("L" if rank == 0 else "R" if rank == len(order) - 1 else str(rank))
                              for rank, idx in enumerate(order)}
                for idx, (cx, cy, score, (x, y, w, h), live) in enumerate(detections):
                    color = (0, 255, 0) if live else (0, 165, 255)
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    tag = side_label[idx]
                    cv2.putText(display, f"{tag} {score:.2f}" if live else f"{tag} held",
                                (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if len(detections) >= 2:
                    x1, y1 = detections[order[0]][0], detections[order[0]][1]
                    x2, y2 = detections[order[-1]][0], detections[order[-1]][1]
                    pixel_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    status = f"{len(detections)} detected  dist={pixel_dist:.1f}px  {pixel_dist / REAL_DISTANCE_CM:.1f}px/cm"
                    if focal_length_px is not None:
                        cam_dist_cm = (REAL_DISTANCE_CM * focal_length_px) / pixel_dist
                        status += f"  cam_dist={cam_dist_cm:.1f}cm"
                else:
                    status = f"{len(detections)} detected"
                cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            for (x, y, w, h) in boxes:
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 1)
            if dragging and drag_start and drag_end:
                cv2.rectangle(display, drag_start, drag_end, (0, 165, 255), 1)

            cv2.putText(display, f"{len(boxes)} box(es)  {'LOCKED' if tracker else 'not calibrated'}",
                        (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                boxes.clear()
                tracker = None
                if os.path.exists(TEMPLATES_PATH):
                    os.remove(TEMPLATES_PATH)
                print("Cleared boxes and deleted saved templates -- box fresh ones and press 'c'.")
            elif key == ord('c'):
                templates = make_templates(frame)
                if templates is None:
                    print("No boxes yet -- drag one first.")
                else:
                    tracker = StarTracker(templates)
                    print(f"Locked in {len(templates)} template(s).")
            elif key == ord('s'):
                if tracker is None:
                    print("Calibrate first ('c') before dumping debug info.")
                else:
                    dump_debug(frame, tracker, detections)
            elif key == ord('f'):
                if pixel_dist is None:
                    print("Need both stars detected right now to calibrate camera distance.")
                else:
                    try:
                        print("Measure each star separately -- 'L'/'R' match the tags shown on screen.")
                        left_cm = float(input("  Distance to the L star (cm): "))
                        right_cm = float(input("  Distance to the R star (cm): "))
                        avg_cm = (left_cm + right_cm) / 2
                        estimate = pixel_dist * avg_cm / REAL_DISTANCE_CM
                        focal_samples.append(estimate)
                        save_focal_samples(focal_samples)
                        focal_length_px = float(np.mean(focal_samples))
                        std = float(np.std(focal_samples)) if len(focal_samples) > 1 else 0.0
                        print(f"  sample #{len(focal_samples)}: {estimate:.1f}px "
                              f"(avg_cm={avg_cm:.1f}, L={left_cm}, R={right_cm})")
                        print(f"  estimate now {focal_length_px:.1f}px "
                              f"from {len(focal_samples)} sample(s), std={std:.1f}px")
                        if len(focal_samples) < 3:
                            print("  take a couple more at different distances for a steadier estimate.")
                    except ValueError:
                        print("Not a number -- calibration cancelled.")
            elif key == ord('x'):
                focal_samples.clear()
                focal_length_px = None
                if os.path.exists(FOCAL_SAMPLES_PATH):
                    os.remove(FOCAL_SAMPLES_PATH)
                print("Cleared focal-length samples.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
