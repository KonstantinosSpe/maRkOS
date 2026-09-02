"""Teach ``item_tracking`` a new item: drag a box around it in several views and press the save key.

Run: python vision/experiments/teach_item.py
"""

import os

import cv2
import numpy as np
from item_tracking import ItemTracker, clear_crops, load_crops, save_crop

from markos_vision.config import ITEM_DEBUG_DIR as DEBUG_DIR
from markos_vision.perception.camera import open_camera

box = None
drag_start = None
drag_end = None
dragging = False


def on_mouse(event, x, y, flags, param):
    global box, drag_start, drag_end, dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start = drag_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        x0, x1 = sorted((drag_start[0], x))
        y0, y1 = sorted((drag_start[1], y))
        if x1 - x0 >= 12 and y1 - y0 >= 12:
            box = (x0, y0, x1 - x0, y1 - y0)
            print(f"Box: {box}")
        drag_start = drag_end = None


def draw_result(display, tracker, found):
    # Thin orange = passed the size/shape checks; thick = the one it picked.
    for corners in tracker.candidates:
        cv2.polylines(display, [corners.astype(np.int32)], True, (0, 140, 255), 1)
    if found is None:
        cv2.putText(display, "item not found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        return
    color = (0, 255, 0) if found["live"] else (0, 165, 255)
    cv2.polylines(display, [found["corners"].astype(np.int32)], True, color, 2)
    label = (f"item  {found['long']:.0f}x{found['short']:.0f}px  turned {found['angle']:+.0f}deg"
             if found["live"] else "item held")
    cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def dump_debug(frame, tracker, found):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    annotated = frame.copy()
    draw_result(annotated, tracker, found)
    if found is None:
        print(f"\n--- debug snapshot: item NOT found, {len(tracker.candidates)} candidate blob(s) ---")
    else:
        print(f"\n--- debug snapshot: {found['long']:.0f}x{found['short']:.0f}px, turned {found['angle']:+.0f}deg, "
              f"{'live' if found['live'] else 'HELD (stale)'}, {len(tracker.candidates)} candidate blob(s) ---")
    cv2.imwrite(f"{DEBUG_DIR}/live_raw.png", frame)
    cv2.imwrite(f"{DEBUG_DIR}/live_annotated.png", annotated)
    if tracker.mask is not None:
        cv2.imwrite(f"{DEBUG_DIR}/live_mask.png", tracker.mask)
    print(f"Saved -> {DEBUG_DIR}/live_raw.png, live_annotated.png, live_mask.png\n")


def main():
    global box
    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return

    win = "drag a box around the item | c=add view  r=forget all  m=show mask  s=debug dump  q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    crops = load_crops()
    tracker = ItemTracker(crops) if crops else None
    show_mask = False
    if tracker is not None:
        print(f"Loaded {len(crops)} saved view(s) -- {tracker.describe()}")
        print("Detecting live already.")
    else:
        print("Put the item on the desk, drag a box around it, then press 'c'.")
    print("Only the cropped item is saved (item_crops/) -- never the full frame.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            display = frame.copy()
            found = None

            if tracker is not None:
                found = tracker.find(frame)
                if show_mask and tracker.mask is not None:
                    display = cv2.cvtColor(tracker.mask, cv2.COLOR_GRAY2BGR)
                draw_result(display, tracker, found)

            if box is not None:
                bx, by, bw, bh = box
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
            if dragging and drag_start and drag_end:
                cv2.rectangle(display, drag_start, drag_end, (0, 165, 255), 1)
            cv2.putText(display, f"{len(crops)} view(s) taught", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask
            elif key == ord('r'):
                clear_crops()
                crops = []
                tracker = None
                box = None
                print("Forgot every view of the item.")
            elif key == ord('c'):
                if box is None:
                    print("Drag a box around the item first.")
                    continue
                bx, by, bw, bh = box
                save_crop(frame[by:by + bh, bx:bx + bw].copy())
                crops = load_crops()
                tracker = ItemTracker(crops)
                box = None
                print(f"Added view {len(crops)} ({bw}x{bh}px) -- {tracker.describe()}")
            elif key == ord('s'):
                if tracker is None:
                    print("Teach it first ('c') before dumping debug info.")
                else:
                    dump_debug(frame, tracker, found)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
