"""The bottle finder window: watch what the detector makes of the camera picture, and enrol bottles.

Shows the YOLO outline, the foot and cap positions and, for a rejected detection, the reason. Keys: ``a`` enrols the
bottle in view (name, height, cap height, foot diameter), ``x`` marks a false alarm, ``t`` lists the enrolled types,
``h`` toggles the neck-shape test, ``s`` saves debugging images, ``q`` quits. The drawing and enrolment helpers are
shared with the hover and calibration windows.

Run: python -m markos_vision.apps.find_bottle
"""

import os
import time
import traceback

import cv2
import numpy as np

from markos_vision.config import BOTTLE_DEBUG_DIR as DEBUG_DIR
from markos_vision.perception.bottle_types import reach_note
from markos_vision.perception.bottle_yolo import YoloBottleFinder
from markos_vision.perception.camera import open_camera
from markos_vision.perception.star_tracking import load_focal_length

REPORT_EVERY_S = 5.0
CAP_SMOOTH = 0.5
CAP_JUMP_PX = 40


def pt(p):
    return int(p[0]), int(p[1])


def smooth_cap(prev, new):
    """The model's outline wobbles a pixel or two frame to frame; the arm
    shouldn't chase that."""
    if prev is None or abs(new["cap"][0] - prev["cap"][0]) + abs(new["cap"][1] - prev["cap"][1]) > CAP_JUMP_PX:
        return dict(new)
    out = dict(new)
    for k in ("cap_w", "cap_h", "top_y", "bottom_y", "height_px"):
        out[k] = prev[k] + CAP_SMOOTH * (new[k] - prev[k])
    out["cap"] = tuple(o + CAP_SMOOTH * (n - o) for o, n in zip(prev["cap"], new["cap"]))
    return out


def draw(display, finder, found, cap, focal_px):
    for (x, y, w, h), reason in finder.rejected:
        cv2.rectangle(display, (int(x), int(y)), (int(x + w), int(y + h)), (200, 0, 200), 1)
        cv2.putText(display, reason, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 200), 1)
    lib = finder.library
    neck = "neck test on" if finder.shape_check else "neck test OFF"
    cv2.putText(display, f"{len(lib.types)} type(s), {len(lib.negatives)} false alarm(s) known   {neck}",
                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if found is None:
        cv2.putText(display, "no bottle", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        return

    fresh = found["missed"] == 0
    color = (0, 165, 255) if not fresh else ((0, 255, 0) if found["verified"] else (0, 255, 255))
    x, y, w, h = found["box"]
    cv2.rectangle(display, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
    if found["polygon"] is not None:
        cv2.polylines(display, [np.round(found["polygon"]).astype(np.int32)], True, color, 2)   # the model's outline
    cv2.circle(display, pt(found["base"]), 5, (255, 200, 0), -1)                                # on the desk
    if cap is not None:
        cx, cy = cap["cap"]
        cv2.rectangle(display, pt((cx - cap["cap_w"] / 2, cy - cap["cap_h"] / 2)),
                      pt((cx + cap["cap_w"] / 2, cy + cap["cap_h"] / 2)), (255, 0, 255), 2)     # the cap
        cv2.circle(display, pt(cap["cap"]), 4, (0, 255, 255), -1)                               # where to grip

    if not fresh:
        label = f"bottle held ({found['missed']} frames)"
    elif found["verified"]:
        label = f"{found['type']['name']}  match {found['sim']:.2f}"
        if cap is not None and focal_px:
            label += f"  {focal_px * found['type']['height_cm'] / cap['height_px']:.0f} cm away"
    else:
        label = ("bottle -- press a to give it a name and height" if not lib.has_types
                 else f"unknown bottle (best match {found['sim']:.2f}) -- press a to give it a name and height")
    if cap is not None:
        label += f"   cap via {cap['method']}"
    cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def enrol(finder, found):
    if found is None or found["missed"] != 0:
        print("Get the bottle outlined and steady first, then press 'a'.")
        return
    existing = ", ".join(t["name"] for t in finder.library.types)
    print("\n--- add a bottle type" + (f" (known: {existing})" if existing else "") + " ---")
    try:
        name = input("  Name (reuse a name to add another look at the same bottle): ").strip()
        height = float(input("  Real height standing up, cap included (cm): "))
        cap_h = input("  Cap height (cm) [Enter = 3]: ").strip()
        cap_h = float(cap_h) if cap_h else 3.0
        known = next((t.get("diameter_cm") for t in finder.library.types if t["name"] == name), None)
        default_note = f" [Enter = {known:g}]" if known else " [Enter = skip: the arm then assumes the calibration bottle's]"
        text = input("  Diameter of its foot, where it stands on the desk (cm)" + default_note + ": ").strip()
        diameter = float(text) if text else None
        if diameter is not None and not 1.0 <= diameter <= 20.0:
            print("That doesn't look like a bottle's diameter in cm -- nothing saved.")
            return
    except ValueError:
        print("Not a number -- nothing saved.")
        return
    if not name:
        print("No name -- nothing saved.")
        return
    t = finder.enrol(name, height, cap_h, diameter)
    if t is None:
        print("Couldn't take a signature from that box -- nothing saved.")
        return
    print(f"Saved '{name}': {len(t['samples'])} look(s). Turn the bottle and press 'a' again with the same name "
          "to add more.")
    note = reach_note(height, cap_h)
    if note:
        print(f"  Reach: {note}")
    print()


def list_types(finder):
    lib = finder.library
    if not lib.types:
        print("No bottle types yet -- outline one and press 'a'.")
    for t in lib.types:
        width = f", {t['diameter_cm']:.1f} cm across" if t.get("diameter_cm") else ""
        print(f"  {t['name']}: {t['height_cm']:.1f} cm tall{width}, cap {t['cap_height_cm']:.1f} cm, {len(t['samples'])} look(s)")
        note = reach_note(t["height_cm"], t["cap_height_cm"])
        if note:
            print(f"      {note}")
    print(f"  {len(lib.negatives)} false alarm(s) remembered")


def main():
    print("Loading the YOLO model...")
    finder = YoloBottleFinder()
    cap = open_camera()
    if not cap.isOpened():
        print("FAILED: could not open camera")
        return
    focal_px = load_focal_length()

    win = "bottle finder | a=add bottle  x=not a bottle  t=list types  h=neck test  s=dump  q=quit"
    if finder.library.has_types:
        print(f"Loaded {len(finder.library.types)} bottle type(s) -- only bottles that look like one of them are accepted.")
    else:
        print("No bottle types yet: put a bottle in view, and once it's outlined press 'a' to name it and enter its height.")
    print("If it ever outlines the robot (or anything else) as a bottle, press 'x' and it will never follow that again.")
    print("Green = enrolled bottle, yellow = a bottle that isn't (or isn't yet) enrolled, orange = held from a moment ago,")
    print("purple = rejected (with the reason). Magenta box = the cap, yellow dot = where to grip.")

    last_report = time.time()
    cap_state = None
    last_error = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                found = finder.find(frame)
            except Exception as exc:  # noqa: BLE001 -- one bad frame shouldn't end the session
                found = None
                if repr(exc) != last_error:
                    last_error = repr(exc)
                    print(f"[error on a frame, carrying on] {last_error}")
                    traceback.print_exc()

            if found is None:
                cap_state = None
            elif found["missed"] == 0:
                cap_state = smooth_cap(cap_state, found["cap"])

            display = frame.copy()
            draw(display, finder, found, cap_state, focal_px)
            cv2.imshow(win, display)

            if time.time() - last_report >= REPORT_EVERY_S:
                print(finder.summary())
                last_report = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                enrol(finder, found)
            elif key == ord('x'):
                if finder.mark_not_a_bottle():
                    cap_state = None
                    print("Remembered that as NOT a bottle -- it won't be followed again.")
                else:
                    print("Nothing tracked right now to mark.")
            elif key == ord('t'):
                list_types(finder)
            elif key == ord('h'):
                finder.shape_check = not finder.shape_check
                print(f"Neck-shape test {'ON' if finder.shape_check else 'OFF'}.")
            elif key == ord('s'):
                os.makedirs(DEBUG_DIR, exist_ok=True)
                cv2.imwrite(f"{DEBUG_DIR}/live_raw.png", frame)
                cv2.imwrite(f"{DEBUG_DIR}/live_annotated.png", display)
                print(f"Saved -> {DEBUG_DIR}/live_raw.png, live_annotated.png"
                      f"  ({'found' if found else 'nothing found'}, {len(finder.rejected)} rejected)")
    finally:
        print("Final:", finder.summary())
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
