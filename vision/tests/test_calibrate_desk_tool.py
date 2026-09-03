"""Drives ``calibrate_desk.main()`` end to end.

A fake camera shows a bottle at six spots, key presses and typed measurements are scripted. The tool must collect the
points, refuse bad input, survive undo, and save a usable calibration.
"""

import builtins
import io
import math
from contextlib import redirect_stdout

import cv2
import numpy as np

from markos_vision.apps import calibrate_desk
from markos_vision.geometry.desk_calibration import DeskCalibration
from markos_vision.perception import bottle_yolo
from markos_vision.perception.bottle_types import BottleLibrary

F, CX, CY = 600.0, 320.0, 240.0
CAM = np.array([0.0, 85.0, 32.0])
PITCH = math.radians(18)
SPOTS = [(-20, 27), (0, 30), (20, 27), (-14, 34), (14, 34), (0, 37)]
FRAMES_PER_SPOT = 40
CAPTURE_AT = 28


def project(x, y, z=0.0):
    f = np.array([0.0, -math.cos(PITCH), -math.sin(PITCH)])
    d = np.array([0.0, math.sin(PITCH), -math.cos(PITCH)])
    q = np.array([x, y, z]) - CAM
    zc = q @ f
    return CX + F * q[0] / zc, CY + F * (q @ d) / zc


def poly_at(x, y, height=33.0):
    u, v = project(x, y)
    _, vt = project(x, y, height)
    h, w = v - vt, 6.5 / height * (v - vt)
    return np.array([(u - .23 * w, vt), (u + .23 * w, vt), (u + .5 * w, vt + .34 * h), (u + .5 * w, v), (u - .5 * w, v),
                     (u - .5 * w, vt + .34 * h)], float)


def box_of(p):
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def test_the_tool_collects_points_refuses_bad_input_undoes_and_saves_a_usable_calibration(monkeypatch, tmp_path):
    holder, tick = {}, {"n": 0}
    # The author measures from the RIGHT corner of the box end face as seen from the laptop: u = how far RIGHT of the corner,
    # v = how far out from the end face. The axis is built in at (-9.0, -21.9), so the true spot (x, y) is typed as
    # u = x - 9, v = y - 21.9.
    answers = iter(["-29 5.1",  # spot 1 (true -20, 27)
                    "banana",  # spot 2: not numbers -> must be refused
                    "-9 8.1",  # spot 2 again (true 0, 30)
                    "11, 5.1",  # spot 3 (true 20, 27): a comma is fine
                    "-23 12.1", "5 12.1", "-9 15.1"])

    def frame():
        spot = min(tick["n"] // FRAMES_PER_SPOT, len(SPOTS) - 1)
        p = poly_at(*SPOTS[spot])
        holder["insts"] = [{"id": 1, "box": box_of(p), "conf": .8, "polygon": p}]
        img = np.full((480, 640, 3), (40, 70, 90), np.uint8)
        cv2.fillPoly(img, [np.round(p).astype(np.int32)], (60, 170, 60))
        return img

    class Camera:
        def isOpened(self):
            return True

        def release(self):
            pass

        def read(self):
            tick["n"] += 1
            return True, frame()

    keys = {}
    for i in range(len(SPOTS)):
        keys[i * FRAMES_PER_SPOT + CAPTURE_AT] = ord("c")
    keys[1 * FRAMES_PER_SPOT + CAPTURE_AT + 1] = ord("c")  # the banana at spot 2 is refused, so it needs another 'c'
    keys[len(SPOTS) * FRAMES_PER_SPOT - 1] = ord("t")
    keys[len(SPOTS) * FRAMES_PER_SPOT + 2] = ord("u")  # undo the last point...
    keys[len(SPOTS) * FRAMES_PER_SPOT + 4] = ord("q")

    path = str(tmp_path / "desk.json")
    finder = bottle_yolo.YoloBottleFinder(run_model=lambda fr: holder["insts"], library=BottleLibrary(path=None))
    monkeypatch.setattr(calibrate_desk, "YoloBottleFinder", lambda *a, **k: finder)
    monkeypatch.setattr(calibrate_desk, "open_camera", lambda: Camera())
    monkeypatch.setattr(calibrate_desk, "DeskCalibration", lambda: DeskCalibration(path=path))
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)  # no display in CI: the headless OpenCV has no windows
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: keys.get(tick["n"], 255))
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    out = io.StringIO()
    with redirect_stdout(out):
        calibrate_desk.main()
    text = out.getvalue()

    cal = DeskCalibration(path=path)
    assert "Type two numbers" in text, "the bad entry ('banana') wasn't refused"
    assert len(cal.points) == 5, f"expected 6 captured minus 1 undone = 5, got {len(cal.points)}"
    assert cal.ready

    worst = 0.0  # the saved mapping must put a bottle at a fresh spot where it really is
    for x, y in [(6, 32), (-9, 30), (12, 29)]:
        px, py = poly_at(x, y)[[3, 4], :].mean(axis=0)[0], poly_at(x, y)[:, 1].max()
        ex, ey = cal.to_desk(px, py)
        worst = max(worst, math.hypot(ex - x, ey - y) * 10)
    assert worst < 10

    # the built-in axis was applied without asking; u counted rightward and v toward the camera matches the picture
    assert cal.axis == [-9.0, -21.9]
    assert not cal.mirrored
    assert "Robot axis: u -9.0, v -21.9" in text
