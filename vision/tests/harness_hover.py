"""A harness that runs the real hover loop (``hover.run_camera``) without a camera, a robot or a YOLO model.

A fake camera produces synthetic frames of a bottle standing at a known spot, a stand-in for the segmentation model returns
that bottle's outline, and a fake ROS link records what would have been sent to the arm.
"""

import math

import cv2
import numpy as np

from markos_vision.apps import hover
from markos_vision.perception import bottle_yolo, camera, star_tracking
from markos_vision.perception.bottle_types import BottleLibrary

GREEN = (60, 170, 60)
HEIGHT_CM, CAP_CM = 33.0, 3.0  # a 2 L soda: exactly reachable, so no fallback muddies a comparison


class FakeRos:
    """Records what the hover loop would send: (plan, frame number) pairs, and the RViz marker info."""

    def __init__(self):
        self.sent, self.markers, self.closed, self.frame = [], [], False, 0
        self.joints = self

    def get_subscription_count(self):
        return 1

    def send_joints(self, plan):
        self.sent.append((dict(plan), self.frame))

    def send_markers(self, info, status, base_id=0, ns="bottle"):
        self.markers.append((dict(info), status))

    def close(self):
        self.closed = True

    @property
    def last_plan(self):
        return self.sent[-1][0]

    @property
    def last_marker(self):
        return self.markers[-1][0]


def box_of(polygon):
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def bottle_polygon(project, x_cm, y_cm, height_cm=HEIGHT_CM):
    """The bottle's outline as a camera would see it: foot at the projected desk point, height and width to scale.
    project(x, y, z=0) -> (pixel x, pixel y)."""
    u, v = project(x_cm, y_cm)
    _, vt = project(x_cm, y_cm, height_cm)
    h = v - vt
    w = 6.5 / height_cm * h
    return np.array([(u - .23 * w, vt), (u + .23 * w, vt), (u + .23 * w, vt + .07 * h), (u + .26 * w, vt + .07 * h),
                     (u + .26 * w, vt + .22 * h), (u + .5 * w, vt + .34 * h), (u + .5 * w, v), (u - .5 * w, v),
                     (u - .5 * w, vt + .34 * h), (u - .26 * w, vt + .22 * h), (u - .26 * w, vt + .07 * h),
                     (u - .23 * w, vt + .07 * h)], float)


def desk_frame(rng, polygon, brightness=80):
    """A noisy brown desk with the bottle's outline filled in green."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[..., 0] = 10
    hsv[..., 1] = np.clip(90 + rng.normal(0, 8, (480, 640)), 40, 150)
    hsv[..., 2] = np.clip(brightness + rng.normal(0, 8, (480, 640)), 40, 130)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.fillPoly(frame, [np.round(polygon).astype(np.int32)], GREEN)
    return cv2.add(frame, rng.integers(0, 5, frame.shape, dtype=np.uint8))


def gripper_radius_mm(plan):
    """Forward kinematics of the planner's own model: how far from the base axis the plan puts the grasp point."""
    from markos import hw_config as hw
    t1 = math.radians(plan["D"])
    t2 = math.radians(plan["D"] + plan["B"] + hw.GRASP_ANGLE_DEG)
    return abs(hw.UPPER_ARM * math.sin(t1) + hw.FOREARM_TO_GRASP * math.sin(t2))


def run_hover_camera(monkeypatch, tmp_path, *, calibration, make_frame, keys, star_base=None, star_detections=None, log_every=None):
    """Run ``hover.run_camera`` until the scripted 'q'. Returns the FakeRos.

    make_frame() -> (frame, outline polygon), called once per camera read.
    keys: {frame number: key code}, e.g. {60: ord("g"), 110: ord("q")}.
    star_base: replaces the averaged star position (hover.StarBase); star_detections: callable(frame number) -> the star
    tracker's detections, to feed the real star-lock path (default: the stars are never seen).
    """
    holder = {}
    state = {"n": 0}
    ros = FakeRos()

    def next_frame():
        frame, polygon = make_frame()
        holder["insts"] = [{"id": 1, "box": box_of(polygon), "conf": .8, "polygon": polygon}]
        return frame

    finder = bottle_yolo.YoloBottleFinder(run_model=lambda fr: holder["insts"], library=BottleLibrary(path=None))
    for _ in range(6):  # let it see the bottle, then teach it what this bottle is
        finder.find(next_frame())
    finder.enrol("2l soda", HEIGHT_CM, CAP_CM)

    class Camera:
        def isOpened(self):
            return True

        def release(self):
            pass

        def read(self):
            state["n"] += 1
            ros.frame = state["n"]
            return True, next_frame()

    class Tracker:
        def __init__(self, templates):
            pass

        def match(self, gray):
            return star_detections(state["n"]) if star_detections is not None else []

    pick_config = hover.PickConfig
    monkeypatch.setattr(hover, "PickConfig", lambda: pick_config(path=None))  # fresh defaults, not saved nudges
    monkeypatch.setattr(bottle_yolo, "YoloBottleFinder", lambda *a, **k: finder)
    monkeypatch.setattr(camera, "open_camera", lambda: Camera())
    monkeypatch.setattr(hover, "Ros", lambda: ros)
    monkeypatch.setattr(hover, "DeskCalibration", lambda: calibration)
    monkeypatch.setattr(hover, "HOVER_LOG", str(tmp_path / "hover.log"))
    monkeypatch.setattr(hover, "PROBE_LOG", str(tmp_path / "probe.log"))
    if log_every is not None:
        monkeypatch.setattr(hover, "LOG_EVERY_S", log_every)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: keys.get(state["n"], 255))
    if star_base is not None:
        monkeypatch.setattr(hover, "StarBase", lambda: star_base)
    # the hover loop needs star templates and a focal length to start; these come from a one-time setup on the real machine
    monkeypatch.setattr(star_tracking, "load_templates", lambda: [np.zeros((20, 20), np.uint8)])
    monkeypatch.setattr(star_tracking, "load_focal_length", lambda: 600.0)
    monkeypatch.setattr(star_tracking, "StarTracker", Tracker)
    hover.run_camera(None)
    ros.log_path = tmp_path / "hover.log"
    return ros
