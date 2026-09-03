"""Adding tape spots with the laptop somewhere other than where the calibration was done.

The stars must place the laptop, the new points must be converted into the calibration camera's view, and the mapping must
stay accurate in the NEW area too (not just where the first points were).
"""

import contextlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest

from markos_vision.apps import calibrate_desk
from markos_vision.geometry import camera_model as cm
from markos_vision.geometry.desk_calibration import DeskCalibration

F, K1 = 600.0, -0.06
STARS_TRUE = [(-3.95, 21.9, 3.5), (3.95, 21.9, 3.5)]
STARS_MEASURED = [(-3.97, 21.88, 3.52), (3.93, 21.92, 3.47)]  # the tape measurement is a little off, as it would be
# the strip the tape calibration covered, and the new area beyond it (x = cm right of the axis, y = toward the laptop)
STRIP = [(18, 8), (22, -8), (26, 0), (18, -14), (22, 14), (20, 2), (24, -12), (26, 12)]
NEW = [(30, 14), (30, 0), (30, -14), (34, 14), (34, 0), (34, -14)]
BETWEEN = [(32, 7), (32, -7), (36, 3)]


def make_camera(pos, h=30.0, pitch=18.0, wobble=0.0, aim=(0.0, 10.0)):
    yaw = math.atan2(-(aim[0] - pos[0]), -(aim[1] - pos[1])) + math.radians(wobble)
    return cm.CameraModel(F, K1, pos[0], pos[1], h, yaw, math.radians(pitch))


class Scenario:
    def __init__(self):
        self.rng = np.random.default_rng(41)
        self.a = make_camera((28.0, 75.0))
        self.b = make_camera((36.0, 83.0), wobble=4.0)  # the laptop has moved 10 cm and turned a little
        self.cal = DeskCalibration(path=None)
        for x, y in STRIP:
            px, py = self.a.project((x, y, 0.0)) + self.rng.normal(0, 0.6, 2)
            self.cal.add(px, py, x, y)
        self.cal.set_axis(0.0, 0.0)
        self.cal.set_star_geometry(STARS_MEASURED)
        self.learn_stars(self.a)
        assert self.cal.star_cam is not None

    def star_pixels(self, camera, noise=0.12):
        return [camera.project(s) + self.rng.normal(0, noise, 2) for s in STARS_TRUE]

    def learn_stars(self, camera, star_lock=None):
        tracker = SimpleNamespace(match=lambda gray: [(*p, 0.9, (int(p[0]) - 10, int(p[1]) - 10, 20, 20), True)
                                                      for p in self.star_pixels(camera)])
        capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None,
                                  read=lambda: (True, np.zeros((480, 640, 3), np.uint8)))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            calibrate_desk.learn_stars(self.cal, capture, tracker, 560.0, star_lock)
        return out.getvalue()

    def lock_on(self, camera):
        lock = cm.StarLock(self.cal)
        for _ in range(100):
            if lock.update(self.star_pixels(camera)):
                break
        assert lock.locked, lock.detail
        return lock

    def error_mm(self, camera, points, noise=0.6):
        out = []
        for x, y in points:
            px, py = camera.project((x, y, 0.0)) + self.rng.normal(0, noise, 2)
            out.append(math.hypot(*(np.array(self.cal.to_desk(px, py)) - (x, y))) * 10)
        return out


@pytest.fixture(scope="module")
def scenario():
    s = Scenario()
    s.lock = s.lock_on(s.b)
    s.extrapolated = s.error_mm(s.b, NEW)  # before any point is added in the new area

    s.converted_px = []
    for x, y in NEW:
        seen_b = s.b.project((x, y, 0.0)) + s.rng.normal(0, 0.6, 2)
        pa = np.array(calibrate_desk.calibration_pixel(s.lock, *seen_b))
        s.converted_px.append(np.hypot(*(pa - s.a.project((x, y, 0.0)))))

    for x, y in NEW:  # add the new spots the way the tool does, with the laptop at B
        seen_b = s.b.project((x, y, 0.0)) + s.rng.normal(0, 0.6, 2)
        s.cal.add(*calibrate_desk.calibration_pixel(s.lock, *seen_b), x, y)
    s.after_new = s.error_mm(s.b, NEW + BETWEEN)  # includes places between them that were never added
    s.after_old = s.error_mm(s.b, STRIP)

    # `s` afterwards, still with the laptop at B: the camera is re-fitted from the remembered star pixels
    s.refit_text = s.learn_stars(s.b, s.lock)
    s.refit_params = list(s.cal.star_cam["params"])
    s.lock_after_refit = s.lock_on(s.b)
    s.final_new = s.error_mm(s.b, NEW + BETWEEN[:2])
    s.final_old = s.error_mm(s.b, STRIP)
    return s


def test_the_stars_place_the_moved_laptop(scenario):
    assert calibrate_desk.laptop_displaced(scenario.lock)


def test_a_pixel_seen_from_the_new_position_converts_to_where_the_calibration_camera_saw_it(scenario):
    assert max(scenario.converted_px) < 4.0


def test_points_added_from_the_new_position_make_the_new_area_accurate(scenario):
    assert max(scenario.after_new) < 15
    assert max(scenario.after_old) < 15, "and the area calibrated first must not get worse"


def test_pressing_s_again_refits_the_camera_from_remembered_star_pixels(scenario):
    assert "remembered" in scenario.refit_text and "Camera fitted" in scenario.refit_text
    height = scenario.refit_params[4]
    assert height == pytest.approx(30.0, abs=2.0)
    assert max(scenario.final_new) < 15 and max(scenario.final_old) < 15


def test_with_the_laptop_back_where_it_was_nothing_is_converted(scenario):
    lock_a = scenario.lock_on(scenario.a)
    assert not calibrate_desk.laptop_displaced(lock_a)
