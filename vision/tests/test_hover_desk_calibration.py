"""The real hover loop with a fixed-laptop desk calibration.

A synthetic slanted webcam sees a bottle standing at a known spot on the desk. The arm must be commanded to the right place,
never before ``g`` is pressed, and never when the camera has been moved since calibration.
"""

import math

import numpy as np
import pytest
from harness_hover import bottle_polygon, desk_frame, gripper_radius_mm, run_hover_camera

from markos_vision.geometry.desk_calibration import DeskCalibration

F, CX, CY = 600.0, 320.0, 240.0
CAM = np.array([0.0, 85.0, 32.0])
PITCH = math.radians(18)
MARKS = [(-20, 27), (0, 30), (20, 27), (-14, 34), (14, 34), (0, 37), (-24, 20), (24, 20)]
KEYS = {25: ord("g"), 70: ord("q")}


def project(x, y, z=0.0):
    f = np.array([0.0, -math.cos(PITCH), -math.sin(PITCH)])
    d = np.array([0.0, math.sin(PITCH), -math.cos(PITCH)])
    q = np.array([x, y, z]) - CAM
    zc = q @ f
    return CX + F * q[0] / zc, CY + F * (q @ d) / zc


class FakeStars:
    """Stand-in for the star tracker's averaged position."""

    ready = True

    def __init__(self, mid, sep):
        self.mid, self.sep, self.n = mid, sep, 999

    def update(self, left, right):
        pass


@pytest.fixture
def rng():
    return np.random.default_rng(31)


def make_calibration(rng, stars=None):
    cal = DeskCalibration(path=None)
    for x, y in MARKS:
        u, v = project(x, y)
        cal.add(u + rng.normal(0, .5), v + rng.normal(0, .5), x, y)
    if stars:
        cal.set_star_reference(*stars)
    return cal


def run(monkeypatch, tmp_path, rng, bottle_xy, calibration, star_base=None, keys=KEYS):
    def make_frame():
        polygon = bottle_polygon(project, *bottle_xy)
        return desk_frame(rng, polygon), polygon

    return run_hover_camera(monkeypatch, tmp_path, calibration=calibration, make_frame=make_frame, keys=keys, star_base=star_base)


@pytest.mark.parametrize("xy", [(8.0, 32.0), (-12.0, 33.0), (0.0, 31.0)], ids=["right", "left", "straight"])
def test_the_arm_is_sent_over_a_bottle_standing_at_a_known_spot(monkeypatch, tmp_path, rng, xy):
    truth_r = math.hypot(*xy) * 10
    truth_bearing = math.degrees(math.atan2(xy[0], xy[1]))
    ros = run(monkeypatch, tmp_path, rng, xy, make_calibration(rng))
    assert ros.sent, f"nothing was sent for a bottle at {xy}"
    assert ros.sent[0][1] >= 25, "the arm was commanded before 'g' was pressed"
    info = ros.last_marker
    # the bottle's estimated position; the gripper itself can only go where the arm reaches
    assert info["cap_radius_mm"] == pytest.approx(truth_r, abs=15)
    assert info["cap_azimuth_deg"] == pytest.approx(truth_bearing, abs=2.0)
    assert gripper_radius_mm(ros.last_plan) > 0
    assert ros.closed


def test_nothing_is_sent_when_the_camera_has_been_moved_since_calibration(monkeypatch, tmp_path, rng):
    calibration = make_calibration(rng, stars=((320.0, 200.0), 70.0))
    moved = FakeStars((350.0, 205.0), 70.0)  # the stars appear 30 px away from where they were
    ros = run(monkeypatch, tmp_path, rng, (8.0, 32.0), calibration, star_base=moved)
    assert not ros.sent


def test_the_arm_is_sent_when_the_stars_are_where_they_were(monkeypatch, tmp_path, rng):
    calibration = make_calibration(rng, stars=((320.0, 200.0), 70.0))
    steady = FakeStars((321.0, 200.5), 70.2)
    ros = run(monkeypatch, tmp_path, rng, (8.0, 32.0), calibration, star_base=steady)
    assert ros.sent


def test_without_a_desk_calibration_the_rough_star_estimate_is_used(monkeypatch, tmp_path, rng):
    ros = run(monkeypatch, tmp_path, rng, (8.0, 32.0), DeskCalibration(path=None), star_base=FakeStars((320.0, 200.0), 70.0))
    assert ros.sent, "fallback mode should still command the arm (where it goes is not checked here)"
