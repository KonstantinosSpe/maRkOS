"""Regression: a laptop webcam often looks level or slightly UP.

The camera solver once only accepted cameras looking down, so it threw away the correct answer and the moved-laptop step
refused to lock. This uses the author's real fitted geometry: f 556, 20.7 cm high, tilted -2.6 degrees (looking slightly
up), stars near the left edge of the picture.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from markos_vision.geometry import camera_model as cm
from markos_vision.geometry.desk_calibration import DeskCalibration

F, K1 = 556.0, -0.06
STARS = [(-4.0, 21.9, 5.05), (3.85, 21.9, 5.05)]
TAPE = [(22, 21.9), (22, 1.9), (22, -8.1), (22, 6.9), (18, 11.9), (18, 21.9), (18, 1.9), (18, -8.1), (22, 11.9), (26, -18.1)]
SPOTS = [(24.0, 10.0), (20.0, -4.0), (26.0, -12.0), (19.0, 18.0)]
MOVES = [
    ("same place", 0, 0, 0),
    ("10 cm right", 10, 0, 0),
    ("10 cm back, turned 8 deg", 0, 10, 8),
    ("12 cm left, turned -10 deg", -12, 0, -10),
]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(12)
    camera_a = cm.CameraModel(F, K1, 23.4, 79.3, 20.7, math.radians(-1.2), math.radians(-2.6))
    cal = DeskCalibration(path=None)
    for x, y in TAPE:
        px, py = camera_a.project((x, y, 0.0)) + rng.normal(0, 0.7, 2)
        cal.add(px, py, x, y)
    cal.set_axis(0.0, 0.0)
    tape = [(p["px"], p["py"], p["x_cm"], p["y_cm"]) for p in cal.points]
    observed = [camera_a.project(s) + rng.normal(0, 0.1, 2) for s in STARS]
    fitted, rms, offsets = cm.fit_calibration_pose(tape, STARS, observed, f0=600.0)
    return SimpleNamespace(rng=rng, true=camera_a, cal=cal, fitted=fitted, rms=rms, offsets=offsets)


def test_an_upward_looking_camera_is_not_fitted_as_looking_down(setup):
    assert math.degrees(setup.fitted.pitch) < 1.0
    assert setup.fitted.h == pytest.approx(20.7, abs=3.0)


@pytest.mark.parametrize("name,dx,dy,dyaw", MOVES, ids=[m[0] for m in MOVES])
def test_a_moved_laptop_is_placed_from_the_stars_alone(setup, name, dx, dy, dyaw):
    a, rng = setup.true, setup.rng
    b = cm.CameraModel(F, K1, a.pos[0] + dx, a.pos[1] + dy, a.h, a.yaw + math.radians(dyaw), a.pitch)
    observed = [b.project(s) + rng.normal(0, 0.1, 2) for s in STARS]
    cam_b, rms_b, tilt_change = cm.locate_new_pose(setup.fitted, STARS, observed, setup.offsets)
    assert rms_b < 0.8 and abs(tilt_change) < 2.0

    mapping = cm.carry_over_mapping(setup.fitted, cam_b, lambda px, py: setup.cal.to_desk(px, py))
    errors_mm = [
        math.hypot(*(np.array(mapping(*(b.project((x, y, 0.0)) + rng.normal(0, 0.7, 2)))) - np.array([x, y]))) * 10
        for x, y in SPOTS
    ]
    assert max(errors_mm) < 25.0


def test_the_star_lock_locks_on_this_geometry(setup):
    a, rng, cal = setup.true, setup.rng, setup.cal
    b = cm.CameraModel(F, K1, a.pos[0] + 10.0, a.pos[1], a.h, a.yaw, a.pitch)
    cal.star3d = [list(s) for s in STARS]
    cal.set_star_camera(setup.fitted.params, setup.offsets, setup.rms)
    lock = cm.StarLock(cal)
    for _ in range(45):
        lock.update([b.project(s) + rng.normal(0, 0.1, 2) for s in STARS])
    assert lock.locked, lock.detail
