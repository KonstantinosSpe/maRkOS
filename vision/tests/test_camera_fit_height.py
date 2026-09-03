"""A ruler-measured lens height pins the camera fit; without one the fit is as free as before."""

import math

import numpy as np
import pytest

from markos_vision.geometry import camera_model as cm

TRUE = cm.CameraModel(640.0, -0.05, 30.0, 100.0, 24.0, 0.0, math.radians(4.0))
STARS = [(-3.95, 21.9, 5.0), (3.95, 21.9, 5.0)]
SPOTS = [(x, y) for x in (22, 26, 30, 34) for y in (-14, 0, 14)]


@pytest.fixture(scope="module")
def observations():
    rng = np.random.default_rng(9)
    tape = []
    for x, y in SPOTS:
        px, py = TRUE.project((x, y, 0.0)) + rng.normal(0, 0.8, 2)
        tape.append((px, py, x, y))
    star_px = [TRUE.project(s) + rng.normal(0, 0.2, 2) for s in STARS]
    return tape, star_px


def test_pinning_to_the_true_height_costs_almost_nothing(observations):
    tape, star_px = observations
    free, rms_free, _ = cm.fit_calibration_pose(tape, STARS, star_px, f0=600.0)
    pinned, rms_pin, _ = cm.fit_calibration_pose(tape, STARS, star_px, f0=600.0, height=24.0)
    assert abs(pinned.h - 24.0) < 0.5
    assert rms_pin < rms_free + 0.3
    assert abs(pinned.f - 640) <= abs(free.f - 640) + 10  # and it does not make the focal length worse


def test_a_wrong_ruler_reading_is_followed_but_the_data_pulls_back(observations):
    tape, star_px = observations
    wrong, _, _ = cm.fit_calibration_pose(tape, STARS, star_px, f0=600.0, height=27.0)
    assert 26.0 < wrong.h < 27.5
