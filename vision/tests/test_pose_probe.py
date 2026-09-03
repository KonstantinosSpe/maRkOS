"""The aim probe's fit.

The fit must predict where the gripper lands (also at poses it was NOT fitted on) from noisy "camera" readings, and the
recommended settings must actually cancel the arm's error.
"""

import math

import numpy as np
import pytest

from markos_vision.apps import pose_probe as pp
from markos_vision.geometry import pick_geometry as pg

# the arm's centre 2.5 cm right / 4 cm back, 7 degrees off, turning 1.3x what it is told, reaching 3 cm too long
TRUE = np.array([25.0, -40.0, 7.0, 1.30, 30.0])
# (E command deg, radius mm) inside the working zone, none of them among the fitted poses
HELD_OUT = [(-25.0, 300.0), (-10.0, 280.0), (0.0, 320.0), (15.0, 290.0), (28.0, 330.0), (-30.0, 335.0), (5.0, 275.0)]
TEST_BOTTLES = [(300.0, -50.0), (280.0, -160.0), (330.0, 20.0), (270.0, 150.0), (310.0, -100.0)]  # mm from the ASSUMED axis


@pytest.fixture(scope="module")
def poses():
    return pp.plans()


def measure(poses, rng, true, noise_mm):
    out = []
    for p in poses:
        x, y = pp.predict(true, p["E"], p["radius_p"])
        out.append({"bearing": p["bearing"], "radius_p": p["radius_p"], "e_cmd": p["E"],
                    "x_mm": x + rng.normal(0, noise_mm), "y_mm": y + rng.normal(0, noise_mm)})
    return out


def held_out_error(p, true):
    worst = 0.0
    for e, r in HELD_OUT:
        a, b = pp.predict(p, e, r), pp.predict(true, e, r)
        worst = max(worst, math.hypot(a[0] - b[0], a[1] - b[1]))
    return worst


def test_there_are_enough_reachable_poses(poses):
    assert len(poses) >= 5


def test_an_arm_with_no_error_is_reported_as_such(poses):
    rng = np.random.default_rng(11)
    none = np.array([0, 0, 0, 1.0, 0])
    p0, _ = pp.fit(measure(poses, rng, none, 3.0))
    assert abs(p0[2]) < 1.5 and abs(p0[3] - 1) < 0.06
    assert held_out_error(p0, none) < 12


def test_a_big_error_is_recovered_and_predicted_at_unseen_poses_despite_noisy_readings(poses):
    rng = np.random.default_rng(11)
    worst_held, worst_off, worst_k = 0.0, 0.0, 0.0
    for _ in range(30):
        p, _ = pp.fit(measure(poses, rng, TRUE, 5.0))
        worst_held = max(worst_held, held_out_error(p, TRUE))
        worst_off = max(worst_off, abs(p[2] - TRUE[2]))
        worst_k = max(worst_k, abs(p[3] - TRUE[3]))
    # the turn scale alone is loosely determined; what counts is the prediction
    assert worst_off < 2.5 and worst_k < 0.2 and worst_held < 15


def test_the_recommended_settings_cancel_the_arms_error(poses):
    rng = np.random.default_rng(11)
    p, _ = pp.fit(measure(poses, rng, TRUE, 2.0))
    rec = pp.recommend(p)

    def misses(cfg, axis_shift, radius_offset):
        out = []
        for target_x, target_y in TEST_BOTTLES:
            tx, ty = (target_x - axis_shift[0], target_y - axis_shift[1])  # the corrected program sees the moved axis
            bearing, radius = math.degrees(math.atan2(tx, ty)), math.hypot(tx, ty) + radius_offset  # the hover adds its offset
            plan, info = pg.hover_plan(bearing, radius, 22.0, 1.8, cfg)
            if plan is None:
                continue
            gx, gy = pp.predict(TRUE, plan["E"], info["radius_mm"])  # the true arm, given that command
            out.append(math.hypot(gx - target_x, gy - target_y))
        return out

    corrected = pg.PickConfig(path=None)
    corrected.values.update(e_sign=1, e_zero_deg=rec["e_zero_deg"], e_gain=rec["e_gain"], radius_offset_mm=0.0,
                            base_height_cm=0.0, hover_clearance_mm=209.0)
    uncorrected = pg.PickConfig(path=None)
    uncorrected.values.update(e_sign=1, e_zero_deg=-90.0, e_gain=1.0, radius_offset_mm=0.0, base_height_cm=0.0,
                              hover_clearance_mm=209.0)

    after = misses(corrected, (p[0], p[1]), rec["radius_offset_mm"])
    before = misses(uncorrected, (0.0, 0.0), 0.0)
    assert len(after) == len(TEST_BOTTLES)
    assert max(after) < 15
    assert min(before) > 3 * max(after), "without the correction the same bottles are missed by centimetres"
