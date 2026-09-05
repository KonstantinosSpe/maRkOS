"""ik: the two-link planar solve for the arm's shoulder (D) and elbow (B), and the response curve for gesture-driven targets."""

import math

import ik
import pytest


def forward(j3_deg, j5_deg):
    """Where the gripper mount ends up (r, y above the shoulder) for joint angles as ik_planar defines them:
    J3 is link 1 from vertical, J5 is link 2 relative to link 1."""
    t1 = math.radians(j3_deg)
    t2 = math.radians(j3_deg + j5_deg)
    return ik.A_LEN * math.sin(t1) + ik.B_LEN * math.sin(t2), ik.A_LEN * math.cos(t1) + ik.B_LEN * math.cos(t2)


REACHABLE = [(150.0, 100.0), (200.0, 0.0), (120.0, -60.0), (250.0, 60.0), (100.0, 200.0), (180.0, -120.0)]


@pytest.mark.parametrize("r,y", REACHABLE)
@pytest.mark.parametrize("elbow", [+1, -1])
def test_the_solved_angles_put_the_gripper_mount_at_the_target(r, y, elbow):
    j3, j5 = ik.ik_planar(r, y, elbow)
    got_r, got_y = forward(j3, j5)
    assert (got_r, got_y) == pytest.approx((r, y), abs=0.5)


def test_a_target_beyond_full_stretch_is_pulled_back_instead_of_failing():
    j3, j5 = ik.ik_planar(2 * (ik.A_LEN + ik.B_LEN), 0.0, +1)
    assert math.isfinite(j3) and math.isfinite(j5)
    got_r, _ = forward(j3, j5)
    assert got_r < ik.A_LEN + ik.B_LEN


def test_the_two_elbow_configurations_are_mirror_images_of_each_other():
    up, down = ik.ik_planar(200.0, 50.0, +1), ik.ik_planar(200.0, 50.0, -1)
    assert up[1] == pytest.approx(-down[1])


def test_solve_reach_ik_stays_inside_the_joint_limits():
    for height in (ik.HEIGHT_FLOOR, ik.SHOULDER_H, ik.HEIGHT_CEIL):
        for reach in (ik.R_FLOOR, 200.0, 350.0):
            j3, j5 = ik.solve_reach_ik(height, reach)
            assert ik.LIMITS["J3"][0] <= j3 <= ik.LIMITS["J3"][1]
            assert ik.LIMITS["J5"][0] <= j5 <= ik.LIMITS["J5"][1]


def test_solve_reach_ik_prefers_the_pose_with_the_shoulder_closer_to_vertical():
    j3, _ = ik.solve_reach_ik(ik.SHOULDER_H + 100.0, 150.0)
    other = ik.ik_planar(150.0, 100.0, -1)[0], ik.ik_planar(150.0, 100.0, +1)[0]
    assert abs(j3) == pytest.approx(min(abs(o) for o in other), abs=0.01)


def test_clamp_reach_keeps_the_point_inside_the_working_envelope():
    for height in (ik.HEIGHT_FLOOR, ik.SHOULDER_H, ik.HEIGHT_CEIL - 10):
        for r in (0.0, 100.0, 300.0, 1000.0):
            clamped = ik.clamp_reach(height, r)
            assert clamped >= ik.R_FLOOR
            assert math.hypot(clamped, height - ik.SHOULDER_H) <= ik.REACH_MAX + 1e-6 or clamped == ik.R_FLOOR


def test_the_response_curve_has_a_dead_zone_then_a_curved_ramp():
    assert ik.response(0.0) == 0.0
    assert ik.response(ik.DEADZONE * 0.99) == 0.0
    assert ik.response(1.0) == pytest.approx(1.0) and ik.response(-1.0) == pytest.approx(-1.0)
    assert 0.0 < ik.response(0.5) < 0.5, "curved (quadratic), not linear: half a deflection is well under half the speed"
    assert ik.response(-0.5) == pytest.approx(-ik.response(0.5))


def test_clamp():
    assert ik.clamp(5, 0, 3) == 3 and ik.clamp(-5, 0, 3) == 0 and ik.clamp(2, 0, 3) == 2
