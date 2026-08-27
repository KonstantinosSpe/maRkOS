"""
The planner has to say "no" as often as it says "yes": most of what a generic
IK would return is outside this arm's real hinge ranges.
"""

import math

from markos import hw_config as cfg
from markos import reach


def forward(d_deg, b_deg):
    """Grasp point (r, height above base_link) from the two hinge angles."""
    t1, t2 = math.radians(d_deg), math.radians(d_deg + b_deg + cfg.GRASP_ANGLE_DEG)
    r = cfg.UPPER_ARM * math.sin(t1) + cfg.FOREARM_TO_GRASP * math.sin(t2)
    y = cfg.UPPER_ARM * math.cos(t1) + cfg.FOREARM_TO_GRASP * math.cos(t2)
    return r, cfg.BASE_TO_SHOULDER + y


def test_solution_reproduces_the_point_it_was_asked_for():
    for azimuth, radius, height in [(60, 280, 300), (0, 300, 400), (-100, 320, 260)]:
        plan = reach.plan_target(azimuth, radius, height)
        assert plan is not None, (azimuth, radius, height)
        r, h = forward(plan["D"], plan["B"])
        assert abs(h - height) < 0.5
        assert abs(abs(r) - radius) < 0.5


def test_joint_angles_stay_inside_the_hardware_ranges():
    plan = reach.plan_target(60, 280, 300)
    assert reach.D_LIMITS_DEG[0] <= plan["D"] <= reach.D_LIMITS_DEG[1]
    assert reach.B_LIMITS_DEG[0] <= plan["B"] <= reach.B_LIMITS_DEG[1]
    assert abs(plan["E"]) <= reach.E_LIMIT_DEG


def test_table_level_and_close_to_the_base_are_unreachable():
    assert reach.plan_target(90, 340, 20) is None      # desk level, base on the desk
    assert reach.plan_target(90, 150, 300) is None     # too close to fold that low


def test_azimuth_near_zero_needs_a_turn_E_cannot_make():
    # Leaning away from a point at azimuth ~0 needs E at ~180, far past the ~102 degrees it can turn.
    assert reach.plan_target(0, 200, 640, lean="away") is None
    assert reach.plan_target(30, 200, 640, lean="away") is None
    assert reach.plan_target(100, 200, 640, lean="away") is not None
    assert reach.plan_target(0, 200, 640, lean="toward") is not None


def test_lean_limits_the_answer_to_one_pose():
    toward = reach.plan_target(100, 200, 640, lean="toward")
    away = reach.plan_target(100, 200, 640, lean="away")
    assert toward is not None and away is not None
    assert toward["E"] == 100 and abs(abs(away["E"] - 100) - 180) < 1e-9     # turned toward the spot, or the other way round
    assert reach.plan_target(100, 200, 640) == toward                        # no preference: toward first, as before
    # with the gripper hanging at right angles to the forearm only the facing pose gets down to 26 cm out
    assert reach.plan_target(100, 280, 300, lean="toward") is not None
    assert reach.plan_target(100, 280, 300, lean="away") is None


def test_the_arm_reaches_down_to_a_bottle_cap_from_18_cm_out():
    assert reach.lowest_height(260) is not None and 150 <= reach.lowest_height(260) <= 175
    assert reach.lowest_height(200) <= 175      # it gets down to about 15 cm above its base from 18 cm out
    assert reach.lowest_height(140) > 500       # closer in than that it cannot fold down
    assert reach.lowest_height(400) is None     # and out past about 37 cm it cannot get there at all


def test_a_gripper_that_points_down_off_the_forearm_line(monkeypatch):
    # fingers 60 mm back along the forearm and 150 mm out to the side: the grasp point is not on the forearm's line
    along, across = -60.0, 150.0
    monkeypatch.setattr(cfg, "FOREARM_TO_GRASP", math.hypot(along, across))
    monkeypatch.setattr(cfg, "GRASP_ANGLE_DEG", math.degrees(math.atan2(across, along)))
    found = 0
    for azimuth in (0, 60):
        for radius in range(250, 420, 20):
            for height in range(150, 700, 25):
                plan = reach.plan_target(azimuth, radius, float(height))
                if plan is None:
                    continue
                found += 1
                r, h = forward(plan["D"], plan["B"])
                assert abs(h - height) < 0.5 and abs(abs(r) - radius) < 0.5, (azimuth, radius, height)
                assert reach.D_LIMITS_DEG[0] <= plan["D"] <= reach.D_LIMITS_DEG[1]
                assert reach.B_LIMITS_DEG[0] <= plan["B"] <= reach.B_LIMITS_DEG[1]
    assert found > 20


def test_d_limits_are_mirrored_because_d_runs_backwards_on_the_real_arm():
    lo, hi = (s / cfg.stepsPerDegD for s in cfg.STEP_LIMITS["D"])
    assert cfg.SIGN["D"] == -1
    assert abs(reach.D_LIMITS_DEG[0] + hi) < 1e-9 and abs(reach.D_LIMITS_DEG[1] + lo) < 1e-9
    assert reach.B_LIMITS_DEG[0] < 0 < reach.B_LIMITS_DEG[1] + 10   # B keeps its sign
