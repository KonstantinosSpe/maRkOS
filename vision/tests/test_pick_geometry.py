"""The chain from a picture to joint angles: locating the cap, planning the hover, and what happens out of reach."""

import math
import time

import pytest

from markos import hw_config as hw
from markos_vision.geometry.pick_geometry import PickConfig, hover_plan, locate_cap, urdf_xyz

F = 600.0
CX = 320.0
BOOST = (22.0, 1.8)  # the enrolled bottle: its cap is only 21 cm up


def project(x_cm, z_cm):
    return CX + F * x_cm / z_cm


def scene(robot_x, robot_z, bottle_x, bottle_z, height_cm, noise=(0, 0, 0)):
    """Pixel measurements a pinhole camera at the origin would make."""
    star_sep = F * 7.9 / robot_z + noise[0]
    star_mid = (project(robot_x, robot_z), 200.0)
    cap = (project(bottle_x, bottle_z) + noise[1], 150.0)
    sil_h = F * height_cm / bottle_z + noise[2]
    return cap, sil_h, star_mid, star_sep


def gripper_position(d_deg, b_deg):
    """Forward kinematics of the planner's own model: where the grasp point ends up (radius, height above base_link), mm."""
    t1, t2 = math.radians(d_deg), math.radians(d_deg + b_deg + hw.GRASP_ANGLE_DEG)
    r = hw.UPPER_ARM * math.sin(t1) + hw.FOREARM_TO_GRASP * math.sin(t2)
    y = hw.UPPER_ARM * math.cos(t1) + hw.FOREARM_TO_GRASP * math.cos(t2)
    return abs(r), hw.BASE_TO_SHOULDER + y


@pytest.fixture
def cfg():
    return PickConfig(path=None)


# ── locating the cap from the stars and the bottle's known height ──────────────────────────────────────────────────

def test_locate_cap_recovers_bearing_and_radius_from_exact_pixels(cfg):
    cap, sil_h, mid, sep = scene(-10, 90, 5, 65, 30)
    out = locate_cap(cap, sil_h, mid, sep, F, 30.0, cfg)
    assert out["bearing_deg"] == pytest.approx(math.degrees(math.atan2(15, 25)), abs=0.05)
    assert out["radius_mm"] == pytest.approx(math.hypot(15, 25) * 10, abs=0.5)


def test_the_base_axis_can_sit_behind_the_star_plane(cfg):
    cap, sil_h, mid, sep = scene(-10, 90, 5, 65, 30)
    cfg.values["star_to_axis_cm"] = 4.0
    out = locate_cap(cap, sil_h, mid, sep, F, 30.0, cfg)
    assert out["radius_mm"] == pytest.approx(math.hypot(15, 29) * 10, abs=0.5)


# ── pixels -> joints -> where the gripper ends up ───────────────────────────────────────────────────────────────────

def test_planned_joints_put_the_gripper_over_the_cap(cfg):
    cap, sil_h, mid, sep = scene(-10, 90, 12, 60, 32)
    loc = locate_cap(cap, sil_h, mid, sep, F, 32.0, cfg)
    plan, info = hover_plan(loc["bearing_deg"], loc["radius_mm"], 32.0, 3.0, cfg)
    assert plan is not None, info
    r, h = gripper_position(plan["D"], plan["B"])
    assert r == pytest.approx(loc["radius_mm"], abs=1)
    assert h == pytest.approx(info["height_mm"], abs=1)


def test_a_bottle_right_next_to_the_base_says_why_it_cannot_be_reached(cfg):
    plan, info = hover_plan(15.0, 150, 21.0, 3.0, cfg, go_above=False)
    assert plan is None and "why" in info


BOOST_CASES = [
    ("boost on the desk, 335 mm, bearing 30", 30, 335, *BOOST),
    ("boost, 350 mm (cap too low there)", 30, 350, *BOOST),
    ("boost, 250 mm (too close to the base)", 30, 250, *BOOST),
    ("boost, 420 mm (further than the arm)", 30, 420, *BOOST),
    ("boost dead ahead (bearing 0), 330 mm", 0, 330, *BOOST),
    ("2 L soda, 340 mm (fully reachable)", 40, 340, 33.0, 3.0),
    ("boost, 150 mm (right next to the base)", 60, 150, *BOOST),
]


@pytest.mark.parametrize("name,bearing,radius,height,cap", BOOST_CASES, ids=[c[0] for c in BOOST_CASES])
def test_hover_always_finds_a_plan_and_the_plan_is_consistent(cfg, name, bearing, radius, height, cap):
    """Out of exact reach the planner goes as close as it can instead of giving up, and its joints are self-consistent."""
    plan, info = hover_plan(bearing, radius, height, cap, cfg)
    assert plan is not None, info["why"]
    r, h = gripper_position(plan["D"], plan["B"])
    assert r == pytest.approx(info["radius_mm"], abs=1.5)
    assert h == pytest.approx(info["height_mm"], abs=1.5)
    assert h >= info["wanted_mm"] - 1


def test_a_repeat_call_for_an_out_of_reach_bottle_is_cached(cfg):
    """The planner runs on every camera frame, so the search for the nearest reachable spot must be remembered."""
    hover_plan(30, 420, 22.0, 1.8, cfg)
    t0 = time.perf_counter()
    for _ in range(200):
        hover_plan(30, 420, 22.0, 1.8, cfg)
    per_call_ms = (time.perf_counter() - t0) / 200 * 1000
    assert per_call_ms < 5.0


# ── the base-turn gain scales only E ────────────────────────────────────────────────────────────────────────────────

def test_e_gain_scales_only_the_commanded_base_turn(cfg):
    cfg.values.update(e_sign=1, e_zero_deg=-90.0)
    assert cfg["e_gain"] == 1.0
    base, info0 = hover_plan(45.0, 310.0, *BOOST, cfg)
    assert base["E"] == pytest.approx(-45.0)  # bearing 45 -> the base turns 45 degrees the other way from straight right

    cfg.values["e_gain"] = 1.25
    scaled, info1 = hover_plan(45.0, 310.0, *BOOST, cfg)
    assert scaled["E"] == pytest.approx(base["E"] * 1.25)
    assert scaled["D"] == base["D"] and scaled["B"] == base["B"]
    assert info1["azimuth_deg"] == info0["azimuth_deg"], "the gain must not move where the planner thinks the gripper is going"

    cfg.values["e_gain"] = 0.8
    turned, _ = hover_plan(120.0, 310.0, *BOOST, cfg)
    assert turned["E"] == pytest.approx(30.0 * 0.8)
    straight, _ = hover_plan(90.0, 310.0, *BOOST, cfg)  # straight to the arm's right: no turn, whatever the gain
    assert straight["E"] == pytest.approx(0.0)


# ── the RViz marker frame and the saved configuration ───────────────────────────────────────────────────────────────

def test_the_rviz_marker_lands_where_the_model_puts_the_gripper():
    x, y, z = urdf_xyz(90.0, 330.0, 300.0)  # planner azimuth 90 with the arm leaning away puts the tip on the +X axis
    assert (x, y, z) == pytest.approx((0.33, 0.0, 0.3))
    x, y, _ = urdf_xyz(60.0, 325.0, 280.0)
    assert math.degrees(math.atan2(y, x)) == pytest.approx(-30.0)


def test_nudges_are_saved_and_reloaded(tmp_path):
    path = str(tmp_path / "pick_config.json")
    c = PickConfig(path=path)
    c.nudge("e_zero_deg", 3.0)
    c.nudge("radius_offset_mm", -5.0)
    c.flip_e()
    again = PickConfig(path=path)
    assert (again["e_zero_deg"], again["radius_offset_mm"], again["e_sign"]) == (3.0, -5.0, -1)
