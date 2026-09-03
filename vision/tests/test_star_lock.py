"""Placing a moved laptop from the two red stars, and the hover loop that uses it.

The tape calibration is done once with the laptop at A, and the two stars are learned there. The laptop is then put
somewhere else, at the same height and lid angle, and the mapping has to follow from the stars alone. Covers the
calibration tool's `s` step, the ``StarLock`` (locking, refusing, noticing a bump), and the real hover loop.
"""

import builtins
import contextlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
from harness_hover import bottle_polygon, desk_frame, gripper_radius_mm, run_hover_camera

from markos_vision.apps import calibrate_desk
from markos_vision.geometry import camera_model as cm
from markos_vision.geometry.desk_calibration import DeskCalibration

F, K1 = 600.0, -0.06
# The desk frame is the measuring frame here: origin at the robot's axis, x right, y toward the laptop. The stars are on
# the box end face (y = 21.9), 7.9 cm apart, 3.5 cm up: measured to the millimetre in real life, 2 mm off here.
STARS_TRUE = [(-3.95, 21.9, 3.5), (3.95, 21.9, 3.5)]
STARS_MEASURED = [(-3.97, 21.88, 3.52), (3.93, 21.92, 3.47)]
TAPE = [(20, 8), (32, 8), (26, 0), (36, 0), (22, -8), (34, -8), (28, -14), (38, -14), (18, 2), (44, 8), (48, -2)]
BOTTLE_SPOTS = [(31.0, 9.0), (34.0, -2.0), (36.0, -10.0), (30.0, 0.0), (26.0, 14.0)]
PLACES = [
    ("same place", (28.0, 70.0), 30.0, 18.0, 0.0),
    ("10 cm right, 10 cm back", (38.0, 80.0), 30.0, 18.0, 0.0),
    ("nearer (58 cm), turned 8 deg", (20.0, 58.0), 30.0, 18.0, 8.0),
    ("left side of the desk", (8.0, 62.0), 30.0, 18.0, 0.0),
    ("far back (92 cm), turned", (42.0, 92.0), 30.0, 18.0, -8.0),
    ("same place, lid angle 1.5 deg different", (38.0, 80.0), 30.0, 19.5, 0.0),
]


def make_camera(pos, h=30.0, pitch=18.0, wobble=0.0, aim=(0.0, 10.0)):
    yaw = math.atan2(-(aim[0] - pos[0]), -(aim[1] - pos[1])) + math.radians(wobble)
    return cm.CameraModel(F, K1, pos[0], pos[1], h, yaw, math.radians(pitch))


class World:
    def __init__(self):
        self.rng = np.random.default_rng(31)
        self.a = make_camera((28.0, 70.0))
        self.cal = DeskCalibration(path=None)
        for x, y in TAPE:
            px, py = self.a.project((x, y, 0.0)) + self.rng.normal(0, 0.7, 2)
            self.cal.add(px, py, x, y)
        self.cal.set_axis(0.0, 0.0)

    def star_pixels(self, camera, noise=0.12):
        return [camera.project(s) + self.rng.normal(0, noise, 2) for s in STARS_TRUE]

    def lock(self, camera, frames=100, noise=0.12):
        lock = cm.StarLock(self.cal)
        for _ in range(frames):
            if lock.update(self.star_pixels(camera, noise)):
                break
        return lock


@pytest.fixture(scope="module")
def world():
    """The calibration tool's `s` step, driven with a fake camera and typed answers."""
    w = World()
    tracker = SimpleNamespace(match=lambda gray: [(*p, 0.9, (int(p[0]) - 10, int(p[1]) - 10, 20, 20), True)
                                                  for p in w.star_pixels(w.a)])
    capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None, read=lambda: (True, np.zeros((480, 640, 3), np.uint8)))
    answers = iter(["-3.97 21.88 3.52", "3.93 21.92 3.47"])
    real_input, builtins.input = builtins.input, lambda prompt="": next(answers)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            calibrate_desk.learn_stars(w.cal, capture, tracker, 560.0)
    finally:
        builtins.input = real_input
    w.learn_output = out.getvalue()
    return w


@pytest.fixture(autouse=True)
def laptop_where_it_was(world):
    world.cal.live_map = None


# ── the calibration tool's `s` step ─────────────────────────────────────────────────────────────────────────────────

def test_the_s_step_fits_the_camera_from_the_tape_spots_and_the_stars(world):
    assert world.cal.star_cam is not None and world.cal.star3d is not None, world.learn_output
    params = world.cal.star_cam["params"]  # the true camera: height 30, tilt 18, focal 600
    assert params[4] == pytest.approx(30.0, abs=3.0)
    assert math.degrees(params[6]) == pytest.approx(18.0, abs=2.0)
    assert params[0] == pytest.approx(600.0, abs=40.0)


# ── the StarLock: the laptop somewhere else ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,pos,height,pitch,wobble", PLACES, ids=[p[0] for p in PLACES])
def test_a_moved_laptop_is_located_from_the_stars_and_bottles_are_placed_accurately(world, name, pos, height, pitch, wobble):
    b = make_camera(pos, height, pitch, wobble)
    lock = world.lock(b)
    assert lock.locked, lock.detail
    errors = [math.hypot(*(np.array(world.cal.to_desk(*(b.project((x, y, 0.0)) + world.rng.normal(0, 0.7, 2)))) - (x, y))) * 10
              for x, y in BOTTLE_SPOTS]
    assert max(errors) < 30.0


def test_it_refuses_when_the_left_and_right_stars_are_mixed_up(world):
    b = make_camera((38.0, 80.0))
    lock = cm.StarLock(world.cal)
    for _ in range(60):
        lock.update(list(reversed(world.star_pixels(b))))
    assert not lock.locked and world.cal.live_map is None


def test_it_refuses_a_big_height_change_that_could_pass_as_a_tilt_change(world):
    lock = world.lock(make_camera((38.0, 80.0), 39.0, 18.0), frames=60)  # 9 cm higher than calibrated
    assert not lock.locked


def test_it_refuses_while_the_picture_is_jittering_as_if_the_laptop_were_carried(world):
    lock = world.lock(make_camera((38.0, 80.0)), frames=60, noise=4.0)
    assert not lock.locked


def test_it_says_it_needs_both_stars_when_they_are_not_in_view(world):
    lock = cm.StarLock(world.cal)
    for _ in range(5):
        lock.update(None)
    assert not lock.locked and "both stars" in lock.detail


def test_a_bump_is_noticed_and_it_relocks_by_itself(world):
    b = make_camera((38.0, 80.0))
    lock = world.lock(b, frames=60)
    assert lock.locked
    for _ in range(30):
        lock.update(world.star_pixels(b))
    assert lock.locked and lock.events == 0

    pushed = cm.CameraModel(F, K1, 42.0, 80.0, 30.0, b.yaw, b.pitch)  # pushed 4 cm, not re-aimed
    noticed = None
    for i in range(1, 40):
        lock.update(world.star_pixels(pushed))
        if lock.events:
            noticed = i
            break
    assert noticed is not None and not lock.locked and world.cal.live_map is None
    for _ in range(60):
        lock.update(world.star_pixels(pushed))
    assert lock.locked


# ── persistence ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_star_geometry_and_the_fitted_camera_are_saved_and_adding_a_point_asks_for_s_again(world, tmp_path):
    path = str(tmp_path / "desk.json")
    first = DeskCalibration(path=path)
    for x, y in TAPE:
        first.add(*(world.a.project((x, y, 0.0))), x, y)
    first.set_axis(0.0, 0.0)
    first.set_star_geometry(STARS_MEASURED)
    first.set_star_camera(world.cal.star_cam["params"], world.cal.star_cam["offsets"], world.cal.star_cam["rms"])
    again = DeskCalibration(path=path)
    assert again.star3d == first.star3d and again.star_cam["params"] == first.star_cam["params"]
    again.add(*(world.a.project((30, 3, 0.0))), 30, 3)
    assert any("stars were learned" in note for note in again.notes())


# ── the real hover loop, laptop moved: the stars are all that is known about where ──────────────────────────────────

BOTTLE = (34.0, -2.0)
# aimed between the robot and the bottle, like a person would, so both the stars and the bottle are in the picture
HOVER_PLACES = [("40 cm right, 20 cm back", (40.0, 92.0)), ("nearer and to the left", (22.0, 84.0)),
                ("right of where it was calibrated", (48.0, 78.0))]


def hover(world, monkeypatch, tmp_path, camera, star_visible=True, go_at=60):
    def make_frame():
        polygon = bottle_polygon(lambda x, y, z=0.0: camera.project((x, y, z)), *BOTTLE)
        return desk_frame(world.rng, polygon, brightness=80), polygon

    def detections(n):
        if not star_visible:
            return []
        return [(*p, 0.9, (int(p[0]) - 10, int(p[1]) - 10, 20, 20), True) for p in world.star_pixels(camera)]

    return run_hover_camera(monkeypatch, tmp_path, calibration=world.cal, make_frame=make_frame,
                            keys={go_at: ord("g"), 110: ord("q")}, star_detections=detections)


@pytest.mark.parametrize("name,pos", HOVER_PLACES, ids=[p[0] for p in HOVER_PLACES])
def test_the_hover_loop_puts_the_arm_over_the_bottle_with_the_laptop_moved(world, monkeypatch, tmp_path, name, pos):
    truth_r = math.hypot(*BOTTLE) * 10
    truth_bearing = math.degrees(math.atan2(BOTTLE[0], BOTTLE[1]))
    ros = hover(world, monkeypatch, tmp_path, make_camera(pos, aim=(17.0, 8.0)))
    assert ros.sent, f"{name}: nothing was sent"
    assert gripper_radius_mm(ros.last_plan) == pytest.approx(truth_r, abs=30)
    assert ros.last_marker["cap_azimuth_deg"] == pytest.approx(truth_bearing, abs=3.0)


def test_nothing_is_sent_when_the_stars_are_not_visible(world, monkeypatch, tmp_path):
    ros = hover(world, monkeypatch, tmp_path, make_camera((40.0, 92.0), aim=(17.0, 8.0)), star_visible=False, go_at=20)
    assert not ros.sent
