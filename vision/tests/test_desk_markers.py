"""Printed desk markers: rebuilding the pixel -> desk mapping wherever the laptop is put, with no tape.

Tape-calibrate once at position A, learn the printed markers there, then put the camera somewhere else (a different spot,
a different angle, even a slightly different tilt) and rebuild the mapping from the markers alone. How well does a bottle's
desk position come out, and does it notice the camera being moved?
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from markos_vision.geometry import desk_markers as dm
from markos_vision.geometry.desk_calibration import DeskCalibration

F, CX, CY, K1 = 600.0, 320.0, 240.0, -0.06
WORKSPACE = (32.0, -2.0)
SIDE = 10.0
# Markers lie flat on the desk beside the workspace (desk frame: x right, y toward the laptop, cm from the robot's axis)
MARKERS = {0: (28.0, 6.0), 1: (42.0, 6.0), 2: (28.0, -14.0), 3: (42.0, -14.0)}
BOTTLE_SPOTS = [(31.0, 9.0), (34.0, -2.0), (36.0, -10.0), (30.0, 0.0)]  # where a bottle might stand (true desk cm)
TAPE_SPOTS = [(20, 8), (32, 8), (26, 0), (36, 0), (22, -8), (34, -8), (28, -14), (38, -14), (18, 2)]


def make_camera(cam_xy, cam_h, pitch_deg, yaw_deg):
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    fwd = np.array([-math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(right, fwd)
    position = np.array([cam_xy[0], cam_xy[1], cam_h])

    def project(x, y, z=0.0):
        q = np.array([x, y, z]) - position
        zc = q @ fwd
        xn, yn = (q @ right) / zc, (q @ down) / zc
        rr = xn * xn + yn * yn
        return np.array([CX + F * xn * (1 + K1 * rr), CY + F * yn * (1 + K1 * rr)])

    return project


def look_at(pos, height, pitch, wobble_deg=0.0):
    """A camera at pos that a person has turned toward the workspace (yaw only; pitch and height as given)."""
    yaw = math.degrees(math.atan2(-(WORKSPACE[0] - pos[0]), -(WORKSPACE[1] - pos[1]))) + wobble_deg
    return make_camera(pos, height, pitch, yaw)


def render(rng, project, markers=MARKERS, hide=(), noise=3):
    img = np.full((480, 640), 95, np.uint8)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    dictionary = cv2.aruco.getPredefinedDictionary(dm.DICTIONARY)
    for mid, (cx, cy) in markers.items():
        if mid in hide:
            continue
        marker = cv2.aruco.generateImageMarker(dictionary, mid, 96)
        pad = 16
        canvas = np.full((96 + 2 * pad, 96 + 2 * pad), 255, np.uint8)
        canvas[pad:pad + 96, pad:pad + 96] = marker
        half = SIDE / 2 * canvas.shape[0] / 96
        src = np.float32([[0, 0], [canvas.shape[1], 0], [canvas.shape[1], canvas.shape[0]], [0, canvas.shape[0]]])
        dst = np.float32([project(cx - half, cy - half), project(cx + half, cy - half),
                          project(cx + half, cy + half), project(cx - half, cy + half)])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(canvas, matrix, (640, 480), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(canvas, 255), matrix, (640, 480), flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    img = cv2.add(cv2.GaussianBlur(img, (0, 0), 0.7), rng.integers(0, noise + 1, img.shape, dtype=np.uint8))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def world():
    """A tape calibration at position A (0.7 px of noise on the dot) and the markers learned from there."""
    rng = np.random.default_rng(17)
    a = look_at((28.0, 70.0), 30.0, 18.0)
    cal = DeskCalibration(path=None)
    for x, y in TAPE_SPOTS:
        px, py = a(x, y)
        cal.add(px + rng.normal(0, 0.7), py + rng.normal(0, 0.7), x, y)
    cal.set_axis(0.0, 0.0)
    acc = dm.Accumulator()
    for _ in range(dm.LEARN_FRAMES):
        acc.add(dm.detect(render(rng, a)))
    learned = dm.learn(cal, acc)
    return SimpleNamespace(rng=rng, a=a, cal=cal, learned=learned)


def test_the_markers_are_learned_at_the_calibration_position(world):
    assert sorted(world.learned) == [0, 1, 2, 3]
    worst = 0.0
    for mid, corners in world.learned.items():
        cx, cy = MARKERS[mid]
        truth = [(cx - SIDE / 2, cy - SIDE / 2), (cx + SIDE / 2, cy - SIDE / 2), (cx + SIDE / 2, cy + SIDE / 2),
                 (cx - SIDE / 2, cy + SIDE / 2)]
        worst = max(worst, max(math.hypot(u - tx, v - ty) * 10 for (u, v), (tx, ty) in zip(corners, truth)))
    assert worst < 8, "the learned corners carry the tape calibration's own error, about 5 mm"


PLACES = [
    ("same place again", (28.0, 70.0), 30.0, 18.0, 0.0),
    ("10 cm right, 10 cm farther back", (38.0, 80.0), 30.0, 18.0, 0.0),
    ("nearer (58 cm), turned 8 deg off", (20.0, 58.0), 30.0, 18.0, 8.0),
    ("farther (92 cm), turned 8 deg the other way", (42.0, 92.0), 30.0, 18.0, -8.0),
    ("off to the left side of the desk", (8.0, 62.0), 30.0, 18.0, 0.0),
    ("same spot but tilted 4 deg steeper", (28.0, 70.0), 30.0, 22.0, 0.0),
    ("higher (35 cm) and tilted 15 deg", (30.0, 72.0), 35.0, 15.0, 5.0),
]


@pytest.mark.parametrize("name,pos,height,pitch,yaw", PLACES, ids=[p[0] for p in PLACES])
def test_a_moved_laptop_is_placed_from_the_markers_alone(world, name, pos, height, pitch, yaw):
    project = look_at(pos, height, pitch, yaw)
    world.cal.live_H = None
    lock = dm.MarkerLock(world.cal)
    frames = 0
    while not lock.locked and frames < 80:
        lock.update(render(world.rng, project))
        frames += 1
    assert lock.locked, lock.detail
    errors = []
    for x, y in BOTTLE_SPOTS:
        px, py = project(x, y)  # the bottle's foot in this camera view
        got = world.cal.to_desk(px + world.rng.normal(0, 0.7), py + world.rng.normal(0, 0.7))
        errors.append(math.hypot(got[0] - x, got[1] - y) * 10)
    assert max(errors) < 15


def test_a_bump_of_the_laptop_is_noticed_and_it_relocks_by_itself(world):
    rng, cal = world.rng, world.cal
    project = look_at((28.0, 70.0), 30.0, 18.0)
    lock = dm.MarkerLock(cal)
    for _ in range(60):
        lock.update(render(rng, project))
    assert lock.locked
    for _ in range(30):  # camera untouched: stays locked, no false alarms
        lock.update(render(rng, project))
    assert lock.locked and lock.events == 0

    yaw_a = math.degrees(math.atan2(-(WORKSPACE[0] - 28.0), -(WORKSPACE[1] - 70.0)))
    pushed = make_camera((32.0, 70.0), 30.0, 18.0, yaw_a)  # 4 cm sideways, NOT re-aimed: a real bump
    noticed = None
    for i in range(1, 40):
        lock.update(render(rng, pushed))
        if lock.events:
            noticed = i
            break
    assert noticed is not None and not lock.locked and cal.live_H is None
    for _ in range(60):
        lock.update(render(rng, pushed))
    assert lock.locked


@pytest.mark.parametrize("hidden,should_lock", [((3,), True), ((2, 3), False)], ids=["one of four hidden", "two of four hidden"])
def test_a_hidden_marker_is_tolerated_but_two_are_not(world, hidden, should_lock):
    project = look_at((38.0, 80.0), 30.0, 18.0)
    lock = dm.MarkerLock(world.cal)
    for _ in range(50):
        lock.update(render(world.rng, project, hide=hidden))
    assert lock.locked == should_lock


def test_a_marker_moved_after_learning_is_caught_not_trusted(world):
    moved = dict(MARKERS)
    moved[1] = (MARKERS[1][0] + 4.0, MARKERS[1][1])
    lock = dm.MarkerLock(world.cal)
    for _ in range(50):
        lock.update(render(world.rng, look_at((38.0, 80.0), 30.0, 18.0), markers=moved))
    assert not lock.locked


def test_markers_are_saved_and_changing_the_points_asks_for_them_to_be_relearned(world, tmp_path):
    path = str(tmp_path / "desk.json")
    first = DeskCalibration(path=path)
    for x, y in TAPE_SPOTS:
        first.add(*world.a(x, y), x, y)
    first.set_axis(0, 0)
    first.set_markers(world.learned)
    again = DeskCalibration(path=path)
    assert sorted(again.markers) == [0, 1, 2, 3] and again.markers_from_points == len(TAPE_SPOTS)
    again.add(*world.a(30, 3), 30, 3)
    assert any("relearn" in note for note in again.notes())


def test_the_printable_sheet_is_read_back_by_the_detector(tmp_path):
    sheet = dm.make_sheet(str(tmp_path / "sheet.png"))
    found = dm.detect(cv2.imread(sheet, cv2.IMREAD_GRAYSCALE))
    assert sorted(found) == [0, 1, 2, 3]
