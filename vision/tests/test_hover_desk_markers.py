"""The real hover loop with the laptop somewhere other than where the tape calibration was done.

The desk mapping has to come from the printed markers: nothing is sent while they are not locked, and once locked the arm is
sent to where the bottle really is.
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from harness_hover import bottle_polygon, run_hover_camera

from markos_vision.geometry import desk_markers as dm
from markos_vision.geometry.desk_calibration import DeskCalibration

F, CX, CY, K1 = 600.0, 320.0, 240.0, -0.06
WORKSPACE = (32.0, -2.0)
SIDE = 10.0
# Beside and in front of the bottle area (the bottle stands around (35, -4)), never behind it as seen from the laptop
MARKERS = {0: (22.0, 12.0), 1: (46.0, 12.0), 2: (16.0, -6.0), 3: (52.0, -6.0)}
BOTTLE = (35.0, -4.0)
TAPE_SPOTS = [(20, 8), (32, 8), (26, 0), (36, 0), (22, -8), (34, -8), (28, -14), (38, -14), (18, 2),
              (44, 8), (48, -2), (20, 14), (46, -8)]


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


def look_at(pos, height=30.0, pitch=18.0):
    yaw = math.degrees(math.atan2(-(WORKSPACE[0] - pos[0]), -(WORKSPACE[1] - pos[1])))
    return make_camera(pos, height, pitch, yaw)


def scene(rng, project, bottle_xy, hide=()):
    """The desk with the printed markers laid flat, and the bottle standing on it (None: no bottle)."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[..., 0] = 10
    hsv[..., 1] = np.clip(90 + rng.normal(0, 8, (480, 640)), 40, 150)
    hsv[..., 2] = np.clip(95 + rng.normal(0, 8, (480, 640)), 40, 130)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    dictionary = cv2.aruco.getPredefinedDictionary(dm.DICTIONARY)
    for mid, (cx, cy) in MARKERS.items():
        if mid in hide:
            continue
        marker = cv2.aruco.generateImageMarker(dictionary, mid, 96)
        canvas = np.full((128, 128), 255, np.uint8)
        canvas[16:112, 16:112] = marker
        half = SIDE / 2 * 128 / 96
        src = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
        dst = np.float32([project(cx - half, cy - half), project(cx + half, cy - half),
                          project(cx + half, cy + half), project(cx - half, cy + half)])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(canvas, matrix, (640, 480), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(canvas, 255), matrix, (640, 480), flags=cv2.INTER_NEAREST)
        img[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]
    polygon = None
    if bottle_xy is not None:
        polygon = bottle_polygon(project, *bottle_xy)
        cv2.fillPoly(img, [np.round(polygon).astype(np.int32)], (60, 170, 60))
    return cv2.add(img, rng.integers(0, 4, img.shape, dtype=np.uint8)), polygon


@pytest.fixture(scope="module")
def world():
    """The one-time setup at laptop position A: a tape calibration, then the markers learned from there."""
    rng = np.random.default_rng(23)
    a = look_at((28.0, 70.0))
    cal = DeskCalibration(path=None)
    for x, y in TAPE_SPOTS:  # includes some spots out toward the markers, so their learned positions are as good as the rest
        px, py = a(x, y)
        cal.add(px + rng.normal(0, 0.7), py + rng.normal(0, 0.7), x, y)
    cal.set_axis(0.0, 0.0)
    acc = dm.Accumulator()
    for _ in range(dm.LEARN_FRAMES):
        acc.add(dm.detect(scene(rng, a, None)[0]))  # no bottle in view while the markers are learned
    assert len(dm.learn(cal, acc)) == 4
    return SimpleNamespace(rng=rng, a=a, cal=cal)


def hover(world, monkeypatch, tmp_path, project, hide=(), go_at=60):
    world.cal.live_H = None

    def make_frame():
        frame, polygon = scene(world.rng, project, BOTTLE, hide)
        return frame, polygon

    return run_hover_camera(monkeypatch, tmp_path, calibration=world.cal, make_frame=make_frame,
                            keys={go_at: ord("g"), 110: ord("q")}, log_every=0.0)


PLACES = [("10 cm right, 10 cm back", (38.0, 80.0)), ("nearer and to the left", (18.0, 60.0)), ("far left side", (8.0, 64.0))]


@pytest.mark.parametrize("name,pos", PLACES, ids=[p[0] for p in PLACES])
def test_the_hover_loop_places_the_bottle_from_the_markers_when_the_laptop_has_moved(world, monkeypatch, tmp_path, name, pos):
    truth_r = math.hypot(*BOTTLE) * 10
    truth_bearing = math.degrees(math.atan2(BOTTLE[0], BOTTLE[1]))
    ros = hover(world, monkeypatch, tmp_path, look_at(pos))
    assert ros.sent, f"{name}: nothing was sent"
    assert ros.last_marker["cap_radius_mm"] == pytest.approx(truth_r, abs=15)
    assert ros.last_marker["cap_azimuth_deg"] == pytest.approx(truth_bearing, abs=2.0)


def test_nothing_is_sent_while_fewer_than_three_markers_are_visible(world, monkeypatch, tmp_path):
    ros = hover(world, monkeypatch, tmp_path, look_at((38.0, 80.0)), hide=(0, 1, 2), go_at=20)  # only one visible: cannot lock
    assert not ros.sent


def test_three_of_four_markers_are_enough(world, monkeypatch, tmp_path):
    ros = hover(world, monkeypatch, tmp_path, look_at((38.0, 80.0)), hide=(3,))
    assert ros.sent


def test_without_learned_markers_it_behaves_as_before_with_the_tape_calibration(world, monkeypatch, tmp_path):
    saved = world.cal.markers
    world.cal.markers = {}
    try:
        ros = hover(world, monkeypatch, tmp_path, world.a)
    finally:
        world.cal.markers = saved
    assert ros.last_marker["cap_radius_mm"] == pytest.approx(math.hypot(*BOTTLE) * 10, abs=15)
