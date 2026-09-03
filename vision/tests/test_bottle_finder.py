"""YoloBottleFinder: which detections to believe.

The segmentation model is replaced by a stand-in that returns scripted outlines, so the decision logic is tested with no model:
the neck-shape test, bans, cap position from the outline, enrolled types, holding through dropouts, re-checks, and odd output.
"""

import cv2
import numpy as np
import pytest

from markos_vision.perception.bottle_types import BottleLibrary
from markos_vision.perception.bottle_yolo import RECHECK_EVERY, YoloBottleFinder

GREEN = (60, 170, 60)
ORANGE = (40, 130, 230)
GREY = (165, 165, 165)
ROBOT_X, BOTTLE_X = 110, 430


def bottle_poly(cx, top):
    return np.array([(cx - 15, top), (cx + 15, top), (cx + 15, top + 20), (cx + 14, top + 20), (cx + 14, top + 50),
                     (cx + 32, top + 80), (cx + 32, top + 190), (cx - 32, top + 190), (cx - 32, top + 80),
                     (cx - 14, top + 50), (cx - 14, top + 20), (cx - 15, top + 20)], float)


def column_poly(cx, top, width=58, height=300):
    return np.array([(cx - width / 2, top), (cx + width / 2, top), (cx + width / 2, top + height),
                     (cx - width / 2, top + height)], float)


def box_of(poly):
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


class Scene:
    """A desk with a tall grey column standing in for the robot (the original false alarm) and a real bottle."""

    def __init__(self, seed=4):
        self.rng = np.random.default_rng(seed)
        self.bottle_colour, self.robot, self.bottle, self.bottle_id = GREEN, True, True, 2
        self.bottle_as_column, self.conf_robot, self.conf_bottle = False, 0.9, 0.7
        self.holder = {}
        self.finder = None

    def desk(self):
        hsv = np.zeros((480, 640, 3), np.uint8)
        hsv[..., 0] = 10 + self.rng.integers(-2, 3, (480, 640))
        hsv[..., 1] = np.clip(90 + self.rng.normal(0, 12, (480, 640)), 40, 150)
        hsv[..., 2] = np.clip(80 + self.rng.normal(0, 10, (480, 640)), 40, 120)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def frame_with(self, shapes):
        img = self.desk()
        for poly, colour in shapes:
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)], colour)
        return cv2.add(img, self.rng.integers(0, 6, img.shape, dtype=np.uint8))

    def scene(self):
        parts, insts = [], []
        if self.robot:
            robot = column_poly(ROBOT_X, 60)
            parts.append((robot, GREY))
            insts.append({"id": 1, "box": box_of(robot), "conf": self.conf_robot, "polygon": robot})
        if self.bottle:
            bottle = column_poly(BOTTLE_X, 120, 64, 190) if self.bottle_as_column else bottle_poly(BOTTLE_X, 120)
            parts.append((bottle, GREY if self.bottle_as_column else self.bottle_colour))
            insts.append({"id": self.bottle_id, "box": box_of(bottle), "conf": self.conf_bottle, "polygon": bottle})
        return self.frame_with(parts), insts

    def make(self):
        self.finder = YoloBottleFinder(run_model=lambda frame: self.holder["insts"], library=BottleLibrary(path=None))
        return self.finder

    def step(self, n=1):
        out = None
        for _ in range(n):
            frame, self.holder["insts"] = self.scene()
            out = self.finder.find(frame)
        return out


@pytest.fixture
def scene():
    s = Scene()
    s.make()
    return s


def test_the_robot_outlined_as_a_bottle_is_rejected_by_the_neck_test(scene):
    out = scene.step(20)
    assert out["cx"] == pytest.approx(BOTTLE_X, abs=5)
    assert any("column" in reason for _, reason in scene.finder.rejected)


def test_without_the_neck_test_the_robot_outscores_the_real_bottle(scene):
    """The original complaint, kept as a check that the neck test is what fixes it."""
    scene.finder.shape_check = False
    out = scene.step(20)
    assert out["cx"] == pytest.approx(ROBOT_X, abs=5)


def test_marking_a_false_alarm_bans_that_track_and_the_finder_moves_on(scene):
    scene.finder.shape_check = False
    assert scene.step(20)["cx"] == pytest.approx(ROBOT_X, abs=5)
    assert scene.finder.mark_not_a_bottle()
    assert scene.step(20)["cx"] == pytest.approx(BOTTLE_X, abs=5)


def test_the_cap_position_comes_straight_from_the_models_outline(scene):
    cap = scene.step(20)["cap"]
    assert cap["method"] == "yolo mask"
    assert cap["cap"][0] == pytest.approx(BOTTLE_X, abs=2) and cap["cap"][1] == pytest.approx(130, abs=4)
    assert cap["cap_w"] == pytest.approx(30, abs=6)


def test_an_enrolled_bottle_is_verified_and_named_with_its_height(scene):
    scene.step(5)
    bottle_type = scene.finder.enrol("green tea", 19.0, 2.5)
    out = scene.step(10)
    assert out["verified"] and out["type"]["name"] == "green tea" and out["type"]["height_cm"] == 19.0
    assert bottle_type["silhouette_aspect"] == pytest.approx(190 / 64, abs=0.15)


def test_something_that_is_not_the_enrolled_bottle_never_becomes_the_target(scene):
    scene.step(5)
    scene.finder.enrol("green tea", 19.0, 2.5)
    scene.step(10)
    scene.bottle = False
    out = scene.step(30)
    assert out is None or out["missed"] > 0, "the robot was picked up as the target"


def test_a_real_bottle_that_is_not_the_enrolled_one_is_kept_as_unknown_not_thrown_out(scene):
    scene.robot = False
    scene.bottle_colour, scene.bottle_id = GREEN, 3
    scene.step(3)
    scene.finder.enrol("green tea", 19.0, 2.5)  # the library now has a (green) type
    scene.bottle_colour, scene.bottle_id = ORANGE, 4
    out = scene.step(15)
    assert out is not None and out["verified"] is False and out["type"] is None
    assert [reason for _, reason in scene.finder.rejected] == []


def test_brief_dropouts_are_held_then_let_go(scene):
    assert scene.step(5)["missed"] == 0
    scene.bottle = False
    answers = [scene.step(1) for _ in range(14)]
    held = [a for a in answers if a is not None]
    assert 0 < len(held) <= 10 and answers[-1] is None


def test_an_accepted_track_that_turns_into_a_column_is_dropped_after_two_failed_rechecks(scene):
    scene.robot = False
    scene.step(5)
    scene.finder.enrol("green tea", 19.0, 2.5)
    scene.step(RECHECK_EVERY + 2)
    assert scene.step(1)["verified"]
    scene.bottle_as_column = True
    dropped_after = None
    for i in range(1, 60):
        out = scene.step(1)
        if out is None or out["missed"] > 0:
            dropped_after = i
            break
    assert dropped_after is not None and dropped_after <= 3 * RECHECK_EVERY


def test_a_dark_cap_that_does_not_separate_from_the_desk_is_placed_from_the_enrolled_ratio(scene):
    scene.robot = False
    scene.step(5)
    scene.finder.enrol("green tea", 19.0, 2.5)
    scene.step(5)
    cx, top = BOTTLE_X, 120
    no_cap = np.array([(cx - 14, top + 20), (cx + 14, top + 20), (cx + 14, top + 50), (cx + 32, top + 80),
                       (cx + 32, top + 190), (cx - 32, top + 190), (cx - 32, top + 80), (cx - 14, top + 50)], float)
    inst = {"id": 2, "box": box_of(no_cap), "conf": 0.7, "polygon": no_cap}  # the outline stops at the neck
    scene.finder._run_model = lambda frame: [inst]
    out = scene.finder.find(scene.frame_with([(bottle_poly(BOTTLE_X, 120), GREEN)]))
    assert "height prior" in out["cap"]["method"]
    assert out["cap"]["cap"][1] == pytest.approx(130, abs=6)


def test_it_never_crashes_on_odd_model_output():
    weird = [
        {"id": None, "box": (-30.0, -20.0, 80.0, 200.0), "conf": 0.9, "polygon": None},
        {"id": 7, "box": (600.0, 400.0, 100.0, 300.0), "conf": 0.9, "polygon": np.array([[600, 400], [700, 400], [700, 700]], float)},
        {"id": 8, "box": (300.0, 100.0, 1.0, 1.0), "conf": 0.9, "polygon": np.array([[300, 100], [301, 100], [301, 101]], float)},
        {"id": 9, "box": (10.0, 10.0, 50.0, 120.0), "conf": 0.5, "polygon": np.array([[10, 10], [60, 10]], float)},
    ]
    finder = YoloBottleFinder(run_model=lambda frame: weird, library=BottleLibrary(path=None))
    blank = Scene().frame_with([])
    for _ in range(20):
        finder.find(blank)
