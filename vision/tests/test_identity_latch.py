"""The identity latch.

A recognised bottle keeps its identity while it stays in the same place, even if a shadow changes its colours (the arm
hovering over it does exactly that); a bottle that moves, or a different track, has to be recognised afresh.
"""

import cv2
import numpy as np
import pytest

from markos_vision.perception import bottle_yolo
from markos_vision.perception.bottle_types import BottleLibrary

GREEN, SHADOWED = (60, 170, 60), (40, 60, 130)  # BGR: the enrolled look, and a very different one


def poly(cx, foot_y, h=134.0, w=41.0):
    top = foot_y - h
    return np.array([(cx - .23 * w, top), (cx + .23 * w, top), (cx + .23 * w, top + .07 * h), (cx + .26 * w, top + .07 * h),
                     (cx + .26 * w, top + .22 * h), (cx + .5 * w, top + .34 * h), (cx + .5 * w, foot_y), (cx - .5 * w, foot_y),
                     (cx - .5 * w, top + .34 * h), (cx - .26 * w, top + .22 * h), (cx - .26 * w, top + .07 * h),
                     (cx - .23 * w, top + .07 * h)], float)


def box_of(p):
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


class Bench:
    def __init__(self):
        self.rng = np.random.default_rng(5)
        self.holder = {}
        self.finder = bottle_yolo.YoloBottleFinder(run_model=lambda fr: self.holder["insts"], library=BottleLibrary(path=None))
        self.home = poly(300.0, 400.0)
        for _ in range(6):
            self.look(self.home, GREEN, frames=1)
        self.finder.enrol("2l soda", 33.0, 3.0)

    def frame(self, p, colour):
        hsv = np.zeros((480, 640, 3), np.uint8)
        hsv[..., 0], hsv[..., 1] = 10, np.clip(90 + self.rng.normal(0, 8, (480, 640)), 40, 150)
        hsv[..., 2] = np.clip(95 + self.rng.normal(0, 8, (480, 640)), 40, 130)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.fillPoly(img, [np.round(p).astype(np.int32)], colour)
        return cv2.add(img, self.rng.integers(0, 4, img.shape, dtype=np.uint8))

    def look(self, p, colour, tid=1, frames=12):
        """Feed the same scene for a while (the finder re-judges a track every 8 frames) and return the last answer."""
        out = None
        for _ in range(frames):
            self.holder["insts"] = [{"id": tid, "box": box_of(p), "conf": .6, "polygon": p}]
            out = self.finder.find(self.frame(p, colour))
        return out


@pytest.fixture
def bench():
    return Bench()


def test_a_bottle_that_looks_as_enrolled_is_verified_without_the_latch(bench):
    r = bench.look(bench.home, GREEN)
    assert r["verified"] and not r["latched"]


def test_a_shadow_that_changes_its_colours_does_not_lose_it(bench):
    bench.look(bench.home, GREEN)
    r = bench.look(bench.home, SHADOWED)
    assert r["sim"] < 0.7, "the test's 'shadowed' look is not different enough to be a test"
    assert r["verified"] and r["latched"] and r["type"]["name"] == "2l soda"


def test_a_small_nudge_keeps_the_latch(bench):
    bench.look(bench.home, GREEN)
    bench.look(bench.home, SHADOWED)
    r = bench.look(poly(300.0 + 3, 400.0 + 2), SHADOWED)
    assert r["verified"] and r["latched"]


def test_a_bottle_moved_far_has_to_be_recognised_afresh(bench):
    bench.look(bench.home, GREEN)
    bench.look(bench.home, SHADOWED)
    r = bench.look(poly(300.0 + 45, 400.0), SHADOWED)
    assert not r["verified"]


def test_it_is_recognised_normally_again_once_it_looks_right(bench):
    bench.look(bench.home, GREEN)
    bench.look(bench.home, SHADOWED)
    bench.look(poly(300.0 + 45, 400.0), SHADOWED)
    r = bench.look(bench.home, GREEN)
    assert r["verified"] and not r["latched"]


def test_a_different_track_at_the_same_spot_does_not_inherit_the_latch(bench):
    bench.look(bench.home, GREEN)
    bench.look(bench.home, SHADOWED)
    r = bench.look(bench.home, SHADOWED, tid=2)
    assert not r["verified"]


def test_the_latch_expires_so_a_swapped_bottle_cannot_inherit_it_forever(bench, monkeypatch):
    assert bench.look(bench.home, GREEN, tid=3)["verified"]
    monkeypatch.setattr(bottle_yolo, "LATCH_SECONDS", 0.0)
    r = bench.look(bench.home, SHADOWED, tid=3)
    assert not r["verified"]
