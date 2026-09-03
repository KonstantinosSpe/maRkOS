"""Per-bottle neck-shape calibration.

A real nearly-straight-sided bottle measured a neck ratio of 0.84-0.89 on the model's soft outline. A fixed cut-off in the
middle of that flickered between 'ok' and 'not bottle-shaped'. Calibrating per bottle must keep it steady, and a straight
column of the same colour must still be turned down.
"""

import cv2
import numpy as np

from markos_vision.perception.bottle_types import BottleLibrary
from markos_vision.perception.bottle_yolo import YoloBottleFinder

GREEN = (60, 170, 60)
JITTER = 0.8  # px of wobble in the neck width; the real bottle moved its ratio by about +-0.03
CX, TOP = 430, 120


def fat_bottle(rng, jitter):
    nh = 27.0 + rng.normal(0, jitter)  # neck half-width: ~54 px of a 64 px body
    return np.array([(CX - 15, TOP), (CX + 15, TOP), (CX + 15, TOP + 20), (CX + nh, TOP + 20), (CX + nh, TOP + 50),
                     (CX + 32, TOP + 80), (CX + 32, TOP + 190), (CX - 32, TOP + 190), (CX - 32, TOP + 80),
                     (CX - nh, TOP + 50), (CX - nh, TOP + 20), (CX - 15, TOP + 20)], float)


def column():
    return np.array([(CX - 32, TOP), (CX + 32, TOP), (CX + 32, TOP + 190), (CX - 32, TOP + 190)], float)


def box_of(p):
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


class Bench:
    def __init__(self, library):
        self.rng = np.random.default_rng(9)
        self.shape = "bottle"
        self.holder = {}
        self.finder = YoloBottleFinder(run_model=lambda frame: self.holder["insts"], library=library)

    def desk(self):
        hsv = np.zeros((480, 640, 3), np.uint8)
        hsv[..., 0] = 10 + self.rng.integers(-2, 3, (480, 640))
        hsv[..., 1] = np.clip(90 + self.rng.normal(0, 12, (480, 640)), 40, 150)
        hsv[..., 2] = np.clip(80 + self.rng.normal(0, 10, (480, 640)), 40, 120)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def run(self, n):
        outs = []
        for _ in range(n):
            poly = fat_bottle(self.rng, JITTER) if self.shape == "bottle" else column()
            frame = self.desk()
            cv2.fillPoly(frame, [np.round(poly).astype(np.int32)], GREEN)
            frame = cv2.add(frame, self.rng.integers(0, 6, frame.shape, dtype=np.uint8))
            self.holder["insts"] = [{"id": 1, "box": box_of(poly), "conf": 0.8, "polygon": poly}]
            outs.append(self.finder.find(frame))
        return outs


def test_an_unenrolled_fat_necked_bottle_is_tracked_throughout_and_never_called_a_column():
    bench = Bench(BottleLibrary(path=None))
    outs = bench.run(60)
    assert all(o is not None for o in outs), "an unenrolled fat-necked bottle flickered out"
    assert not any("column" in reason for _, reason in bench.finder.rejected)


def test_an_enrolled_bottle_is_judged_against_its_own_measured_ratio_and_stays_verified():
    bench = Bench(BottleLibrary(path=None))
    bench.run(5)
    bench.finder.enrol("boost-like", 22.0, 1.8)
    outs = bench.run(80)
    verified = sum(1 for o in outs if o is not None and o["verified"] and o["missed"] == 0)
    assert verified == 80, "the enrolled bottle flickered between ok and not-a-bottle"


def test_a_straight_column_of_the_same_colour_in_the_same_place_is_still_turned_down():
    bench = Bench(BottleLibrary(path=None))
    bench.run(5)
    bench.finder.enrol("boost-like", 22.0, 1.8)
    bench.run(20)
    bench.shape = "column"
    outs = bench.run(40)
    target = [o for o in outs[-12:] if o is not None and o["missed"] == 0]
    assert not target


def test_the_soft_outline_of_a_fat_necked_bottle_gives_a_neck_ratio_in_the_range_seen_in_use():
    bench = Bench(BottleLibrary(path=None))
    ratios = [o["cap"]["neck_ratio"] for o in bench.run(40)]
    assert 0.75 < min(ratios) and max(ratios) < 0.97, "the simulated bottle should look like the real one (0.84-0.89)"
