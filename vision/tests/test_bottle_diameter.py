"""The camera sees the NEAR EDGE of the bottle's round foot; the tape measures its CENTRE.

Simulated with real pinhole cameras that see the lowest point of a foot's rim circle. Checks how far off the bottle's axis is
placed with and without the diameter correction: with the calibration camera, with a fatter bottle, and with the laptop moved.
"""

import contextlib
import io
import json
import math

import numpy as np
import pytest

from markos_vision.apps import calibrate_desk
from markos_vision.geometry import camera_model as cm
from markos_vision.geometry.desk_calibration import DeskCalibration
from markos_vision.geometry.pick_geometry import PickConfig, bottle_radius_cm, locate_on_desk

F, K1 = 651.0, -0.10
STARS = [(-3.95, 21.9, 3.5), (3.95, 21.9, 3.5)]
R_CAL = 2.75  # the 5.5 cm bottle the calibration was measured with
SPOTS = [(22, 12), (22, 2), (22, -8), (28, 12), (28, 2), (28, -8), (34, 12), (34, 2), (34, -8), (30, -3)]
TEST_POINTS = [(24, 8), (26, -4), (30, 6), (32, -6), (33, 10), (25, 0), (29, -1), (31, 3)]


def camera(pos, h=24.0, pitch=0.0, aim=(28.0, 2.0)):
    yaw = math.atan2(-(aim[0] - pos[0]), -(aim[1] - pos[1]))
    return cm.CameraModel(F, K1, pos[0], pos[1], h, yaw, math.radians(pitch))


def contact_pixel(cam, x, y, r):
    """Ground truth by brute force: the lowest point of the foot's rim circle in the picture (independent of the library)."""
    best = None
    for theta in np.linspace(0, 2 * math.pi, 1440, endpoint=False):
        p = cam.project((x + r * math.cos(theta), y + r * math.sin(theta), 0.0))
        if best is None or p[1] > best[1]:
            best = p
    return best


class FakeTracker:
    def __init__(self, get_camera):
        self.get_camera = get_camera

    def match(self, gray):
        return [(*self.get_camera().project(s), 0.9, (0, 0, 20, 20), True) for s in STARS]


class FakeCapture:
    def isOpened(self):
        return True

    def release(self):
        pass

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)


class Rig:
    """A desk calibration made with the calibration bottle, plus the camera it was made with."""

    def __init__(self, tape_noise_px=0.3, seed=7, diameter_known=True):
        self.rng = np.random.default_rng(seed)
        self.true_a = camera((-5.0, 100.0))
        self.cal = DeskCalibration(path=None)
        self.cal.set_axis(0.0, 0.0)
        for x, y in SPOTS:
            self.cal.add(*(contact_pixel(self.true_a, x, y, R_CAL) + self.rng.normal(0, tape_noise_px, 2)), x, y)
        if diameter_known:
            self.cal.set_bottle_diameter(2 * R_CAL)
        self.cal.set_star_geometry(STARS)
        self.learn_stars()

    def learn_stars(self):
        with contextlib.redirect_stdout(io.StringIO()):
            calibrate_desk.learn_stars(self.cal, FakeCapture(), FakeTracker(lambda: self.true_a), F)

    def lock_on(self, cam):
        lock = cm.StarLock(self.cal)
        for _ in range(100):
            if lock.update([cam.project(s) + self.rng.normal(0, 0.1, 2) for s in STARS]):
                break
        assert lock.locked, lock.detail
        return lock

    def errors_mm(self, cam, r_true, r_arg, points=TEST_POINTS):
        out = []
        for x, y in points:
            px, py = contact_pixel(cam, x, y, r_true)
            out.append(math.hypot(*(np.array(self.cal.to_desk(px, py, r_arg)) - (x, y))) * 10)
        return np.array(out)

    @contextlib.contextmanager
    def diameter_unknown(self):
        saved = self.cal.bottle_diameter_cm
        self.cal.bottle_diameter_cm = None
        try:
            yield
        finally:
            self.cal.set_bottle_diameter(saved)


@pytest.fixture(scope="module")
def rig():
    return Rig()


@pytest.fixture(autouse=True)
def fixed_camera(rig):
    rig.cal.live_map = None  # every test starts with the laptop where it was calibrated


def test_calibration_fit_is_tight(rig):
    assert max(rig.cal.residuals_mm()) < 4.0
    assert rig.cal.star_cam["rms"] < 0.6


def test_the_calibration_bottle_seen_from_the_calibration_camera_is_unchanged(rig):
    left_out, given = rig.errors_mm(rig.true_a, R_CAL, None), rig.errors_mm(rig.true_a, R_CAL, R_CAL)
    assert given.max() < 8 and left_out.max() < 8
    assert abs(given - left_out).max() < 1.0


def test_a_fatter_bottle_is_placed_correctly_only_when_its_diameter_is_used(rig):
    not_used = rig.errors_mm(rig.true_a, 5.0, None)
    used = rig.errors_mm(rig.true_a, 5.0, 5.0)
    assert not_used.mean() > 15  # the near edge of a 10 cm foot is 2.25 cm nearer than the calibration bottle's
    assert used.max() < 8


def test_a_moved_laptop_places_the_calibration_bottle_better_with_the_diameter(rig):
    moved = camera((25.0, 85.0))
    with rig.diameter_unknown():
        rig.lock_on(moved)
        before = rig.errors_mm(moved, R_CAL, None)
    rig.lock_on(moved)
    after = rig.errors_mm(moved, R_CAL, R_CAL)
    assert after.mean() <= before.mean() + 0.5, "the correction must not make the calibration bottle worse"
    assert after.max() < 8


def test_a_moved_laptop_places_a_fat_bottle_correctly_with_its_diameter(rig):
    moved = camera((25.0, 85.0))
    with rig.diameter_unknown():
        rig.lock_on(moved)
        before = rig.errors_mm(moved, 5.0, None)
    rig.lock_on(moved)
    after = rig.errors_mm(moved, 5.0, 5.0)
    assert before.mean() > 2 * after.mean()
    assert after.max() < 10


def test_a_laptop_moved_far_round_to_the_side_needs_the_correction_most(rig):
    far = camera((60.0, 70.0))
    with rig.diameter_unknown():
        rig.lock_on(far)
        before = rig.errors_mm(far, R_CAL, None)
    rig.lock_on(far)
    after = rig.errors_mm(far, R_CAL, R_CAL)
    assert after.mean() < before.mean() / 3
    assert after.max() < 8


def test_a_tape_spot_added_with_the_laptop_moved_is_converted_with_the_edge_taken_into_account(rig):
    moved = camera((25.0, 85.0))
    lock = rig.lock_on(moved)
    with_correction, without = [], []
    for x, y in [(30, 14), (30, 0), (34, -8), (36, 6)]:
        seen = contact_pixel(moved, x, y, R_CAL)
        wanted = contact_pixel(rig.true_a, x, y, R_CAL)  # what the calibration camera saw for the same spot
        with_correction.append(np.hypot(*(np.array(calibrate_desk.calibration_pixel(lock, *seen)) - wanted)))
        with rig.diameter_unknown():
            lock_without = rig.lock_on(moved)
            without.append(np.hypot(*(np.array(calibrate_desk.calibration_pixel(lock_without, *seen)) - wanted)))
        rig.lock_on(moved)
    assert max(with_correction) < max(without) and max(with_correction) < 4.0


def test_locate_on_desk_uses_the_enrolled_types_diameter(rig):
    px, py = contact_pixel(rig.true_a, 30.0, 0.0, 5.0)
    cfg = PickConfig(path=None)
    plain = locate_on_desk((px, py), rig.cal, cfg)
    sized = locate_on_desk((px, py), rig.cal, cfg, bottle_radius_cm({"diameter_cm": 10.0}))
    off_plain = math.hypot(plain["x_cm"] - 30.0, plain["y_cm"]) * 10
    off_sized = math.hypot(sized["x_cm"] - 30.0, sized["y_cm"]) * 10
    assert off_sized < 4 and off_plain > 15


def test_a_bottle_type_without_a_diameter_is_taken_as_the_calibration_bottle():
    assert bottle_radius_cm({"name": "x"}) is None and bottle_radius_cm(None) is None
    assert bottle_radius_cm({"diameter_cm": 5.5}) == 2.75


def test_the_diameter_is_saved_and_an_old_file_without_it_still_maps_as_before(rig, tmp_path):
    path = str(tmp_path / "desk_calibration.json")
    rig.cal.path = path
    rig.cal.save()
    back = DeskCalibration(path=path)
    assert back.bottle_diameter_cm == 5.5 and back.bottle_radius_cm == 2.75
    px, py = contact_pixel(rig.true_a, 30.0, 0.0, 5.0)
    assert back.to_desk(px, py, 5.0) == rig.cal.to_desk(px, py, 5.0)

    data = json.load(open(path))
    data.pop("bottle_diameter_cm")
    json.dump(data, open(path, "w"))
    older = DeskCalibration(path=path)
    assert older.bottle_diameter_cm is None
    assert older.to_desk(px, py, 5.0) == older.to_desk(px, py)


def test_a_camera_fitted_before_the_diameter_was_known_is_flagged_and_refitting_fixes_it():
    """Such a camera has the near edge bent into it: the moved-laptop mapping must not apply an edge correction to it."""
    rig = Rig(tape_noise_px=0.0, diameter_known=False)  # noise-free: this compares two fits
    rig.cal.set_bottle_diameter(2 * R_CAL)  # told afterwards
    assert rig.cal.star_camera_radius_cm is None
    assert any("fitted without the bottle's diameter" in note for note in rig.cal.notes())

    moved = camera((25.0, 85.0))
    rig.lock_on(moved)
    held_off = rig.errors_mm(moved, R_CAL, R_CAL)
    with rig.diameter_unknown():
        rig.lock_on(moved)
        plain = rig.errors_mm(moved, R_CAL, None)
    assert abs(held_off.mean() - plain.mean()) < 0.5  # identical: nothing was applied

    rig.learn_stars()  # press s again: now the camera models the foot
    assert rig.cal.star_camera_radius_cm == R_CAL
    assert not any("fitted without" in note for note in rig.cal.notes())
    rig.lock_on(moved)
    assert rig.errors_mm(moved, R_CAL, R_CAL).mean() < held_off.mean()


def test_the_camera_models_foot_pixel_agrees_with_brute_force(rig):
    """The lowest point is flat in y, so compare y tightly; x is only as fine as the brute force's 0.25 degree sampling."""
    for x, y in TEST_POINTS:
        for r in (2.75, 5.0):
            exact = contact_pixel(rig.true_a, x, y, r)
            found = rig.true_a.foot_pixel(x, y, r)
            assert found[1] == pytest.approx(exact[1], abs=0.01)
            assert found[0] == pytest.approx(exact[0], abs=0.2)
