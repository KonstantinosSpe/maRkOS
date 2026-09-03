"""The desk calibration against a synthetic laptop webcam looking at the desk from a slant, with lens distortion and noise.

What the calibration recovers is compared with the tape-measured truth, in the workspace the arm really has, and its
diagnostics (leave-one-out, coverage advice, extrapolation and camera-moved flags, saving) are checked.
"""

import math

import numpy as np
import pytest

from markos_vision.geometry.desk_calibration import DeskCalibration, base_contact
from markos_vision.geometry.pick_geometry import PickConfig, locate_on_desk

F, CX, CY = 600.0, 320.0, 240.0
CAM = np.array([0.0, 85.0, 32.0])  # cm: 85 cm in front of the base (toward +y), 32 cm above the desk
PITCH = math.radians(18)  # looking down this much
K1 = -0.06  # mild barrel distortion, typical of a laptop webcam
# Where the tape-measured marks sit (x right, y toward the camera), all around the arm's working ring
MARKS = [(-20, 27), (0, 30), (20, 27), (-14, 34), (14, 34), (0, 37), (-24, 20), (24, 20)]


def project(x_cm, y_cm, z_cm=0.0):
    """Pixel of a point on the desk, for a camera at CAM looking toward -y, pitched down."""
    f = np.array([0.0, -math.cos(PITCH), -math.sin(PITCH)])
    r = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, math.sin(PITCH), -math.cos(PITCH)])
    q = np.array([x_cm, y_cm, z_cm]) - CAM
    zc = q @ f
    xn, yn = (q @ r) / zc, (q @ d) / zc
    rr = xn * xn + yn * yn
    xn, yn = xn * (1 + K1 * rr), yn * (1 + K1 * rr)
    return CX + F * xn, CY + F * yn


@pytest.fixture
def rng():
    return np.random.default_rng(12)


def calibrate(rng, marks, noise_px, path=None):
    cal = DeskCalibration(path=path)
    for x, y in marks:
        px, py = project(x, y)
        cal.add(px + rng.normal(0, noise_px), py + rng.normal(0, noise_px), x, y)
    return cal


def field_error_mm(rng, cal, noise_px, trials=300):
    """Error over random bottle spots across the working area, measuring the bottle with pixel noise too."""
    errs = []
    cfg = PickConfig(path=None)
    for _ in range(trials):
        r = rng.uniform(29, 37)
        b = math.radians(rng.uniform(-45, 45))
        x, y = r * math.sin(b), r * math.cos(b)
        px, py = project(x, y)
        loc = locate_on_desk((px + rng.normal(0, noise_px), py + rng.normal(0, noise_px)), cal, cfg)
        errs.append(math.hypot(loc["x_cm"] - x, loc["y_cm"] - y) * 10)
    return np.array(errs)


# ── accuracy ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_noiseless_points_are_reproduced_to_what_lens_distortion_alone_costs(rng):
    errors = field_error_mm(rng, calibrate(rng, MARKS, 0.0), 0.0)
    assert errors.max() < 6.0


def test_realistic_pixel_noise_stays_well_inside_what_the_arm_tolerates(rng):
    errors = field_error_mm(rng, calibrate(rng, MARKS, 1.0), 1.0)
    assert np.percentile(errors, 95) < 15.0


# ── diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_four_points_fit_exactly_and_say_nothing_so_leave_one_out_needs_more(rng):
    cal = calibrate(rng, MARKS[:4], 1.0)
    assert cal.leave_one_out_mm() == []
    assert max(cal.residuals_mm()) < 0.01


def test_leave_one_out_is_never_more_optimistic_than_the_residuals(rng):
    cal = calibrate(rng, MARKS, 1.0)
    assert max(cal.leave_one_out_mm()) >= max(cal.residuals_mm()) - 1e-6


def test_a_mis_measured_point_shows_up_in_the_leave_one_out_figures(rng):
    bad = DeskCalibration(path=None)
    for i, (x, y) in enumerate(MARKS):
        px, py = project(x, y)
        bad.add(px + rng.normal(0, 0.7), py + rng.normal(0, 0.7), x + (4.0 if i == 3 else 0.0), y)  # #4 read 4 cm wrong
    assert bad.leave_one_out_mm()[3] > 25


def test_points_along_a_line_are_flagged(rng):
    line = DeskCalibration(path=None)
    for x, y in [(0, 25), (0, 30), (0, 35), (0, 40), (0, 45)]:
        line.add(*project(x, y), x, y)
    assert any("line" in note for note in line.notes())


def test_well_spread_points_give_no_advice(rng):
    assert calibrate(rng, MARKS, 0.5).notes() == []


def test_a_spot_far_outside_the_marks_is_flagged_as_extrapolated(rng):
    cal = calibrate(rng, MARKS, 0.5)
    assert cal.inside(*project(0, 32))
    assert not cal.inside(*project(60, 8))


def test_camera_moved_is_flagged_by_the_stars_but_not_by_small_wobble(rng):
    cal = calibrate(rng, MARKS, 0.5)
    cal.set_star_reference((320.0, 200.0), 70.0)
    assert cal.camera_moved((321.5, 201.0), 70.5) is None
    assert "moved" in cal.camera_moved((332.0, 200.0), 70.0)
    assert "spacing" in cal.camera_moved((320.0, 200.0), 76.0)


# ── saving and the contact point ────────────────────────────────────────────────────────────────────────────────────

def test_a_calibration_is_saved_and_reloaded(rng, tmp_path):
    path = str(tmp_path / "desk.json")
    cal = calibrate(rng, MARKS, 0.5, path=path)
    cal.set_star_reference((320.0, 200.0), 70.0)
    again = DeskCalibration(path=path)
    assert again.ready and len(again.points) == 8 and again.stars["sep"] == 70.0
    assert again.to_desk(*project(5, 31))[0] == pytest.approx(cal.to_desk(*project(5, 31))[0], abs=1e-6)


def test_the_contact_point_is_the_outlines_lowest_point_centred_over_the_bottom_rows():
    poly = np.array([(100, 50), (130, 50), (130, 200), (128, 202), (102, 202), (100, 200)], float)
    cx, cy = base_contact(poly, (100, 50, 30, 152))
    assert cx == pytest.approx(115, abs=1) and cy == pytest.approx(202, abs=1)
    assert base_contact(None, (100, 50, 30, 152)) == (115.0, 202.0)


# ── measuring from somewhere other than the base axis (e.g. a corner of the robot's box) ────────────────────────────

def calibrate_in_frame(rng, to_frame, axis_in_frame, noise=0.5):
    cal = DeskCalibration(path=None)
    for x, y in MARKS:
        px, py = project(x, y)
        u, v = to_frame(x, y)
        cal.add(px + rng.normal(0, noise), py + rng.normal(0, noise), u, v)
    if axis_in_frame is not None:
        cal.set_axis(*axis_in_frame)
    return cal


def worst_error_mm(cal, expect):
    worst = 0.0
    for x, y in [(5, 31), (-12, 33), (14, 28), (0, 36)]:
        got = cal.to_desk(*project(x, y))
        worst = max(worst, math.hypot(got[0] - expect(x, y)[0], got[1] - expect(x, y)[1]) * 10)
    return worst


def test_a_shifted_origin_is_handled(rng):
    # the corner is 17 cm to one side of the axis and the axis is 6 cm in from the front face
    cal = calibrate_in_frame(rng, lambda x, y: (x + 17, y + 6), (17, 6))
    assert worst_error_mm(cal, lambda x, y: (x, y)) < 8.0


def test_a_shifted_and_turned_origin_is_handled(rng):
    th = math.radians(30)
    cal = calibrate_in_frame(rng, lambda x, y: (x * math.cos(th) - y * math.sin(th) + 9, x * math.sin(th) + y * math.cos(th) - 4), (9, -4))
    assert worst_error_mm(cal, lambda x, y: (x * math.cos(th) - y * math.sin(th), x * math.sin(th) + y * math.cos(th))) < 8.0


def test_swapped_measuring_axes_are_detected_as_a_mirrored_frame_and_flipped(rng):
    mirror = calibrate_in_frame(rng, lambda x, y: (y + 6, x + 17), (6, 17))
    assert mirror.mirrored, "a swapped (left-handed) frame wasn't detected"
    assert worst_error_mm(mirror, lambda x, y: (-y, x)) < 8.0
    for x, y in [(5, 31), (-12, 33)]:  # once flipped, distances from the axis are preserved
        assert math.hypot(*mirror.to_desk(*project(x, y))) == pytest.approx(math.hypot(x, y), abs=0.8)


def test_the_axis_can_be_entered_after_the_points_without_refitting(rng):
    late = calibrate_in_frame(rng, lambda x, y: (x + 17, y + 6), None)
    before = late.to_desk(*project(5, 31))
    late.set_axis(17, 6)
    after = late.to_desk(*project(5, 31))
    assert after[0] == pytest.approx(5, abs=0.8) and after[1] == pytest.approx(31, abs=0.8)
    assert abs(before[0] - after[0]) > 5


def test_coverage_advice_looks_at_the_spread_around_the_axis_not_the_origin(rng):
    assert calibrate_in_frame(rng, lambda x, y: (x + 17, y + 6), (17, 6)).notes() == []


def test_the_axis_is_saved_and_reloaded(tmp_path):
    path = str(tmp_path / "desk_axis.json")
    c1 = DeskCalibration(path=path)
    for x, y in MARKS:
        c1.add(*project(x, y), x + 17, y + 6)
    c1.set_axis(17, 6)
    c2 = DeskCalibration(path=path)
    assert c2.axis == [17.0, 6.0]
    assert c2.to_desk(*project(5, 31))[0] == pytest.approx(5, abs=0.8)
