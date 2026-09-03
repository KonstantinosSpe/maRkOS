"""Star tracking: sub-pixel refinement of the template match."""

import cv2
import numpy as np

from markos_vision.perception import star_tracking
from markos_vision.perception.star_tracking import StarTracker


def blob(rng, cx, cy, size=22.0):
    """A star-ish bright shape rendered at a fractional position (anti-aliased by supersampling)."""
    big = np.zeros((480 * 4, 640 * 4), np.uint8)
    pts = []
    for k in range(10):
        r = size * 4 * (1.0 if k % 2 == 0 else 0.42)
        a = np.pi * k / 5 - np.pi / 2
        pts.append((cx * 4 + r * np.cos(a), cy * 4 + r * np.sin(a)))
    cv2.fillPoly(big, [np.round(np.array(pts)).astype(np.int32)], 255)
    img = cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA)
    img = np.clip(img.astype(float) * 0.8 + 60 + rng.normal(0, 2, img.shape), 0, 255)
    return img.astype(np.uint8)


def test_subpixel_refinement_beats_whole_pixel_matching():
    rng = np.random.default_rng(2)
    ref = blob(rng, 200.0, 150.0)
    tpl = ref[120:181, 170:231].copy()  # 61x61 template around the star at integer position
    tracker_centre = (170 + 61 / 2, 120 + 61 / 2)  # what the template centre is, in the reference frame

    errs_int, errs_sub = [], []
    for _ in range(60):
        tx, ty = rng.uniform(100, 500), rng.uniform(100, 350)
        frame = blob(rng, tx, ty)
        tracker = StarTracker([tpl])
        for _ in range(star_tracking.STABLE_FRAMES + 1):
            det = tracker.match(frame)
        cx, cy = det[0][0], det[0][1]
        exp_x, exp_y = tx + (tracker_centre[0] - 200.0), ty + (tracker_centre[1] - 150.0)
        errs_sub.append(np.hypot(cx - exp_x, cy - exp_y))
        # what whole-pixel matching would have given
        res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        errs_int.append(np.hypot(loc[0] + 61 / 2 - exp_x, loc[1] + 61 / 2 - exp_y))

    assert np.mean(errs_sub) < 0.6 * np.mean(errs_int)
    assert np.max(errs_sub) < 0.5
