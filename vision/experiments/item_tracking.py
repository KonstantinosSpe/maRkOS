"""Tracks a small item you have taught it by colour: saved crops define the colour range, blobs are filtered by size and shape.

An early experiment in picking up something other than a bottle; see ``teach_item.py`` to teach it.
"""

import glob
import math
import os

import cv2
import numpy as np

from markos_vision.config import ITEM_CROPS as CROP_DIR

V_MARGIN = 10  # slack under the dimmest taught item pixel, for lighting drift
S_MARGIN = 10  # slack over the most colourful taught item pixel
CORE_TRIM = 0.2  # ignore this fraction of each side of a taught crop: the box edge catches desk
MIN_AREA_FRAC = 0.5  # of the smallest taught view -- the item shrinks with distance
MAX_AREA_FRAC = 3.0  # of the largest taught view
MIN_AREA_PX = 80
MIN_SHORT_SIDE_PX = 6
MAX_ASPECT = 4.0  # the taught views run about 1.3 to 3; tilt and rotation stretch it further
MIN_SOLIDITY = 0.7  # how much of its own bounding rectangle the blob fills
IDEAL_ASPECT = 1.6
STABLE_FRAMES = 3
POS_TOLERANCE_PX = 15  # unlike the stars, the item is allowed to be moving
GRACE_FRAMES = 10  # keep reporting the last known position for this many consecutive misses


def _view_number(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def load_crops():
    paths = sorted(glob.glob(f"{CROP_DIR}/view_*.png"), key=_view_number)
    return [c for c in (cv2.imread(p) for p in paths) if c is not None]


def save_crop(crop):
    os.makedirs(CROP_DIR, exist_ok=True)
    existing = glob.glob(f"{CROP_DIR}/view_*.png")
    n = max((_view_number(p) for p in existing), default=-1) + 1
    cv2.imwrite(f"{CROP_DIR}/view_{n}.png", crop)


def clear_crops():
    for p in glob.glob(f"{CROP_DIR}/view_*.png"):
        os.remove(p)


class ItemTracker:
    """Finds the taught item by being bright and pale against the desk, then
    reading position, size and rotation off the blob's outline. That's
    deliberately not appearance matching: the item is a plain block with
    almost no texture, so there's nothing for template or feature matching to
    hold on to, but a blob is the same blob at any distance or angle.

    What "bright and pale" means is learned from the taught crops, and the
    size limits from how big the item looked in them. Held-steady and
    brief-dropout handling follows StarTracker."""

    def __init__(self, crops):
        cores = []
        for crop in crops:
            h, w = crop.shape[:2]
            my, mx = int(h * CORE_TRIM), int(w * CORE_TRIM)
            core = crop[my:h - my, mx:w - mx]
            if core.size:
                cores.append(cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3))
        if not cores:
            raise ValueError("no usable taught views")
        px = np.concatenate(cores)
        self.v_lo = int(max(40, np.percentile(px[:, 2], 2) - V_MARGIN))
        self.s_hi = int(min(255, np.percentile(px[:, 1], 98) + S_MARGIN))

        areas = [int(cv2.countNonZero(self._mask(c))) for c in crops]
        areas = [a for a in areas if a > 0]
        self.min_area = max(MIN_AREA_PX, int(MIN_AREA_FRAC * min(areas)))
        self.max_area = int(MAX_AREA_FRAC * max(areas))

        self.candidates = []  # every blob that passed the size/shape checks last frame, for drawing
        self.mask = None
        self._pending = None
        self._pending_count = 0
        self._last = None
        self._missed = 0

    def describe(self):
        return f"V>={self.v_lo}, S<={self.s_hi}, blob area {self.min_area}..{self.max_area}px"

    def _mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 0, self.v_lo), (179, self.s_hi, 255))

    def _blobs(self, frame_bgr):
        mask = self._mask(frame_bgr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # The item's shading and its grid face punch holes in the mask.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if not self.min_area <= area <= self.max_area:
                continue
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
            long_side, short_side = max(rw, rh), min(rw, rh)
            if short_side < MIN_SHORT_SIDE_PX:
                continue
            aspect = long_side / short_side
            solidity = area / (rw * rh)
            if aspect > MAX_ASPECT or solidity < MIN_SOLIDITY:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(c))
            edge = box[1] - box[0] if np.linalg.norm(box[1] - box[0]) >= np.linalg.norm(box[2] - box[1]) \
                else box[2] - box[1]
            angle = math.degrees(math.atan2(edge[1], edge[0]))
            angle = (angle + 90) % 180 - 90  # a block looks the same turned 180 deg
            blobs.append({
                "cx": cx, "cy": cy, "long": long_side, "short": short_side, "area": area,
                "angle": angle, "corners": box, "score": solidity / (1 + abs(aspect - IDEAL_ASPECT)),
            })
        return blobs

    def find(self, frame_bgr):
        """Dict with cx, cy, long/short (side lengths, px), area, angle
        (degrees, of the long side), corners, score and live -- or None."""
        blobs = self._blobs(frame_bgr)
        self.candidates = [b["corners"] for b in blobs]
        best = max(blobs, key=lambda b: b["score"]) if blobs else None
        return self._stabilise(best)

    def _stabilise(self, hit):
        if hit is not None:
            p = self._pending
            if p is not None and math.hypot(hit["cx"] - p["cx"], hit["cy"] - p["cy"]) <= POS_TOLERANCE_PX:
                self._pending_count += 1
            else:
                self._pending_count = 1
            self._pending = hit
            if self._pending_count >= STABLE_FRAMES:
                self._last, self._missed = hit, 0
                return {**hit, "live": True}
        else:
            self._pending, self._pending_count = None, 0

        if self._last is not None and self._missed < GRACE_FRAMES:
            self._missed += 1
            return {**self._last, "live": False}
        self._last = None
        return None
