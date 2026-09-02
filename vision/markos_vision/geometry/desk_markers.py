"""Printed ArUco markers taped flat on the desk, as an alternative to placing the laptop from the stars.

Their positions are learned once through the tape calibration (``learn``); afterwards ``MarkerLock`` rebuilds the
pixel -> desk mapping from whichever markers are in view, wherever the camera is put, with no tape. Run this module to
write the printable marker sheets.
"""

import math
import sys

import cv2
import numpy as np

from markos_vision.config import MARKER_SHEETS, ensure_parent
from markos_vision.geometry.desk_calibration import _apply, _fit

DICTIONARY = cv2.aruco.DICT_4X4_50
LEARN_FRAMES = 25  # frames averaged when recording where the markers are
LOCK_FRAMES = 20  # frames averaged when locking the mapping in a new camera position
MIN_SEEN_FRAC = 0.6  # a marker has to show up in this share of those frames to count
MIN_MARKERS = 3  # the mapping is only built from three or more markers: one small marker can't fix a whole desk
STABLE_PX = 1.5  # a marker's corners may wander this much (std) over the lock frames; more means the camera is moving
MAX_FIT_MM = 12.0  # the markers must agree with each other to this well, or one has been moved. Their learned
                   # positions carry the tape calibration's own error (~5 mm), so this can't be much tighter
MOVED_PX = 3.0  # markers this far from where they were locked mean the camera has moved...
MOVED_FRAMES = 5  # ...for this many frames in a row

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    return _detector


def detect(frame):
    """{marker id: 4x2 array of its corners in pixels}, in the marker's own
    corner order (which stays the same wherever the camera is)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = _get_detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.flatten(), corners)}


class Accumulator:
    """Gathers marker detections over a run of frames, so a position rests on
    an average and not on one noisy frame."""

    def __init__(self):
        self.frames = 0
        self.seen = {}

    def add(self, detections):
        self.frames += 1
        for mid, c in detections.items():
            self.seen.setdefault(mid, []).append(c)

    def mean(self):
        """{id: mean corners} for markers seen often enough, and {id: worst corner wander (px)}."""
        means, wander = {}, {}
        for mid, cs in self.seen.items():
            if len(cs) >= MIN_SEEN_FRAC * self.frames:
                stack = np.array(cs)
                means[mid] = stack.mean(axis=0)
                wander[mid] = float(stack.std(axis=0).max())
        return means, wander


def learn(calib, accumulator):
    """Records where each visible marker's corners are, in the calibration's
    measuring frame, using the tape calibration as it stands. Do this once,
    with the camera where the calibration was done. Returns {id: corners} that
    were learned (empty if the calibration isn't ready)."""
    if not calib.ready:
        return {}
    means, wander = accumulator.mean()
    learned = {mid: [list(_apply(calib.H, x, y)) for x, y in corners]
               for mid, corners in means.items() if wander[mid] < STABLE_PX}
    if len(learned) >= MIN_MARKERS:   # fewer than that can't rebuild a mapping later, so don't keep them
        calib.set_markers(learned)
    return learned


class MarkerLock:
    """Rebuilds the desk mapping from the printed markers wherever the camera
    happens to be, so the laptop can go anywhere. It collects LOCK_FRAMES of the
    markers, fits pixel -> desk from their corners, and hands that to the
    calibration (calib.live_H). Afterwards it watches the markers: if they
    shift in the picture the camera has moved, and it starts locking again --
    the arm isn't used in between."""

    def __init__(self, calib):
        self.calib = calib
        self.events = 0  # how many times the camera was found to have moved
        self.seen = {}
        self.reset()

    def reset(self):
        self.acc = Accumulator()
        self.H = None
        self.locked_px = None
        self.problem = ""  # why the last attempt to lock was refused; stays on screen while it tries again
        self.detail = "looking for the desk markers"
        self._off = 0
        self.calib.live_H = None

    @property
    def locked(self):
        return self.H is not None

    def update(self, frame):
        known = {i: c for i, c in detect(frame).items() if i in self.calib.markers}
        self.seen = known   # for drawing
        if self.H is None:
            self.acc.add(known)
            if self.acc.frames >= LOCK_FRAMES:
                self._try_lock()
            else:
                progress = f"locking on the desk markers {self.acc.frames}/{LOCK_FRAMES} (seeing {sorted(known)})"
                self.detail = f"{self.problem} -- {progress}" if self.problem else progress
        else:
            self._watch(known)
        return self.locked

    def _try_lock(self):
        means, wander = self.acc.mean()
        moving = [i for i, w in wander.items() if w > STABLE_PX]
        if len(means) < MIN_MARKERS:
            self.problem = (f"need {MIN_MARKERS} desk markers, seeing {len(means)} {sorted(means)}: "
                            "keep them in view and unblocked")
        elif moving:
            self.problem = f"the picture is moving (marker {moving[0]} wobbled {wander[moving[0]]:.1f} px): hold the laptop still"
        else:
            src = np.vstack([means[i] for i in sorted(means)])
            dst = np.vstack([np.array(self.calib.markers[i]) for i in sorted(means)])
            src_pts = [{"px": x, "py": y, "x_cm": u, "y_cm": v} for (x, y), (u, v) in zip(src, dst)]
            H = _fit(src_pts)
            errs = ([math.hypot(*(np.array(_apply(H, x, y)) - (u, v))) * 10 for (x, y), (u, v) in zip(src, dst)]
                    if H is not None else [float("inf")])
            if max(errs) > MAX_FIT_MM:
                self.problem = (f"the markers don't agree with where they were learned ({max(errs):.0f} mm off): "
                                "one may have been moved -- put them back, or learn them again with m")
            else:
                self.H, self.locked_px = H, means
                self.calib.live_H = H
                self.problem = ""
                self.detail = f"locked on {len(means)} desk markers {sorted(means)}"
                self._off = 0
                return
        self.detail = self.problem
        self.acc = Accumulator()   # try again from scratch

    def _watch(self, known):
        seen = [i for i in known if i in self.locked_px]
        if not seen:
            return
        shifts = [float(np.abs(known[i] - self.locked_px[i]).mean()) for i in seen]
        if float(np.median(shifts)) > MOVED_PX:
            self._off += 1
        else:
            self._off = 0
        if self._off >= MOVED_FRAMES:
            self.events += 1
            self.reset()
            self.problem = "the camera moved"
            self.detail = "the camera moved: locking on the desk markers again"


def make_sheet(path, ids=(0, 1, 2, 3), side_cm=9.0, quiet_cm=0.5, dpi=200):
    """A printable A4 page of the markers: 2 columns, sized so they fit with a
    little white around each. Print it to fill the page (any scaling is fine:
    where the markers are gets measured, not assumed) and tape them FLAT."""
    px = dpi / 2.54
    page = np.full((int(29.7 * px), int(21.0 * px)), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    cell = side_cm + 2 * quiet_cm
    for n, mid in enumerate(ids):
        col, row = n % 2, n // 2
        x0 = int(((21.0 - 2 * cell) / 2 + col * cell + quiet_cm) * px)
        y0 = int((1.2 + row * (cell + 1.0) + quiet_cm) * px)
        s = int(side_cm * px)
        page[y0:y0 + s, x0:x0 + s] = cv2.aruco.generateImageMarker(dictionary, mid, s)
        cv2.putText(page, f"marker {mid}", (x0, y0 + s + int(0.45 * px)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        for cx, cy in ((x0 - int(quiet_cm * px), y0 - int(quiet_cm * px)), (x0 + s + int(quiet_cm * px), y0 - int(quiet_cm * px)),
                       (x0 - int(quiet_cm * px), y0 + s + int(quiet_cm * px)), (x0 + s + int(quiet_cm * px), y0 + s + int(quiet_cm * px))):
            cv2.drawMarker(page, (cx, cy), 160, cv2.MARKER_CROSS, int(0.5 * px), 1)   # faint cutting marks
    cv2.imwrite(ensure_parent(path), page)
    return path


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else MARKER_SHEETS
    print("wrote", make_sheet(f"{base}1.png", ids=(0, 1, 2, 3)))
    print("wrote", make_sheet(f"{base}2.png", ids=(4, 5, 6, 7)))
