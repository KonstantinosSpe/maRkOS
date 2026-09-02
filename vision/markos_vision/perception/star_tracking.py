"""Finds the two red stars stuck on the robot's base by template matching.

The stars are a fixed 7.9 cm apart and sit at known positions on the robot, so where they appear in the picture says
where the camera is. ``StarTracker`` matches the saved templates (made with ``apps.select_stars``) with sub-pixel
refinement and keeps a steady position through brief dropouts. ``load_focal_length`` returns the focal-length estimate
saved during that setup.
"""

import os

import cv2
import numpy as np

from markos_vision.config import STAR_FOCAL_SAMPLES as FOCAL_SAMPLES_PATH
from markos_vision.config import STAR_TEMPLATES as TEMPLATES_PATH

REAL_DISTANCE_CM = 7.9
MATCH_THRESHOLD = 0.55  # cv2.TM_CCOEFF_NORMED score needed to accept a match
GRACE_FRAMES = 10  # keep reporting the last known position for this many consecutive misses
STABLE_FRAMES = 6  # a new match position must hold this steady before it's trusted
POS_TOLERANCE_PX = 6  # how much a match can drift frame-to-frame and still count as "steady"


def _subpixel_peak(result, loc):
    """How far the true match peak sits from the whole-pixel maximum, (dx, dy) within +/-0.5 px, from a parabola
    through the peak and its two neighbours on each axis. Whole-pixel positions carry up to half a pixel of error that
    averaging over frames can't remove (a still scene gives the same integer every time)."""
    x, y = loc
    h, w = result.shape
    out = []
    for lo, c, hi, ok in ((result[y, x - 1] if x > 0 else 0, result[y, x], result[y, x + 1] if x < w - 1 else 0, 0 < x < w - 1),
                          (result[y - 1, x] if y > 0 else 0, result[y, x], result[y + 1, x] if y < h - 1 else 0, 0 < y < h - 1)):
        d = lo - 2 * c + hi
        out.append(max(-0.5, min(0.5, 0.5 * (lo - hi) / d)) if ok and d < -1e-9 else 0.0)
    return float(out[0]), float(out[1])


def load_templates():
    if not os.path.exists(TEMPLATES_PATH):
        return None
    data = np.load(TEMPLATES_PATH)
    count = int(data["count"])
    return [data[f"t{i}"] for i in range(count)]


def load_focal_length():
    if not os.path.exists(FOCAL_SAMPLES_PATH):
        return None
    estimates = np.load(FOCAL_SAMPLES_PATH)["estimates"]
    return float(np.mean(estimates)) if len(estimates) else None


class StarTracker:
    """Tracks a fixed set of star templates frame to frame: matches each
    independently, resolves collisions (two templates can't claim the same
    spot -- the higher-scoring one wins), requires a new match position to
    hold steady for STABLE_FRAMES before it's trusted (so a candidate that's
    jumping around each frame never gets reported as a real point), and
    briefly holds a template's last confirmed position through short
    dropouts or not-yet-stable stretches instead of losing it outright."""

    def __init__(self, templates):
        self.templates = templates
        self.state = [{'pos': None, 'bbox': None, 'score': 0.0, 'missed': 0,
                        'pending_pos': None, 'pending_bbox': None, 'pending_score': 0.0,
                        'pending_count': 0}
                       for _ in templates]
        self.min_separation = max(10, int(min(min(t.shape) for t in templates) * 0.75))

    def match(self, frame_gray):
        raw = []
        for i, t in enumerate(self.templates):
            th, tw = t.shape
            if th >= frame_gray.shape[0] or tw >= frame_gray.shape[1]:
                continue
            result = cv2.matchTemplate(frame_gray, t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            sub_x, sub_y = _subpixel_peak(result, max_loc)
            cx, cy = max_loc[0] + sub_x + tw / 2, max_loc[1] + sub_y + th / 2
            raw.append((i, cx, cy, max_val, (max_loc[0], max_loc[1], tw, th)))

        # highest-scoring template claims a spot first; anything else landing on
        # (nearly) the same spot this frame is rejected, not just deduped after
        raw.sort(key=lambda r: r[3], reverse=True)
        claimed = []
        accepted = {}
        for (i, cx, cy, score, bbox) in raw:
            if score < MATCH_THRESHOLD:
                continue
            if any(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 < self.min_separation for ax, ay in claimed):
                continue
            accepted[i] = (cx, cy, score, bbox)
            claimed.append((cx, cy))

        detections = []
        for i in range(len(self.templates)):
            st = self.state[i]
            if i in accepted:
                cx, cy, score, bbox = accepted[i]
                pend = st['pending_pos']
                if pend is not None and ((cx - pend[0]) ** 2 + (cy - pend[1]) ** 2) ** 0.5 <= POS_TOLERANCE_PX:
                    st['pending_count'] += 1
                else:
                    st['pending_count'] = 1
                st.update(pending_pos=(cx, cy), pending_bbox=bbox, pending_score=score)

                if st['pending_count'] >= STABLE_FRAMES:
                    st.update(pos=(cx, cy), bbox=bbox, score=score, missed=0)
                    detections.append((cx, cy, score, bbox, True))
                elif st['pos'] is not None and st['missed'] < GRACE_FRAMES:
                    st['missed'] += 1
                    detections.append((st['pos'][0], st['pos'][1], st['score'], st['bbox'], False))
                elif st['pos'] is not None:
                    st['missed'] += 1
            else:
                st['pending_pos'] = None
                st['pending_count'] = 0
                if st['pos'] is not None and st['missed'] < GRACE_FRAMES:
                    st['missed'] += 1
                    detections.append((st['pos'][0], st['pos'][1], st['score'], st['bbox'], False))
                elif st['pos'] is not None:
                    st['missed'] += 1
        return detections


def star_side_order(detections):
    """Index order sorted by on-screen x position (leftmost first), so 'L'/'R'
    labeling and geometry are always based on where things actually are on
    screen, not on template/detection order."""
    return sorted(range(len(detections)), key=lambda i: detections[i][0])


def star_frame(detections, order):
    """Returns (origin, ex, ey, cm_per_px) for the coordinate frame defined by
    the two star markers: origin is their midpoint, ex/ey are unit vectors
    along and perpendicular to the L-R line. Using the marker line itself as
    the local X axis (instead of assuming screen-horizontal = real-world
    horizontal) keeps the mapping correct even if the camera isn't perfectly
    level with the robot base. Returns None if the two markers coincide."""
    lx, ly = detections[order[0]][0], detections[order[0]][1]
    rx, ry = detections[order[-1]][0], detections[order[-1]][1]
    dx, dy = rx - lx, ry - ly
    px_dist = (dx ** 2 + dy ** 2) ** 0.5
    if px_dist < 1e-6:
        return None
    ex = (dx / px_dist, dy / px_dist)
    ey = (-ex[1], ex[0])
    origin = ((lx + rx) / 2, (ly + ry) / 2)
    cm_per_px = REAL_DISTANCE_CM / px_dist
    return origin, ex, ey, cm_per_px


def to_local_cm(point, frame_geom):
    origin, ex, ey, cm_per_px = frame_geom
    vx, vy = point[0] - origin[0], point[1] - origin[1]
    local_x = (vx * ex[0] + vy * ex[1]) * cm_per_px
    local_y = (vx * ey[0] + vy * ey[1]) * cm_per_px
    return local_x, local_y
