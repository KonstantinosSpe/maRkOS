"""Where a bottle stands on the desk, from where it appears in the picture.

``DeskCalibration`` maps a pixel on the desk to centimetres around the robot's base axis with one projective mapping (a
homography) fitted to a handful of bottle spots measured with a tape. ``base_contact`` finds the pixel where a
bottle's foot meets the desk, and ``ContactFilter`` steadies it. When the stars are learned the calibration also keeps the
camera fitted at calibration time, so a laptop that has been moved can still be placed (see ``camera_model.StarLock``).
"""

import json
import math
import os

import cv2
import numpy as np

from markos_vision.config import DESK_CALIBRATION as CALIB_PATH
from markos_vision.config import ensure_parent
from markos_vision.geometry.camera_model import CameraModel, edge_to_centre

MIN_POINTS = 4
RECOMMENDED_POINTS = 6
MIN_SPREAD_CM = 5.0  # the points must be at least this far across in their narrower direction, or they're nearly a line.
                     # The arm's own work area is a ring only ~7 cm deep, so anything stricter nags on good sets
STAR_MOVED_PX = 6.0  # the stars' midpoint may drift this far before the camera counts as moved...
STAR_MOVED_SEP = 0.04  # ...or their spacing change by this fraction (camera nearer or farther)
HULL_MARGIN = 0.15  # how far outside the calibrated area a bottle can be before it's flagged as extrapolated
BOTTOM_BAND = 0.03  # share of the bottle's height, at the bottom, that's used to find where it meets the desk


def base_contact(polygon, box):
    """Pixel where the bottle meets the desk: the lowest point of its outline,
    centred sideways over the bottom few rows. The rim of a round base is an
    ellipse in the picture, and its lowest point is its middle. Falls back to
    the bottom-centre of the detector's box when there's no outline."""
    if polygon is not None and len(polygon) >= 3:
        pts = np.asarray(polygon, float)
        y_max, y_min = float(pts[:, 1].max()), float(pts[:, 1].min())
        band = pts[pts[:, 1] >= y_max - max(3.0, BOTTOM_BAND * (y_max - y_min))]
        return (float(band[:, 0].min() + band[:, 0].max()) / 2.0, y_max)
    x, y, w, h = box
    return (x + w / 2.0, y + h)


class ContactFilter:
    """Steadies the point where the bottle meets the desk. The model's outline
    wobbles a pixel or two from frame to frame and now and then jumps several,
    which shows up as a dancing dot and a shaky position. A median over the
    last few frames throws the jumps away, then a light exponential average
    smooths what's left. A real move (the raw point staying far from the
    filtered one for a few frames in a row) snaps the filter to the new spot
    at once, so it doesn't crawl after a bottle you've just put down."""

    def __init__(self, window=7, alpha=0.35, jump_px=25.0, jump_frames=3):
        self.window, self.alpha, self.jump_px, self.jump_frames = window, alpha, jump_px, jump_frames
        self.reset()

    def reset(self):
        self.raw = []
        self.value = None
        self._far = 0

    @staticmethod
    def _median(points):
        a = np.asarray(points, float)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def update(self, point):
        self.raw = (self.raw + [(float(point[0]), float(point[1]))])[-self.window:]
        if self.value is None:
            self.value = self._median(self.raw)
            return self.value
        if math.hypot(point[0] - self.value[0], point[1] - self.value[1]) > self.jump_px:
            self._far += 1
        else:
            self._far = 0
        if self._far >= self.jump_frames:
            self.raw = self.raw[-self.jump_frames:]
            self.value = self._median(self.raw)
            self._far = 0
        else:
            m = self._median(self.raw)
            self.value = (self.value[0] + self.alpha * (m[0] - self.value[0]),
                          self.value[1] + self.alpha * (m[1] - self.value[1]))
        return self.value


def _fit(points):
    src = np.float32([[p["px"], p["py"]] for p in points])
    dst = np.float32([[p["x_cm"], p["y_cm"]] for p in points])
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def _apply(H, px, py):
    v = H @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


class DeskCalibration:
    """Maps a pixel on the desk to a position in centimetres around the
    robot's base axis. The desk is a flat plane, so the picture of it is
    related to the real thing by one fixed projective mapping (a
    homography); a few bottle positions measured with a tape pin it down.
    That replaces working the position out from apparent sizes, which turns a
    pixel of error in the stars' 7.9 cm spacing into centimetres.

    The points can be measured from anywhere: the origin of the measuring
    frame is arbitrary (a corner of the robot's box, say), as long as it's
    the same for every point and the base axis's position in that frame is
    given (set_axis). The two measuring directions must be perpendicular.
    If they come out mirrored relative to what the camera sees (a left/right
    mix-up when working out which way is which), that's detected from the
    fit and flipped, so results always come out in the desk frame: origin on
    the base axis, +y toward the camera and +x to the right as the camera
    sees it, when the measuring frame is lined up with that. The bearing is
    atan2(x, y)."""

    def __init__(self, path=CALIB_PATH):
        self.path = path
        self.points = []  # {"px", "py", "x_cm", "y_cm"}: the measured position, in the measuring frame
        self.axis = None  # [u, v]: where the base axis is in the measuring frame; None = the origin is the axis
        self.stars = None  # {"mid": [x, y], "sep": pixels} when the points were taken
        self.H = None
        self.mirrored = False
        # Printed markers fixed on the desk: {id: [[u, v] x4]}, each marker's corners in the measuring frame, learned
        # once through the tape calibration (desk_markers.learn). They let the mapping be rebuilt whenever the
        # camera is somewhere new, without any tape.
        self.markers = {}
        self.markers_from_points = 0  # how many points the mapping had when the markers were learned
        self.live_H = None  # a mapping rebuilt from the markers in the current camera position; wins over H when set
        # The two red stars on the robot, for placing a laptop that has been moved (camera_model.StarLock):
        # star3d = [[u, v, height cm], [u, v, height cm]] as measured, image-left star first; star_cam = the camera
        # as fitted at calibration ({"params", "offsets", "rms", "from_points"}).
        self.star3d = None
        self.star_cam = None
        self.lens_height_cm = None  # the camera lens's height above the desk, measured with a ruler (pins the camera fit)
        # The diameter of the bottle's foot in the bottle the points were measured with. The camera sees the lowest point of
        # that foot, its near edge, while the tape measured to its centre, so each point maps a near edge to a centre.
        self.bottle_diameter_cm = None
        self.live_map = None  # pixel -> desk frame (x, y) from the stars in the current camera position; wins over all
        self._cal_camera = None
        self.load()

    # ── points ─────────────────────────────────────────────────────────────

    def add(self, px, py, x_cm, y_cm):
        self.points.append({"px": float(px), "py": float(py), "x_cm": float(x_cm), "y_cm": float(y_cm)})
        self.fit()
        self.save()

    def undo(self):
        if self.points:
            self.points.pop()
            self.fit()
            self.save()

    def reset(self):
        self.points, self.stars, self.H, self.axis, self.mirrored = [], None, None, None, False
        self.markers, self.markers_from_points, self.live_H = {}, 0, None   # they were learned through the points
        self.star_cam, self.live_map, self._cal_camera = None, None, None   # so was the camera fitted from them (the star positions stay)
        self.save()

    def set_star_geometry(self, star3d):
        self.star3d = [[float(u), float(v), float(z)] for u, v, z in star3d]
        self.star_cam = None   # any camera fitted with the old star positions is void
        self.save()

    def set_star_camera(self, params, offsets, rms, foot_radius_cm=None):
        self.star_cam = {"params": [float(p) for p in params], "offsets": [[float(a) for a in o] for o in offsets],
                         "rms": float(rms), "from_points": len(self.points),
                         "foot_radius_cm": None if foot_radius_cm is None else float(foot_radius_cm)}
        self._cal_camera = None
        self.save()

    def set_bottle_diameter(self, diameter_cm):
        self.bottle_diameter_cm = None if diameter_cm is None else float(diameter_cm)
        self.save()

    @property
    def bottle_radius_cm(self):
        return None if self.bottle_diameter_cm is None else self.bottle_diameter_cm / 2.0

    @property
    def star_camera_radius_cm(self):
        """The foot radius the calibration camera was fitted with, if it is the calibration bottle's, else None. The
        moved-laptop placing only applies the bottle's size with a camera fitted that way."""
        fitted = (self.star_cam or {}).get("foot_radius_cm")
        r = self.bottle_radius_cm
        return r if (fitted is not None and r is not None and abs(fitted - r) < 1e-6) else None

    def calibration_camera(self):
        """The camera as fitted at calibration (camera_model.CameraModel, in the desk frame), or None before the stars were learned."""
        if self._cal_camera is None and self.star_cam:
            self._cal_camera = CameraModel(*self.star_cam["params"])
        return self._cal_camera

    def tape_ground(self, px, py):
        """The plain tape-calibrated mapping in the measuring frame, ignoring any live (moved-laptop) mapping."""
        return _apply(self.H, px, py)

    def set_markers(self, markers):
        self.markers = {int(i): [[float(u), float(v)] for u, v in corners] for i, corners in markers.items()}
        self.markers_from_points = len(self.points)
        self.save()

    def set_axis(self, u_cm, v_cm):
        """Where the robot's base axis is, in the same frame the points were
        measured in. Doesn't need the mapping refitted: only the origin moves."""
        self.axis = [float(u_cm), float(v_cm)]
        self.save()

    def fit(self):
        self.H = _fit(self.points) if len(self.points) >= MIN_POINTS else None
        self.mirrored = self._is_mirrored()
        return self.H is not None

    def _is_mirrored(self):
        """True if the measuring frame is left-handed as the camera sees it.
        The picture's own axes (right, down) are a right-handed pair when seen
        from above the desk, so a mapping that reverses that reverses handedness."""
        if self.H is None:
            return False
        c = np.float32([[p["px"], p["py"]] for p in self.points]).mean(axis=0)
        o, a, b = _apply(self.H, *c), _apply(self.H, c[0] + 1, c[1]), _apply(self.H, c[0], c[1] + 1)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) < 0

    @property
    def ready(self):
        return self.H is not None

    def _to_frame(self, u, v):
        """Measuring-frame coordinates -> desk frame (origin on the base axis, handedness fixed)."""
        au, av = self.axis or (0.0, 0.0)
        return (-1.0 if self.mirrored else 1.0) * (u - au), v - av

    def from_desk(self, x, y):
        """The inverse of _to_frame: a desk-frame position (x, y, cm, centred on the base axis) back into the u, v
        you measure with the tape, so a reading can be compared with the ruler."""
        au, av = self.axis or (0.0, 0.0)
        return au + (-1.0 if self.mirrored else 1.0) * x, av + y

    def to_desk(self, px, py, radius_cm=None):
        """Position in cm on the desk, around the base axis: x to the right
        of the camera's view (when y points toward the camera), y toward it.

        The pixel is where the bottle's foot meets the desk at its near edge (base_contact), and the position returned is
        where the bottle's axis is: the calibration was measured with a bottle of bottle_diameter_cm centred on each mark.
        radius_cm is the radius of the bottle in the picture now (the calibration bottle's when left out). Needs the
        camera fitted at calibration (press s in calibrate_desk.py) to know which way is 'back from the near edge'; without
        it, or without the calibration bottle's diameter, the pixel is mapped as it is."""
        if self.live_map is not None:
            return self.live_map(px, py, radius_cm)   # already in the desk frame
        if self.live_H is not None:
            return self._to_frame(*_apply(self.live_H, px, py))
        x, y = self._to_frame(*_apply(self.H, px, py))
        cam, r_cal = self.calibration_camera(), self.bottle_radius_cm
        if cam is not None and r_cal is not None and radius_cm is not None:
            dx, dy = edge_to_centre(cam, cam, x, y, radius_cm, r_cal)
            x, y = x + dx, y + dy
        return x, y

    # ── how good is it ─────────────────────────────────────────────────────

    def residuals_mm(self):
        """How far each point lands from where it was measured, using all the
        points. Only says the fit is tight; with exactly four points it's
        always zero, which is why the leave-one-out figure exists."""
        return [math.hypot(*(a - b for a, b in zip(_apply(self.H, p["px"], p["py"]), (p["x_cm"], p["y_cm"])))) * 10
                for p in self.points]

    def leave_one_out_mm(self):
        """The honest number: fit without each point in turn and see how far
        off the mapping is at that point. Needs more than MIN_POINTS."""
        if len(self.points) <= MIN_POINTS:
            return []
        errs = []
        for i, p in enumerate(self.points):
            H = _fit(self.points[:i] + self.points[i + 1:])
            if H is None:
                errs.append(float("inf"))
                continue
            x, y = _apply(H, p["px"], p["py"])
            errs.append(math.hypot(x - p["x_cm"], y - p["y_cm"]) * 10)
        return errs

    def notes(self):
        """Things to fix about the point set, in plain words."""
        out = []
        n = len(self.points)
        if n < MIN_POINTS:
            return [f"need at least {MIN_POINTS} points, have {n}"]
        if n < RECOMMENDED_POINTS:
            out.append(f"only {n} points: {RECOMMENDED_POINTS} or more lets it check itself")
        around_axis = [self._to_frame(p["x_cm"], p["y_cm"]) for p in self.points]
        radii = [math.hypot(x, y) for x, y in around_axis]
        bearings = [math.degrees(math.atan2(x, y)) for x, y in around_axis]
        if max(radii) - min(radii) < 6:
            out.append("points are all at nearly the same distance from the base: add some nearer and farther")
        if max(bearings) - min(bearings) < 30:
            out.append("points are all in nearly the same direction: add some to the left and right")
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(np.float32([[p["x_cm"], p["y_cm"]] for p in self.points]))
        if min(side_a, side_b) < MIN_SPREAD_CM:
            out.append(f"the points lie nearly along a line (only {min(side_a, side_b):.0f} cm across): spread them out")
        if self.markers and self.markers_from_points != n:
            out.append(f"the points changed since the desk markers were learned ({self.markers_from_points} then, {n} now): "
                       "press m again to relearn them")
        if self.star_cam and self.star_cam["from_points"] != n:
            out.append(f"the points changed since the stars were learned ({self.star_cam['from_points']} then, {n} now): "
                       "press s again so the moved-laptop placing uses the current calibration")
        elif self.star_cam and self.bottle_radius_cm is not None and self.star_camera_radius_cm is None:
            out.append("the camera was fitted without the bottle's diameter: press s again so moving the laptop takes the "
                       "bottle's near edge into account")
        return out

    def inside(self, px, py):
        """False if this pixel is well outside the area the points cover,
        where the mapping is extrapolating and gets less trustworthy. Says
        yes when the mapping comes from markers in a different camera
        position, where the calibration points' pixels mean nothing."""
        if len(self.points) < MIN_POINTS or self.live_H is not None or self.live_map is not None:
            return True
        hull = cv2.convexHull(np.float32([[p["px"], p["py"]] for p in self.points]))
        margin = HULL_MARGIN * math.sqrt(max(cv2.contourArea(hull), 1.0))
        return cv2.pointPolygonTest(hull, (float(px), float(py)), True) >= -margin

    # ── has the camera moved? ──────────────────────────────────────────────

    def set_star_reference(self, mid, sep):
        self.stars = {"mid": [float(mid[0]), float(mid[1])], "sep": float(sep)}
        self.save()

    def camera_moved(self, mid, sep):
        """A message if the stars are somewhere other than when the points
        were taken (so the mapping no longer applies), else None."""
        if self.stars is None:
            return None
        shift = math.hypot(mid[0] - self.stars["mid"][0], mid[1] - self.stars["mid"][1])
        if shift > STAR_MOVED_PX:
            return f"camera moved: the stars are {shift:.0f} px from where they were during calibration"
        change = abs(sep - self.stars["sep"]) / self.stars["sep"]
        if change > STAR_MOVED_SEP:
            return f"camera moved: the stars' spacing changed {change:.0%} since calibration"
        return None

    # ── saving ─────────────────────────────────────────────────────────────

    def save(self):
        if self.path is None:
            return
        with open(ensure_parent(self.path), "w") as f:
            json.dump({"points": self.points, "axis": self.axis, "stars": self.stars,
                       "markers": {str(i): c for i, c in self.markers.items()},
                       "markers_from_points": self.markers_from_points,
                       "star3d": self.star3d, "star_cam": self.star_cam, "lens_height_cm": self.lens_height_cm,
                       "bottle_diameter_cm": self.bottle_diameter_cm}, f, indent=1)

    def load(self):
        if self.path is None or not os.path.exists(self.path):
            return
        with open(self.path) as f:
            data = json.load(f)
        self.points = data.get("points", [])
        self.axis = data.get("axis")
        self.stars = data.get("stars")
        self.markers = {int(i): c for i, c in data.get("markers", {}).items()}
        self.markers_from_points = data.get("markers_from_points", 0)
        self.star3d = data.get("star3d")
        self.star_cam = data.get("star_cam")
        self.lens_height_cm = data.get("lens_height_cm")
        self.bottle_diameter_cm = data.get("bottle_diameter_cm")
        self._cal_camera = None
        self.fit()
