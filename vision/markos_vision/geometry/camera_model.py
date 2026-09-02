"""A pinhole-camera model of the laptop webcam, and the two things built on it.

* ``fit_calibration_pose`` fits the camera (focal length, lens bend, position, height, turn, tilt) to the tape-measured
  bottle spots and to the two red stars on the robot.
* ``StarLock`` places the camera again from the stars alone after the laptop has been moved, and hands the desk
  calibration a pixel -> desk mapping for the new position (``carry_over_mapping``).

World frame: the desk plane is x, y (cm) with the robot's base axis at the origin, z up.
"""

import math

import numpy as np

CX, CY = 320.0, 240.0
# A laptop webcam often looks level or slightly UP (the screen is tilted back to face you), so a camera tilted a little
# above horizontal is normal and must be accepted; only the mirror-image nonsense solutions are excluded.
MIN_PITCH = math.radians(-35.0)
MAX_PITCH = math.radians(80.0)


def _levenberg_marquardt(fun, x0, iterations=80):
    """A small least-squares solver (numeric Jacobian), so this doesn't need SciPy. Returns (x, residuals)."""
    x = np.array(x0, float)
    r = np.asarray(fun(x), float)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iterations):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            d = np.zeros_like(x)
            d[j] = 1e-5 * max(1.0, abs(x[j]))
            J[:, j] = (np.asarray(fun(x + d)) - np.asarray(fun(x - d))) / (2 * d[j])
        A, g = J.T @ J, J.T @ r
        improved = False
        step = np.zeros_like(x)
        for _ in range(30):
            try:
                step = np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(len(x))), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2 = np.asarray(fun(x + step), float)
            if float(r2 @ r2) < cost:
                x, r, cost, lam, improved = x + step, r2, float(r2 @ r2), max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-9 * (1 + np.linalg.norm(x)):
            break
    return x, r


class CameraModel:
    """A pinhole camera with a little radial lens bend, standing at (X, Y) at height h above the desk, turned by yaw and
    tilted down by pitch. World coordinates are the desk frame: x, y on the desk, z up."""

    def __init__(self, f, k1, X, Y, h, yaw, pitch):
        self.f, self.k1, self.h = float(f), float(k1), float(h)
        self.pos = np.array([X, Y, h], float)
        self.yaw, self.pitch = float(yaw), float(pitch)
        cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        self.fwd = np.array([-sy * cp, -cy * cp, -sp])
        self.right = np.array([cy, -sy, 0.0])
        self.down = np.cross(self.right, self.fwd)

    @property
    def params(self):
        return [self.f, self.k1, float(self.pos[0]), float(self.pos[1]), self.h, self.yaw, self.pitch]

    def project(self, world):
        q = np.asarray(world, float) - self.pos
        zc = q @ self.fwd
        xn, yn = (q @ self.right) / zc, (q @ self.down) / zc
        rr = xn * xn + yn * yn
        return np.array([CX + self.f * xn * (1 + self.k1 * rr), CY + self.f * yn * (1 + self.k1 * rr)])

    def foot_pixel(self, x, y, radius_cm):
        """Where a bottle whose round foot (radius radius_cm) stands with its axis at (x, y) meets the desk in the picture:
        the lowest point of the foot's rim, close to the rim point nearest the camera."""
        if radius_cm <= 0:
            return self.project((x, y, 0.0))

        def low(t):
            return self.project((x + radius_cm * math.cos(t), y + radius_cm * math.sin(t), 0.0))

        theta = math.atan2(self.pos[1] - y, self.pos[0] - x)
        h = 1e-3
        for _ in range(3):                                   # Newton steps to the lowest point; a fixed count keeps it smooth
            a, b, c = low(theta - h)[1], low(theta)[1], low(theta + h)[1]
            curvature = (a - 2.0 * b + c) / (h * h)
            if curvature >= -1e-9:
                break
            theta -= (c - a) / (2.0 * h) / curvature
        return low(theta)

    def ground(self, px, py):
        """Where the ray through this pixel meets the desk, (x, y)."""
        xd, yd = (px - CX) / self.f, (py - CY) / self.f
        xn, yn = xd, yd
        for _ in range(8):                                  # undo the lens bend
            rr = xn * xn + yn * yn
            xn, yn = xd / (1 + self.k1 * rr), yd / (1 + self.k1 * rr)
        ray = self.fwd + xn * self.right + yn * self.down
        t = -self.pos[2] / ray[2]
        p = self.pos + t * ray
        return float(p[0]), float(p[1])


def _from_theta(theta):
    f, k1, X, Y, h, yaw, pitch = theta
    return CameraModel(f, k1, X, Y, h, yaw, pitch)


def fit_calibration_pose(tape, stars, star_pixels, f0=600.0, height=None, height_sigma_cm=0.3, foot_radius_cm=None):
    """The camera as it stood at calibration: fitted to the tape-measured bottle spots (on the desk) and to the stars
    seen from there (at their measured heights). tape: list of (px, py, x, y) with x, y in the desk frame;
    stars: [(x, y, z)] in the same frame; star_pixels: [(px, py)]. height: the lens's height above the desk if it has
    been measured with a ruler: the picture alone can't tell a higher camera from a farther one, so a measured height
    pins that down. foot_radius_cm: the radius of the bottle's foot, if known; the tape spots' pixels are then modelled
    as its near edge (CameraModel.foot_pixel) instead of its axis. Returns (camera, rms px, star offsets)."""
    cx, cy = np.mean([[t[2], t[3]] for t in tape], axis=0)

    def residuals(theta):
        cam = _from_theta(theta)
        out = []
        for px, py, x, y in tape:
            seen = cam.foot_pixel(x, y, foot_radius_cm) if foot_radius_cm else cam.project((x, y, 0.0))
            out.extend(seen - (px, py))
        for (sx, sy, z), sp in zip(stars, star_pixels):
            out.extend(cam.project((sx, sy, z)) - sp)
        out.append((theta[1] / 0.05) * 0.3)                 # keep the lens bend modest unless the data insists
        if height is not None:
            out.append((theta[4] - height) / height_sigma_cm)   # one 'pixel' of misfit per height_sigma_cm off the ruler
        return out

    # Screen many starting guesses cheaply (a few iterations each), then polish only the best few.
    screened = []
    for dist in (40.0, 65.0, 95.0):
        for ang in np.radians(np.arange(0, 360, 30)):
            X, Y = cx + dist * math.sin(ang), cy + dist * math.cos(ang)
            yaw = math.atan2(-(cx - X), -(cy - Y))
            h0 = height if height is not None else 30.0
            x, r = _levenberg_marquardt(residuals, [f0, 0.0, X, Y, h0, yaw, math.atan2(h0, dist)], iterations=6)
            if 5.0 < x[4] < 90.0 and MIN_PITCH < x[6] < MAX_PITCH:
                screened.append((float(r @ r), x))
    if not screened:
        raise ValueError("couldn't fit a camera to those points")
    screened.sort(key=lambda s: s[0])
    best = None
    for _, x0 in screened[:3]:
        x, r = _levenberg_marquardt(residuals, x0, iterations=80)
        if best is None or r @ r < best[1] @ best[1]:
            best = (x, r)
    x, r = best
    n = len(tape) * 2 + len(stars) * 2
    cam = _from_theta(x)
    # What the stars' pixels disagree with the model by, from here. The tape points pin the camera down much better
    # than two close-together stars can, so this disagreement is mostly the error in the hand-measured star positions
    # (and any pixel bias of the star detector). It's kept so it can be taken out again later.
    offsets = [np.asarray(sp, float) - cam.project(s) for s, sp in zip(stars, star_pixels)]
    return cam, float(np.sqrt(np.mean(np.square(r[:n])))), offsets


def locate_new_pose(cam_a, stars, star_pixels, offsets=None, pitch_sigma_deg=0.6):
    """Where the camera is after it's been moved and turned, from the stars alone. Focal length, lens bend and height
    are taken from calibration; position, turn and tilt are found. Four measurements for four unknowns, with a weak
    pull toward the calibration tilt so noise can't invent a big one. `offsets` are the stars' pixel disagreements
    remembered from calibration (fit_calibration_pose): subtracting them means only the CHANGE in what the stars show
    moves the camera, so a slightly wrong star position doesn't. Returns (camera, rms reprojection error in px,
    change of tilt in degrees)."""
    offsets = offsets if offsets is not None else [np.zeros(2)] * len(stars)

    def residuals(theta):
        X, Y, yaw, pitch = theta
        cam = CameraModel(cam_a.f, cam_a.k1, X, Y, cam_a.h, yaw, pitch)
        out = []
        for (sx, sy, z), sp, off in zip(stars, star_pixels, offsets):
            out.extend(cam.project((sx, sy, z)) + off - sp)
        out.append(math.degrees(pitch - cam_a.pitch) / pitch_sigma_deg * 0.3)
        return out

    screened = []
    for dyaw in np.radians(np.arange(-90, 91, 30)):
        for dx, dy in ((0, 0), (25, 0), (-25, 0)):
            x, r = _levenberg_marquardt(residuals, [cam_a.pos[0] + dx, cam_a.pos[1] + dy, cam_a.yaw + dyaw, cam_a.pitch],
                                        iterations=5)
            if MIN_PITCH < x[3] < MAX_PITCH:
                screened.append((float(r @ r), x))
    if not screened:
        raise ValueError("the stars don't fit any camera position")
    screened.sort(key=lambda s: s[0])
    best = None
    for _, x0 in screened[:2]:
        x, r = _levenberg_marquardt(residuals, x0, iterations=60)
        if best is None or r @ r < best[1] @ best[1]:
            best = (x, r)
    x, r = best
    cam = CameraModel(cam_a.f, cam_a.k1, x[0], x[1], cam_a.h, x[2], x[3])
    return cam, float(np.sqrt(np.mean(np.square(r[:-1])))), math.degrees(x[3] - cam_a.pitch)


def _away(cam, x, y):
    """The unit vector on the desk pointing from the camera to the desk point (x, y): the direction from a bottle's near
    edge, which is what the camera sees on the desk, to the bottle's axis."""
    dx, dy = x - cam.pos[0], y - cam.pos[1]
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)


def edge_to_centre(cam_a, cam_b, x, y, radius_cm, radius_cal_cm):
    """What to add to the position the tape calibration gives for a bottle's near edge, at desk point (x, y), to get its
    axis: radius_cm * w_b - radius_cal_cm * w_a, where w_a and w_b are the directions pointing away from the calibration
    camera (cam_a) and from the camera now (cam_b). Zero for the calibration bottle seen from the calibration camera."""
    wbx, wby = _away(cam_b, x, y)
    wax, way = _away(cam_a, x, y)
    return radius_cm * wbx - radius_cal_cm * wax, radius_cm * wby - radius_cal_cm * way


def carry_over_mapping(cam_a, cam_b, ground_a, radius_cal_cm=None):
    """A pixel -> desk mapping for the new camera position that keeps the tape calibration's accuracy. The camera model
    gives the desk point for any pixel in the new view; the tape-calibrated mapping (ground_a: pixel -> (x, y)) is
    more accurate than the model at the calibration position, so the model's error there, at the pixel where the
    calibration camera would see that same desk point, is added back.

    radius_cal_cm is the radius of the bottle the tape calibration was measured with. With it the mapping takes the
    bottle's own radius as a third argument (radius_cm, the calibration bottle's when left out) and returns where its
    axis is instead of where its near edge would be (edge_to_centre)."""

    def mapping(px, py, radius_cm=None):
        x, y = cam_b.ground(px, py)
        pa = cam_a.project((x, y, 0.0))
        mx, my = cam_a.ground(*pa)
        gx, gy = ground_a(*pa)
        x, y = x + (gx - mx), y + (gy - my)
        if radius_cal_cm is not None:
            dx, dy = edge_to_centre(cam_a, cam_b, x, y, radius_cal_cm if radius_cm is None else radius_cm, radius_cal_cm)
            x, y = x + dx, y + dy
        return x, y

    return mapping


class StarLock:
    """Places the moved laptop from the two red stars, so the tape calibration keeps working wherever the laptop is
    put. It averages the stars' pixels over LOCK_FRAMES, finds where the camera must be (locate_new_pose), and hands
    the resulting pixel -> desk mapping to the calibration (calib.live_map). It only trusts the answer if the stars
    fit the calibration well; and it watches the stars afterwards -- if they shift in the picture the camera has
    moved, and it locks again, with the arm unused in between."""

    LOCK_FRAMES = 30
    STABLE_PX = 1.0  # the stars may wander this much (std) over the lock frames; more means the picture is moving
    MAX_RMS_PX = 0.8  # how well the stars must fit; worse means the height or the stars' identity is off
    MAX_TILT_CHANGE_DEG = 4.0  # a height change gets absorbed as a tilt change, so a big apparent tilt change is suspect
    MAX_TURN_DEG = 80.0  # a laptop still looking at the workspace can't have turned more than this...
    MAX_MOVE_CM = 100.0  # ...or been carried farther than this from where it was calibrated
    MOVED_PX = 3.0
    MOVED_FRAMES = 5

    def __init__(self, calib):
        self.calib = calib
        sc = calib.star_cam
        self.cam_a = CameraModel(*sc["params"])
        self.offsets = [np.array(o, float) for o in sc["offsets"]]
        self.stars = [(*calib._to_frame(u, v), z) for u, v, z in calib.star3d]
        self.events = 0
        self.camera = None
        self.reset()

    def reset(self):
        self.samples = []
        self.locked = False
        self.locked_px = None
        self.problem = ""
        self.detail = "looking for both stars"
        self._off = 0
        self.calib.live_map = None

    def update(self, star_pixels):
        """star_pixels: [(x, y) of the star on the left in the picture, (x, y) of the right one], or None when they
        aren't both seen right now."""
        if star_pixels is None:
            if not self.locked:
                self.samples = []
                self.detail = f"{self.problem} -- need both stars in view" if self.problem else "need both stars in view"
            return self.locked
        px = np.array(star_pixels, float)
        if self.locked:
            self._watch(px)
        else:
            self.samples.append(px)
            if len(self.samples) >= self.LOCK_FRAMES:
                self._try_lock()
            else:
                progress = f"locking on the stars {len(self.samples)}/{self.LOCK_FRAMES}"
                self.detail = f"{self.problem} -- {progress}" if self.problem else progress
        return self.locked

    def _try_lock(self):
        stack = np.array(self.samples)
        self.samples = []
        wander = float(stack.std(axis=0).max())
        if wander > self.STABLE_PX:
            self.problem = f"the picture is moving (the stars wobbled {wander:.1f} px): hold the laptop still"
        else:
            mean = stack.mean(axis=0)
            try:
                cam_b, rms, dpitch = locate_new_pose(self.cam_a, self.stars, mean, self.offsets)
            except ValueError as exc:
                self.problem = str(exc)
                self.detail = self.problem
                return
            moved = math.hypot(*(cam_b.pos[:2] - self.cam_a.pos[:2]))
            turned = (math.degrees(cam_b.yaw - self.cam_a.yaw) + 180.0) % 360.0 - 180.0
            if rms > self.MAX_RMS_PX:
                self.problem = (f"the stars don't fit the calibration ({rms:.1f} px off): is the laptop at a different "
                                "height, or are the stars mixed up?")
            elif abs(dpitch) > self.MAX_TILT_CHANGE_DEG:
                self.problem = (f"the tilt seems {dpitch:+.0f} deg different from calibration: did the lid angle or the laptop's "
                                "HEIGHT change? Put it back as it was (or recalibrate and press s again)")
            elif abs(turned) > self.MAX_TURN_DEG or moved > self.MAX_MOVE_CM:
                # Two stars can't tell "the laptop is here" from "it is on the far side of the robot looking back", so
                # anything that would need a move that big is refused rather than believed.
                self.problem = (f"that would mean the laptop moved {moved:.0f} cm and turned {turned:+.0f} deg from where "
                                "it was calibrated: not believable -- are the left and right stars the right way round?")
            else:
                ground_a = lambda px_, py_: self.calib._to_frame(*self.calib.tape_ground(px_, py_))  # noqa: E731
                self.calib.live_map = carry_over_mapping(self.cam_a, cam_b, ground_a, self.calib.star_camera_radius_cm)
                self.camera, self.locked, self.locked_px, self._off = cam_b, True, mean, 0
                self.problem = ""
                self.detail = (f"locked from the stars: laptop {moved:.0f} cm from where it was calibrated, turned "
                               f"{turned:+.0f} deg, tilt {dpitch:+.1f} deg")
                return
        self.detail = self.problem

    def _watch(self, px):
        shift = float(np.abs(px - self.locked_px).mean())
        self._off = self._off + 1 if shift > self.MOVED_PX else 0
        if self._off >= self.MOVED_FRAMES:
            self.events += 1
            self.reset()
            self.problem = "the camera moved"
            self.detail = "the camera moved: locking on the stars again"
