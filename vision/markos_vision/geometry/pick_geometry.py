"""From a position on the desk to joint angles: the glue between the camera's view and the arm's reach planner.

``locate_on_desk`` turns a bottle's foot pixel into a bearing and radius around the robot's base (``locate_cap`` does the
same from the stars alone, the rougher fallback). ``hover_plan`` then asks ``markos.reach`` for the joint angles that put
the gripper above the cap, falling back to the nearest spot the arm can reach and keeping to one arm pose (the "lean")
so a bottle sliding across the workspace does not swing the base around and back. ``PickConfig`` holds the few numbers
that tie the camera's view to the arm's and are tuned live.
"""

import json
import math
import os
from collections import namedtuple
from functools import lru_cache

from markos import reach
from markos_vision.config import PICK_CONFIG as CONFIG_PATH
from markos_vision.config import ensure_parent

STAR_SEPARATION_CM = 7.9  # the two red stars on the robot's base, measured by hand
CAMERA_CX = 320.0  # principal point, taken as the middle of the 640x480 frame
# In the URDF the arm at E=0 leans along -Y, not +X (checked by comparing the planner's joint
# angles against the model's own forward kinematics), so the planner's azimuth and the model's differ by this.
URDF_AZIMUTH_OFFSET_DEG = -90.0


class PickConfig:
    """The few numbers that tie the camera's view to the arm's: they can't be
    known from the geometry alone, so they're nudged live while the arm
    hovers over a bottle and saved."""

    DEFAULTS = {
        "e_sign": 1,               # +1 if a bottle further right in the picture needs E to increase
        "e_zero_deg": 0.0,         # planner azimuth of the direction from the robot toward the camera
        "e_gain": 1.0,             # E commanded per degree the base should turn: above 1 if it falls short, below if it overshoots
        "radius_offset_mm": 0.0,   # added to the measured distance from the base axis
        "star_to_axis_cm": 0.0,    # how much farther the base's axis is from the camera than the stars are
        "hover_clearance_mm": 50.0,  # how far above the cap to hover
        "base_height_cm": 0.0,     # how far the robot's base sits above the surface the bottle stands on
    }

    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.values = dict(self.DEFAULTS)
        if path and os.path.exists(path):
            with open(path) as f:
                self.values.update({k: v for k, v in json.load(f).items() if k in self.DEFAULTS})

    def __getitem__(self, key):
        return self.values[key]

    def nudge(self, key, delta):
        self.values[key] = round(self.values[key] + delta, 3)
        self.save()

    def flip_e(self):
        self.values["e_sign"] = -self.values["e_sign"]
        self.save()

    def save(self):
        if self.path:
            with open(ensure_parent(self.path), "w") as f:
                json.dump(self.values, f, indent=2)


def locate_cap(cap_px, silhouette_height_px, star_mid_px, star_sep_px, focal_px, bottle_height_cm, cfg):
    """Where the bottle's cap is, relative to the robot's base axis, from one
    camera view. The stars (a known 7.9 cm apart, on the base) give the
    camera's distance to the robot, and the bottle's known real height gives
    its distance to the camera -- a pinhole camera only ever needs one known
    size per object. The difference between the two, and the sideways offset,
    place the bottle on the desk around the robot.

    Returns bearing_deg (0 = straight toward the camera, positive to the
    right in the picture), radius_mm from the base axis, and the two camera
    distances."""
    star_dist_cm = focal_px * STAR_SEPARATION_CM / star_sep_px
    axis_dist_cm = star_dist_cm + cfg["star_to_axis_cm"]
    bottle_dist_cm = focal_px * bottle_height_cm / silhouette_height_px
    dx_cm = (cap_px[0] - CAMERA_CX) * bottle_dist_cm / focal_px - (star_mid_px[0] - CAMERA_CX) * star_dist_cm / focal_px
    dz_cm = axis_dist_cm - bottle_dist_cm
    return {
        "bearing_deg": math.degrees(math.atan2(dx_cm, dz_cm)),
        "radius_mm": math.hypot(dx_cm, dz_cm) * 10.0 + cfg["radius_offset_mm"],
        "star_dist_cm": star_dist_cm,
        "bottle_dist_cm": bottle_dist_cm,
    }


def bottle_radius_cm(bottle_type):
    """Half the enrolled bottle's foot diameter, or None if that type doesn't have one (then it's taken to be the size of
    the bottle the desk calibration was measured with)."""
    diameter = (bottle_type or {}).get("diameter_cm")
    return diameter / 2.0 if diameter else None


def locate_on_desk(base_px, calib, cfg, radius_cm=None):
    """Where the bottle stands, from where its base meets the desk in the
    picture and a desk calibration (desk_calibration.DeskCalibration).
    Same result shape as locate_cap: bearing_deg (0 = straight toward the
    camera, positive to the right in the picture) and radius_mm from the base
    axis, plus the desk position itself in cm. The picture shows the near edge
    of the bottle's foot; radius_cm (the foot's radius) gets from there to the
    bottle's axis, which is where the arm has to go."""
    x_cm, y_cm = calib.to_desk(*base_px, radius_cm)
    return {
        "bearing_deg": math.degrees(math.atan2(x_cm, y_cm)),
        "radius_mm": math.hypot(x_cm, y_cm) * 10.0 + cfg["radius_offset_mm"],
        "x_cm": x_cm,
        "y_cm": y_cm,
    }


MAX_HOVER_MM = 800.0
_SEARCH_AZ_DEG = 60      # how far to look for a spot the arm can reach, sideways...
_SEARCH_R_MM = 300       # ...and in distance from the base


def _search_offsets():
    """(azimuth offset, radius offset) pairs, nearest first, so the search
    for somewhere reachable fans outward from where the bottle really is."""
    pairs = [(da, dr) for da in range(-_SEARCH_AZ_DEG, _SEARCH_AZ_DEG + 1, 2)
             for dr in range(-_SEARCH_R_MM, _SEARCH_R_MM + 1, 5)]
    # about 330 mm out, one degree of azimuth is about 6 mm along the desk
    return sorted(pairs, key=lambda p: abs(p[1]) + abs(p[0]) * 6.0)


_OFFSETS = _search_offsets()


_SHIFT_COST = 10.0  # what moving the hover point 1 mm sideways costs, against 1 mm of extra height. Staying over
                    # the bottle matters far more than staying low: a few mm to the side is worth 50+ mm of height,
                    # but a 60 mm miss is not worth avoiding a 300 mm climb


LEAN_SWITCH_MM = 150.0  # turning the base a half turn is a big move for the arm to make over the desk, so it only does
                        # it when the other pose gets that much closer to the wanted spot (in the same cost as above).
                        # From 23 to 30 cm out the two poses hover within 11 cm of each other, so it stays put there;
                        # from 31 cm the turned-away pose is the only one that gets down to the cap, and it turns


@lru_cache(maxsize=8192)
def _nearest_reachable(lean, azimuth, radius_mm, height_mm):
    """The best spot the arm can actually get its gripper to in the given
    pose (reach.LEANS), at or above height_mm: (cost, (E, D, B), azimuth,
    radius, height), or None. Best means the cheapest mix of moving sideways
    off the bottle and going higher, so a spot 5 mm to the side beats a hover
    180 mm higher. Inputs are quantised by the caller so the search happens
    once per neighbourhood, not once per camera frame."""
    best = None
    for da, dr in _OFFSETS:
        shift = _SHIFT_COST * (abs(dr) + abs(da) * 6.0)
        if best is not None and shift >= best[0]:
            break                       # the offsets are sorted by shift, so nothing later can win
        r = radius_mm + dr
        if r <= 0:
            continue
        h = height_mm
        while h <= MAX_HOVER_MM:
            cost = shift + (h - height_mm)
            if best is not None and cost >= best[0]:
                break
            plan = reach.plan_target(azimuth + da, r, h, lean=lean)
            if plan is not None:
                best = (cost, (plan["E"], plan["D"], plan["B"]), azimuth + da, r, h)
                break
            h += 10.0
    return best


_Spot = namedtuple("_Spot", "cost plan azimuth radius height exact")


def _candidate(lean, azimuth, radius_mm, wanted, go_above):
    """A _Spot for one pose: straight over the cap at the wanted height if
    the arm can (exact), else the nearest spot it can reach, else None. The
    nearest-spot search works on rounded numbers, so its cost can be 0 without
    the spot being exactly the wanted one."""
    plan = reach.plan_target(azimuth, radius_mm, wanted, lean=lean)
    if plan is not None:
        return _Spot(0.0, plan, azimuth, radius_mm, wanted, True)
    if not go_above:
        return None
    found = _nearest_reachable(lean, round(azimuth), round(radius_mm / 5.0) * 5.0, round(wanted / 5.0) * 5.0)
    if found is None:
        return None
    cost, (e, d, b), az, r, h = found
    return _Spot(cost, {"E": e, "D": d, "B": b}, az, r, h, False)


def hover_plan(bearing_deg, radius_mm, bottle_height_cm, cap_height_cm, cfg, grasp=False, go_above=True, lean=None):
    """_hover_plan with the base turn scaled by cfg's e_gain: the E it commands is the E the arm should turn to, times
    the gain that says how far the base really goes for what it's told."""
    plan, info = _hover_plan(bearing_deg, radius_mm, bottle_height_cm, cap_height_cm, cfg, grasp, go_above, lean)
    if plan is not None:
        plan = {**plan, "E": plan["E"] * cfg["e_gain"]}
    return plan, info


def _hover_plan(bearing_deg, radius_mm, bottle_height_cm, cap_height_cm, cfg, grasp=False, go_above=True, lean=None):
    """Joint angles (E, D, B in degrees) that put the gripper over the cap --
    hover_clearance above it, or on it if grasp.

    lean is the pose the arm is in now (reach.LEANS: facing the bottle, or
    turned away with the arm leaning back). It stays in that pose, however
    the bottle moves, unless the other pose gets LEAN_SWITCH_MM closer to the
    wanted spot; without this a bottle sliding across the distance where the
    two poses cost about the same swung the whole base round and back.

    If the arm can't get there, and go_above is set, it goes as close as it
    can instead of giving up: the lowest height it can reach directly over
    the bottle, and failing that the nearest spot it can reach. info says
    what happened ('degraded'), where the cap is (cap_*), where the gripper
    is actually headed (azimuth_deg, radius_mm, height_mm) and the pose it
    will be in ('lean'). Returns None for the plan only if go_above is off, or
    nothing is reachable."""
    cap_mm = (bottle_height_cm - cap_height_cm / 2 - cfg["base_height_cm"]) * 10.0
    wanted = cap_mm + (0.0 if grasp else cfg["hover_clearance_mm"])
    azimuth = cfg["e_sign"] * bearing_deg + cfg["e_zero_deg"]
    info = {"cap_azimuth_deg": azimuth, "cap_radius_mm": radius_mm, "cap_mm": cap_mm, "wanted_mm": wanted,
            "azimuth_deg": azimuth, "radius_mm": radius_mm, "height_mm": wanted, "degraded": None, "lean": lean}
    stay = lean or reach.LEANS[0]
    other = reach.LEANS[1] if stay == reach.LEANS[0] else reach.LEANS[0]
    kept = _candidate(stay, azimuth, radius_mm, wanted, go_above)
    if kept is not None and kept.exact:
        info["lean"] = stay
        return kept.plan, info
    turned = _candidate(other, azimuth, radius_mm, wanted, go_above)
    options = [(name, c) for name, c in ((stay, kept), (other, turned)) if c is not None]
    chosen = None
    if options:
        best_name, best = min(options, key=lambda o: (o[1].cost, not o[1].exact))
        keep = kept is not None and kept.cost <= best.cost + LEAN_SWITCH_MM
        info["lean"], chosen = (stay, kept) if keep else (best_name, best)
        if chosen.exact:
            return chosen.plan, info

    low = reach.lowest_height(radius_mm, 90.0)
    if low is None:
        info["why"] = f"{radius_mm:.0f} mm from the base is outside the arm's reach"
    elif wanted < low:
        info["why"] = f"at {radius_mm:.0f} mm the arm can't go below {low:.0f} mm, this needs {wanted:.0f} mm"
    else:
        info["why"] = f"azimuth {azimuth:.0f} deg needs a base turn the arm can't make"
    if chosen is None:
        return None, info

    plan, az, r, h = chosen.plan, chosen.azimuth, chosen.radius, chosen.height
    info.update(azimuth_deg=az, radius_mm=r, height_mm=h)
    shift = math.hypot(r - radius_mm, math.radians(az - azimuth) * radius_mm)   # mm along the desk
    if shift < 3.0:
        info["degraded"] = f"too low to reach here: hovering at {h:.0f} mm, not {wanted:.0f}"
    elif shift < 15.0:
        info["degraded"] = f"just out of exact reach: hovering {h:.0f} mm up, {shift:.0f} mm off the cap"
    else:
        info["degraded"] = f"out of reach: hovering over the nearest spot the arm can reach ({r:.0f} mm out, {h:.0f} mm up)"
    return plan, info


def urdf_xyz(azimuth_deg, radius_mm, height_mm):
    """The same point in the robot model's base_link frame, in metres, for RViz."""
    a = math.radians(azimuth_deg + URDF_AZIMUTH_OFFSET_DEG)
    return radius_mm / 1000.0 * math.cos(a), radius_mm / 1000.0 * math.sin(a), height_mm / 1000.0
