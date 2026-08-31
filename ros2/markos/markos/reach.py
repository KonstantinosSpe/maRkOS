#!/usr/bin/env python3
"""
reach.py — can the arm put the gripper at this point, and with which joints?
============================================================================
The generic 2-link IK in ik.py clamps to +/-90 and +/-150 degrees and never
says "no". This arm's real hinge ranges are much smaller (B bends only about
62 degrees one way), so most of the workspace that IK cheerfully returns
answers for isn't reachable. This module solves the same triangle and then
throws away every answer that falls outside the ranges hw_config.STEP_LIMITS
allow, so a target either comes back with joint angles the arm can hold or
comes back as None.

Frame: heights are above base_link (the surface the robot is bolted to),
radius and azimuth are measured from the E axis. Joint angles follow ik.py:
J3 (D) is link 1 from vertical, J5 (B) is link 2 relative to link 1, and the
sign of D and B against the real hardware is the same assumption
mark1os.py's gesture control makes.
"""

import math
from typing import Dict, List, Optional, Tuple

from . import hw_config as cfg
from .axis_controller import STEPS_PER_DEG


def _model_limits(axis: str, sign: int):
    """The axis's allowed range in the model's angles: the hardware limits, mirrored if this axis turns the
    other way on the real arm than in the model."""
    lo, hi = (s / STEPS_PER_DEG[axis] for s in cfg.STEP_LIMITS[axis])
    return (lo, hi) if sign > 0 else (-hi, -lo)


E_LIMIT_DEG = cfg.STEP_LIMITS["E"][1] / STEPS_PER_DEG["E"]
D_LIMITS_DEG = _model_limits("D", cfg.SIGN["D"])
B_LIMITS_DEG = _model_limits("B", cfg.SIGN["BC"])


def _wrap(deg: float) -> float:
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def _planar(r: float, y: float) -> List[Tuple[float, float]]:
    """Both elbow configurations that put the grasp point at (r, y) relative
    to the shoulder, as (J3, J5) degrees. Empty if the point is farther than
    the arm can stretch or closer than it can fold."""
    a, b = cfg.UPPER_ARM, cfg.FOREARM_TO_GRASP
    d = math.hypot(r, y)
    if d > a + b or d < abs(a - b) or d == 0:
        return []
    el = math.acos((a * a + b * b - d * d) / (2 * a * b))
    psi = math.acos((a * a + d * d - b * b) / (2 * a * d))
    phi = math.atan2(r, y)
    out = []
    for elbow in (+1, -1):
        j3 = math.degrees(phi + psi * elbow)
        j5 = math.degrees(-(math.pi - el) * elbow)
        out.append((_wrap(j3), j5))
    return out


def solve_plane(r: float, height: float) -> Optional[Tuple[float, float]]:
    """(D, B) in degrees for a grasp point r mm out (negative = the arm
    leans the other way) and `height` mm above base_link, or None."""
    for j3, j5 in _planar(r, height - cfg.BASE_TO_SHOULDER):
        b = j5 - cfg.GRASP_ANGLE_DEG        # j5 is the line to the grasp point, which sits off the forearm's own line
        if D_LIMITS_DEG[0] <= j3 <= D_LIMITS_DEG[1] and B_LIMITS_DEG[0] <= b <= B_LIMITS_DEG[1]:
            return j3, b
    return None


LEANS = ("toward", "away")


def plan_target(azimuth_deg: float, radius_mm: float, height_mm: float,
                lean: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Joint angles (E, D, B in degrees) that put the grasp point over a spot
    `radius_mm` from the base axis in direction `azimuth_deg`, at `height_mm`
    above base_link. The arm can reach the point either leaning toward it
    (r > 0, E at the azimuth) or leaning away with the base turned around
    (r < 0, E at azimuth +/- 180); E's travel decides which are allowed.
    `lean` ("toward" or "away") limits the answer to one of the two, so a
    caller that is already in one pose doesn't get handed the other; None
    takes whichever works, toward first."""
    for name, r, e in (("toward", radius_mm, azimuth_deg), ("away", -radius_mm, _wrap(azimuth_deg + 180.0))):
        if lean not in (None, name) or abs(e) > E_LIMIT_DEG:
            continue
        dz = solve_plane(r, height_mm)
        if dz is not None:
            return {"E": e, "D": dz[0], "B": dz[1]}
    return None


def lowest_height(radius_mm: float, azimuth_deg: float = 90.0) -> Optional[float]:
    """The lowest grasp height above base_link the arm can reach at this
    radius (in a direction where E allows either lean), or None if it can't
    get there at any height. Handy for deciding how tall a plinth has to be."""
    for h in range(0, 800, 5):
        if plan_target(azimuth_deg, radius_mm, float(h)) is not None:
            return float(h)
    return None
