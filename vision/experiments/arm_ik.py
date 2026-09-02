#!/usr/bin/env python3
"""
arm_ik.py -- the arm's geometry: link lengths, joint limits, and the 2-link IK
============================================================================
Ported verbatim (naming/comments intact) from ik.py on the maRkOS repo's
Reorganize branch, which was never merged/ported into the ROS2 workspace.
Copied here rather than imported as a package so follow_palm_ros.py doesn't
depend on ROS2 workspace sourcing being exactly right -- this file has zero
dependencies beyond the standard library either way.

A quick word on the physical layout, since it explains where SHOULDER_H,
A_LEN and B_LEN come from. Walking the real arm from the base up to the
gripper, the order is: E (a twist at the base) -> D (the first hinge,
closest to the base -- anatomically the "shoulder") -> A (a second twist,
partway up the arm) -> B/C (the second hinge, closer to the gripper --
anatomically the "elbow") -> gripper. D and B/C are the two hinges that
this file's IK actually solves for; A and E are twists, so they rotate
the whole assembly rather than bending it, and don't factor into the
planar solve below.

Because D is the hinge nearer the base, it gets J3 -- the angle measured
from vertical, i.e. the first link's own angle. B, being the hinge nearer
the gripper, gets J5 -- the angle of the second link relative to the
first one, not to vertical. That mapping (D -> J3, B -> J5) is what
solve_reach_ik hands back, and it's worth remembering if you're ever
tracing a value from here back to a motor: which joint is "first" and
which is "second" is a statement about the physical chain, not an
arbitrary labeling choice.

Requires: nothing beyond the standard library.
"""

import math
from typing import Tuple

# -- Link geometry -----------------------------------------------------
# Pivot-to-pivot distances, in millimetres. Cross-checked against the
# official Thor CAD joint table rather than left as raw tape-measure
# readings, since the tape numbers were within about 1% of CAD anyway and
# CAD locates a pivot pin's center far more precisely than eyeballing one
# with a tape measure can.
SHOULDER_H = 263.0   # E -> D, base twist to shoulder hinge
A_LEN = 194.0         # D -> B/C, shoulder hinge to elbow hinge
B_LEN = 185.0         # B/C -> gripper mount (no CAD entry exists past this
                      # point, so this one's still the direct measurement)

# The two-link solve has a real singularity at full stretch (d == A_LEN +
# B_LEN) and at full fold (d == |A_LEN - B_LEN|) -- the triangle it's
# solving degenerates to a line there, so the elbow angle stops being
# well-defined. Pulling REACH_MIN/REACH_MAX in a little from those exact
# extremes keeps the solver comfortably clear of that edge instead of
# hoping floating-point rounding never lands exactly on it.
REACH_MIN = abs(A_LEN - B_LEN) + 15
REACH_MAX = A_LEN + B_LEN - 10
LIMITS = {"J1": (-165, 165), "J3": (-90, 90), "J5": (-150, 150)}

R_FLOOR = 55.0
HEIGHT_CEIL = SHOULDER_H + math.sqrt(max(REACH_MAX * REACH_MAX - R_FLOOR * R_FLOOR, 0.0))
HEIGHT_FLOOR = SHOULDER_H - 160.0


def clamp(v: float, lo: float, hi: float) -> float:
    """Restrict v to [lo, hi]."""
    return max(lo, min(hi, v))


def ik_planar(r: float, y: float, elbow: int) -> Tuple[float, float]:
    """Solve the 2-link planar triangle for one elbow configuration
    (elbow = +1 or -1, i.e. elbow-up vs elbow-down) and return (J3, J5) in
    degrees. This is the law-of-cosines solve for a two-bar linkage: given
    how far away the target point is (d) and the two link lengths, the
    triangle's angles are fully determined."""
    d = clamp(math.hypot(r, y), abs(A_LEN - B_LEN) + 1, A_LEN + B_LEN - 1)
    cos_el = clamp((A_LEN * A_LEN + B_LEN * B_LEN - d * d) / (2 * A_LEN * B_LEN), -1, 1)
    el = math.acos(cos_el)
    j5 = -(math.pi - el) * elbow
    phi = math.atan2(r, y)
    cos_psi = clamp((A_LEN * A_LEN + d * d - B_LEN * B_LEN) / (2 * A_LEN * d), -1, 1)
    psi = math.acos(cos_psi)
    j3 = phi + psi * elbow
    j3, j5 = math.degrees(j3), math.degrees(j5)
    while j3 > 180:
        j3 -= 360
    while j3 < -180:
        j3 += 360
    return j3, j5


def solve_reach_ik(height: float, reach: float) -> Tuple[float, float]:
    """Pick whichever elbow configuration (up or down) keeps J3 closer to
    vertical, since that's the more natural, less cramped pose for this
    arm, then clamp both joints to their travel limits. This is the
    function _hw_tick calls every gesture-control tick to turn a desired
    (height, reach) point into the D/B motor targets."""
    r, y = reach, height - SHOULDER_H
    best = None
    for e in (+1, -1):
        j3, j5 = ik_planar(r, y, e)
        pen = (abs(j3) - 90) * 10 if abs(j3) > 90 else 0
        score = pen + abs(j3)
        if best is None or score < best[0]:
            best = (score, j3, j5)
    return clamp(best[1], *LIMITS["J3"]), clamp(best[2], *LIMITS["J5"])


def clamp_reach(height: float, r: float) -> float:
    """Given a height, work out how close/far the reach is allowed to be
    so the resulting point stays inside the arm's actual working envelope
    (REACH_MIN/REACH_MAX at that height), and clamp r into that range."""
    dy = height - SHOULDER_H
    hi_sq = REACH_MAX * REACH_MAX - dy * dy
    lo_sq = REACH_MIN * REACH_MIN - dy * dy
    r_hi = math.sqrt(hi_sq) if hi_sq > 0 else 0.0
    r_lo = math.sqrt(lo_sq) if lo_sq > 0 else 0.0
    r_lo = max(r_lo, R_FLOOR)
    if r_hi < r_lo:
        r_hi = r_lo
    return clamp(r, r_lo, r_hi)
