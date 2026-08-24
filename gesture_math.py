#!/usr/bin/env python3
"""
gesture_math.py — turning a MediaPipe hand into pinch/fist/scale numbers
============================================================================
Everything here is pure geometry on hand landmarks: no camera, no serial
port, no Tkinter. Given the 21 landmarks MediaPipe Hands hands back for a
detected hand, these functions answer simple questions like "how far apart
are the thumb and index tip" or "does this look like a closed fist" — the
kind of thing that's easy to unit-test with a few made-up landmark sets,
even without a webcam attached.

Landmark indices follow MediaPipe's own numbering: 0 is the wrist, 4 the
thumb tip, 8/12/16/20 the index/middle/ring/pinky tips, and 6/10/14/18 the
corresponding first knuckle (PIP joint) below each of those tips. 9 is the
middle finger's base knuckle, which turns out to be a convenient stand-in
for "how big does the hand look right now" — see hand_scale below.
"""

import math
from typing import Protocol, Sequence


class _Landmark(Protocol):
    x: float
    y: float


Landmarks = Sequence[_Landmark]

# ── Tunable feel ─────────────────────────────────────────────────────────
# Exponential-moving-average smoothing factors — how much of a new reading
# gets blended in each tick. Higher = snappier but jittery; lower = smoother
# but laggier. Different signals get different amounts of smoothing because
# they're noisy in different ways:
INPUT_SMOOTH = 0.65   # the raw hand position — needs to feel responsive
SCALE_SMOOTH = 0.18   # apparent hand size (used for reach/depth) — noisier
                       # than position, since perspective exaggerates it
OUT_SMOOTH = 0.40      # the solved joint angles going out to the arm
VEL_SMOOTH = 0.30      # extra smoothing on top, specifically for D and B's
                        # velocity. Those two are the only signals computed
                        # as a finite difference of an already-smoothed
                        # angle, and differentiating always re-amplifies
                        # whatever noise survived the first smoothing pass —
                        # this second pass is what keeps the elbow from
                        # feeling jerky compared to the twist axes, which
                        # get their velocity directly with no differencing.

ENGAGE_DEADZONE = 0.035   # how far the hand has to move from its anchor
                           # point (normalized image units) before it counts
                           # as an intentional move. hand_pos is the
                           # thumb-index midpoint — the same two points that
                           # define the pinch gesture itself — so finishing
                           # a pinch naturally nudges hand_pos a little, and
                           # without this deadzone that nudge would read as
                           # a movement command the instant you engage.

PINCH_ON = 0.38     # pinch ratio below which we consider the pinch "closed"
PINCH_OFF = 0.55    # and above which we consider it "released" — two
                     # different thresholds (rather than one) so the toggle
                     # doesn't chatter back and forth when the ratio is
                     # sitting right at the boundary.
GRIP_PINCH_ON = 0.45
GRIP_PINCH_OFF = 0.70
GRIP_SMOOTH = 0.30
GRIP_CLOSED, GRIP_OPEN = 60.0, 105.0   # hand-tested on the real MG996R gripper:
                                        # jogging past 60 toward closed starts
                                        # straining against the mechanism, and
                                        # 105 is a confirmed comfortable full
                                        # open, not just a boot-default guess

FIST_HOLD_S = 1.0     # how long a fist needs to be held before it toggles
                       # freeze/resume — long enough that it isn't tripped
                       # by accident, short enough not to feel sluggish.
FIST_COOLDOWN = 0.6    # a short lockout right after a fist toggle fires, so
                        # the same held fist can't immediately trigger again.

MAX_ROLL_SPEED = 60.0   # ceiling for E's gesture-space rotation speed,
                         # before hw_config's HW_MAX_SPEED clamps it down
                         # again for the real motor output.


def pinch_distance(lm: Landmarks) -> float:
    """Distance between thumb tip (4) and index tip (8), in normalized
    image units."""
    return math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)


def hand_scale(lm: Landmarks) -> float:
    """A stand-in for 'how close is the hand to the camera': the distance
    between the wrist (0) and the middle finger's base knuckle (9) doesn't
    change with hand pose the way a fingertip-to-fingertip measurement
    would, so it tracks depth pretty cleanly on its own."""
    return math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)


def pinch_ratio(lm: Landmarks) -> float:
    """pinch_distance normalized by hand_scale, so the pinch threshold
    means the same thing whether the hand is close to the camera or far
    from it — without this, a pinch made further away would measure as a
    smaller raw distance than the exact same pinch made up close."""
    s = hand_scale(lm)
    return pinch_distance(lm) / s if s > 1e-6 else 1.0


def thumb_pinky_distance(lm: Landmarks) -> float:
    """Distance between thumb tip (4) and pinky tip (20) — the gesture used
    for the gripper toggle, kept separate from the thumb/index pinch so the
    two gestures can't be confused for each other."""
    return math.hypot(lm[4].x - lm[20].x, lm[4].y - lm[20].y)


def is_fist(lm: Landmarks) -> bool:
    """True if at least 4 of the 5 fingers look curled: for each finger,
    compare how far its tip sits from the wrist against how far its own
    PIP joint sits from the wrist. An extended finger's tip is further from
    the wrist than its own knuckle; a curled one folds back in, so the tip
    ends up closer instead. The 0.92 factor gives a little slack rather
    than requiring the tip to be strictly closer, so a fist that isn't
    perfectly closed still reads correctly."""
    def d(i: int) -> float:
        return math.hypot(lm[i].x - lm[0].x, lm[i].y - lm[0].y)
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(1 for tip, pip in fingers if d(tip) < d(pip) * 0.92) >= 4
