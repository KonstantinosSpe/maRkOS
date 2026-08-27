#!/usr/bin/env python3
"""
hw_config.py — calibration and safety numbers for the bridge and the reach planner
============================================================================
Started from the numbers in the desktop app's teleop/hw_config.py. The
steps-per-degree scales of D, E and B have since been re-measured on the
real arm (see below), so the two files differ on purpose. A has no beacon
and stays out of HOMING_ORDER for the same reason it does in the desktop app.

Everything below the steps-per-degree block only matters once the bridge
accepts motion commands. The step limits mirror the firmware's own soft
limits on purpose: the firmware would stop a move at those anyway, but
asking for a position past them just makes it hit the wall over and over.
"""

import math

stepsPerDegA = 2200 / 360
# D, E and B were measured from photos of the arm on 2026-09-20. D: commanded 45 and 75 degrees showed the arm at 54
# and 90, and 75 after the fix showed 74.8, so 3700 steps are about 216 degrees, not 180. B: -30 commanded showed about
# -39, so B moves about 1.3x what it was asked. E: 60 commanded turned the base about 36 degrees for 167 steps and
# about 71 for 270, so about 3.9 steps a degree, good to about 5%.
stepsPerDegB = 2.09
stepsPerDegD = 3700 / 216
stepsPerDegE = 3.87

HOMING_ORDER = ["B", "D", "E"]
HOMING_STEP_TIMEOUT = 75.0   # s — D's worst-case search is ~35-40 s at the
                             # slower homing ramp; this leaves real headroom

# A loses steps under load, so it never gets motion commands from here.
ARMED_AXES = ("E", "D", "B")

# Must match E_DIR_FLIP / B_DIR_FLIP / D_DIR_FLIP in thor_teleop_firmware.ino.
# The firmware reports "moved N" in the physical direction it stepped, which
# for a flipped axis is the opposite sign from the MOVE that was asked for —
# so this table is how the bridge keeps its position counter in the same
# frame as the commands it sends.
FW_DIR_FLIP = {"A": False, "B": False, "D": False, "E": True}

# Real arm against the model, from direction_test.py: +E swung the gripper anticlockwise seen from above, but +D
# tipped the arm to the left while +B tipped it to the right, so D runs backwards (in the model both tip the same
# way). The bridge applies these to what it sends and reports; the planner mirrors D's limits to match.
SIGN = {"E": 1, "D": -1, "BC": 1}

# Allowed range per axis, in steps, in the command frame (the frame MOVE is
# sent in). B and D are the firmware's soft limits pulled in by a hair so a
# clamped target never asks for the exact wall step. The firmware's own
# limits already have margin taken off B's measured 180-step travel, and B
# only has ~62 deg to give — a bigger cushion here costs real reach. E is
# held to +/-460 steps to keep the base's cabling from winding up.
_LIMIT_MARGIN_STEPS = 2
STEP_LIMITS = {
    "B": (-170 + _LIMIT_MARGIN_STEPS, 10 - _LIMIT_MARGIN_STEPS),
    "D": (-1700 + _LIMIT_MARGIN_STEPS, 1500 - _LIMIT_MARGIN_STEPS),
    "E": (-460, 460),           # the step range E has always been held to (165 degrees at the old scale); its real angle is smaller
}

# The most any one axis is sent per MOVE. The firmware blocks until a move
# finishes, so this is what bounds how long the arm is deaf to a stop or a
# changed target — 40 steps is about a fifth of a second at the default
# ramp speed. B is smaller because its steps are ~7x coarser in degrees
# than D's.
MAX_MOVE_STEPS = {"E": 40, "D": 40, "B": 20}

# Geometry the pick planner reasons with, all in mm. The shoulder is the
# D pivot's height above base_link (99 to E, 263 on to D — same chain as
# thor.urdf.xacro and ik.py). The grasp point is where the fingers close on a
# cap, taken from the B/C pivot in the forearm's own directions: ALONG the
# forearm (up when the arm stands at home) and ACROSS it (toward the side the
# arm leans for positive D). On this arm the gripper hangs at right angles to
# the forearm, not in line with it as thor.urdf.xacro has it: with the forearm
# horizontal the fingertips are 182 mm below the B/C axle and 19 mm along the
# forearm (measured from a photo of the arm tipped 75 degrees, 2026-09-20),
# which agrees with the 185 mm ik.py has from a tape. It sets how low and how
# far out the arm can grasp, so re-measure it if the gripper is ever moved.
BASE_TO_SHOULDER = 362.0
UPPER_ARM = 194.0
GRASP_ALONG = 19.0
GRASP_ACROSS = 184.0
FOREARM_TO_GRASP = math.hypot(GRASP_ALONG, GRASP_ACROSS)
GRASP_ANGLE_DEG = math.degrees(math.atan2(GRASP_ACROSS, GRASP_ALONG))   # how far the grasp point is off the forearm's line

# Don't bother the firmware over an error this small; a one-step MOVE costs
# a full ramp-up/ramp-down for almost no motion.
DEADBAND_STEPS = 2

MOVE_ACK_TIMEOUT = 2.0    # s — same recovery window as mark1os.py
COMMAND_TIMEOUT = 0.5     # s — no fresh target for this long and the bridge
                          # stops chasing the old one

# Servo angles hand-tested on the real MG996R; the firmware clamps G<n> to
# the same range, this just keeps the numbers from being sent in the first
# place.
GRIP_CLOSED = 60
GRIP_OPEN = 105
