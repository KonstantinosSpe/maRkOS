#!/usr/bin/env python3
"""
hw_config.py — what's safe to actually send to the motors, and how fast
============================================================================
This file is the arm's safety and calibration state: which axes are
trusted enough to receive real motion commands, how many steps each motor
takes per degree, and the timing constants the homing/streaming logic
leans on. Read it before flipping any HW_CALIBRATED_* flag — each one is a
statement that a person has watched that specific axis move and confirmed
it's safe to drive automatically, not just a config toggle.

A quick map of where each axis stands, since "armed" on its own doesn't
tell the whole story:

  B — 490 steps measured cleanly against a 180° swing, so stepsPerDegB is
      trusted and B is armed.
  E — armed too. Its steps-per-degree comes from a real full-lap
      measurement (the firmware's SWEEP E+/E- command, run home-to-home
      in each direction) rather than an assumed constant, which matters
      because a twist axis has no natural "180° reference" the way a
      hinge does — you either measure a full lap directly, or you're
      guessing.
  D — armed, but newer to real driving than B or E: it shares one 2-link
      IK solve with B (see ik.py's solve_reach_ik), where D takes the
      angle measured from vertical and B takes the angle relative to the
      first link, since D is the hinge closer to the base. Its
      steps-per-degree is recorded from earlier testing but hasn't had as
      much live confirmation as B's, so it's worth a slow, supervised
      JOG test on real hardware before trusting continuous gesture-driven
      motion on it.
  A — still disarmed. A is the axis that loses steps under load, so its
      recorded steps-per-degree can't be trusted the way B's can — the
      motor doesn't reliably complete the rotation you asked for. It also
      isn't wired to any gesture input right now, so this mostly matters
      for anyone re-enabling it later: re-measure at a slower speed first.

Direction sign (whether "+" actually moves the way you'd expect) is a
separate question from arming, covered in the firmware's own bring-up
checklist — the values below are magnitudes only.

None of this gates the plain command console: JOG/CAL/MOVE typed by hand
already work in real, counted steps regardless of these flags. What's
gated is only the automatic path — gesture control computing and sending
its own deltas every tick.
"""

from pathlib import Path

HW_CALIBRATED_A = False
HW_CALIBRATED_B = True
HW_CALIBRATED_D = True
HW_CALIBRATED_E = True

# Steps per degree, per axis — how the firmware's raw step counts translate
# into the angles ik.py and the gesture loop reason about.
stepsPerDegA = 2200 / 360    # not trusted yet — see HW_CALIBRATED_A above
stepsPerDegB = 490 / 180     # measured cleanly
stepsPerDegD = 3700 / 180    # recorded, live-confirmation still building up
stepsPerDegE = 1004 / 360    # from a direct full-lap SWEEP measurement

# Speed ceiling for gesture-driven output, in degrees/second, plus two
# extra per-axis cuts on top of it. B and D are geometrically the two
# joints that move the most output distance for a given command, so they
# get scaled down further than the base ceiling alone would allow —
# without this, they'd feel noticeably twitchier than A and E for what's
# meant to be the same gesture intensity.
HW_MAX_SPEED = 24.0
HW_ELBOW_B_SCALE = 0.3      # B, the elbow hinge
HW_SHOULDER_D_SCALE = 0.3   # D, the shoulder hinge

MOVE_ACK_TIMEOUT = 2.0   # seconds. Each MOVE blocks the firmware until it
                          # finishes, and this side won't send a new delta
                          # for that axis until the matching "[fw] MOVE
                          # <axis> done" line comes back — so if that ack
                          # is ever lost (a burst of serial noise, a line
                          # split at just the wrong moment), the axis needs
                          # some way to recover instead of refusing to move
                          # for the rest of the session. This timeout is
                          # that recovery: generous enough not to fire
                          # during a normal move, short enough to notice.

HOMING_ORDER = ["B", "D", "E"]   # B first, since its beacon sits at an edge
                                  # backed by a working hardware limit
                                  # switch — the simplest, safest one to
                                  # start with. D goes next: its beacon sits
                                  # in the middle of its range with no
                                  # hardware backstop of its own, so it's
                                  # worth having B already homed as a known
                                  # reference before trusting it. E goes
                                  # last, since as a full rotation its
                                  # homing search doesn't care which
                                  # direction the other axes ended up in.
                                  # A has no beacon yet, so it isn't part of
                                  # this sequence at all.
HOMING_STEP_TIMEOUT = 75.0   # seconds to wait for one "[fw] HOME <axis>"
                              # completion line before giving up on the
                              # whole sequence. D's worst case — not found
                              # going one way, retracing, not found the
                              # other way either — covers close to 6000
                              # steps, and homing searches run at a
                              # deliberately slower ramp speed than normal
                              # moves for extra torque margin, which
                              # stretches that worst case out to somewhere
                              # around 35-40 seconds. This leaves real
                              # headroom above that rather than cutting it
                              # close.

PATH_DIR = Path(__file__).with_name("paths")
