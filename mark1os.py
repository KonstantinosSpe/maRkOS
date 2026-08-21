#!/usr/bin/env python3
"""
mark1os.py — unified control app for the Thor arm on the Mark1OS branch
=========================================================================
One Tkinter window combining three things that used to be separate tools:

  1. A command console (JOG/CAL/MOVE/gripper) — same idea as
     thor_teleop_console.py, built in here so everything shares one
     connection and one log.
  2. A "Gesture control" toggle — press it to start webcam hand-tracking
     (the same pinch-clutch velocity scheme as test1_telecontrol.py) and
     drive the real arm from it; press again to stop. The camera loop runs
     as a periodic Tkinter callback (root.after), not a blocking while-loop,
     so it coexists with the console in one process/window.
  3. Path record + playback — every command actually sent to the firmware,
     from EITHER source (typed/JOG/CAL buttons or live gesture control),
     flows through one send function. Hit Record, do something (type
     commands, jog, or drive it with your hand), hit Stop — the whole
     timestamped sequence saves to a .json file. Load Path + Play replays
     it with the same relative timing.

  ────────────────────────────────────────────────────────────────────────
  PHYSICAL AXIS LAYOUT — corrected from direct inspection of the arm
  ────────────────────────────────────────────────────────────────────────
  Base to gripper, the real order is E -> D -> A -> B/C -> gripper. This
  reverses an earlier assumption (B/C proximal, D distal, E at the wrist) —
  it wasn't just link lengths that were wrong, D and B/C had the wrong
  hinge roles too (see Thor_Kinematic_Model.docx Section 3/4). Current
  letter -> physical role:
    E: base twist (was assumed to be the wrist twist).
    D: proximal hinge, i.e. anatomically the "shoulder" — closest to the
       base of the two hinges.
    A: mid-arm twist, between the two hinges (was assumed to be the base
       twist).
    B (+C): distal hinge, i.e. anatomically the "elbow" — closer to the
       gripper of the two hinges. Still driven by two synchronized motors.

  ────────────────────────────────────────────────────────────────────────
  CALIBRATION STATUS — READ BEFORE USING GESTURE CONTROL
  ────────────────────────────────────────────────────────────────────────
  Calibrated PER AXIS, not as one global switch — these are hardware facts
  about each specific motor and don't change with the physical-role
  correction above (a motor's steps-per-degree doesn't care what we call
  the joint it drives):
    B: 490 steps = 180°, measured cleanly.
       stepsPerDegB = 2.72 (trusted).  HW_CALIBRATED_B = True — armed.
    A: 2200 steps was supposed to be a full 360°, but THIS is the axis that
       loses steps under load, so that count does NOT reliably correspond
       to 360°. stepsPerDegA = 6.11 is recorded but explicitly NOT trusted.
       HW_CALIBRATED_A = False — disarmed until A is re-measured, ideally
       at a slower SPEED setting to reduce step loss. Also currently
       receives no gesture signal at all (see A's physical-role note above).
    E: 2200 steps = 360°, measured cleanly. stepsPerDegE = 6.11 (trusted).
       HW_CALIBRATED_E = True — armed. Driven by left/right hand offset.
    D and B (+C) are J3 and J5 of the same 2-link IK (solve_reach_ik solves
       both from one (height, reach) point). Because D is the proximal
       hinge, it gets J3 (the angle measured from vertical); B, being
       distal, gets J5 (the angle relative to the first link) — see
       _hw_tick's docstring for exactly where that assignment happens.
       stepsPerDegD = 20.56 (recorded, not yet confirmed in real use).
       HW_CALIBRATED_D = True — armed by explicit request, but this is the
       FIRST time D has ever been live-driven after being explicitly
       off-limits for most of this project. Needs the updated firmware
       flashed (old firmware ignores MOVE D) and a slow supervised test —
       JOG D by hand first to confirm direction and limits before trusting
       continuous gesture-driven motion.

  Direction SIGN is a separate, still-unverified thing (see the firmware's
  own bring-up checklist) — these are magnitudes only.

  Regardless of per-axis arming: the gesture loop, HUD, and command console
  all work either way. JOG/CAL/MOVE typed through the console are NOT
  gated by any of this — those already work in real, counted steps.

Requires: pip install opencv-python "mediapipe==0.10.9" numpy pyserial
Run:      python mark1os.py     (or .venv\\Scripts\\python.exe mark1os.py)
"""

import json
import math
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import mediapipe as mp
import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("pyserial is not installed.\n\nRun:  pip install pyserial")

# ── Calibration — see the module docstring ──────────────────────────────────
# CORRECTED: the lag/step-loss problem was on motor A, not motor B —
# originally misattributed the other way around. Fixed after user clarification.
# (A's physical role has since been corrected too — see the module docstring's
# PHYSICAL AXIS LAYOUT note; it's the mid-arm twist, not the base twist.)
HW_CALIBRATED_A = False   # A: 2200 steps was meant to be 360 deg, but A is the
                          # axis that loses steps under load — NOT trusted
HW_CALIBRATED_B = True    # B: 490 steps = 180 deg, measured cleanly — trusted, armed

stepsPerDegA = 2200 / 360    # = 6.111...,  UNRELIABLE — recorded but not armed
stepsPerDegB = 490 / 180     # = 2.722...,  measured cleanly — trusted
stepsPerDegD = 3700 / 180    # = 20.556..., recorded but not yet confirmed live
stepsPerDegE = 2200 / 360    # = 6.111...,  measured cleanly — trusted

HW_CALIBRATED_E = True   # lower rotation: clean 2200 steps / 360 deg measurement,
                         # driven by left/right hand offset (freed from A)

HW_CALIBRATED_D = True   # Re-armed: the earlier "up/down unreliable" symptom
                         # traced to the pinch-engage twitch (ENGAGE_DEADZONE),
                         # not a D sign/direction problem — B alone was never
                         # actually confirmed bad. Requires the updated
                         # firmware to be flashed (old firmware silently
                         # ignores MOVE D).

HW_BAUD = 115200
HW_BOOT_DELAY_MS = 2000

# Speed ceilings for gesture-driven output, deg/s. Raised 3x from the
# original bring-up values (8.0 / 0.1 / 0.1) now that direction/limits are
# confirmed working on all axes — those original numbers capped B/D at
# 0.8 deg/s, which is why gesture control felt like it couldn't keep up
# with the hand no matter how good the tracking was. Tune further from here
# based on feel, not blind — this is still well below "fast."
HW_MAX_SPEED = 24.0
HW_ELBOW_B_SCALE = 0.3       # B (distal hinge, "elbow") is mechanically
                             # restricted/fast-for-a-given-command — extra
                             # cut on top of the base ceiling
HW_SHOULDER_D_SCALE = 0.3    # D (proximal hinge, "shoulder"): same extra cut as B
MAX_ROLL_SPEED = 60.0        # deg/s in gesture space, before the HW_MAX_SPEED
                             # ceiling clamps it down for real hardware output

MOVE_ACK_TIMEOUT = 2.0   # seconds. Each MOVE blocks the firmware until it
                         # finishes, and Python won't send a new delta for
                         # that axis until "[fw] MOVE <axis> done" comes back
                         # — if that ack is ever dropped (serial noise, a
                         # missed line split), the axis would otherwise be
                         # stuck refusing to move for the rest of the session.
                         # This unblocks it after a generous timeout instead.

HOMING_ORDER = ["B", "D", "E"]   # B/C first (edge beacon, simplest/safest),
                                  # then D (center beacon, no working hardware
                                  # limit — this is why it goes after B, not
                                  # first: B and E both have real backstops),
                                  # then E (full-rotation beacon, direction-
                                  # independent). A has no beacon yet.
HOMING_STEP_TIMEOUT = 75.0   # seconds to wait for one "[fw] HOME <axis> ..."
                            # completion line before giving up on the whole
                            # sequence. D's worst case (not found going '+',
                            # retrace, not found going '-' either) is close
                            # to 6000 total steps — and HOME <axis> now runs
                            # at the even-slower HOME_VMIN_US/MAX_US ramp
                            # (4200/2600us, not the normal 3000/1800), which
                            # puts the real worst case around 35-40 seconds.
                            # 45s cut that too close in practice — if this
                            # still isn't enough, check what HOME_VMIN_US/
                            # HOME_VMAX_US actually are in the firmware
                            # before just raising this again.

PATH_DIR = Path(__file__).with_name("paths")

# ── Palette — matches thor_teleop_console.py / thor_control.py ─────────────
BG       = "#15171A"
PANEL    = "#1E2126"
PANEL_HI = "#272B32"
EDGE     = "#343941"
INK      = "#E9E7E2"
MUTED    = "#7E858F"
AMBER    = "#FFB020"
GO       = "#54BE81"
WARN     = "#F2C12E"
ALARM    = "#E4572E"
LAMP_OFF = "#2C3138"


def pick_font(candidates, size, weight="normal"):
    import tkinter.font as tkfont
    available = {f.lower() for f in tkfont.families()}
    for name in candidates:
        if name.lower() in available:
            return (name, size, weight)
    return ("TkDefaultFont", size, weight)


# ── Serial link — plain text out, whatever the firmware prints back in ─────
class FirmwareLink:
    def __init__(self):
        self.ser = None
        self.rx = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def open(self, port):
        self.ser = serial.Serial(port, HW_BAUD, timeout=0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _reader(self):
        """A single read failure used to kill this thread permanently,
        forcing a manual reconnect for even a transient hiccup (e.g. a
        stray voltage glitch from mis-wired hardware momentarily upsetting
        the OS-level COM port). Now it logs (throttled, so a repeating
        fault doesn't spam the console) and keeps retrying instead."""
        buf = b""
        last_err_log_t = 0.0
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as exc:
                now = time.time()
                if now - last_err_log_t > 2.0:
                    self.rx.put(("error", f"Serial read failed: {exc} (retrying)"))
                    last_err_log_t = now
                time.sleep(0.3)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                text = raw.decode("ascii", "replace").strip()
                if text:
                    self.rx.put(("line", text))

    def send_line(self, text):
        if not self.is_open:
            return False
        try:
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False


# ── Geometry + IK — ported from test1_telecontrol.py (proven logic) ────────
# CORRECTED CHAIN ORDER (from direct physical inspection, pivot pin center to
# pivot pin center): the physical arm, base to gripper, is E (base twist) ->
# D (proximal hinge, "shoulder") -> A (mid-arm twist) -> B/C (distal hinge,
# "elbow") -> gripper. This is the REVERSE of what earlier analysis assumed
# (which had B/C proximal and D distal, with E at the wrist) — that earlier
# model was wrong about which hinge is closer to the base, not just about
# these three lengths. See Thor_Kinematic_Model.docx Section 3/4 for the
# full derivation under this corrected order.
SHOULDER_H = 263.0   # E -> D pivot-to-pivot. Tape-measured at 260mm; refined to the official
                     # Thor CAD equivalent (joint_2 + joint_3 = 103 + 160 = 263mm), since
                     # that's within 1.1% of the raw measurement and CAD is more precise
                     # than a tape measure + estimating pivot pin centers by eye.
A_LEN = 194.0         # D -> B/C pivot-to-pivot. Tape-measured at 195mm; refined to the
                     # official CAD equivalent (joint_4 + joint_5 = 89.5 + 104.5 = 194mm),
                     # within 0.5% of the raw measurement.
B_LEN = 185.0         # B/C -> gripper mount, pivot-to-pivot. No official CAD equivalent
                     # exists past joint_6 (gripper mount isn't in the joint table) — kept
                     # as directly measured.
REACH_MIN = abs(A_LEN - B_LEN) + 15
REACH_MAX = A_LEN + B_LEN - 10
LIMITS = {"J1": (-165, 165), "J3": (-90, 90), "J5": (-150, 150)}

R_FLOOR = 55.0
HEIGHT_CEIL = SHOULDER_H + math.sqrt(max(REACH_MAX * REACH_MAX - R_FLOOR * R_FLOOR, 0.0))
HEIGHT_FLOOR = SHOULDER_H - 160.0

MAX_J1_SPEED = 80.0
MAX_HEIGHT_SPEED = 160.0
MAX_REACH_SPEED = 140.0
DEADZONE = 0.04
CURVE_POW = 2.0
REACH_GAIN = 4.0   # halved from 8.0 — depth (apparent hand size) is already a
                   # touchy signal on its own (perspective makes small real
                   # forward/back movement swing it a lot), and this gain was
                   # amplifying it 4x harder than up/down's flat 2.0 offset
                   # multiplier, so depth reached full speed for much less
                   # physical movement than height did. Tune further if still
                   # depth-dominant.
INPUT_SMOOTH = 0.65
SCALE_SMOOTH = 0.18
OUT_SMOOTH = 0.40
VEL_SMOOTH = 0.30   # extra EMA pass specifically on B/D's velocity (v_j3/v_j5
                    # in _hw_tick). Those two are the only ones computed as a
                    # raw finite-difference of an already-smoothed angle —
                    # differentiation amplifies whatever noise survives OUT_SMOOTH,
                    # which is the likely source of the jagged elbow motion.
                    # A/E get their velocity directly (no differentiation), so
                    # they don't need this.

ENGAGE_DEADZONE = 0.035   # raw hand-offset radius (normalized image units,
                          # before the *2.0 gain) that must be cleared before
                          # left-right/up-down motion registers at all. hand_pos
                          # is the thumb-index midpoint — the same two points
                          # that define the pinch — so finishing the pinch
                          # motion itself shifts hand_pos slightly and would
                          # otherwise read as an intentional gesture the
                          # instant you engage. Depth (hand_scale) uses
                          # wrist/knuckle landmarks instead, so it's unaffected
                          # and isn't gated by this.

PINCH_ON = 0.38
PINCH_OFF = 0.55
GRIP_PINCH_ON = 0.45
GRIP_PINCH_OFF = 0.70
GRIP_SMOOTH = 0.30
GRIP_CLOSED, GRIP_OPEN = 0.0, 90.0

FIST_HOLD_S = 1.0
FIST_COOLDOWN = 0.6


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def response(x):
    s = abs(x)
    if s < DEADZONE:
        return 0.0
    s = (s - DEADZONE) / (1 - DEADZONE)
    return math.copysign(clamp(s, 0, 1) ** CURVE_POW, x)


def ik_planar(r, y, elbow):
    d = clamp(math.hypot(r, y), abs(A_LEN - B_LEN) + 1, A_LEN + B_LEN - 1)
    cos_el = clamp((A_LEN * A_LEN + B_LEN * B_LEN - d * d) / (2 * A_LEN * B_LEN), -1, 1)
    el = math.acos(cos_el)
    j5 = -(math.pi - el) * elbow
    phi = math.atan2(r, y)
    cos_psi = clamp((A_LEN * A_LEN + d * d - B_LEN * B_LEN) / (2 * A_LEN * d), -1, 1)
    psi = math.acos(cos_psi)
    j3 = phi + psi * elbow
    j3, j5 = math.degrees(j3), math.degrees(j5)
    while j3 > 180: j3 -= 360
    while j3 < -180: j3 += 360
    return j3, j5


def solve_reach_ik(height, reach):
    r, y = reach, height - SHOULDER_H
    best = None
    for e in (+1, -1):
        j3, j5 = ik_planar(r, y, e)
        pen = (abs(j3) - 90) * 10 if abs(j3) > 90 else 0
        score = pen + abs(j3)
        if best is None or score < best[0]:
            best = (score, j3, j5)
    return clamp(best[1], *LIMITS["J3"]), clamp(best[2], *LIMITS["J5"])


def clamp_reach(height, r):
    dy = height - SHOULDER_H
    hi_sq = REACH_MAX * REACH_MAX - dy * dy
    lo_sq = REACH_MIN * REACH_MIN - dy * dy
    r_hi = math.sqrt(hi_sq) if hi_sq > 0 else 0.0
    r_lo = math.sqrt(lo_sq) if lo_sq > 0 else 0.0
    r_lo = max(r_lo, R_FLOOR)
    if r_hi < r_lo: r_hi = r_lo
    return clamp(r, r_lo, r_hi)


def pinch_distance(lm): return math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
def hand_scale(lm): return math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)
def pinch_ratio(lm):
    s = hand_scale(lm)
    return pinch_distance(lm) / s if s > 1e-6 else 1.0
def thumb_pinky_distance(lm): return math.hypot(lm[4].x - lm[20].x, lm[4].y - lm[20].y)

def is_fist(lm):
    def d(i): return math.hypot(lm[i].x - lm[0].x, lm[i].y - lm[0].y)
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(1 for tip, pip in fingers if d(tip) < d(pip) * 0.92) >= 4


# ── Main application ─────────────────────────────────────────────────────
class Mark1OS:
    def __init__(self, root):
        self.root = root
        self.link = FirmwareLink()

        root.title("Mark1OS")
        root.configure(bg=BG)
        root.minsize(900, 700)

        self.f_head = pick_font(["Barlow Condensed", "Oswald", "Segoe UI Semibold",
                                 "DejaVu Sans", "Helvetica"], 11, "bold")
        self.f_body = pick_font(["Segoe UI", "Inter", "DejaVu Sans", "Helvetica"], 10)
        self.f_mono = pick_font(["Cascadia Mono", "Consolas", "JetBrains Mono",
                                 "DejaVu Sans Mono", "Courier New"], 11)

        self.log = None
        self._pending_log = []

        # -- gesture state --
        self.gesture_on = False
        self.cap = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        self.cmd_j1 = 0.0
        self.cmd_height = SHOULDER_H + 10.0
        self.cmd_R = 160.0
        self.engaged = False
        self.frozen = False   # starts ready (no un-lock gesture needed) —
                               # per-axis HW_CALIBRATED_* flags gate real motion regardless
        self.anchor = None
        self.gripper_pos = GRIP_OPEN
        self.grip_closing = False
        self.grip_pinched = False
        self.fist_held = False
        self.fist_start = None
        self.fist_lock_until = 0.0
        self.sm_j1 = self.sm_j3 = self.sm_j5 = None
        self.sm_v3 = self.sm_v5 = 0.0
        self.sm_hx = self.sm_hy = self.sm_scale = None
        self.last_t = None
        self.last_send_t = 0.0
        self.move_out_a = False
        self.move_out_b = False
        self.move_out_d = False
        self.move_out_e = False
        self.move_sent_t = {}
        self._acc_a = 0.0
        self._acc_b = 0.0
        self._acc_d = 0.0
        self._acc_e = 0.0
        self.last_intended = "(none yet)"
        self.last_would_log_t = {}
        self.jog_step_size = 200   # mirrors firmware's default jogStepSteps —
                                   # kept in sync via set_jog_step()

        # -- homing sequence state (see start_homing/_advance_homing) --
        self.homing_queue = []     # remaining axes still to home, in order
        self.homing_axis = None    # axis we're currently waiting on a response for
        self.homing_sent_t = None

        # -- recording state --
        self.recording = False
        self.rec_start_t = 0.0
        self.rec_events = []
        self.play_job = None

        self._build()
        self.root.after(40, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ---------------------------------------------------------

    def _section(self, parent, title):
        wrap = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        tk.Label(wrap, text=title.upper(), bg=PANEL, fg=MUTED, font=self.f_head,
                 anchor="w", padx=12, pady=7).pack(fill="x")
        body = tk.Frame(wrap, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return wrap, body

    def _flat_button(self, parent, text, command, accent=None):
        return tk.Button(parent, text=text, command=command, bg=PANEL_HI,
                         fg=accent or INK, activebackground=EDGE,
                         activeforeground=accent or INK, font=self.f_body,
                         bd=0, padx=10, pady=6, cursor="hand2", highlightthickness=0)

    def _build(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        self._build_topbar(outer)

        mid = tk.Frame(outer, bg=BG)
        mid.pack(fill="both", expand=True, pady=(12, 0))
        mid.columnconfigure(0, weight=3, uniform="c")
        mid.columnconfigure(1, weight=2, uniform="c")
        mid.rowconfigure(0, weight=1)

        left = tk.Frame(mid, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right = tk.Frame(mid, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self._build_jog(left)
        self._build_gesture(left)
        self._build_path(right)
        self._build_console(right)

    def _build_topbar(self, parent):
        bar = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="x", padx=12, pady=10)

        tk.Label(inner, text="PORT", bg=PANEL, fg=MUTED, font=self.f_head).pack(side="left")
        self.port_var = tk.StringVar()
        self.port_menu = tk.OptionMenu(inner, self.port_var, "")
        self.port_menu.configure(bg=PANEL_HI, fg=INK, font=self.f_mono,
                                 activebackground=EDGE, activeforeground=INK,
                                 highlightthickness=0, bd=0, width=14)
        self.port_menu["menu"].configure(bg=PANEL_HI, fg=INK, font=self.f_mono)
        self.port_menu.pack(side="left", padx=(10, 6))
        self._flat_button(inner, "Refresh", self.refresh_ports).pack(side="left")
        self.connect_btn = self._flat_button(inner, "Connect", self.toggle_connect, accent=GO)
        self.connect_btn.pack(side="left", padx=(6, 0))

        self.state_lamp = tk.Label(inner, text="  ", bg=LAMP_OFF, width=2)
        self.state_lamp.pack(side="right", padx=(10, 0))
        self.state_label = tk.Label(inner, text="disconnected", bg=PANEL, fg=MUTED, font=self.f_mono)
        self.state_label.pack(side="right")

        cal_txt = (f"A:{'armed' if HW_CALIBRATED_A else 'disarmed'}  "
                  f"B:{'armed' if HW_CALIBRATED_B else 'disarmed'}  "
                  f"D:{'armed' if HW_CALIBRATED_D else 'disarmed'}  "
                  f"E:{'armed' if HW_CALIBRATED_E else 'disarmed'}")
        cal_col = GO if (HW_CALIBRATED_A and HW_CALIBRATED_B and HW_CALIBRATED_D and HW_CALIBRATED_E) else WARN
        tk.Label(inner, text=cal_txt, bg=PANEL, fg=cal_col, font=self.f_head).pack(side="right", padx=(0, 18))

        self.refresh_ports()

    def _build_jog(self, parent):
        wrap, body = self._section(parent, "Manual jog")
        wrap.pack(fill="x")

        def axis_row(label, letter, cal=True):
            r = tk.Frame(body, bg=PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=PANEL, fg=INK, font=self.f_body, width=18, anchor="w").pack(side="left")
            self._flat_button(r, "-", lambda: self.send(f"JOG {letter}-")).pack(side="left", padx=2)
            self._flat_button(r, "+", lambda: self.send(f"JOG {letter}+")).pack(side="left", padx=2)
            if cal:
                self._flat_button(r, "record", lambda l=letter: self.send(f"CAL {l}"), accent=AMBER).pack(side="left", padx=(8, 0))

        axis_row("A (mid-arm twist)", "A")
        axis_row("B (elbow, distal)", "B")
        axis_row("D (shoulder, proximal)", "D")
        axis_row("E (base twist)", "E")
        axis_row("Gripper", "G", cal=False)

        distrow = tk.Frame(body, bg=PANEL)
        distrow.pack(fill="x", pady=3)
        tk.Label(distrow, text="JOG step size (all axes)", bg=PANEL, fg=INK, font=self.f_body,
                 width=22, anchor="w").pack(side="left")
        self.jogstep_var = tk.IntVar(value=self.jog_step_size)
        scale = tk.Scale(distrow, from_=1, to=500, orient="horizontal", variable=self.jogstep_var,
                         command=self._on_jogstep_drag, bg=PANEL, fg=INK, troughcolor=PANEL_HI,
                         highlightthickness=0, bd=0, font=self.f_mono, length=170,
                         activebackground=AMBER, showvalue=False, sliderrelief="flat")
        scale.pack(side="left", padx=(6, 8))
        scale.bind("<ButtonRelease-1>", self._on_jogstep_release)
        self.jogstep_entry_var = tk.StringVar(value=str(self.jog_step_size))
        entry = tk.Entry(distrow, textvariable=self.jogstep_entry_var, width=5, bg=PANEL_HI, fg=INK,
                         insertbackground=AMBER, font=self.f_mono, bd=0, justify="center")
        entry.pack(side="left")
        entry.bind("<Return>", self._on_jogstep_entry)
        entry.bind("<FocusOut>", self._on_jogstep_entry)
        self.dist_label = tk.Label(distrow, text=f"step size: {self.jog_step_size} steps",
                                   bg=PANEL, fg=MUTED, font=self.f_mono)
        self.dist_label.pack(side="left", padx=(10, 0))

        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self._flat_button(row, "CAL STOP", lambda: self.send("CAL STOP"), accent=WARN).pack(side="left")
        for n in (1, 10, 50, 200, 500):
            self._flat_button(row, str(n), lambda n=n: self.set_jog_step(n)).pack(side="left", padx=2)

        homerow = tk.Frame(body, bg=PANEL)
        homerow.pack(fill="x", pady=(8, 0))
        self._flat_button(homerow, "Find home (B → D → E)", self.start_homing, accent=GO).pack(side="left")
        self._flat_button(homerow, "status", lambda: self.send("HOME?")).pack(side="left", padx=(4, 0))

        speedrow = tk.Frame(body, bg=PANEL)
        speedrow.pack(fill="x", pady=(6, 0))
        tk.Label(speedrow, text="Jog speed", bg=PANEL, fg=INK, font=self.f_body,
                 width=18, anchor="w").pack(side="left")
        # (vmin, vmax) presets — bigger microseconds = slower. vmin=start/end
        # ramp floor, vmax=cruise. Matches Markos_basic.ino's own terminology.
        # "default" (3000/1800) now matches the firmware's own startup value
        # — confirmed on real hardware to track noticeably better than the
        # old 2200/900, which is kept here as "fast" instead of dropped.
        for label, vmin, vmax in [("default", 3000, 1800), ("fast", 2200, 900), ("fastest", 1400, 500)]:
            self._flat_button(speedrow, label,
                              lambda a=vmin, b=vmax: self.send(f"SPEED {a} {b}")).pack(side="left", padx=2)

    def _build_gesture(self, parent):
        wrap, body = self._section(parent, "Gesture control")
        wrap.pack(fill="x", pady=(12, 0))

        self.gesture_btn = self._flat_button(body, "Start gesture control", self.toggle_gesture, accent=GO)
        self.gesture_btn.pack(side="left")
        self.gesture_status = tk.Label(body, text="off", bg=PANEL, fg=MUTED, font=self.f_mono)
        self.gesture_status.pack(side="left", padx=(12, 0))

        tk.Label(wrap, text="Pinch thumb+index to engage (hand offset = speed). "
                            "Pinch thumb+pinky to toggle gripper. Hold a fist ~1s "
                            "to freeze/resume. Real-arm output stays off until "
                            "calibration is confirmed (see banner above).",
                 bg=PANEL, fg=MUTED, font=self.f_body, anchor="w",
                 wraplength=380, justify="left").pack(fill="x", padx=12, pady=(0, 10))

    def _build_path(self, parent):
        wrap, body = self._section(parent, "Path record / playback")
        wrap.pack(fill="x")

        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x", pady=3)
        self.rec_btn = self._flat_button(row, "● Record", self.toggle_record, accent=ALARM)
        self.rec_btn.pack(side="left")
        self._flat_button(row, "Save...", self.save_path).pack(side="left", padx=4)
        self._flat_button(row, "Load...", self.load_path).pack(side="left", padx=4)
        self._flat_button(row, "Play", self.play_path).pack(side="left", padx=4)
        self._flat_button(row, "Stop playback", self.stop_playback).pack(side="left", padx=4)

        self.path_status = tk.Label(body, text="idle — 0 events loaded", bg=PANEL, fg=MUTED, font=self.f_mono)
        self.path_status.pack(anchor="w", pady=(6, 0))
        self.loaded_path = []
        self.loaded_path_name = None

    def _build_console(self, parent):
        wrap, body = self._section(parent, "Console")
        wrap.pack(fill="both", expand=True, pady=(12, 0))

        self.log = tk.Text(body, height=18, bg="#101215", fg=INK, insertbackground=AMBER,
                           font=self.f_mono, bd=0, padx=10, pady=8, wrap="none", state="disabled")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("tx", foreground="#6FA8DC")
        self.log.tag_configure("rx", foreground=INK)
        self.log.tag_configure("err", foreground=ALARM)
        self.log.tag_configure("sys", foreground=MUTED)

        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self.cmd_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self.cmd_var, bg=PANEL_HI, fg=INK,
                         font=self.f_mono, insertbackground=AMBER, bd=0)
        entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 6))
        entry.bind("<Return>", self._send_typed)
        self._flat_button(row, "Send", self._send_typed).pack(side="left")

        pending, self._pending_log = self._pending_log, []
        for text, tag in pending:
            self.write_log(text, tag)

    # -- serial -----------------------------------------------------------

    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        menu = self.port_menu["menu"]
        menu.delete(0, "end")
        for port in ports:
            menu.add_command(label=port, command=lambda v=port: self.port_var.set(v))
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        elif not ports:
            self.port_var.set("")
        self.write_log(f"Found {len(ports)} serial port(s)", "sys")

    def toggle_connect(self):
        if self.link.is_open:
            self.link.close()
            self.connect_btn.configure(text="Connect", fg=GO)
            self.state_label.configure(text="disconnected", fg=MUTED)
            self.state_lamp.configure(bg=LAMP_OFF)
            self.write_log("Disconnected", "sys")
            return
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No port", "Select a serial port first.")
            return
        try:
            self.link.open(port)
        except Exception as exc:
            messagebox.showerror("Connection failed", str(exc))
            return
        self.connect_btn.configure(text="Disconnect", fg=ALARM)
        self.state_label.configure(text="waiting for boot", fg=WARN)
        self.state_lamp.configure(bg=WARN)
        self.write_log(f"Opened {port} at {HW_BAUD}. Waiting {HW_BOOT_DELAY_MS/1000:.1f}s...", "sys")
        self.root.after(HW_BOOT_DELAY_MS, self._finish_connect)

    def _finish_connect(self):
        if not self.link.is_open:
            return
        try:
            self.link.ser.reset_input_buffer()
            self.link.ser.reset_output_buffer()
        except Exception:
            pass
        self.state_label.configure(text="connected", fg=GO)
        self.state_lamp.configure(bg=GO)
        self.write_log("Ready.", "sys")

    def send(self, text):
        """Single instrumented send path — everything (typed, JOG buttons,
        gesture control) goes through here so recording can capture any
        source uniformly."""
        if self.recording:
            self.rec_events.append({"t": time.time() - self.rec_start_t, "cmd": text})
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        self.write_log(f"> {text}", "tx")
        self.link.send_line(text)

    def _send_typed(self, _event=None):
        text = self.cmd_var.get().strip()
        if text:
            self.send(text)
            self.cmd_var.set("")

    def _pump(self):
        while True:
            try:
                kind, payload = self.link.rx.get_nowait()
            except queue.Empty:
                break
            self.write_log(payload, "err" if kind == "error" else "rx")
            if payload.startswith("[fw] MOVE A done"):
                self.move_out_a = False
            elif payload.startswith("[fw] MOVE B done"):
                self.move_out_b = False
            elif payload.startswith("[fw] MOVE D done"):
                self.move_out_d = False
            elif payload.startswith("[fw] MOVE E done"):
                self.move_out_e = False
            # Uppercase "[fw] HOME <axis>" is the action command's completion
            # line (found/FAILED/already-home) — distinct from lowercase
            # "[fw] home ..." which is HOME?'s status-only reply.
            if self.homing_axis is not None and payload.startswith(f"[fw] HOME {self.homing_axis}"):
                self.homing_axis = None
                self.root.after(150, self._advance_homing)
        if (self.homing_axis is not None and self.homing_sent_t is not None
                and (time.time() - self.homing_sent_t) > HOMING_STEP_TIMEOUT):
            self.write_log(f"[app] no response for HOME {self.homing_axis} after "
                           f"{HOMING_STEP_TIMEOUT:.0f}s — aborting homing sequence", "err")
            self.homing_axis = None
            self.homing_queue = []
        self.root.after(40, self._pump)

    def write_log(self, text, tag="rx"):
        if self.log is None:
            self._pending_log.append((text, tag))
            return
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 800:
            self.log.delete("1.0", "300.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    # -- path record / playback -------------------------------------------

    def toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.rec_events = []
            self.rec_start_t = time.time()
            self.rec_btn.configure(text="■ Stop recording", fg=ALARM)
            self.write_log("Recording started", "sys")
        else:
            self.rec_btn.configure(text="● Record", fg=ALARM)
            self.write_log(f"Recording stopped — {len(self.rec_events)} events captured", "sys")

    def save_path(self):
        if not self.rec_events:
            messagebox.showinfo("Nothing to save", "No recorded events yet.")
            return
        PATH_DIR.mkdir(exist_ok=True)
        fname = filedialog.asksaveasfilename(initialdir=PATH_DIR, defaultextension=".json",
                                             filetypes=[("Path files", "*.json")])
        if not fname:
            return
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(self.rec_events, f, indent=2)
        self.write_log(f"Saved {len(self.rec_events)} events -> {fname}", "sys")

    def load_path(self):
        PATH_DIR.mkdir(exist_ok=True)
        fname = filedialog.askopenfilename(initialdir=PATH_DIR, filetypes=[("Path files", "*.json")])
        if not fname:
            return
        with open(fname, encoding="utf-8") as f:
            self.loaded_path = json.load(f)
        self.loaded_path_name = Path(fname).name
        self.path_status.configure(text=f"loaded {self.loaded_path_name} — {len(self.loaded_path)} events")
        self.write_log(f"Loaded {len(self.loaded_path)} events from {fname}", "sys")

    def play_path(self):
        if not self.loaded_path:
            messagebox.showinfo("No path loaded", "Load a path file first.")
            return
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        self.stop_playback()
        self.write_log(f"Playing {self.loaded_path_name} ({len(self.loaded_path)} events)", "sys")
        t0 = time.time()

        def schedule(i):
            if i >= len(self.loaded_path):
                self.write_log("Playback finished", "sys")
                return
            ev = self.loaded_path[i]
            delay_ms = max(0, int((ev["t"] - (time.time() - t0)) * 1000))
            self.play_job = self.root.after(delay_ms, lambda: (self.send(ev["cmd"]), schedule(i + 1)))

        schedule(0)

    def stop_playback(self):
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    # -- gesture control ----------------------------------------------------

    def toggle_gesture(self):
        self.gesture_on = not self.gesture_on
        if self.gesture_on:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                                 min_detection_confidence=0.7, min_tracking_confidence=0.6)
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("camera did not open (index 0) — in use by another "
                                       "program, or wrong device index?")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.write_log(f"[gesture] FAILED to start: {exc!r}", "err")
                self.write_log("If this mentions mediapipe/protobuf, you're likely running "
                               "with the wrong Python — use .venv\\Scripts\\python.exe mark1os.py",
                               "err")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.hands = None
                self.gesture_on = False
                self.gesture_btn.configure(text="Start gesture control", fg=GO)
                self.gesture_status.configure(text="off (failed to start)", fg=ALARM)
                return
            self.last_t = time.time()
            self.frozen = False
            self.gesture_btn.configure(text="Stop gesture control", fg=ALARM)
            self.gesture_status.configure(text="listening — pinch thumb+index to engage", fg=GO)
            self.write_log("Gesture control ON — ready immediately, no fist un-lock needed "
                           "(hold a fist ~1s any time to freeze/resume)", "sys")
            self.root.after(15, self._gesture_tick)
        else:
            self._stop_gesture()

    def _stop_gesture(self):
        self.gesture_on = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.gesture_btn.configure(text="Start gesture control", fg=GO)
        self.gesture_status.configure(text="off", fg=MUTED)
        self.write_log("Gesture control OFF", "sys")

    def _gesture_tick(self):
        """Thin wrapper: guarantees the tick loop reschedules itself even if
        a frame throws, so one bad frame can't silently freeze gesture
        control forever (Tkinter's after() callbacks abort quietly on an
        exception, and the reschedule used to be the last line of the
        function — any error before it meant no more ticks, ever, with no
        visible sign why)."""
        if not self.gesture_on or self.cap is None:
            return
        try:
            self._gesture_tick_body()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.write_log(f"[gesture] tick error: {exc!r}", "err")
        finally:
            if self.gesture_on:
                self.root.after(15, self._gesture_tick)

    def _gesture_tick_body(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cw = w - 260
        cam = frame[:, :cw]
        now = time.time()
        dt = clamp(now - self.last_t, 0, 0.1)
        self.last_t = now

        res = self.hands.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        v_j1 = v_h = v_r = v_e = 0.0
        fist_now = False
        hand_pos = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            self.mp_draw.draw_landmarks(cam, res.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

            fist_now = is_fist(lm)
            if fist_now and not self.fist_held:
                if self.fist_start is None and now >= self.fist_lock_until:
                    self.fist_start = now
                if self.fist_start is not None and now - self.fist_start >= FIST_HOLD_S:
                    self.fist_held = True
                    self.fist_start = None
                    self.frozen = not self.frozen
                    self.fist_lock_until = now + FIST_COOLDOWN
                    self.engaged = False
                    self.anchor = None
                    self.gesture_status.configure(
                        text="FROZEN" if self.frozen else "listening",
                        fg=(WARN if self.frozen else GO))
                    self.write_log(f"[fist] {'OFF (frozen)' if self.frozen else 'ON'}", "sys")
            elif not fist_now:
                self.fist_held = False
                self.fist_start = None

            # Fist-hold charging ring — restored from the original script:
            # an arc around the wrist that fills up as you hold the fist,
            # so you can see the ~1s hold registering in real time.
            if fist_now:
                wx, wy = int(lm[0].x * cw), int(lm[0].y * h)
                cv2.circle(cam, (wx, wy), 30, (40, 48, 70), 2, cv2.LINE_AA)
                if self.fist_start is not None:
                    frac = clamp((now - self.fist_start) / FIST_HOLD_S, 0.0, 1.0)
                    cv2.ellipse(cam, (wx, wy), (30, 30), -90, 0, 360 * frac,
                               (80, 180, 255), 3, cv2.LINE_AA)
                    label = "hold..." if frac < 1 else "RELEASE"
                else:
                    label = "fist"
                cv2.putText(cam, label, (wx - 26, wy - 38), cv2.FONT_HERSHEY_SIMPLEX,
                           0.55, (80, 180, 255), 2, cv2.LINE_AA)

            ctrl_ok = (not self.frozen) and (not fist_now)
            if not ctrl_ok:
                self.sm_hx = self.sm_hy = self.sm_scale = None
            if ctrl_ok:
                ratio = pinch_ratio(lm)
                raw_hx, raw_hy = (lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2
                raw_scale = hand_scale(lm)
                if self.sm_hx is None:
                    self.sm_hx, self.sm_hy, self.sm_scale = raw_hx, raw_hy, raw_scale
                self.sm_hx += INPUT_SMOOTH * (raw_hx - self.sm_hx)
                self.sm_hy += INPUT_SMOOTH * (raw_hy - self.sm_hy)
                self.sm_scale += SCALE_SMOOTH * (raw_scale - self.sm_scale)
                hand_pos, scale = (self.sm_hx, self.sm_hy), self.sm_scale

                if not self.engaged and ratio < PINCH_ON:
                    self.engaged = True
                    self.anchor = (hand_pos[0], hand_pos[1], scale)
                elif self.engaged and ratio > PINCH_OFF:
                    self.engaged = False
                    self.anchor = None

                if self.engaged and self.anchor is not None:
                    raw_dx = hand_pos[0] - self.anchor[0]
                    raw_dy = self.anchor[1] - hand_pos[1]
                    if math.hypot(raw_dx, raw_dy) < ENGAGE_DEADZONE:
                        raw_dx = raw_dy = 0.0
                    ox = raw_dx * 2.0
                    oy = raw_dy * 2.0
                    v_j1 = response(clamp(ox, -1, 1)) * MAX_J1_SPEED
                    v_h = response(clamp(oy, -1, 1)) * MAX_HEIGHT_SPEED
                    sc = (scale - self.anchor[2]) * REACH_GAIN
                    v_r = response(clamp(sc, -1, 1)) * MAX_REACH_SPEED
                    # E mirrors A's left/right offset (same anchor, same
                    # response curve) — matches the original main's
                    # offset-from-anchor movement style rather than a twist.
                    v_e = response(clamp(ox, -1, 1)) * MAX_ROLL_SPEED

                    j3c, j5c = solve_reach_ik(self.cmd_height, clamp_reach(self.cmd_height, self.cmd_R))
                    phi = math.radians(j3c + j5c)
                    step = v_r * dt
                    self.cmd_j1 += v_j1 * dt
                    self.cmd_height = clamp(self.cmd_height + v_h * dt + step * math.cos(phi),
                                            HEIGHT_FLOOR, HEIGHT_CEIL)
                    self.cmd_R = clamp_reach(self.cmd_height, self.cmd_R + step * math.sin(phi))

                grip_d = thumb_pinky_distance(lm) / max(scale, 1e-6)
                if not self.grip_pinched and grip_d < GRIP_PINCH_ON:
                    self.grip_pinched = True
                    self.grip_closing = not self.grip_closing
                    self.write_log(f"[gripper] {'CLOSE' if self.grip_closing else 'OPEN'}", "sys")
                elif self.grip_pinched and grip_d > GRIP_PINCH_OFF:
                    self.grip_pinched = False

                # Engage-ring + reach(distance) bar, same cues as the original script.
                if hand_pos is not None and self.anchor is not None:
                    ap = (int(self.anchor[0] * cw), int(self.anchor[1] * h))
                    hp = (int(hand_pos[0] * cw), int(hand_pos[1] * h))
                    cv2.circle(cam, ap, int(h / 2), (40, 48, 64), 1, cv2.LINE_AA)
                    moving = (abs(v_j1) + abs(v_h) + abs(v_r)) > 1
                    col = (80, 255, 120) if moving else (90, 100, 120)
                    cv2.line(cam, ap, hp, col, 2, cv2.LINE_AA)
                    cv2.circle(cam, hp, 9, col, 2, cv2.LINE_AA)
                    bx = hp[0] + 22
                    cv2.rectangle(cam, (bx, hp[1] - 40), (bx + 8, hp[1] + 40), (50, 58, 76), 1, cv2.LINE_AA)
                    fillv = int(clamp(v_r / MAX_REACH_SPEED, -1, 1) * 40)
                    rc = (80, 200, 255) if abs(v_r) > 1 else (90, 100, 120)
                    cv2.rectangle(cam, (bx, hp[1]), (bx + 8, hp[1] - fillv), rc, -1)
                    cv2.putText(cam, "FWD" if v_r > 1 else ("BACK" if v_r < -1 else "dist"),
                               (bx - 6, hp[1] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rc, 1, cv2.LINE_AA)
        else:
            self.engaged = False
            self.anchor = None
            self.sm_hx = self.sm_hy = self.sm_scale = None

        grip_target = GRIP_CLOSED if self.grip_closing else GRIP_OPEN
        self.gripper_pos += GRIP_SMOOTH * (grip_target - self.gripper_pos)

        j3_raw, j5_raw = solve_reach_ik(self.cmd_height, clamp_reach(self.cmd_height, self.cmd_R))
        if self.sm_j1 is None:
            self.sm_j1, self.sm_j3, self.sm_j5 = self.cmd_j1, j3_raw, j5_raw
        self.sm_j1 += OUT_SMOOTH * (self.cmd_j1 - self.sm_j1)
        self.sm_j3 += OUT_SMOOTH * (j3_raw - self.sm_j3)
        self.sm_j5 += OUT_SMOOTH * (j5_raw - self.sm_j5)

        self._hw_tick(now, v_j1, self.sm_j3, self.sm_j5, v_e, self.gripper_pos)

        self._draw_sidebar(frame, cw, w, h, v_j1, v_h, v_r, v_e, j3_raw, j5_raw)

        # Large, impossible-to-miss state banner directly on the camera image
        # (not just small sidebar text) — same idea as the original script's
        # centered banners for FROZEN/LISTENING/ENGAGED.
        if self.frozen:
            cv2.putText(cam, "FROZEN - hold fist ~1s to resume", (cw // 2 - 220, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 80, 230), 2, cv2.LINE_AA)
        elif self.engaged:
            cv2.putText(cam, "ENGAGED", (cw // 2 - 80, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 220, 120), 2, cv2.LINE_AA)
        else:
            cv2.putText(cam, "listening - pinch thumb+index to engage", (cw // 2 - 260, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Mark1OS - gesture control", frame)
        cv2.waitKey(1)

    def _draw_sidebar(self, frame, cw, w, h, v_j1, v_h, v_r, v_e, j3v, j5v):
        """Full numeric HUD, ported from the original test1_telecontrol.py /
        1.4.py sidebar — velocity, commanded pose (including reach/distance),
        solved joints, hardware status, gripper, key legend."""
        px = cw
        cv2.rectangle(frame, (px, 0), (w, h), (16, 18, 22), -1)
        cv2.line(frame, (px, 0), (px, h), (32, 38, 52), 1)

        def L(t, x, y, c=(150, 155, 165), s=0.40):
            cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)

        L("MARK1OS", px + 10, 26, (80, 180, 255), 0.46)
        if self.frozen:
            st, sc = "FROZEN", (60, 80, 230)
        elif self.engaged:
            st, sc = "ENGAGED", (80, 220, 120)
        else:
            st, sc = "listening", (90, 95, 110)
        L(st, px + 10, 50, sc, 0.5)

        L("Velocity", px + 10, 84, (55, 70, 100), 0.42)
        L(f"J1   : {v_j1:+6.1f} d/s", px + 14, 104, (80, 255, 120) if abs(v_j1) > 1 else (90, 95, 110), 0.34)
        L(f"hgt  : {v_h:+6.1f} mm/s", px + 14, 124, (80, 255, 120) if abs(v_h) > 1 else (90, 95, 110), 0.34)
        L(f"dist : {v_r:+6.1f} mm/s", px + 14, 144, (80, 255, 120) if abs(v_r) > 1 else (90, 95, 110), 0.34)
        L(f"E    : {v_e:+6.1f} d/s", px + 14, 164, (80, 255, 120) if abs(v_e) > 1 else (90, 95, 110), 0.34)

        L("Commanded pose", px + 10, 178, (55, 70, 100), 0.42)
        turns = int(self.cmd_j1 // 360)
        wrapped = self.cmd_j1 % 360.0
        L(f"J1 : {self.cmd_j1:+7.1f}", px + 14, 198, (120, 180, 230), 0.36)
        L(f"  = {turns:+d} turns {wrapped:5.1f}", px + 14, 214, (90, 110, 150), 0.32)
        L(f"hgt: {self.cmd_height:5.0f}mm", px + 14, 232, (120, 180, 230), 0.36)
        L(f"dist(R): {self.cmd_R:5.0f}mm", px + 14, 250, (120, 180, 230), 0.36)

        L("Solved joints", px + 10, 282, (55, 70, 100), 0.42)
        for i, (n, vv) in enumerate([("J1", self.cmd_j1), ("J3", j3v), ("J5", j5v)]):
            L(f"{n}: {vv:+6.1f}", px + 14, 302 + i * 20, (120, 180, 230), 0.36)

        L("Hardware", px + 10, 366, (55, 70, 100), 0.42)
        connected = self.link.is_open
        all_armed = HW_CALIBRATED_A and HW_CALIBRATED_B and HW_CALIBRATED_D and HW_CALIBRATED_E
        hcol = (80, 220, 120) if (all_armed and connected) else ((230, 80, 80) if not connected else (230, 200, 80))
        L(f"link: {'connected' if connected else 'disconnected'}", px + 14, 386, hcol, 0.34)
        a_txt = "A:armed" if HW_CALIBRATED_A else "A:disarmed"
        b_txt = "B:armed" if HW_CALIBRATED_B else "B:disarmed"
        d_txt = "D:armed" if HW_CALIBRATED_D else "D:disarmed"
        e_txt = "E:armed" if HW_CALIBRATED_E else "E:disarmed"
        L(f"{a_txt}  {b_txt}  {d_txt}  {e_txt}", px + 14, 404,
          (80, 220, 120) if all_armed else (230, 180, 80), 0.32)
        if self.recording:
            L(f"REC ({len(self.rec_events)} events)", px + 14, 422, (230, 80, 80), 0.34)
        L(f"last: {self.last_intended}", px + 14, 440, (150, 170, 200), 0.30)

        gtxt = "OPEN" if self.gripper_pos >= GRIP_OPEN - 1 else ("CLOSED" if self.gripper_pos <= GRIP_CLOSED + 1 else f"{self.gripper_pos:.0f}")
        L(f"Gripper: {gtxt}", px + 10, h - 62, (80, 200, 120) if self.gripper_pos > GRIP_CLOSED + 1 else (200, 120, 80), 0.40)
        L("hold fist=freeze  thumb+pinky=grip", px + 10, h - 26, (60, 70, 90), 0.28)
        L("pinch thumb+index=engage, move hand=speed", px + 10, h - 10, (60, 70, 90), 0.28)

    def _axis_ready(self, axis, move_out_attr):
        """True if axis can accept a new MOVE: either nothing's in flight,
        or the last one has gone unacknowledged past MOVE_ACK_TIMEOUT (a
        dropped "[fw] MOVE <axis> done" line would otherwise stall this
        axis forever, since it would never see the ack it's waiting for)."""
        if not getattr(self, move_out_attr):
            return True
        sent_t = self.move_sent_t.get(axis)
        if sent_t is not None and (time.time() - sent_t) > MOVE_ACK_TIMEOUT:
            setattr(self, move_out_attr, False)
            self.write_log(f"[warn] no ack for MOVE {axis} after {MOVE_ACK_TIMEOUT:.0f}s — unblocking", "err")
            return True
        return False

    def _hw_tick(self, now, v_j1, j3_now, j5_now, v_e, grip_target_deg):
        """Speed-limited MOVE streaming. Always COMPUTES the intended delta
        for A/B/D/E every tick, regardless of calibration — that's what makes
        gesture-driven movement visible (sidebar + console) and recordable
        even before real hardware output is armed. Only the actual serial
        TRANSMISSION is gated per-axis (HW_CALIBRATED_A/B/D/E); once armed,
        the same wait-for-done pacing as before applies per axis (never
        queue a second move behind one still executing).

        D (J3, proximal/shoulder hinge) and B (J5, distal/elbow hinge) come
        from the SAME (height, reach) point via solve_reach_ik — they're the
        two joints of one 2-link IK solve, not independent gestures. J3 is
        the angle measured from vertical (the FIRST hinge in the chain), J5
        is the angle relative to the first link (the SECOND hinge) — since D
        is physically closer to the base, J3 drives D and J5 drives B, the
        reverse of a naive A_LEN/B_LEN-order mapping."""
        dt = (now - self.last_send_t) if self.last_send_t else 0.0
        self.last_send_t = now
        if not hasattr(self, "_prev_j3"):
            self._prev_j3 = j3_now
            self._prev_j5 = j5_now
        v_j3 = (j3_now - self._prev_j3) / dt if dt > 1e-6 else 0.0
        v_j5 = (j5_now - self._prev_j5) / dt if dt > 1e-6 else 0.0
        self._prev_j3 = j3_now
        self._prev_j5 = j5_now

        # Extra smoothing pass on the differentiated signals only (B/D) —
        # differentiation amplifies noise, this is what was making the
        # elbows feel jagged compared to A/E's direct (non-differentiated)
        # velocity.
        self.sm_v3 += VEL_SMOOTH * (v_j3 - self.sm_v3)
        self.sm_v5 += VEL_SMOOTH * (v_j5 - self.sm_v5)

        v1 = clamp(v_j1, -HW_MAX_SPEED, HW_MAX_SPEED)
        # v_d/v_b named for the MOTOR they drive, not the joint-solve variable
        # they come from — D is the proximal/first hinge (J3, from vertical),
        # B is the distal/second hinge (J5, relative to the first link). See
        # the chain-order note above and in _hw_tick's docstring.
        v_d = clamp(self.sm_v3, -HW_MAX_SPEED * HW_SHOULDER_D_SCALE, HW_MAX_SPEED * HW_SHOULDER_D_SCALE)
        v_b = clamp(self.sm_v5, -HW_MAX_SPEED * HW_ELBOW_B_SCALE, HW_MAX_SPEED * HW_ELBOW_B_SCALE)
        ve = clamp(v_e, -HW_MAX_SPEED, HW_MAX_SPEED)

        # Fractional steps carry forward instead of being discarded each tick.
        # B in particular has a 10x extra speed cut (HW_ELBOW_B_SCALE) on
        # top of the base ceiling, so its per-tick delta is well under 0.5 —
        # int(round(...)) truncated it to 0 on essentially every tick, so B
        # could never accumulate a full step no matter how long you gestured.
        self._acc_a += v1 * stepsPerDegA * dt
        if (not HW_CALIBRATED_A) or self._axis_ready("A", "move_out_a"):
            dA = int(self._acc_a)
            if dA != 0:
                self._acc_a -= dA
                if HW_CALIBRATED_A:
                    self.move_out_a = True
                self._emit_move("A", dA)

        self._acc_b += v_b * stepsPerDegB * dt
        if (not HW_CALIBRATED_B) or self._axis_ready("B", "move_out_b"):
            dB = int(self._acc_b)
            if dB != 0:
                self._acc_b -= dB
                if HW_CALIBRATED_B:
                    self.move_out_b = True
                self._emit_move("B", dB)

        self._acc_d += v_d * stepsPerDegD * dt
        if (not HW_CALIBRATED_D) or self._axis_ready("D", "move_out_d"):
            dD = int(self._acc_d)
            if dD != 0:
                self._acc_d -= dD
                if HW_CALIBRATED_D:
                    self.move_out_d = True
                self._emit_move("D", dD)

        self._acc_e += ve * stepsPerDegE * dt
        if (not HW_CALIBRATED_E) or self._axis_ready("E", "move_out_e"):
            dE = int(self._acc_e)
            if dE != 0:
                self._acc_e -= dE
                if HW_CALIBRATED_E:
                    self.move_out_e = True
                self._emit_move("E", dE)

    def set_jog_step(self, n):
        """Single source of truth for JOGSTEP — called from the slider, the
        entry box, or anywhere else. Clamped to the slider's own 1-500
        range. Keeps the slider and entry box in sync with each other and
        sends the actual JOGSTEP command; does not move any axis itself."""
        self.jog_step_size = max(1, min(500, int(n)))
        self.dist_label.configure(text=f"step size: {self.jog_step_size} steps")
        if hasattr(self, "jogstep_var"):
            self.jogstep_var.set(self.jog_step_size)
            self.jogstep_entry_var.set(str(self.jog_step_size))
        self.send(f"JOGSTEP {self.jog_step_size}")

    def _on_jogstep_drag(self, val):
        # Live label/entry feedback while dragging — doesn't send JOGSTEP on
        # every tick (that would flood the serial link); the actual send
        # happens on release, via _on_jogstep_release.
        self.jogstep_entry_var.set(str(int(float(val))))

    def _on_jogstep_release(self, _event=None):
        self.set_jog_step(self.jogstep_var.get())

    def _on_jogstep_entry(self, _event=None):
        try:
            n = int(self.jogstep_entry_var.get())
        except ValueError:
            n = self.jog_step_size
        self.set_jog_step(n)

    def start_homing(self):
        """Kicks off the B -> D -> E homing sequence (see HOMING_ORDER) —
        one HOME <axis> at a time, waiting for that axis's own completion
        line before sending the next, since the firmware blocks on each
        HOME call until it's done."""
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        if self.homing_axis is not None:
            self.write_log("[app] Homing already in progress", "err")
            return
        self.homing_queue = list(HOMING_ORDER)
        self.write_log(f"[app] Homing sequence starting: {' -> '.join(HOMING_ORDER)}", "sys")
        self._advance_homing()

    def _advance_homing(self):
        if not self.homing_queue:
            self.homing_axis = None
            self.write_log("[app] Homing sequence complete", "sys")
            return
        self.homing_axis = self.homing_queue.pop(0)
        self.homing_sent_t = time.time()
        self.send(f"HOME {self.homing_axis}")

    def _emit_move(self, axis, delta):
        """Emit a gesture-computed MOVE command. Always updates the live
        'last command' display and (if recording) the path log, so both
        work regardless of calibration. Only actually transmits bytes to
        the real arm once that axis's HW_CALIBRATED_<axis> is True —
        otherwise it's logged (time-throttled, not every tick) as what
        WOULD have been sent."""
        cmd = f"MOVE {axis}{delta}"
        self.last_intended = cmd
        if self.recording:
            self.rec_events.append({"t": time.time() - self.rec_start_t, "cmd": cmd})
        armed = {"A": HW_CALIBRATED_A, "B": HW_CALIBRATED_B, "D": HW_CALIBRATED_D,
                 "E": HW_CALIBRATED_E}.get(axis, False)
        if armed:
            self.move_sent_t[axis] = time.time()
            self.write_log(f"> {cmd}", "tx")
            self.link.send_line(cmd)
        else:
            last = self.last_would_log_t.get(axis, 0.0)
            now = time.time()
            if now - last >= 0.15:
                self.write_log(f"[would send, uncalibrated] {cmd}", "sys")
                self.last_would_log_t[axis] = now

    def _on_close(self):
        if self.gesture_on:
            self._stop_gesture()
        try:
            self.link.close()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    Mark1OS(root)
    root.mainloop()


if __name__ == "__main__":
    main()
