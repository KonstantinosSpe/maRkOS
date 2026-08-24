#!/usr/bin/env python3
"""
mark1os.py — the Thor arm's control app: console, gesture control, path record/playback
============================================================================
This is the Tkinter app that ties the project's pieces together into one
window with one shared connection to the firmware. Three things live here:

  1. A command console (JOG/CAL/MOVE/gripper) for talking to the firmware
     directly, typed or via buttons.
  2. A gesture-control toggle: turn it on and drive the real arm with a
     webcam and one hand, using a pinch-to-engage / hand-offset-as-velocity
     scheme. The camera loop runs as a periodic Tkinter callback rather
     than a blocking loop, so it can live in the same window as the
     console instead of needing its own process.
  3. Path record and playback: every command actually sent to the
     firmware, from either source, flows through one function (send /
     _emit_move), so Record captures the whole session — typed commands,
     jogging, or gesture control — as one timestamped list you can save,
     reload, and play back with the same relative timing.

The geometry and IK live in ik.py, the hand-landmark math in
gesture_math.py, the serial connection in serial_link.py, and the
per-axis calibration/safety state in hw_config.py — this file is mostly
the GUI, plus the logic that ties those pieces together for a live
session. If you're trying to work out which axis is safe to drive right
now, hw_config.py's docstring is the place to look, not here.

Requires: pip install opencv-python "mediapipe==0.10.9" numpy pyserial
Run:      python mark1os.py     (or .venv\\Scripts\\python.exe mark1os.py)
"""

import json
import math
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

try:
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("pyserial is not installed.\n\nRun:  pip install pyserial")

from gesture_math import (
    ENGAGE_DEADZONE,
    FIST_COOLDOWN,
    FIST_HOLD_S,
    GRIP_CLOSED,
    GRIP_OPEN,
    GRIP_PINCH_OFF,
    GRIP_PINCH_ON,
    GRIP_SMOOTH,
    INPUT_SMOOTH,
    MAX_ROLL_SPEED,
    OUT_SMOOTH,
    PINCH_OFF,
    PINCH_ON,
    SCALE_SMOOTH,
    VEL_SMOOTH,
    hand_scale,
    is_fist,
    pinch_ratio,
    thumb_pinky_distance,
)
from hw_config import (
    HOMING_ORDER,
    HOMING_STEP_TIMEOUT,
    HW_CALIBRATED_A,
    HW_CALIBRATED_B,
    HW_CALIBRATED_D,
    HW_CALIBRATED_E,
    HW_ELBOW_B_SCALE,
    HW_MAX_SPEED,
    HW_SHOULDER_D_SCALE,
    MOVE_ACK_TIMEOUT,
    PATH_DIR,
    stepsPerDegA,
    stepsPerDegB,
    stepsPerDegD,
    stepsPerDegE,
)
from ik import (
    HEIGHT_CEIL,
    HEIGHT_FLOOR,
    MAX_HEIGHT_SPEED,
    MAX_J1_SPEED,
    MAX_REACH_SPEED,
    REACH_GAIN,
    SHOULDER_H,
    clamp,
    clamp_reach,
    response,
    solve_reach_ik,
)
from serial_link import HW_BAUD, HW_BOOT_DELAY_MS, FirmwareLink

# ── Palette — a dark console look, tuned for a workshop/lab setting where ──
# the app usually sits next to the arm rather than in a bright office.
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


def pick_font(candidates: List[str], size: int, weight: str = "normal") -> Tuple[str, int, str]:
    import tkinter.font as tkfont
    available = {f.lower() for f in tkfont.families()}
    for name in candidates:
        if name.lower() in available:
            return (name, size, weight)
    return ("TkDefaultFont", size, weight)


# ── Main application ─────────────────────────────────────────────────────
class Mark1OS:
    def __init__(self, root: tk.Tk) -> None:
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

        self.log: Optional[tk.Text] = None
        self._pending_log: List[Tuple[str, str]] = []

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
        self.frozen = True    # starts frozen — a fist has to be held ~1s
                               # before the very first move, rather than the
                               # arm being ready to go the instant a hand is
                               # detected on camera startup
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
        self.move_sent_t: Dict[str, float] = {}
        self._acc_a = 0.0
        self._acc_b = 0.0
        self._acc_d = 0.0
        self._acc_e = 0.0
        self.last_intended = "(none yet)"
        self.last_would_log_t: Dict[str, float] = {}
        self.jog_step_size = 200   # mirrors the firmware's own default
                                    # jogStepSteps — the two are kept in
                                    # sync through set_jog_step()

        # -- homing sequence state (see start_homing/_advance_homing) --
        self.homing_queue: List[str] = []   # axes still left to home, in order
        self.homing_axis: Optional[str] = None   # axis we're currently waiting on
        self.homing_sent_t: Optional[float] = None

        # -- recording state --
        self.recording = False
        self.rec_start_t = 0.0
        self.rec_events: List[Dict[str, Any]] = []
        self.play_job = None

        self._build()
        self.root.after(40, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ---------------------------------------------------------

    def _section(self, parent: tk.Widget, title: str) -> Tuple[tk.Frame, tk.Frame]:
        wrap = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        tk.Label(wrap, text=title.upper(), bg=PANEL, fg=MUTED, font=self.f_head,
                 anchor="w", padx=12, pady=7).pack(fill="x")
        body = tk.Frame(wrap, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return wrap, body

    def _flat_button(self, parent: tk.Widget, text: str, command: Callable[[], None],
                      accent: Optional[str] = None) -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg=PANEL_HI,
                         fg=accent or INK, activebackground=EDGE,
                         activeforeground=accent or INK, font=self.f_body,
                         bd=0, padx=10, pady=6, cursor="hand2", highlightthickness=0)

    def _build(self) -> None:
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

    def _build_topbar(self, parent: tk.Widget) -> None:
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

    def _build_jog(self, parent: tk.Widget) -> None:
        wrap, body = self._section(parent, "Manual jog")
        wrap.pack(fill="x")

        def axis_row(label: str, letter: str, cal: bool = True, grip: bool = False) -> None:
            r = tk.Frame(body, bg=PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=PANEL, fg=INK, font=self.f_body, width=18, anchor="w").pack(side="left")
            self._flat_button(r, "-", lambda: self.send(f"JOG {letter}-")).pack(side="left", padx=2)
            self._flat_button(r, "+", lambda: self.send(f"JOG {letter}+")).pack(side="left", padx=2)
            if cal:
                self._flat_button(r, "record", lambda l=letter: self.send(f"CAL {l}"), accent=AMBER).pack(side="left", padx=(8, 0))
            if grip:
                # Jump straight to the hand-tested open/closed positions
                # instead of nudging 3 degrees at a time — the firmware
                # clamps G<n> to GRIP_MIN/MAX anyway, so these can't
                # overshoot even if the constants ever change.
                self._flat_button(r, "Open", lambda: self.send(f"G{int(GRIP_OPEN)}"), accent=GO).pack(side="left", padx=(8, 2))
                self._flat_button(r, "Close", lambda: self.send(f"G{int(GRIP_CLOSED)}"), accent=WARN).pack(side="left", padx=2)

        axis_row("A (mid-arm twist)", "A")
        axis_row("B (elbow, distal)", "B")
        axis_row("D (shoulder, proximal)", "D")
        axis_row("E (base twist)", "E")
        axis_row("Gripper", "G", cal=False, grip=True)

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
        # (vmin, vmax) presets — bigger microseconds means slower. vmin is
        # the start/end ramp floor, vmax is the cruise speed, same
        # terminology the firmware itself uses. "default" (3000/1800)
        # matches the firmware's own startup speed and is what tracks most
        # reliably on real hardware; "fast" and "fastest" trade some of
        # that reliability for speed, for situations where that trade is
        # worth it.
        for label, vmin, vmax in [("default", 3000, 1800), ("fast", 2200, 900), ("fastest", 1400, 500)]:
            self._flat_button(speedrow, label,
                              lambda a=vmin, b=vmax: self.send(f"SPEED {a} {b}")).pack(side="left", padx=2)

    def _build_gesture(self, parent: tk.Widget) -> None:
        wrap, body = self._section(parent, "Gesture control")
        wrap.pack(fill="x", pady=(12, 0))

        self.gesture_btn = self._flat_button(body, "Start gesture control", self.toggle_gesture, accent=GO)
        self.gesture_btn.pack(side="left")
        self.gesture_status = tk.Label(body, text="off", bg=PANEL, fg=MUTED, font=self.f_mono)
        self.gesture_status.pack(side="left", padx=(12, 0))

        tk.Label(wrap, text="Starts FROZEN — hold a fist ~1s to begin. Pinch "
                            "thumb+index to engage (hand offset = speed). "
                            "Pinch thumb+pinky to toggle gripper. Hold a fist ~1s "
                            "to freeze/resume anytime. Real-arm output stays off "
                            "until calibration is confirmed (see banner above).",
                 bg=PANEL, fg=MUTED, font=self.f_body, anchor="w",
                 wraplength=380, justify="left").pack(fill="x", padx=12, pady=(0, 10))

    def _build_path(self, parent: tk.Widget) -> None:
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
        self.loaded_path: List[Dict[str, Any]] = []
        self.loaded_path_name: Optional[str] = None

    def _build_console(self, parent: tk.Widget) -> None:
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

    def refresh_ports(self) -> None:
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

    def toggle_connect(self) -> None:
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

    def _finish_connect(self) -> None:
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

    def send(self, text: str) -> None:
        """The one path every outgoing command goes through — typed, JOG
        buttons, or gesture control — so recording can capture whichever
        source is active without needing to know which one it was."""
        if self.recording:
            self.rec_events.append({"t": time.time() - self.rec_start_t, "cmd": text})
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        self.write_log(f"> {text}", "tx")
        self.link.send_line(text)

    def _send_typed(self, _event: Optional[tk.Event] = None) -> None:
        text = self.cmd_var.get().strip()
        if text:
            self.send(text)
            self.cmd_var.set("")

    def _pump(self) -> None:
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
            # An uppercase "[fw] HOME <axis>" line is the action command's
            # completion line (found / FAILED / already-home) — different
            # from the lowercase "[fw] home ..." line, which is HOME?'s
            # status-only reply and shouldn't advance the homing sequence.
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

    def write_log(self, text: str, tag: str = "rx") -> None:
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

    def toggle_record(self) -> None:
        self.recording = not self.recording
        if self.recording:
            self.rec_events = []
            self.rec_start_t = time.time()
            self.rec_btn.configure(text="■ Stop recording", fg=ALARM)
            self.write_log("Recording started", "sys")
        else:
            self.rec_btn.configure(text="● Record", fg=ALARM)
            self.write_log(f"Recording stopped — {len(self.rec_events)} events captured", "sys")

    def save_path(self) -> None:
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

    def load_path(self) -> None:
        PATH_DIR.mkdir(exist_ok=True)
        fname = filedialog.askopenfilename(initialdir=PATH_DIR, filetypes=[("Path files", "*.json")])
        if not fname:
            return
        with open(fname, encoding="utf-8") as f:
            self.loaded_path = json.load(f)
        self.loaded_path_name = Path(fname).name
        self.path_status.configure(text=f"loaded {self.loaded_path_name} — {len(self.loaded_path)} events")
        self.write_log(f"Loaded {len(self.loaded_path)} events from {fname}", "sys")

    def play_path(self) -> None:
        if not self.loaded_path:
            messagebox.showinfo("No path loaded", "Load a path file first.")
            return
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        self.stop_playback()
        self.write_log(f"Playing {self.loaded_path_name} ({len(self.loaded_path)} events)", "sys")
        t0 = time.time()

        def schedule(i: int) -> None:
            if i >= len(self.loaded_path):
                self.write_log("Playback finished", "sys")
                return
            ev = self.loaded_path[i]
            delay_ms = max(0, int((ev["t"] - (time.time() - t0)) * 1000))
            self.play_job = self.root.after(delay_ms, lambda: (self.send(ev["cmd"]), schedule(i + 1)))

        schedule(0)

    def stop_playback(self) -> None:
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    # -- gesture control ----------------------------------------------------

    def toggle_gesture(self) -> None:
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
            self.frozen = True
            self.gesture_btn.configure(text="Stop gesture control", fg=ALARM)
            self.gesture_status.configure(text="FROZEN — hold a fist ~1s to begin", fg=WARN)
            self.write_log("Gesture control ON — starting FROZEN, hold a fist ~1s to begin "
                           "(hold again any time to freeze/resume)", "sys")
            self.root.after(15, self._gesture_tick)
        else:
            self._stop_gesture()

    def _stop_gesture(self) -> None:
        self.gesture_on = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.gesture_btn.configure(text="Start gesture control", fg=GO)
        self.gesture_status.configure(text="off", fg=MUTED)
        self.write_log("Gesture control OFF", "sys")

    def _gesture_tick(self) -> None:
        """Runs one frame of gesture control, then always reschedules the
        next tick — even if this frame raised an exception. That matters
        because a Tkinter after() callback aborts silently on an unhandled
        exception: if the reschedule only happened at the very end of a
        long function, one bad frame partway through would quietly stop
        the ticks for good, with nothing on screen to say why. Wrapping the
        real work in try/except/finally keeps a single bad frame from
        taking the whole loop down with it."""
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

    def _gesture_tick_body(self) -> None:
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

            # An arc around the wrist that fills up while the fist is held,
            # so the ~1s hold has a visible countdown instead of the freeze
            # just suddenly happening with no warning.
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
                    # Negated: moving the hand up was driving height down and
                    # vice versa at home. Flipped here, at the single source
                    # both D and B/C's targets are computed from, so both
                    # stay correctly matched to each other rather than
                    # needing two separate firmware-side flips.
                    oy = -raw_dy * 2.0
                    v_j1 = response(clamp(ox, -1, 1)) * MAX_J1_SPEED
                    v_h = response(clamp(oy, -1, 1)) * MAX_HEIGHT_SPEED
                    sc = (scale - self.anchor[2]) * REACH_GAIN
                    v_r = response(clamp(sc, -1, 1)) * MAX_REACH_SPEED
                    # E is driven by the same left/right hand offset as J1
                    # (same anchor, same response curve) rather than by a
                    # separate twist gesture — left/right already reads as
                    # rotational, so there's no need to teach a second motion
                    # just for E.
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

                # Engage ring plus a reach/distance bar — visual feedback
                # for how the current gesture is being read.
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

        # A large banner directly on the camera image, not just small
        # sidebar text — the current state (FROZEN / ENGAGED / listening)
        # is the one thing you need to be able to read at a glance, without
        # hunting around the frame for it.
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

    def _draw_sidebar(self, frame: Any, cw: int, w: int, h: int,
                       v_j1: float, v_h: float, v_r: float, v_e: float,
                       j3v: float, j5v: float) -> None:
        """The numeric HUD down the right side of the camera feed:
        velocity, commanded pose (height/reach), the solved joint angles,
        hardware arm/disarm status, gripper state, and a short key
        legend."""
        px = cw
        cv2.rectangle(frame, (px, 0), (w, h), (16, 18, 22), -1)
        cv2.line(frame, (px, 0), (px, h), (32, 38, 52), 1)

        def L(t: str, x: int, y: int, c: Tuple[int, int, int] = (150, 155, 165), s: float = 0.40) -> None:
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

    def _axis_ready(self, axis: str, move_out_attr: str) -> bool:
        """True if this axis can accept a new MOVE: either nothing is
        currently in flight, or the last one has gone unacknowledged past
        MOVE_ACK_TIMEOUT. Without that timeout, a single dropped
        "[fw] MOVE <axis> done" line would leave this axis waiting for an
        ack that's never coming, stuck refusing to move for the rest of
        the session."""
        if not getattr(self, move_out_attr):
            return True
        sent_t = self.move_sent_t.get(axis)
        if sent_t is not None and (time.time() - sent_t) > MOVE_ACK_TIMEOUT:
            setattr(self, move_out_attr, False)
            self.write_log(f"[warn] no ack for MOVE {axis} after {MOVE_ACK_TIMEOUT:.0f}s — unblocking", "err")
            return True
        return False

    def _hw_tick(self, now: float, v_j1: float, j3_now: float, j5_now: float,
                 v_e: float, grip_target_deg: float) -> None:
        """Speed-limited MOVE streaming. This always computes the intended
        delta for A/B/D/E every tick, regardless of calibration — that's
        what makes gesture-driven movement visible (sidebar + console) and
        recordable even before an axis is armed for real hardware output.
        Only the actual serial transmission is gated per axis
        (HW_CALIBRATED_A/B/D/E); once armed, the same wait-for-done pacing
        as before applies — never queue a second move behind one that's
        still executing.

        The gripper doesn't fit that same pattern — there's no MOVE-style
        relative delta for it, only the firmware's absolute G<n> command —
        so instead of accumulating a delta, this just sends grip_target_deg
        directly whenever it's changed since the last tick. gripper.write()
        on the firmware side is instant regardless of whether the servo can
        physically reach that angle, so there's no blocking/pacing concern
        the way there is for the stepper axes.

        D (J3, the shoulder hinge) and B (J5, the elbow hinge) come from
        the same (height, reach) point via solve_reach_ik — they're the two
        joints of one 2-link IK solve, not two independent gestures. J3 is
        the angle measured from vertical, which belongs to whichever hinge
        is first in the chain; J5 is the angle relative to the first link,
        which belongs to the second hinge. Since D sits closer to the base,
        it's the first hinge and gets J3; B, further out, gets J5."""
        dt = (now - self.last_send_t) if self.last_send_t else 0.0
        self.last_send_t = now
        if not hasattr(self, "_prev_j3"):
            self._prev_j3 = j3_now
            self._prev_j5 = j5_now
        v_j3 = (j3_now - self._prev_j3) / dt if dt > 1e-6 else 0.0
        v_j5 = (j5_now - self._prev_j5) / dt if dt > 1e-6 else 0.0
        self._prev_j3 = j3_now
        self._prev_j5 = j5_now

        # An extra smoothing pass on the differentiated signals only (B/D):
        # differentiating always re-amplifies whatever noise survived the
        # first smoothing pass, which is what made the elbow feel jagged
        # compared to A/E — those get their velocity directly, with no
        # differencing involved.
        self.sm_v3 += VEL_SMOOTH * (v_j3 - self.sm_v3)
        self.sm_v5 += VEL_SMOOTH * (v_j5 - self.sm_v5)

        v1 = clamp(v_j1, -HW_MAX_SPEED, HW_MAX_SPEED)
        # v_d/v_b are named for the motor they drive, not the joint-solve
        # variable they come from — D is the first hinge (J3, measured from
        # vertical), B is the second hinge (J5, measured relative to the
        # first link). See the chain-order note above.
        v_d = clamp(self.sm_v3, -HW_MAX_SPEED * HW_SHOULDER_D_SCALE, HW_MAX_SPEED * HW_SHOULDER_D_SCALE)
        v_b = clamp(self.sm_v5, -HW_MAX_SPEED * HW_ELBOW_B_SCALE, HW_MAX_SPEED * HW_ELBOW_B_SCALE)
        ve = clamp(v_e, -HW_MAX_SPEED, HW_MAX_SPEED)

        # Fractional steps accumulate and carry forward between ticks
        # instead of being thrown away each time. B in particular gets an
        # extra speed cut on top of the base ceiling (HW_ELBOW_B_SCALE), so
        # its per-tick delta is often well under half a step — rounding
        # down to an integer every tick without carrying the remainder
        # forward would mean B almost never accumulates a whole step, no
        # matter how long you gesture.
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

        grip_angle = int(round(clamp(grip_target_deg, GRIP_CLOSED, GRIP_OPEN)))
        if not hasattr(self, "_last_grip_sent"):
            self._last_grip_sent = None
        if grip_angle != self._last_grip_sent:
            self._last_grip_sent = grip_angle
            self.send(f"G{grip_angle}")

    def set_jog_step(self, n: int) -> None:
        """The single place JOGSTEP actually gets set, called from the
        slider, the entry box, or anywhere else that needs to change it —
        keeps the slider and entry box in sync with each other and sends
        the command itself; it doesn't move any axis on its own."""
        self.jog_step_size = max(1, min(500, int(n)))
        self.dist_label.configure(text=f"step size: {self.jog_step_size} steps")
        if hasattr(self, "jogstep_var"):
            self.jogstep_var.set(self.jog_step_size)
            self.jogstep_entry_var.set(str(self.jog_step_size))
        self.send(f"JOGSTEP {self.jog_step_size}")

    def _on_jogstep_drag(self, val: str) -> None:
        # Updates the label/entry live while dragging, but doesn't send
        # JOGSTEP on every tick — that would flood the serial link. The
        # actual send happens once, on release, via _on_jogstep_release.
        self.jogstep_entry_var.set(str(int(float(val))))

    def _on_jogstep_release(self, _event: Optional[tk.Event] = None) -> None:
        self.set_jog_step(self.jogstep_var.get())

    def _on_jogstep_entry(self, _event: Optional[tk.Event] = None) -> None:
        try:
            n = int(self.jogstep_entry_var.get())
        except ValueError:
            n = self.jog_step_size
        self.set_jog_step(n)

    def start_homing(self) -> None:
        """Kicks off the homing sequence in HOMING_ORDER, one HOME <axis>
        at a time — waiting for that axis's own completion line before
        sending the next, since the firmware blocks on each HOME call
        until it's done."""
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        if self.homing_axis is not None:
            self.write_log("[app] Homing already in progress", "err")
            return
        self.homing_queue = list(HOMING_ORDER)
        self.write_log(f"[app] Homing sequence starting: {' -> '.join(HOMING_ORDER)}", "sys")
        self._advance_homing()

    def _advance_homing(self) -> None:
        if not self.homing_queue:
            self.homing_axis = None
            self.write_log("[app] Homing sequence complete", "sys")
            return
        self.homing_axis = self.homing_queue.pop(0)
        self.homing_sent_t = time.time()
        self.send(f"HOME {self.homing_axis}")

    def _emit_move(self, axis: str, delta: int) -> None:
        """Emits a gesture-computed MOVE command. Always updates the live
        'last command' display and, if recording, the path log — both of
        those work regardless of calibration. Only actually transmits
        bytes to the real arm once that axis's HW_CALIBRATED_<axis> is
        True; until then it's logged, time-throttled rather than every
        tick, as what would have been sent."""
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

    def _on_close(self) -> None:
        if self.gesture_on:
            self._stop_gesture()
        try:
            self.link.close()
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Mark1OS(root)
    root.mainloop()


if __name__ == "__main__":
    main()
