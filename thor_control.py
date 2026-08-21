#!/usr/bin/env python3
"""
Thor Arm Control Panel
======================

A serial control panel for AngelLM's Thor GRBL 0.9j firmware running on an
Arduino Mega 2560.

Requires:  pip install pyserial
Run with:  python thor_control.py

------------------------------------------------------------------------------
AXIS REFERENCE (this firmware is 7-axis; you are using 5 motors on 4 joints)

  internal   G-code   joint                 STEP  DIR   LIMIT
  --------   ------   -------------------   ----  ---   -----
  A (0)      A        upper rotation         28    36     42
  B (1)      B        upper elbow motor 1    26    34     44
  C (2)      C        upper elbow motor 2    24    32     44  (shared with B)
  D (3)      D        elbow                  22    30     48
  E (4)      X        lower rotation         23    31     49  (logic inverted)
  F (5)      Y        wrist - not built      25    33     47
  G (6)      Z        wrist - not built      27    35     47

The lower rotation reads inverted at the hardware level; config.h corrects it
with INVERT_LIMIT_PIN_MASK, so the Lim readout below shows it the same way
round as every other joint.

Pins 47 (F/G) are unconnected and will always read triggered. That is expected.
------------------------------------------------------------------------------
"""

import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import messagebox

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is not installed.\n\nRun:  pip install pyserial")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BAUD = 115200
STATUS_INTERVAL_MS = 200          # how often to send '?'
BOOT_DELAY_MS = 2500              # wait for the bootloader before polling
DEFAULT_FEED = 100.0              # units/min
DEFAULT_STEP = 1.0                # units per jog press
STEP_CHOICES = [0.1, 0.5, 1.0, 5.0, 10.0]

# Joint -> the axis words that move it.
# Each entry: (display name, [(gcode letter, sign), ...])
JOINTS = [
    ("Upper rotation", [("A", 1)]),
    ("Upper elbow",    [("B", 1), ("C", 1)]),   # C sign flipped if mirror is on
    ("Elbow",          [("D", 1)]),
    ("Lower rotation", [("X", 1)]),
]

# Which internal axis index reports each joint's position (for the readout)
JOINT_AXIS_INDEX = [0, 1, 3, 4]

AXIS_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
UNUSED_AXES = {5, 6}              # F and G - no hardware fitted

GRIP_MIN, GRIP_MAX = 0, 1000      # config.h sets SPINDLE_MAX_RPM 1000.0


# ---------------------------------------------------------------------------
# Palette - industrial pendant, not generic dark mode
# ---------------------------------------------------------------------------

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
STOP     = "#B32424"
STOP_HI  = "#D42C2C"
LAMP_OFF = "#2C3138"


def pick_font(candidates, size, weight="normal"):
    """Return the first font family that exists on this machine."""
    import tkinter.font as tkfont
    available = {f.lower() for f in tkfont.families()}
    for name in candidates:
        if name.lower() in available:
            return (name, size, weight)
    return ("TkDefaultFont", size, weight)


# ---------------------------------------------------------------------------
# Serial link
# ---------------------------------------------------------------------------

class GrblLink:
    """Owns the serial port and a background reader thread."""

    def __init__(self):
        self.ser = None
        self.rx = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def open(self, port):
        self.ser = serial.Serial(port, BAUD, timeout=0.2)
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
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as exc:
                self.rx.put(("error", f"Serial read failed: {exc}"))
                return
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
            with self._lock:
                self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False

    def send_raw(self, data: bytes):
        """Realtime characters - no line ending, picked out of the stream."""
        if not self.is_open:
            return False
        try:
            with self._lock:
                self.ser.write(data)
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False


# ---------------------------------------------------------------------------
# Status line parsing
# ---------------------------------------------------------------------------

STATUS_RE = re.compile(r"^<([^,>]+)(.*)>$")


def parse_status(line):
    """Turn '<Idle,MPos:0.000,...,Lim:1101111>' into a dict, or None."""
    m = STATUS_RE.match(line.strip())
    if not m:
        return None

    out = {"state": m.group(1), "mpos": [], "lim": None}
    rest = m.group(2)

    def grab(key):
        idx = rest.find(key + ":")
        if idx < 0:
            return []
        values = []
        for token in rest[idx + len(key) + 1:].split(","):
            if ":" in token:
                break
            values.append(token)
        return values

    for value in grab("MPos"):
        try:
            out["mpos"].append(float(value))
        except ValueError:
            pass

    lim = grab("Lim")
    if lim:
        out["lim"] = lim[0]

    return out


def lim_bit(lim_string, axis_index):
    """
    Lim is printed MSB-first across 7 digits, so the leftmost character is
    axis 6 (G) and the rightmost is axis 0 (A).
    """
    if not lim_string or len(lim_string) < 7:
        return None
    return lim_string[6 - axis_index] == "1"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ThorPanel:

    def __init__(self, root):
        self.root = root
        self.link = GrblLink()
        self.last_state = "—"

        root.title("Thor Arm Control")
        root.configure(bg=BG)
        root.minsize(880, 660)

        self.f_head = pick_font(["Barlow Condensed", "Oswald", "Segoe UI Semibold",
                                 "DejaVu Sans", "Helvetica"], 11, "bold")
        self.f_body = pick_font(["Segoe UI", "Inter", "DejaVu Sans", "Helvetica"], 10)
        self.f_mono = pick_font(["Cascadia Mono", "Consolas", "JetBrains Mono",
                                 "DejaVu Sans Mono", "Courier New"], 11)
        self.f_read = pick_font(["Cascadia Mono", "Consolas", "JetBrains Mono",
                                 "DejaVu Sans Mono", "Courier New"], 15, "bold")

        self.step_var = tk.DoubleVar(value=DEFAULT_STEP)
        self.feed_var = tk.StringVar(value=str(int(DEFAULT_FEED)))
        self.mirror_c = tk.BooleanVar(value=True)
        self.show_status_lines = tk.BooleanVar(value=False)
        self.keys_armed = tk.BooleanVar(value=False)
        self.grip_var = tk.IntVar(value=0)

        self.log = None
        self._pending_log = []
        self.poll_enabled = False

        self._build()
        self._bind_keys()

        self.root.after(40, self._pump)
        self.root.after(STATUS_INTERVAL_MS, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ------------------------------------------------------------

    def _section(self, parent, title):
        wrap = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE,
                        highlightthickness=1)
        head = tk.Label(wrap, text=title.upper(), bg=PANEL, fg=MUTED,
                        font=self.f_head, anchor="w", padx=12, pady=7)
        head.pack(fill="x")
        body = tk.Frame(wrap, bg=PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return wrap, body

    def _build(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        self._build_topbar(outer)

        middle = tk.Frame(outer, bg=BG)
        middle.pack(fill="both", expand=True, pady=(12, 0))
        middle.columnconfigure(0, weight=3, uniform="col")
        middle.columnconfigure(1, weight=2, uniform="col")
        middle.rowconfigure(0, weight=1)

        left = tk.Frame(middle, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right = tk.Frame(middle, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self._build_jog(left)
        self._build_gripper(left)
        self._build_limits(right)
        self._build_actions(right)
        self._build_console(outer)

    def _build_topbar(self, parent):
        bar = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE,
                       highlightthickness=1)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="x", padx=12, pady=10)

        tk.Label(inner, text="PORT", bg=PANEL, fg=MUTED,
                 font=self.f_head).pack(side="left")

        self.port_var = tk.StringVar()
        self.port_menu = tk.OptionMenu(inner, self.port_var, "")
        self.port_menu.configure(bg=PANEL_HI, fg=INK, font=self.f_mono,
                                 activebackground=EDGE, activeforeground=INK,
                                 highlightthickness=0, bd=0, width=14)
        self.port_menu["menu"].configure(bg=PANEL_HI, fg=INK, font=self.f_mono)
        self.port_menu.pack(side="left", padx=(10, 6))

        self._flat_button(inner, "Refresh", self.refresh_ports).pack(side="left")

        self.connect_btn = self._flat_button(inner, "Connect", self.toggle_connect,
                                             accent=GO)
        self.connect_btn.pack(side="left", padx=(6, 0))

        # State readout, right-hand side
        self.state_lamp = tk.Label(inner, text="  ", bg=LAMP_OFF, width=2)
        self.state_lamp.pack(side="right", padx=(10, 0))
        self.state_label = tk.Label(inner, text="disconnected", bg=PANEL,
                                    fg=MUTED, font=self.f_mono)
        self.state_label.pack(side="right")

        # E-stop gets its own weight
        stop = tk.Button(inner, text="■  STOP", command=self.estop,
                         bg=STOP, fg="#FFFFFF", activebackground=STOP_HI,
                         activeforeground="#FFFFFF", font=self.f_read,
                         bd=0, padx=22, pady=4, cursor="hand2")
        stop.pack(side="right", padx=(0, 18))

        self.refresh_ports()

    def _build_jog(self, parent):
        wrap, body = self._section(parent, "Jog")
        wrap.pack(fill="x")

        # step size row
        steps = tk.Frame(body, bg=PANEL)
        steps.pack(fill="x", pady=(4, 10))
        tk.Label(steps, text="Step", bg=PANEL, fg=MUTED,
                 font=self.f_head).pack(side="left", padx=(0, 8))
        for value in STEP_CHOICES:
            rb = tk.Radiobutton(
                steps, text=f"{value:g}", variable=self.step_var, value=value,
                bg=PANEL, fg=INK, selectcolor=PANEL_HI, font=self.f_mono,
                activebackground=PANEL, activeforeground=AMBER,
                indicatoron=False, width=5, bd=0, padx=6, pady=3)
            rb.pack(side="left", padx=2)

        tk.Label(steps, text="Feed", bg=PANEL, fg=MUTED,
                 font=self.f_head).pack(side="left", padx=(16, 6))
        feed = tk.Entry(steps, textvariable=self.feed_var, width=7,
                        bg=PANEL_HI, fg=AMBER, font=self.f_mono,
                        insertbackground=AMBER, bd=0, justify="right")
        feed.pack(side="left", ipady=3)
        tk.Label(steps, text="/min", bg=PANEL, fg=MUTED,
                 font=self.f_body).pack(side="left", padx=(4, 0))

        # one row per joint
        self.pos_labels = []
        for index, (name, _) in enumerate(JOINTS):
            row = tk.Frame(body, bg=PANEL)
            row.pack(fill="x", pady=3)

            tk.Label(row, text=name, bg=PANEL, fg=INK, font=self.f_body,
                     anchor="w", width=15).pack(side="left")

            self._jog_button(row, "◀", index, -1).pack(side="left", padx=2)
            self._jog_button(row, "▶", index, +1).pack(side="left", padx=2)

            readout = tk.Label(row, text="—", bg=PANEL, fg=AMBER,
                               font=self.f_read, anchor="e", width=11)
            readout.pack(side="right")
            self.pos_labels.append(readout)

        opts = tk.Frame(body, bg=PANEL)
        opts.pack(fill="x", pady=(10, 0))
        self._check(opts, "Mirror C (elbow motors oppose)",
                    self.mirror_c).pack(side="left")
        self._check(opts, "Keyboard jog", self.keys_armed).pack(side="left",
                                                                padx=(16, 0))

        tk.Label(body, text="Keys: 1/2 · 3/4 · 5/6 · 7/8 by joint   ·   Esc stops",
                 bg=PANEL, fg=MUTED, font=self.f_body,
                 anchor="w").pack(fill="x", pady=(6, 0))

    def _build_gripper(self, parent):
        wrap, body = self._section(parent, "Gripper")
        wrap.pack(fill="x", pady=(12, 0))

        row = tk.Frame(body, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))

        scale = tk.Scale(row, from_=GRIP_MIN, to=GRIP_MAX, orient="horizontal",
                         variable=self.grip_var, bg=PANEL, fg=AMBER,
                         troughcolor=PANEL_HI, highlightthickness=0, bd=0,
                         font=self.f_mono, activebackground=AMBER,
                         showvalue=True, length=240,
                         command=lambda _v: None)
        scale.pack(side="left", fill="x", expand=True)

        self._flat_button(row, "Send", self.send_grip).pack(side="left", padx=(10, 4))
        self._flat_button(row, "Release", lambda: self.send("M5")).pack(side="left")

        tk.Label(body, text="M3 S0–S1000.  Servo PWM pin is unverified — "
                            "confirm before wiring the signal lead.",
                 bg=PANEL, fg=MUTED, font=self.f_body, anchor="w",
                 wraplength=420, justify="left").pack(fill="x", pady=(8, 0))

    def _build_limits(self, parent):
        wrap, body = self._section(parent, "Limit switches")
        wrap.pack(fill="x")

        grid = tk.Frame(body, bg=PANEL)
        grid.pack(fill="x", pady=4)

        self.lamps = []
        for axis in range(7):
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=0, column=axis, padx=4)

            lamp = tk.Label(cell, text=" ", bg=LAMP_OFF, width=3, height=1,
                            highlightbackground=EDGE, highlightthickness=1)
            lamp.pack()
            colour = MUTED if axis in UNUSED_AXES else INK
            tk.Label(cell, text=AXIS_LABELS[axis], bg=PANEL, fg=colour,
                     font=self.f_mono).pack(pady=(3, 0))
            self.lamps.append(lamp)

        tk.Label(body, text="E is inverted in firmware and shown corrected.  "
                            "F and G have no switch fitted and stay lit.",
                 bg=PANEL, fg=MUTED, font=self.f_body, anchor="w",
                 wraplength=300, justify="left").pack(fill="x", pady=(8, 0))

    def _build_actions(self, parent):
        wrap, body = self._section(parent, "Machine")
        wrap.pack(fill="x", pady=(12, 0))

        grid = tk.Frame(body, bg=PANEL)
        grid.pack(fill="x", pady=4)
        for column in range(2):
            grid.columnconfigure(column, weight=1, uniform="btn")

        def cell(text, command, row, column, accent=None):
            btn = self._flat_button(grid, text, command, accent=accent)
            btn.grid(row=row, column=column, sticky="ew", padx=3, pady=3)

        cell("Unlock  $X", lambda: self.send("$X"), 0, 0)
        cell("Settings  $$", lambda: self.send("$$"), 0, 1)
        cell("Zero work position", self.zero_here, 1, 0)
        cell("Return to zero", self.go_zero, 1, 1)
        cell("Hold  !", lambda: self.link.send_raw(b"!"), 2, 0)
        cell("Resume  ~", lambda: self.link.send_raw(b"~"), 2, 1)
        cell("Home  $H", self.home, 3, 0, accent=WARN)
        cell("Keep motors on  $1=0", lambda: self.send("$1=0"), 3, 1)
        cell("Reboot board (DTR)", self.reboot_board, 4, 0)
        cell("Clear console", self.clear_log, 4, 1)

    def reboot_board(self):
        """Pulse DTR to reset the Mega, the same way the IDE does."""
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return
        self.poll_enabled = False
        try:
            self.link.ser.dtr = False
            self.root.after(120, self._finish_reboot)
        except Exception as exc:
            self.write_log(f"Could not toggle DTR: {exc}", "err")
            self.poll_enabled = True

    def _finish_reboot(self):
        try:
            self.link.ser.dtr = True
        except Exception:
            pass
        self.write_log("Reset pulse sent — waiting for Grbl.", "sys")
        self.root.after(BOOT_DELAY_MS, self._start_polling)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _build_console(self, parent):
        wrap, body = self._section(parent, "Console")
        wrap.pack(fill="both", expand=True, pady=(12, 0))

        self.log = tk.Text(body, height=9, bg="#101215", fg=INK,
                           insertbackground=AMBER, font=self.f_mono,
                           bd=0, padx=10, pady=8, wrap="none",
                           state="disabled")
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
        self._check(row, "show status polls",
                    self.show_status_lines).pack(side="left", padx=(12, 0))

        # replay anything logged before this widget existed
        pending, self._pending_log = self._pending_log, []
        for text, tag in pending:
            self.write_log(text, tag)

    # -- small widget helpers ---------------------------------------------

    def _flat_button(self, parent, text, command, accent=None):
        return tk.Button(parent, text=text, command=command,
                         bg=PANEL_HI, fg=accent or INK,
                         activebackground=EDGE, activeforeground=accent or INK,
                         font=self.f_body, bd=0, padx=12, pady=6,
                         cursor="hand2", highlightthickness=0)

    def _jog_button(self, parent, glyph, joint_index, direction):
        return tk.Button(parent, text=glyph,
                         command=lambda: self.jog(joint_index, direction),
                         bg=PANEL_HI, fg=AMBER, activebackground=EDGE,
                         activeforeground=AMBER, font=self.f_read,
                         bd=0, width=3, pady=2, cursor="hand2",
                         highlightthickness=0)

    def _check(self, parent, text, variable):
        return tk.Checkbutton(parent, text=text, variable=variable, bg=PANEL,
                              fg=MUTED, selectcolor=PANEL_HI, font=self.f_body,
                              activebackground=PANEL, activeforeground=INK,
                              bd=0, highlightthickness=0)

    # -- serial plumbing ---------------------------------------------------

    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        menu = self.port_menu["menu"]
        menu.delete(0, "end")
        for port in ports:
            menu.add_command(label=port,
                             command=lambda value=port: self.port_var.set(value))
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        elif not ports:
            self.port_var.set("")
        self.write_log(f"Found {len(ports)} serial port(s)", "sys")

    def toggle_connect(self):
        if self.link.is_open:
            self.link.close()
            self.poll_enabled = False
            self.connect_btn.configure(text="Connect", fg=GO)
            self.state_label.configure(text="disconnected", fg=MUTED)
            self.state_lamp.configure(bg=LAMP_OFF)
            for label in self.pos_labels:
                label.configure(text="—")
            for lamp in self.lamps:
                lamp.configure(bg=LAMP_OFF)
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
        self.write_log(f"Opened {port} at {BAUD}. Board is resetting — holding off "
                       f"for {BOOT_DELAY_MS/1000:.1f}s so the bootloader can hand "
                       f"over to Grbl.", "sys")
        self.root.after(BOOT_DELAY_MS, self._start_polling)

    def _start_polling(self):
        """Called once the bootloader has had time to hand off to Grbl."""
        if not self.link.is_open:
            return
        try:
            self.link.ser.reset_input_buffer()
            self.link.ser.reset_output_buffer()
        except Exception:
            pass
        self.poll_enabled = True
        self.state_label.configure(text="polling", fg=MUTED)
        self.write_log("Polling started. If nothing comes back, press the reset "
                       "button on the Mega.", "sys")

    def send(self, text):
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

    def _poll(self):
        if self.link.is_open and self.poll_enabled:
            self.link.send_raw(b"?")
        self.root.after(STATUS_INTERVAL_MS, self._poll)

    def _pump(self):
        while True:
            try:
                kind, payload = self.link.rx.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self.write_log(payload, "err")
                continue

            status = parse_status(payload)
            if status:
                self.apply_status(status)
                if self.show_status_lines.get():
                    self.write_log(payload, "rx")
            else:
                tag = "err" if payload.lower().startswith(("error", "alarm")) else "rx"
                self.write_log(payload, tag)

        self.root.after(40, self._pump)

    # -- status display ----------------------------------------------------

    def apply_status(self, status):
        state = status["state"]
        if state != self.last_state:
            self.last_state = state
            colour = {"Idle": GO, "Run": AMBER, "Home": AMBER,
                      "Hold": WARN, "Alarm": ALARM}.get(state, MUTED)
            self.state_label.configure(text=state.lower(), fg=colour)
            self.state_lamp.configure(bg=colour)

        for slot, axis in enumerate(JOINT_AXIS_INDEX):
            if axis < len(status["mpos"]):
                self.pos_labels[slot].configure(text=f"{status['mpos'][axis]:.3f}")

        lim = status["lim"]
        if lim:
            for axis in range(7):
                triggered = lim_bit(lim, axis)
                if axis in UNUSED_AXES:
                    colour = EDGE if triggered else LAMP_OFF
                else:
                    colour = ALARM if triggered else LAMP_OFF
                self.lamps[axis].configure(bg=colour)

    def write_log(self, text, tag="rx"):
        if self.log is None:
            self._pending_log.append((text, tag))
            return
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        # keep the buffer from growing without bound
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    # -- motion ------------------------------------------------------------

    def _feed(self):
        try:
            value = float(self.feed_var.get())
            return max(1.0, value)
        except ValueError:
            self.write_log("Feed rate is not a number — using 100", "err")
            return 100.0

    def jog(self, joint_index, direction):
        if not self.link.is_open:
            self.write_log("Not connected", "err")
            return

        step = self.step_var.get() * direction
        words = []
        for letter, sign in JOINTS[joint_index][1]:
            value = step * sign
            if letter == "C" and self.mirror_c.get():
                value = -value
            words.append(f"{letter}{value:.3f}")

        self.send(f"G91 G1 {' '.join(words)} F{self._feed():.0f}")
        self.link.send_line("G90")

    def zero_here(self):
        self.send("G92 A0 B0 C0 D0 X0")

    def go_zero(self):
        if not messagebox.askokcancel(
                "Return to zero",
                "Every joint moves back to work zero at the current feed rate.\n\n"
                "Check the arm has clearance before continuing."):
            return
        self.send(f"G90 G1 A0 B0 C0 D0 X0 F{self._feed():.0f}")

    def home(self):
        if not messagebox.askokcancel(
                "Run homing cycle",
                "Homing order: upper elbow, elbow, upper rotation, lower rotation.\n\n"
                "Each joint seeks its switch and can travel a long way to find it. "
                "Nothing stops a joint moving the wrong direction except you.\n\n"
                "Keep a hand on the stop button."):
            return
        self.send("$H")

    def send_grip(self):
        self.send(f"M3 S{self.grip_var.get()}")

    def estop(self):
        if self.link.send_raw(b"\x18"):
            self.write_log("Soft reset sent. Send $X to unlock before moving again.",
                           "err")

    # -- keyboard ----------------------------------------------------------

    def _bind_keys(self):
        pairs = [("1", 0, -1), ("2", 0, +1),
                 ("3", 1, -1), ("4", 1, +1),
                 ("5", 2, -1), ("6", 2, +1),
                 ("7", 3, -1), ("8", 3, +1)]
        for key, joint, direction in pairs:
            self.root.bind(
                key,
                lambda _e, j=joint, d=direction: self.jog(j, d)
                if self.keys_armed.get() else None)
        self.root.bind("<Escape>", lambda _e: self.estop())

    def _on_close(self):
        try:
            self.link.close()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    ThorPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
