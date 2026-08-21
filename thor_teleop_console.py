#!/usr/bin/env python3
"""
thor_teleop_console.py
=======================
Manual command console for thor_teleop_firmware.ino — the direct-drive
rebuild (digitalWrite/delayMicroseconds bit-banging, matching the user's
own previously-working Markos_basic.ino, NOT AccelStepper, NOT Grbl — so
this is a different tool from thor_control.py). Built specifically because
the Arduino IDE Serial Monitor was fighting the user (typing not
registering, port conflicts) — this is one dedicated window for both
calibration and raw command testing, styled after thor_control.py's own
console so it should feel familiar.

Requires:  pip install pyserial
Run with:  python thor_teleop_console.py

Do NOT run the Arduino Serial Monitor at the same time — a serial port can
only be held open by one program at once.

Protocol reminder (see thor_teleop_firmware.ino for the full picture):
  JOG A+/A-/B+/B-/D+/D-/E+/E-/G+/G-  — small, precise, BLOCKING nudge.
                                        Bounded by construction — there is
                                        no async state to get stuck running.
  JOGSTEP <n>                        — set the nudge size (raw steps), live
  G<angle>                           — absolute gripper set, e.g. G0 / G90
  CAL A / CAL B / CAL D / CAL E      — select axis, start recording (does
                                        not move anything by itself — you
                                        move it with JOG while it's recording)
  CAL STOP                           — stop recording, report exact steps

  D (elbow) is wired up here for calibration/manual-jog use ONLY.

  There is no continuous live-teleop in this build and so no "stop all"
  button — every move is already short and bounded. Physical power/reset
  is the real stop mechanism if something looks wrong.

Calibration results get appended to calibration_log.jsonl next to this
script, one JSON object per line, each with a timestamp and whatever note
was in the Note field when it was recorded.
"""

import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

CAL_LOG_PATH = Path(__file__).with_name("calibration_log.jsonl")
CAL_DONE_RE = re.compile(r"\[fw\] CAL DONE axis=(\w+) steps=(-?\d+)")
CAL_START_RE = re.compile(r"\[fw\] CAL START axis=(\w+)")

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is not installed.\n\nRun:  pip install pyserial")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BAUD = 115200
BOOT_DELAY_MS = 2000   # Arduino resets when the port opens; wait for setup()


# ---------------------------------------------------------------------------
# Palette — matches thor_control.py
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
# Serial link — plain text lines out, whatever the firmware prints back in
# ---------------------------------------------------------------------------

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
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ConsoleApp:

    def __init__(self, root):
        self.root = root
        self.link = FirmwareLink()

        root.title("Thor Teleop Console")
        root.configure(bg=BG)
        root.minsize(680, 620)

        self.f_head = pick_font(["Barlow Condensed", "Oswald", "Segoe UI Semibold",
                                 "DejaVu Sans", "Helvetica"], 11, "bold")
        self.f_body = pick_font(["Segoe UI", "Inter", "DejaVu Sans", "Helvetica"], 10)
        self.f_mono = pick_font(["Cascadia Mono", "Consolas", "JetBrains Mono",
                                 "DejaVu Sans Mono", "Courier New"], 11)

        self.log = None
        self._pending_log = []

        self._build()
        self.root.after(40, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout --------------------------------------------------------

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
        self._build_calibration(outer)
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

        self.state_lamp = tk.Label(inner, text="  ", bg=LAMP_OFF, width=2)
        self.state_lamp.pack(side="right", padx=(10, 0))
        self.state_label = tk.Label(inner, text="disconnected", bg=PANEL,
                                    fg=MUTED, font=self.f_mono)
        self.state_label.pack(side="right")

        tk.Label(inner, text="Every move is short + bounded — no in-flight "
                             "motion to stop via software. Power/reset is "
                             "the real stop.", bg=PANEL, fg=MUTED,
                 font=self.f_body).pack(side="right", padx=(0, 18))

        self.refresh_ports()

    def _build_calibration(self, parent):
        wrap, body = self._section(parent, "Calibration")
        wrap.pack(fill="x", pady=(12, 0))

        def axis_row(label, axis_letter, jog_only=False):
            r = tk.Frame(body, bg=PANEL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, bg=PANEL, fg=INK, font=self.f_body,
                     width=20, anchor="w").pack(side="left")
            self._flat_button(r, "− nudge",
                              lambda: self.send(f"JOG {axis_letter}-")).pack(side="left", padx=2)
            self._flat_button(r, "+ nudge",
                              lambda: self.send(f"JOG {axis_letter}+")).pack(side="left", padx=2)
            if not jog_only:
                self._flat_button(r, "● record this axis",
                                  lambda a=axis_letter: self.send(f"CAL {a}"),
                                  accent=AMBER).pack(side="left", padx=(10, 0))

        axis_row("Axis A (base)", "A")
        axis_row("Axis B (upper elbow)", "B")
        axis_row("Axis D (elbow, CAL-only)", "D")
        axis_row("Axis E (lower rotation)", "E")
        axis_row("Gripper", "G", jog_only=True)

        steprow = tk.Frame(body, bg=PANEL)
        steprow.pack(fill="x", pady=(8, 0))
        tk.Label(steprow, text="Jog step size", bg=PANEL, fg=INK, font=self.f_body,
                 width=20, anchor="w").pack(side="left")
        for n in (50, 200, 500, 1000, 2000):
            self._flat_button(steprow, str(n),
                              lambda n=n: self.send(f"JOGSTEP {n}")).pack(side="left", padx=2)
        self.jogstep_var = tk.StringVar()
        tk.Entry(steprow, textvariable=self.jogstep_var, bg=PANEL_HI, fg=INK,
                 font=self.f_mono, insertbackground=AMBER, bd=0, width=8
                 ).pack(side="left", padx=(10, 4), ipady=3)
        self._flat_button(steprow, "set",
                          self._send_custom_jogstep).pack(side="left")

        statusrow = tk.Frame(body, bg=PANEL)
        statusrow.pack(fill="x", pady=(10, 0))
        self._flat_button(statusrow, "CAL STOP  (stop + report + save)",
                          lambda: self.send("CAL STOP"), accent=WARN).pack(side="left")
        self.rec_label = tk.Label(statusrow, text="not recording", bg=PANEL,
                                  fg=MUTED, font=self.f_mono)
        self.rec_label.pack(side="left", padx=(12, 0))

        noterow = tk.Frame(body, bg=PANEL)
        noterow.pack(fill="x", pady=(8, 0))
        tk.Label(noterow, text="Note", bg=PANEL, fg=MUTED,
                 font=self.f_body).pack(side="left")
        self.note_var = tk.StringVar()
        tk.Entry(noterow, textvariable=self.note_var, bg=PANEL_HI, fg=INK,
                 font=self.f_mono, insertbackground=AMBER, bd=0
                 ).pack(side="left", fill="x", expand=True, ipady=4, padx=(8, 0))

        tk.Label(body, text="Each nudge moves a small, exact, calibration-independent "
                             "number of raw steps. To measure: pick '● record this "
                             "axis' at your reference point, nudge to wherever you're "
                             "measuring to, then CAL STOP — the exact step count gets "
                             "saved to calibration_log.jsonl along with the Note above.",
                 bg=PANEL, fg=MUTED, font=self.f_body, anchor="w",
                 wraplength=600, justify="left").pack(fill="x", pady=(8, 0))

    def _build_console(self, parent):
        wrap, body = self._section(parent, "Console")
        wrap.pack(fill="both", expand=True, pady=(12, 0))

        self.log = tk.Text(body, height=16, bg="#101215", fg=INK,
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
        entry.focus_set()

        self._flat_button(row, "Send", self._send_typed).pack(side="left")

        quick = tk.Frame(body, bg=PANEL)
        quick.pack(fill="x", pady=(6, 0))
        tk.Label(quick, text="Quick:", bg=PANEL, fg=MUTED,
                 font=self.f_body).pack(side="left")
        self._flat_button(quick, "G0 (grip closed)",
                          lambda: self.send("G0")).pack(side="left", padx=4)
        self._flat_button(quick, "G90 (grip open)",
                          lambda: self.send("G90")).pack(side="left", padx=4)

        pending, self._pending_log = self._pending_log, []
        for text, tag in pending:
            self.write_log(text, tag)

    def _flat_button(self, parent, text, command, accent=None):
        return tk.Button(parent, text=text, command=command,
                         bg=PANEL_HI, fg=accent or INK,
                         activebackground=EDGE, activeforeground=accent or INK,
                         font=self.f_body, bd=0, padx=10, pady=6,
                         cursor="hand2", highlightthickness=0)

    # -- serial plumbing -------------------------------------------------

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
        self.write_log(f"Opened {port} at {BAUD}. Waiting "
                       f"{BOOT_DELAY_MS/1000:.1f}s for the Arduino to reset "
                       f"and run setup()...", "sys")
        self.root.after(BOOT_DELAY_MS, self._finish_connect)

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

    def _send_custom_jogstep(self):
        text = self.jogstep_var.get().strip()
        if text.isdigit():
            self.send(f"JOGSTEP {text}")
        elif text:
            self.write_log(f"'{text}' isn't a positive integer", "err")

    def _pump(self):
        while True:
            try:
                kind, payload = self.link.rx.get_nowait()
            except queue.Empty:
                break
            self.write_log(payload, "err" if kind == "error" else "rx")
            if kind != "error":
                self._watch_calibration(payload)
        self.root.after(40, self._pump)

    def _watch_calibration(self, line):
        m = CAL_START_RE.search(line)
        if m:
            self.rec_label.configure(text=f"recording axis {m.group(1)}...", fg=AMBER)
            return
        m = CAL_DONE_RE.search(line)
        if m:
            axis, steps = m.group(1), int(m.group(2))
            self.rec_label.configure(text="not recording", fg=MUTED)
            self._save_calibration(axis, steps)

    def _save_calibration(self, axis, steps):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "axis": axis,
            "steps": steps,
            "note": self.note_var.get().strip(),
        }
        try:
            with open(CAL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self.write_log(f"Saved to {CAL_LOG_PATH.name}: {entry}", "sys")
        except Exception as exc:
            self.write_log(f"Could not save calibration result: {exc}", "err")

    def write_log(self, text, tag="rx"):
        if self.log is None:
            self._pending_log.append((text, tag))
            return
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        try:
            self.link.close()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    ConsoleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
