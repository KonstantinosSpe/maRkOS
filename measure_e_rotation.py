#!/usr/bin/env python3
"""
measure_e_rotation.py — measures E's true full-rotation step count
======================================================================
Standalone, minimal script: starting from home, sweeps E a full lap in
each direction (using the firmware's SWEEP E+/E- command, which forces
a specific direction rather than picking whichever looks shorter) and
reports how many steps a real full rotation actually takes, each way.

This is the "get the numbers first" step before building the full 1/50
ladder test for E — the ladder's point spacing depends on knowing E's
real full-rotation distance rather than assuming the old 2200 figure.

Requires: pip install pyserial (already in .venv)
Run:      .venv\\Scripts\\python.exe measure_e_rotation.py
"""

import re
import sys
import time

import serial
from serial.tools import list_ports

BAUD = 115200
SWEEP_RE = re.compile(r"\[fw\] SWEEP E found, zeroed, steps=(-?\d+)")


class LineReader:
    """See calibrate_steps.py for why this needs to persist its buffer
    across calls rather than resetting it every time."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""

    def read_line(self, deadline):
        while True:
            if b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                line = raw.decode("ascii", "replace").strip()
                if line:
                    return line
                continue
            if time.time() >= deadline:
                return None
            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk


def pick_port():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} - {p.description}")
    raw = input("Port index or name (e.g. 0 or COM5): ").strip()
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(ports):
            return ports[idx].device
        print(f"No port at index {idx}.")
        sys.exit(1)
    for p in ports:
        if p.device.upper() == raw.upper():
            return p.device
    print(f"'{raw}' isn't one of the listed ports — using it anyway.")
    return raw


def send(ser, text):
    print("  >", text)
    ser.write((text + "\n").encode("ascii"))


def wait_for(reader, prefix, timeout_s):
    deadline = time.time() + timeout_s
    while True:
        line = reader.read_line(deadline)
        if line is None:
            return None
        print("  <", line)
        if line.startswith(prefix):
            return line


def home_e(reader):
    send(reader.ser, "HOME E")
    line = wait_for(reader, "[fw] HOME E", 60.0)
    if line is None:
        print("Timed out waiting for HOME E — check the connection.")
        sys.exit(1)
    if "FAILED" in line:
        print("HOME E failed — can't start the sweep without a confirmed true home.")
        sys.exit(1)


def sweep_e(reader, sign):
    send(reader.ser, f"SWEEP E{sign}")
    line = wait_for(reader, "[fw] SWEEP E", 90.0)
    if line is None:
        print(f"Timed out waiting for SWEEP E{sign}.")
        return None
    m = SWEEP_RE.match(line)
    if m:
        return int(m.group(1))
    print(f"SWEEP E{sign} did not find the beacon within one lap + margin.")
    return None


def main():
    port = pick_port()
    ser = serial.Serial(port, BAUD, timeout=0.2)
    time.sleep(2.0)
    ser.reset_input_buffer()
    reader = LineReader(ser)
    print(f"Connected to {port}.\n")

    print("=== Establishing true home before the sweep ===")
    home_e(reader)

    print("\n=== SWEEP E+ (full lap forward) ===")
    plus = sweep_e(reader, "+")

    print("\n=== Re-homing before the other direction ===")
    home_e(reader)

    print("\n=== SWEEP E- (full lap backward) ===")
    minus = sweep_e(reader, "-")

    ser.close()

    print("\n=== Results ===")
    print(f"  + direction: {plus if plus is not None else 'FAILED'} steps")
    print(f"  - direction: {minus if minus is not None else 'FAILED'} steps")
    if plus is not None and minus is not None and (plus + minus) > 0:
        print(f"  difference: {abs(plus - minus)} steps ({100*abs(plus-minus)/((plus+minus)/2):.1f}% of average)")


if __name__ == "__main__":
    main()
