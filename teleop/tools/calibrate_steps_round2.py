#!/usr/bin/env python3
"""
calibrate_steps_round2.py — targeted recheck + E's missing '+' side
=======================================================================
Second calibration pass, saved to its own dataset (step_accuracy_data_v2.json)
rather than the round-1 file — same round-trip method as calibrate_steps.py
(HOME -> MOVE N -> HOME, comparing commanded N against the real return
distance), but a much smaller, deliberately targeted point set instead of
the full 1/50 ladder:

  D  (-): 34, 68, 102, 136 — round 1's two flagged outliers (34, 68,
     adjacent rungs 1-2 of that ladder) plus two more rungs beyond the
     cluster on each side, to see whether the anomaly is localized or
     part of a broader pattern. Rung 0 and below don't exist, so this
     side is naturally clipped rather than symmetric.
     Also 1530, 1564, 1598, 1632, 1666 — a THIRD outlier, found only
     after combining round 1 + this file into the v3 "complete picture"
     dataset: D-1598 (rung 47 of 50) came back discrepancy=-53, near the
     far end of D's range where the two near-origin outliers' explanation
     (the beacon's resolution floor) doesn't apply — this one needs its
     own direct recheck, same +/-2-rungs pattern as before.
  B  (-): 10, 14, 17, 20, 24 — rung 5 of that ladder (17, also flagged)
     +/- 2 rungs.
  E  (-): 50, 60, 70, 80, 90 — rung 7 of that ladder +/- 2 rungs.
  E  (+): the FULL 50-point ladder up to 502 — round 1 only ever tested
     E's '-' side; '+' was never covered at all. Safe to test now that
     doHomeSearch's E branch picks whichever direction is actually
     shorter rather than always searching '+' — the outbound direction
     doesn't need to be handled as a special case anymore.

Requires: pip install pyserial (see requirements/teleop.txt)
Run:      python teleop/tools/calibrate_steps_round2.py
"""

import json
import re
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

BAUD = 115200
OUT_PATH = Path(__file__).resolve().parents[1] / "measurements" / "step_accuracy_data_v2.json"


def ladder(full_way, n=50):
    """n evenly spaced magnitudes from full_way/n up to full_way,
    deduplicated (small full_way values can round to repeats)."""
    seen = []
    for i in range(1, n + 1):
        m = round(full_way * i / n)
        if m > 0 and (not seen or seen[-1] != m):
            seen.append(m)
    return seen


PLAN = {
    "D": {"-": [34, 68, 102, 136, 1530, 1564, 1598, 1632, 1666]},
    "B": {"-": [10, 14, 17, 20, 24]},
    "E": {"-": [50, 60, 70, 80, 90], "+": ladder(502)},
}

HOME_FOUND_RE = re.compile(r"\[fw\] HOME (\w) found.*steps=(-?\d+)")
HOME_ALREADY_RE = re.compile(r"\[fw\] HOME (\w) already at home, steps=(-?\d+)")
MOVE_DONE_RE = re.compile(r"\[fw\] MOVE (\w) done, moved (-?\d+)")


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
    # Typed the device name directly (e.g. "COM5") instead of its index.
    for p in ports:
        if p.device.upper() == raw.upper():
            return p.device
    print(f"'{raw}' isn't one of the listed ports — using it anyway.")
    return raw


class LineReader:
    """Wraps a serial port and hands back one complete line at a time,
    carrying any leftover bytes forward across calls in self.buf — see
    calibrate_steps.py for why a naive reset-every-call version silently
    drops data."""

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


def send(ser, text):
    print("  >", text)
    ser.write((text + "\n").encode("ascii"))


def home(reader, axis, timeout_s):
    send(reader.ser, f"HOME {axis}")
    deadline = time.time() + timeout_s
    while True:
        line = reader.read_line(deadline)
        if line is None:
            return None, "timeout"
        print("  <", line)
        m = HOME_FOUND_RE.match(line)
        if m and m.group(1) == axis:
            return int(m.group(2)), "found"
        m = HOME_ALREADY_RE.match(line)
        if m and m.group(1) == axis:
            return int(m.group(2)), "already"
        if line.startswith(f"[fw] HOME {axis} FAILED"):
            return None, "failed"


def move(reader, axis, n, timeout_s=15.0):
    send(reader.ser, f"MOVE {axis}{n}")
    deadline = time.time() + timeout_s
    while True:
        line = reader.read_line(deadline)
        if line is None:
            return None
        print("  <", line)
        m = MOVE_DONE_RE.match(line)
        if m and m.group(1) == axis:
            return int(m.group(2))


def run_point(reader, axis, direction, magnitude):
    commanded = magnitude if direction == "+" else -magnitude
    result = {"axis": axis, "direction": direction, "commanded": magnitude,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    _, status = home(reader, axis, timeout_s=60.0)
    if status not in ("found", "already"):
        result["error"] = f"pre-home {status}"
        return result

    moved_out = move(reader, axis, commanded)
    if moved_out is None:
        result["error"] = "move timeout"
        return result
    result["moved_out"] = moved_out

    actual_return, status = home(reader, axis, timeout_s=60.0)
    if status != "found":
        result["error"] = f"return {status}"
        return result

    result["actual_return"] = actual_return
    result["discrepancy"] = abs(moved_out) - actual_return
    return result


def main():
    port = pick_port()
    ser = serial.Serial(port, BAUD, timeout=0.2)
    time.sleep(2.0)
    ser.reset_input_buffer()
    reader = LineReader(ser)
    print(f"Connected to {port}.")

    results = []
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"Resuming — {len(results)} points already in {OUT_PATH.name}")
        except Exception:
            pass

    done = {(r["axis"], r["direction"], r["commanded"]) for r in results if "error" not in r}

    total = sum(len(mags) for plan in PLAN.values() for mags in plan.values())
    n = len(done)
    for axis, plan in PLAN.items():
        for direction, magnitudes in plan.items():
            for mag in magnitudes:
                key = (axis, direction, mag)
                if key in done:
                    continue
                n += 1
                print(f"\n=== [{n}/{total}] {axis} {direction}{mag} ===")
                r = run_point(reader, axis, direction, mag)
                print("  result:", r)
                results.append(r)
                OUT_PATH.write_text(json.dumps(results, indent=2))

    ser.close()
    ok = [r for r in results if "error" not in r]
    print(f"\nDone. {len(results)} points total ({len(ok)} clean) saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
