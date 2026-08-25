#!/usr/bin/env python3
"""
calibrate_steps.py — coarse step-accuracy sweep for D, B, E
=============================================================
Measures real step accuracy using the homing beacons as ground truth,
instead of trusting the open-loop step counters blindly.

For each test point (axis, direction, commanded magnitude):
  1. HOME <axis>   — establish a true zero via the beacon.
  2. MOVE <axis> N — travel exactly N commanded steps outward.
  3. HOME <axis>   — search back to true zero; the ACTUAL number of
                     steps that search takes is the real answer to
                     "how far did the arm actually travel."

Comparing commanded N against the actual return distance reveals real
step loss/gain at that point in the range — this is the raw data a
future correction "matrix" would be built from. Results are saved
after every point (not just at the end) to
teleop/measurements/step_accuracy_data.json, so a
run that's interrupted partway through doesn't lose anything already
measured.

Full systematic ladder (this version): 50 evenly spaced points per
direction, at 1/50, 2/50, ... 50/50 of each direction's real, confirmed
full range — not the earlier coarse hand-picked magnitude lists. Full
ranges used, all directly hand-verified or measured, not assumed:
  D: 1500 steps '+', 1700 steps '-' (D_SOFT_MAX/MIN — hand-tested safe
     range, home is centered). Run '+' completely before starting '-'.
  B: 170 steps '-' only (B_SOFT_MIN — home is at one edge, backed by the
     real LIM_B hardware switch; real physical range is 180 steps
     edge-to-edge, hand-measured, with a small margin subtracted).
  E: 502 steps '-' only — HALF of the real measured full lap (1004 steps,
     via the firmware's SWEEP E+/E-, replacing the old never-reverified
     2200 assumption). Capped at half, not the full lap: past 180°, an
     outbound '-' move retraces the same physical positions the long way
     around instead of covering new ground, and it's exactly where the
     return search's shortest-path estimate starts flip-flopping between
     directions (confirmed — an earlier run testing past 502 produced
     wild, inconsistent discrepancies right around there).

B and E's outbound direction is '-' for the same reason as before: their
homing search finds the beacon by searching '+', so '-' outbound is what
puts the beacon on the short, direct path back (see doHomeSearch's E
branch in the firmware for the exception — E's search is now smart about
picking whichever direction is actually shorter, but the outbound test
move itself still needs to go somewhere, and '-' keeps results comparable
to the coarse-pass data already collected).

This is a LOT of points (100 + 50 + 50 = 200), each needing two homing
searches plus a move — expect a long, unattended run. Results save after
every single point, and a rerun skips whatever's already recorded, so an
interrupted run loses nothing.

Requires: pip install pyserial (see requirements/teleop.txt)
Run:      python teleop/tools/calibrate_steps.py
"""

import json
import re
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

BAUD = 115200
OUT_PATH = Path(__file__).resolve().parents[1] / "measurements" / "step_accuracy_data.json"


def ladder(full_way, n=50):
    """n evenly spaced magnitudes from full_way/n up to full_way,
    deduplicated (small full_way values can round to repeats)."""
    seen = []
    for i in range(1, n + 1):
        m = round(full_way * i / n)
        if m > 0 and (not seen or seen[-1] != m):
            seen.append(m)
    return seen


# Full ranges — see module docstring for where each number comes from.
# E is capped at HALF a rotation (502, not the full 1004): past that point
# an outbound '-' move is retracing the same physical positions the long
# way around rather than covering new ground, and it's exactly where the
# return search's shortest-path estimate starts flip-flopping between
# directions — capping here avoids the ambiguity outright rather than
# trying to out-think it.
PLAN = {
    "D": {"+": ladder(1500), "-": ladder(1700)},
    "B": {"-": ladder(170)},
    "E": {"-": ladder(502)},
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
    carrying any leftover bytes forward across calls in self.buf. A naive
    version that resets its buffer on every call silently drops data
    whenever two lines arrive in the same underlying read() — which is
    common here, since the firmware often prints an intermediate
    "LIMIT HIT mid-move" line immediately followed by the actual
    "HOME <axis> found..." line with no delay between them. That was
    exactly what caused early runs to time out waiting for a "found" line
    that had, in fact, already arrived and been thrown away."""

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
                continue  # blank line — keep looking, don't burn a retry
            if time.time() >= deadline:
                return None
            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk


def send(ser, text):
    print("  >", text)
    ser.write((text + "\n").encode("ascii"))


def home(reader, axis, timeout_s):
    """Sends HOME <axis>, waits for its completion line. Returns
    (actual_steps, status) — status is 'found', 'already', 'failed', or
    'timeout'."""
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
    """Sends MOVE <axis><n>, waits for its completion line. Returns the
    actual signed steps completed, or None on timeout."""
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
    time.sleep(2.0)  # board resets when the port opens — let it boot
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
