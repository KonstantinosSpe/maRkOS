"""Does the arm lose steps?

The steppers are open loop: nothing reports where the arm really is, so a step lost under load is invisible and the
software's idea of the position slowly drifts from the real one. The firmware's homing is the one absolute reference: it
drives an axis until its home sensor triggers and says how many steps that took.

This runs the arm through repeated moves like the hover's (through the same controller the bridge uses), sends every
axis back to 0, then re-homes them one at a time. If the arm really is at 0, each axis reports "already at home". If it
lost steps, the axis has to travel to the sensor and the firmware says how far: that is the drift.

The home sensors are zones, not points, so a small drift can hide inside one; a first step probes how big each zone is
(move a known number of steps away, home, see what the firmware reports), so the later numbers can be read fairly.

Needs the arm homed-able and the bridge NOT running (one owner of the serial port).
Status: written but not yet run, neither on the real arm nor against the firmware simulator.

    python -m markos_vision.apps.drift_test --go             # the real arm; moves it for several minutes
    python -m markos_vision.apps.drift_test --cycles 3 9     # (dry) prints what it would do
"""
import argparse
import queue
import re
import time

from markos import hw_config as cfg
from markos.axis_controller import AxisController

HOME_RE = re.compile(r"^\[fw\] HOME (\w) (already at home|found|FAILED)(.*)$")
STEPS_RE = re.compile(r"steps=(\d+)")
# model degrees, the way the hover sends them (the sign of each axis is applied below, like the bridge does)
POSE_1 = {"E": 30.0, "D": 60.0, "B": -15.0}
POSE_2 = {"E": -20.0, "D": 70.0, "B": -25.0}
ZERO = {"E": 0.0, "D": 0.0, "B": 0.0}
PROBE_STEPS = {"D": 40, "B": 15, "E": 40}     # small known displacements for the zone check, in steps
SIGN_KEY = {"E": "E", "D": "D", "B": "BC"}


class Rig:
    def __init__(self, link, say=print):
        self.link, self.say = link, say
        self.ctl = AxisController()
        self.lines = []
        self.limit_hits = 0

    def poll(self):
        got = []
        while True:
            try:
                kind, payload = self.link.rx.get_nowait()
            except queue.Empty:
                return got
            if kind == "error":
                raise RuntimeError(payload)
            self.lines.append(payload)
            if "SOFT LIMIT" in payload:
                self.limit_hits += 1
            self.ctl.on_ack(payload)
            got.append(payload)

    def home(self, axis, timeout=110.0):
        """HOME one axis and return what the firmware said."""
        self.link.send_line(f"HOME {axis}")
        t0 = time.time()
        while time.time() - t0 < timeout:
            for line in self.poll():
                m = HOME_RE.match(line)
                if m and m.group(1) == axis:
                    s = STEPS_RE.search(line)
                    if m.group(2) != "FAILED":
                        self.ctl.mark_homed(axis)
                    return {"axis": axis, "result": m.group(2), "steps": int(s.group(1)) if s else None, "line": line}
            time.sleep(0.05)
        raise TimeoutError(f"no answer to HOME {axis} after {timeout:.0f} s")

    def nudge(self, axis, steps, timeout=20.0):
        """One MOVE of a known number of steps in the command frame, straight to the firmware (for the zone check)."""
        self.ctl.in_flight.clear()
        self.link.send_line(f"MOVE {axis}{steps}")
        t0 = time.time()
        while time.time() - t0 < timeout:
            for line in self.poll():
                if re.match(rf"^\[fw\] MOVE {axis} done", line):
                    return line
            time.sleep(0.02)
        raise TimeoutError(f"no ack for MOVE {axis}{steps}")

    def arrived(self, model_deg):
        if self.ctl.in_flight:
            return False
        for axis, deg in model_deg.items():
            want = round(deg * cfg.SIGN[SIGN_KEY[axis]] * self.ctl.STEPS_PER_DEG[axis]) if hasattr(self.ctl, "STEPS_PER_DEG") else None
            pos = self.ctl.pos[axis]
            if want is None:
                want = self.ctl.target[axis]
            if pos is None or abs(pos - want) >= cfg.DEADBAND_STEPS:
                return False
        return True

    def go(self, model_deg, timeout=90.0):
        """Chase a pose the way the bridge does: refresh the targets, send the MOVEs the controller asks for, read the acks."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            now = time.time()
            for axis, deg in model_deg.items():
                self.ctl.set_target_deg(axis, deg * cfg.SIGN[SIGN_KEY[axis]], now)
            line = self.ctl.next_move(now)
            if line:
                self.link.send_line(line)
            self.poll()
            if self.ctl.fault:
                raise RuntimeError(self.ctl.fault)
            if all(self.ctl.pos[a] is not None and abs(self.ctl.pos[a] - self.ctl.target[a]) < cfg.DEADBAND_STEPS
                   for a in model_deg) and not self.ctl.in_flight:
                return time.time() - t0
            time.sleep(0.02)
        raise TimeoutError(f"the arm did not reach {model_deg} within {timeout:.0f} s (believed at "
                           f"{ {a: self.ctl.pos[a] for a in model_deg} })")

    def hold(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.poll()
            time.sleep(0.05)

    def cycle(self):
        t0 = time.time()
        self.go(POSE_1)
        self.hold(0.8)
        self.go(POSE_2)
        self.hold(0.8)
        self.go(POSE_1)
        self.go(ZERO)
        self.hold(0.8)
        return time.time() - t0


def describe(r, axis):
    spd = cfg.stepsPerDegD if axis == "D" else cfg.stepsPerDegB if axis == "B" else cfg.stepsPerDegE
    if r["result"] == "already at home":
        return "already at home (any drift is inside the sensor's zone)"
    if r["result"] == "FAILED":
        return "HOME FAILED: " + r["line"]
    return f"had to travel {r['steps']} steps to the sensor = {r['steps'] / spd:.1f} deg   [{r['line'][5:]}]"


def run(link, cycles=(3, 9), say=print, cycle_hook=None):
    rig = Rig(link, say)
    time.sleep(cfg_boot_delay())
    link.reset_buffers()
    results = {"reference": {}, "zone": {}, "blocks": [], "seconds_per_cycle": []}

    say("== 0. Homing every axis: the reference. (The arm moves; D can take up to 40 s.)")
    for axis in "BDE":
        r = rig.home(axis)
        results["reference"][axis] = r
        say(f"   {axis}: {r['line'][5:]}")

    say("\n== 1. Sensor zone check: move a known small distance away from home, home again, see what the firmware says.")
    for axis in "DBE":
        for sign in (+1, -1):
            k = sign * PROBE_STEPS[axis]
            rig.nudge(axis, k)
            r = rig.home(axis)
            results["zone"][(axis, k)] = r
            say(f"   {axis} {k:+d} steps away -> {r['line'][5:]}")

    for block, n in enumerate(cycles, 1):
        say(f"\n== {block + 1}. {n} back-and-forth cycles (home -> pose 1 -> pose 2 -> pose 1 -> home), then re-home each axis.")
        for i in range(n):
            secs = rig.cycle()
            results["seconds_per_cycle"].append(secs)
            say(f"   cycle {i + 1}/{n} took {secs:.0f} s; the controller believes B/D/E are at "
                f"{[rig.ctl.pos[a] for a in 'BDE']} steps (should be 0)")
            if cycle_hook:
                cycle_hook(block, i)
        found = {}
        for axis in "BDE":
            r = rig.home(axis)
            found[axis] = r
            say(f"   {axis}: {describe(r, axis)}")
        results["blocks"].append({"cycles": n, "found": found})
    results["limit_hits"] = rig.limit_hits
    results["lines"] = rig.lines
    return results


def cfg_boot_delay():
    from markos.serial_link import HW_BOOT_DELAY_MS
    return HW_BOOT_DELAY_MS / 1000.0


def summary(results, say=print):
    say("\n== Summary ==")
    zone = {}
    for (axis, k), r in results["zone"].items():
        zone.setdefault(axis, []).append((k, r))
    for axis in "DBE":
        bits = []
        for k, r in zone.get(axis, []):
            bits.append(f"{k:+d} steps -> " + ("at home" if r["result"] == "already at home" else f"{r['steps']} steps to find it"))
        say(f"  {axis} sensor zone check: " + "; ".join(bits))
    for b in results["blocks"]:
        say(f"  after {b['cycles']} cycles:")
        for axis in "BDE":
            say(f"    {axis}: {describe(b['found'][axis], axis)}")
    if results["limit_hits"]:
        say(f"  soft-limit hits during the test: {results['limit_hits']}")
    if results["seconds_per_cycle"]:
        say(f"  a cycle takes about {sum(results['seconds_per_cycle']) / len(results['seconds_per_cycle']):.0f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--cycles", type=int, nargs="+", default=[3, 9])
    p.add_argument("--go", action="store_true", help="really run it: the arm homes and then makes repeated moves")
    args = p.parse_args()
    total = sum(args.cycles)
    print(f"Plan: home all axes, a sensor-zone check, then {args.cycles} cycles (about {total * 22 // 60 + 2} minutes of moving), "
          f"re-homing after each block to read the drift.")
    if not args.go:
        print("Dry run: nothing moved. Add --go to run it on the arm.")
        return
    from markos.serial_link import FirmwareLink
    link = FirmwareLink()
    link.open(args.port)
    try:
        results = run(link, tuple(args.cycles))
        summary(results)
    except KeyboardInterrupt:
        print("\nStopped by you. The arm holds where it is; the software's position is no longer trustworthy: re-home.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
