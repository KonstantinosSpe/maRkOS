#!/usr/bin/env python3
"""
axis_controller.py — decides which MOVE goes out next, with no ROS in it
============================================================================
The bridge node hands this a target per axis and asks "what should I send
now?". Keeping that decision free of rclpy is what lets the same code be
run against sim_firmware.py in a plain pytest, so the safety behaviour
(rate cap, limits, blocked-direction memory, lost-ack handling) is checked
before it's ever pointed at the real arm.

Positions are in steps, in the command frame — the frame MOVE is sent in.
For an axis whose firmware direction is flipped (E), the firmware reports
its acks in the opposite sign; on_ack() converts back so pos never has to
care.
"""

import math
import re
from typing import Dict, Optional

from . import hw_config as cfg

STEPS_PER_DEG = {
    "A": cfg.stepsPerDegA,
    "B": cfg.stepsPerDegB,
    "D": cfg.stepsPerDegD,
    "E": cfg.stepsPerDegE,
}

MOVE_JOG_DONE_RE = re.compile(r"^\[fw\] (MOVE|JOG) (\w) done, moved (-?\d+)")


def _sign(n: float) -> int:
    return (n > 0) - (n < 0)


class AxisController:
    def __init__(self) -> None:
        # None until that axis has been homed this run — there's no
        # trustworthy reference to report an angle from, or drive to,
        # before that.
        self.pos: Dict[str, Optional[int]] = {a: None for a in STEPS_PER_DEG}
        self.target: Dict[str, Optional[int]] = {a: None for a in cfg.ARMED_AXES}
        self.blocked_dir: Dict[str, int] = {a: 0 for a in cfg.ARMED_AXES}
        self.in_flight: Dict[str, int] = {}
        self._sent_t = 0.0
        self._target_t = 0.0
        self._last_grip: Optional[int] = None
        self._last_grip_t = float("-inf")
        self.fault: Optional[str] = None
        self._fault_reported = False

    # ── state the node feeds in ────────────────────────────────────────────

    def mark_homed(self, axis: str) -> None:
        self.pos[axis] = 0
        if axis in self.target:
            self.target[axis] = None
            self.blocked_dir[axis] = 0

    def invalidate(self) -> None:
        """Forget where everything is. Used when an ack goes missing: the
        firmware may or may not have finished that move, so every counter
        is suspect until the arm is homed again."""
        for axis in self.pos:
            self.pos[axis] = None
        self.cancel_targets()
        self.in_flight.clear()

    def cancel_targets(self) -> None:
        for axis in self.target:
            self.target[axis] = None

    def set_target_deg(self, axis: str, deg: float, now: float) -> bool:
        """Returns True if the request had to be pulled back inside the
        axis's allowed range."""
        if axis not in self.target:
            return False
        lo, hi = cfg.STEP_LIMITS[axis]
        wanted = round(deg * STEPS_PER_DEG[axis])
        clamped = max(lo, min(hi, wanted))
        self.target[axis] = clamped
        self._target_t = now
        return clamped != wanted

    def on_ack(self, line: str) -> None:
        m = MOVE_JOG_DONE_RE.match(line)
        if not m:
            return
        kind, axis, moved = m.group(1), m.group(2), int(m.group(3))
        if self.pos.get(axis) is None:
            return
        delta = -moved if cfg.FW_DIR_FLIP.get(axis, False) else moved
        self.pos[axis] += delta

        asked = self.in_flight.pop(axis, None) if kind == "MOVE" else None
        if asked is None or axis not in self.blocked_dir:
            return
        if delta == 0:
            # Hit a limit or beacon: pushing the same way again just
            # bumps it again, so leave that direction alone until the
            # target is somewhere else.
            self.blocked_dir[axis] = _sign(asked)
            return
        self.blocked_dir[axis] = 0
        if _sign(delta) != _sign(asked):
            self.fault = (
                f"{axis} moved {delta:+d} steps after MOVE {axis}{asked:+d} — "
                "FW_DIR_FLIP in hw_config.py doesn't match the firmware"
            )
        elif abs(delta) > abs(asked):
            self.fault = f"{axis} moved {delta:+d} steps but only {asked:+d} were asked for"

    # ── what to send ───────────────────────────────────────────────────────

    def next_move(self, now: float) -> Optional[str]:
        if self.fault is not None:
            return None

        if self.in_flight:
            if now - self._sent_t > cfg.MOVE_ACK_TIMEOUT:
                axes = ",".join(self.in_flight)
                self.fault = (
                    f"no ack for MOVE on {axes} after {cfg.MOVE_ACK_TIMEOUT:.0f}s — "
                    "position unknown, re-home before moving again"
                )
                self.invalidate()
            return None

        if now - self._target_t > cfg.COMMAND_TIMEOUT:
            self.cancel_targets()
            return None

        parts = {}
        for axis in cfg.ARMED_AXES:
            pos, tgt = self.pos[axis], self.target[axis]
            if pos is None or tgt is None:
                continue
            err = tgt - pos
            if abs(err) < cfg.DEADBAND_STEPS:
                continue
            if self.blocked_dir[axis] == _sign(err):
                continue
            self.blocked_dir[axis] = 0
            cap = cfg.MAX_MOVE_STEPS[axis]
            parts[axis] = max(-cap, min(cap, err))

        if not parts:
            return None
        self.in_flight = parts
        self._sent_t = now
        return "MOVE " + " ".join(f"{axis}{n}" for axis, n in parts.items())

    def gripper_line(self, openness: float, now: float) -> Optional[str]:
        """openness 0.0 = closed, 1.0 = open. Only returns a command when
        the servo angle would actually change, and not faster than the servo
        can follow anyway."""
        openness = max(0.0, min(1.0, openness))
        angle = round(cfg.GRIP_CLOSED + openness * (cfg.GRIP_OPEN - cfg.GRIP_CLOSED))
        if angle == self._last_grip or now - self._last_grip_t < 0.1:
            return None
        self._last_grip = angle
        self._last_grip_t = now
        return f"G{angle}"

    # ── state the node reads out ───────────────────────────────────────────

    def take_fault(self) -> Optional[str]:
        """The fault message, once. The fault itself stays latched — motion
        stays refused — until clear_fault()."""
        if self.fault is None or self._fault_reported:
            return None
        self._fault_reported = True
        return self.fault

    def clear_fault(self) -> None:
        self.fault = None
        self._fault_reported = False

    def position_deg(self, axis: str) -> Optional[float]:
        steps = self.pos.get(axis)
        return None if steps is None else steps / STEPS_PER_DEG[axis]

    def position_rad(self, axis: str) -> Optional[float]:
        deg = self.position_deg(axis)
        return None if deg is None else math.radians(deg)
