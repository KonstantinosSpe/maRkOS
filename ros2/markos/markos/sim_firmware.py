#!/usr/bin/env python3
"""
sim_firmware.py — a stand-in for the Arduino, speaking the same protocol
============================================================================
Same surface as FirmwareLink (open/close/send_line/rx), so the bridge node
can run against it with `-p mock:=true` when the arm isn't plugged in. It
copies the parts of thor_teleop_firmware.ino that the bridge's safety logic
actually leans on: MOVE is blocking and one axis at a time, replies are
in the *physical* direction (so a flipped axis reports the opposite sign),
and the soft limits stop a move early.

It's a check on the bridge's own logic, not on the firmware — it can't say
anything about real step accuracy or whether a sign flag is right for the
hardware.
"""

import queue
import re
import threading
import time
from typing import Dict, Optional, Tuple

STEPS_PER_SECOND = 230.0     # roughly the firmware's default ramp speed
HOME_SECONDS = 1.0

SOFT_LIMITS = {"B": (-170, 10), "D": (-1700, 1500)}
NO_LIMIT = (-2_000_000_000, 2_000_000_000)

_AXIS_CMD_RE = re.compile(r"([ABDE])(-?\d+)")


class SimFirmware:
    def __init__(
        self,
        speedup: float = 1.0,
        flips: Optional[Dict[str, bool]] = None,
        start_pos: Optional[Dict[str, int]] = None,
        walls: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        self.speedup = speedup
        self.flips = {"A": False, "B": False, "D": False, "E": True}
        self.flips.update(flips or {})
        # Physical position counters, like posA/posB/posD/posE in the .ino.
        self.pos = {"A": 0, "B": 0, "D": 0, "E": 0}
        self.pos.update(start_pos or {})
        # Extra stops beyond the soft limits, for testing a blocked axis.
        self.limits = {**SOFT_LIMITS, **(walls or {})}
        self.gripper = 105
        self.log = []

        self.rx: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._cmds: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_open(self) -> bool:
        return self._thread is not None

    def open(self, port: str) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def reset_buffers(self) -> None:
        pass

    def send_line(self, text: str) -> bool:
        if not self.is_open:
            return False
        self.log.append(text)
        self._cmds.put(text)
        return True

    # ── the firmware side ──────────────────────────────────────────────────

    def _worker(self) -> None:
        # One command at a time, start to finish — the real firmware is
        # single-threaded and blocks inside every move.
        while not self._stop.is_set():
            try:
                line = self._cmds.get(timeout=0.05)
            except queue.Empty:
                continue
            self._handle(line.strip())

    def _wait(self, steps: int) -> None:
        time.sleep(abs(steps) / STEPS_PER_SECOND / self.speedup)

    def _reply(self, text: str) -> None:
        self.rx.put(("line", text))

    def _handle(self, line: str) -> None:
        if line.startswith("MOVE "):
            for axis, n in _AXIS_CMD_RE.findall(line[5:]):
                moved = self._move(axis, int(n))
                self._reply(f"[fw] MOVE {axis} done, moved {moved}")
        elif line.startswith("HOME "):
            axis = line[5:].strip()
            if axis in self.pos:
                time.sleep(HOME_SECONDS / self.speedup)
                self.pos[axis] = 0
                self._reply(f"[fw] HOME {axis} found")
        elif len(line) >= 2 and line[0] == "G" and line[1:].isdigit():
            self.gripper = max(60, min(105, int(line[1:])))
            self._reply(f"[fw] grip -> {self.gripper}")

    def _move(self, axis: str, n: int) -> int:
        physical = -n if self.flips[axis] else n
        lo, hi = self.limits.get(axis, NO_LIMIT)
        step = 1 if physical >= 0 else -1
        done = 0
        for _ in range(abs(physical)):
            nxt = self.pos[axis] + step
            if nxt < lo or nxt > hi:
                self._reply("[fw] SOFT LIMIT HIT mid-move - stopping early")
                break
            self.pos[axis] = nxt
            done += step
        self._wait(done)
        return done
