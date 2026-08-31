#!/usr/bin/env python3
"""
serial_link.py — the one connection to the firmware, ported from mark1os.py
============================================================================
Identical to the Tkinter app's serial_link.py: reads in a background
thread and hands finished lines back through a queue, so whatever's
driving it (a GUI event loop there, a ROS2 timer here) never blocks on
serial I/O directly.
"""

import queue
import threading
import time
from typing import Optional, Tuple

try:
    import serial
except ImportError:
    raise SystemExit("pyserial is not installed.\n\nRun:  pip install pyserial") from None

HW_BAUD = 115200
HW_BOOT_DELAY_MS = 2000


class FirmwareLink:
    def __init__(self) -> None:
        self.ser: Optional["serial.Serial"] = None
        self.rx: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self, port: str) -> None:
        self.ser = serial.Serial(port, HW_BAUD, timeout=0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self) -> None:
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

    def _reader(self) -> None:
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

    def reset_buffers(self) -> None:
        # Whatever the Arduino printed while it was booting (and any
        # half-line from before the port was opened) isn't an answer to
        # anything we sent.
        if self.ser is None:
            return
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def send_line(self, text: str) -> bool:
        if not self.is_open:
            return False
        try:
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False
