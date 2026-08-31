#!/usr/bin/env python3
"""
serial_link.py — the one connection to the firmware, shared by everything
============================================================================
FirmwareLink owns the serial port. It reads in a background thread and
hands finished lines back through a queue, so the GUI thread can poll it
without ever blocking on I/O — Tkinter has no good way to "await" a slow
call, so this is the usual pattern for mixing a blocking-by-nature serial
port with a GUI event loop.

Requires: pip install pyserial
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
        """Runs in its own thread for as long as the port is open. A single
        read error doesn't take this thread down — it logs the problem
        (throttled, so a fault that keeps repeating doesn't flood the
        console) and just keeps trying, since something like a stray
        voltage glitch briefly upsetting the OS-level COM port is usually
        transient and shouldn't force a full manual reconnect to recover
        from."""
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

    def send_line(self, text: str) -> bool:
        if not self.is_open:
            return False
        try:
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(("error", f"Serial write failed: {exc}"))
            return False
