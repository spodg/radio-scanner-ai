"""
BCD436HP serial interface for Raspberry Pi.
Same GLG polling logic as the PC version, just Linux serial paths.
"""

import time
import threading
from dataclasses import dataclass, field

import serial


@dataclass
class ReceptionState:
    active: bool = False
    frequency: str = ""
    modulation: str = ""
    system_name: str = ""
    group_name: str = ""
    channel_name: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.system_name, self.group_name, self.channel_name) if p]
        return " / ".join(parts) if parts else "(unknown)"


class ScannerSerial:
    def __init__(self, port, baud=115200, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None

    def open(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(0.2)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def command(self, cmd: str) -> str:
        if not self._ser:
            raise RuntimeError("Serial port not open.")
        self._ser.reset_input_buffer()
        self._ser.write((cmd + "\r").encode("ascii"))
        raw = self._ser.read_until(b"\r")
        return raw.decode("ascii", errors="replace").strip("\r\n ")

    def get_model(self) -> str:
        resp = self.command("MDL")
        return resp.split(",", 1)[1] if "," in resp else resp

    def glance(self) -> ReceptionState:
        resp = self.command("GLG")
        return self._parse_glg(resp)

    @staticmethod
    def _parse_glg(resp: str) -> ReceptionState:
        st = ReceptionState()
        if not resp or not resp.startswith("GLG"):
            return st
        fields = resp.split(",")
        if len(fields) < 13:
            def g(i):
                return fields[i].strip() if i < len(fields) else ""
            st.frequency = g(1)
            st.modulation = g(2)
            st.system_name = g(5)
            st.group_name = g(6)
            st.channel_name = g(7)
            sql = g(8)
            st.tags = {"sql": sql, "mut": g(9)}
            st.active = (sql == "1") and bool(st.frequency)
            return st

        left = fields[:5]
        right = fields[-5:]
        middle = fields[5:-5]

        st.frequency = left[1].strip()
        st.modulation = left[2].strip()
        st.system_name = middle[0].strip() if len(middle) >= 1 else ""
        st.group_name = middle[1].strip() if len(middle) >= 2 else ""
        st.channel_name = ",".join(middle[2:]).strip() if len(middle) >= 3 else ""

        sql = right[0].strip()
        st.active = (sql == "1") and bool(st.frequency)
        st.tags = {
            "att": left[3].strip(),
            "ctcss_dcs": left[4].strip(),
            "sql": sql,
            "mut": right[1].strip(),
            "sys_tag": right[2].strip(),
            "chan_tag": right[3].strip(),
            "p25nac": right[4].strip(),
        }
        return st


class ScannerPoller(threading.Thread):
    def __init__(self, scanner, interval, on_start, on_stop, min_duration=0.0):
        super().__init__(daemon=True)
        self.scanner = scanner
        self.interval = interval
        self.on_start = on_start
        self.on_stop = on_stop
        self.min_duration = min_duration
        self._stop_evt = threading.Event()
        self._active = False
        self._active_state = None
        self._active_since = 0.0
        # Stuck channel watchdog
        self.max_hold_sec = 60  # max seconds before forcing channel change
        self._avoided = {}  # freq -> avoid_until timestamp

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                state = self.scanner.glance()
            except Exception as e:
                print(f"[poller] serial error: {e}")
                time.sleep(1.0)
                continue

            now = time.time()
            if state.active and not self._active:
                self._active = True
                self._active_state = state
                self._active_since = now
                self.on_start(state)
            elif self._active and not state.active:
                duration = now - self._active_since
                ended = self._active_state
                self._active = False
                self._active_state = None
                if duration >= self.min_duration:
                    self.on_stop(ended, duration)
            elif self._active and state.active:
                self._active_state = state
                # Stuck channel watchdog: if on same freq too long, temp avoid it
                duration = now - self._active_since
                if duration > self.max_hold_sec:
                    freq = state.frequency
                    name = state.display_name
                    print(f"[watchdog] Stuck on {freq} ({name}) for {duration:.0f}s, sending temp L/O")
                    # First, end the current capture so audio is saved
                    ended = self._active_state
                    self._active = False
                    self._active_state = None
                    if duration >= self.min_duration:
                        self.on_stop(ended, duration)
                    # Now send temporary lockout
                    try:
                        self.scanner.command("KEY,L/O,P")
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"[watchdog] Error sending L/O: {e}")
                    self._avoided[freq] = now + 300  # record for logging

            # Periodic screen update callback (if set)
            if hasattr(self, 'on_poll') and self.on_poll:
                self.on_poll(state)

            self._stop_evt.wait(self.interval)
