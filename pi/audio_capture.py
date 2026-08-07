"""
Audio capture for Raspberry Pi using ALSA (via subprocess to arecord).

On the Pi we use arecord instead of sounddevice/portaudio because:
  - No need to compile portaudio from source on ARM
  - arecord is already installed on Raspbian
  - Lower overhead for continuous capture

The architecture: a background arecord process writes raw PCM to stdout,
which we read into a ring buffer. When a capture starts, we snapshot the
pre-buffer and begin accumulating new samples.
"""

import os
import subprocess
import threading
import struct
import numpy as np
from collections import deque


class ALSAAudioCapture:
    def __init__(self, device="hw:1,0", sample_rate=16000, channels=1,
                 prebuffer_sec=0.5):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.prebuffer_sec = prebuffer_sec
        self._proc = None
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread = None

        # Ring buffer for pre-buffering
        prebuf_samples = int(prebuffer_sec * sample_rate)
        self._ring = deque(maxlen=prebuf_samples)

        # Active captures
        self._captures = {}
        self._next_id = 0

    def start(self):
        """Start the arecord process and reader thread."""
        self._start_arecord()
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _start_arecord(self):
        """Launch the arecord subprocess."""
        cmd = [
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", str(self.channels),
            "-t", "raw",
            "--buffer-size", "8192",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def _reader(self):
        """Continuously read samples from arecord and distribute.
        Auto-restarts arecord if it dies unexpectedly."""
        import time
        import sys
        chunk_samples = 1024
        chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample
        restart_count = 0
        max_restarts = 50

        while self._running:
            if not self._proc or self._proc.poll() is not None:
                # arecord died — restart it
                restart_count += 1
                if restart_count > max_restarts:
                    print("[audio] Too many arecord restarts, giving up.",
                          file=sys.stderr, flush=True)
                    break
                stderr_out = ""
                if self._proc and self._proc.stderr:
                    try:
                        stderr_out = self._proc.stderr.read().decode(errors="replace")[:200]
                    except Exception:
                        pass
                print(f"[audio] arecord died (restart #{restart_count}). "
                      f"stderr: {stderr_out.strip() or '(none)'}",
                      file=sys.stderr, flush=True)
                time.sleep(2)  # brief pause before restart
                try:
                    self._start_arecord()
                    print(f"[audio] arecord restarted on {self.device}",
                          file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[audio] Failed to restart arecord: {e}",
                          file=sys.stderr, flush=True)
                    time.sleep(5)
                continue

            raw = self._proc.stdout.read(chunk_bytes)
            if not raw:
                # EOF — arecord probably died, loop will restart it
                continue

            # Reset restart counter on successful read
            restart_count = 0

            # Convert to float32
            samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32767.0
            with self._lock:
                # Feed ring buffer
                for s in samples:
                    self._ring.append(s)
                # Feed active captures
                for buf in self._captures.values():
                    buf.append(samples.copy())

    def begin_capture(self) -> int:
        """Start a new capture, seeded with the pre-buffer."""
        with self._lock:
            cid = self._next_id
            self._next_id += 1
            # Seed with pre-buffer
            prebuf = np.array(list(self._ring), dtype="float32")
            self._captures[cid] = [prebuf] if prebuf.size > 0 else []
        return cid

    def end_capture(self, cid: int) -> np.ndarray:
        """End a capture and return the audio as a float32 array."""
        with self._lock:
            chunks = self._captures.pop(cid, [])
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).astype("float32")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
