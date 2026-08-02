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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def _reader(self):
        """Continuously read samples from arecord and distribute."""
        chunk_samples = 1024
        chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample
        while self._running and self._proc:
            raw = self._proc.stdout.read(chunk_bytes)
            if not raw:
                break
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
