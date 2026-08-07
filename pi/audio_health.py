"""
Audio capture health monitor and auto-recovery.

Provides:
  - find_usb_audio_device() — auto-detect the correct ALSA device
  - AudioHealthMonitor — watchdog that detects silent capture and restarts arecord
"""

import os
import subprocess
import time
import threading
import re


# USB ID of our scanner sound card (C-Media Electronics)
SCANNER_USB_VENDOR = "0d8c"
SCANNER_USB_PRODUCT = "0014"


def find_usb_audio_device() -> str:
    """
    Auto-detect the ALSA device name for the scanner USB audio card.
    
    Searches /proc/asound/ for a card matching our USB vendor:product ID.
    Returns a plughw device string like "plughw:2,0" or "plughw:scanner,0".
    
    Falls back to checking all cards sequentially if procfs method fails.
    """
    # Method 1: Check /proc/asound/cards and match USB ID
    try:
        cards_path = "/proc/asound"
        if os.path.isdir(cards_path):
            for entry in os.listdir(cards_path):
                if not entry.startswith("card"):
                    continue
                card_num = entry.replace("card", "")
                usbid_path = os.path.join(cards_path, entry, "usbid")
                if os.path.exists(usbid_path):
                    with open(usbid_path) as f:
                        usbid = f.read().strip().lower()
                    if usbid == f"{SCANNER_USB_VENDOR}:{SCANNER_USB_PRODUCT}":
                        # Check if this card has a custom name (from udev rule)
                        id_path = os.path.join(cards_path, entry, "id")
                        if os.path.exists(id_path):
                            with open(id_path) as f:
                                card_id = f.read().strip()
                            if card_id and card_id != "Device":
                                return f"plughw:{card_id},0"
                        return f"plughw:{card_num},0"
    except Exception:
        pass

    # Method 2: Parse arecord -l output
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, timeout=5)
        for line in out.splitlines():
            if "USB Audio" in line or "C-Media" in line:
                m = re.search(r"card (\d+):", line)
                if m:
                    return f"plughw:{m.group(1)},0"
    except Exception:
        pass

    # Fallback: try common card numbers
    for card in range(4):
        dev = f"plughw:{card},0"
        if _test_device(dev):
            return dev

    return "plughw:1,0"  # last resort fallback


def _test_device(device: str) -> bool:
    """Quick test if an ALSA capture device is functional."""
    try:
        proc = subprocess.Popen(
            ["arecord", "-D", device, "-f", "S16_LE", "-r", "16000",
             "-c", "1", "-t", "raw", "-d", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(timeout=3)
        return proc.returncode == 0 and len(stdout) > 1000
    except Exception:
        return False


class AudioHealthMonitor:
    """
    Watchdog that monitors audio capture health.
    
    Detects when the audio stream goes silent (arecord dies or device changes)
    and restarts the capture process automatically.
    
    Usage:
        monitor = AudioHealthMonitor(audio_capture_instance)
        monitor.start()
    """

    # If we get this many consecutive empty captures, something is wrong
    EMPTY_THRESHOLD = 10
    # How often to check health (seconds)
    CHECK_INTERVAL = 30
    # Max consecutive empty-audio events before restart
    MAX_EMPTY_BEFORE_RESTART = 15

    def __init__(self, audio_capture, config_module):
        self.audio = audio_capture
        self.config = config_module
        self._stop = threading.Event()
        self._thread = None
        self._empty_count = 0
        self._last_good_capture = time.time()
        self._restarts = 0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="audio-health"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def report_capture(self, had_audio: bool):
        """Called by the scanner after each capture attempt."""
        if had_audio:
            self._empty_count = 0
            self._last_good_capture = time.time()
        else:
            self._empty_count += 1

    def _run(self):
        """Periodic health check."""
        print("[audio-health] Watchdog started.")
        while not self._stop.is_set():
            self._stop.wait(self.CHECK_INTERVAL)
            if self._stop.is_set():
                break

            # Check: have we had audio recently?
            silence_duration = time.time() - self._last_good_capture

            if self._empty_count >= self.MAX_EMPTY_BEFORE_RESTART:
                print(f"[audio-health] WARNING: {self._empty_count} consecutive "
                      f"empty captures ({silence_duration:.0f}s silence). "
                      f"Restarting audio capture...")
                self._restart_audio()
                self._empty_count = 0

    def _restart_audio(self):
        """Restart the arecord process with the correct device."""
        self._restarts += 1
        print(f"[audio-health] Restart #{self._restarts}")

        # Re-detect the correct device
        new_device = find_usb_audio_device()
        old_device = self.audio.device

        if new_device != old_device:
            print(f"[audio-health] Device changed: {old_device} -> {new_device}")
            self.audio.device = new_device

        # Stop and restart the arecord subprocess
        try:
            self.audio.stop()
        except Exception:
            pass

        time.sleep(1)

        try:
            self.audio.start()
            print(f"[audio-health] Audio capture restarted on {self.audio.device}")
            self._last_good_capture = time.time()
        except Exception as e:
            print(f"[audio-health] Failed to restart: {e}")
