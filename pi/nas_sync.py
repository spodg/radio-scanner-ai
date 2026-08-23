"""
Lightweight NAS clip sync for Pi 3 (1GB RAM).

Moves clips from LOCAL_CLIPS_DIR to CLIPS_DIR (NAS) when NAS is available.
Designed to use minimal memory: no os.walk, no recursive scans, processes
one date folder at a time with sleeps between files.

Also provides get_clips_dir() for the scanner to decide where to write.
"""

import os
import time
import shutil
import threading
from pathlib import Path

import config
import scanner_db

# =============================================================================
# State
# =============================================================================
_nas_online = False


def nas_available() -> bool:
    return _nas_online


def get_clips_dir() -> str:
    """Return NAS clips dir if available, else local."""
    if _nas_online:
        return config.CLIPS_DIR
    return config.LOCAL_CLIPS_DIR


# =============================================================================
# NAS check (fast, no allocations)
# =============================================================================
def _check_nas() -> bool:
    """Check NAS mount + write access."""
    try:
        if not os.path.ismount(config.NAS_MOUNT):
            return False
        # Quick write test
        test = os.path.join(config.CLIPS_DIR, ".nas_ok")
        with open(test, "w") as f:
            f.write("1")
        os.remove(test)
        return True
    except (OSError, IOError):
        return False


# =============================================================================
# Sync worker
# =============================================================================
class NasSyncWorker:
    """
    Ultra-lightweight background sync.
    
    Every NAS_CHECK_INTERVAL seconds:
      1. Check if NAS is mounted (single ismount call)
      2. If yes and local clips exist, move ONE file at a time with a
         small sleep between to avoid memory spikes from large copies
    
    Memory usage: ~0 (no file lists held in memory, no os.walk)
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="nas-sync"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        global _nas_online
        # Initial delay — let scanner start first
        self._stop.wait(30)

        while not self._stop.is_set():
            try:
                _nas_online = _check_nas()

                if _nas_online:
                    self._sync_one_pass()
            except Exception:
                pass

            self._stop.wait(config.NAS_CHECK_INTERVAL)

    def _sync_one_pass(self):
        """Move local clips to NAS, one file at a time. Stops on any error."""
        local_dir = config.LOCAL_CLIPS_DIR
        nas_dir = config.CLIPS_DIR

        if not os.path.isdir(local_dir):
            return

        # List only date subdirectories (YYYYMMDD format)
        try:
            entries = os.listdir(local_dir)
        except OSError:
            return

        for date_folder in sorted(entries):
            if not date_folder.isdigit() or len(date_folder) != 8:
                continue

            src_dir = os.path.join(local_dir, date_folder)
            if not os.path.isdir(src_dir):
                continue

            dst_dir = os.path.join(nas_dir, date_folder)

            try:
                files = os.listdir(src_dir)
            except OSError:
                continue

            for filename in files:
                if self._stop.is_set():
                    return
                if not filename.endswith((".wav", ".mp3")):
                    continue

                src = os.path.join(src_dir, filename)
                dst = os.path.join(dst_dir, filename)

                try:
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(src, dst)
                    os.remove(src)
                    # Update DB clip path
                    self._update_db(src, dst)
                except (OSError, IOError):
                    # NAS went away mid-copy — abort this pass
                    return

                # Yield CPU between files (avoids memory pressure from buffered I/O)
                time.sleep(0.1)

            # Remove empty date folder
            try:
                if not os.listdir(src_dir):
                    os.rmdir(src_dir)
            except OSError:
                pass

    def _update_db(self, old_path, new_path):
        """Update clip path in DB."""
        try:
            with scanner_db.get_db() as conn:
                conn.execute(
                    "UPDATE transmissions SET clip = ? WHERE clip = ?",
                    (new_path, old_path),
                )
        except Exception:
            pass
