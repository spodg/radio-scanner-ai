"""
NAS availability checker and clip sync worker.

Provides:
  - nas_available() — quick check if NAS is mounted and writable
  - get_clips_dir() — returns NAS path or local fallback
  - NasSyncWorker — background thread that moves local clips to NAS when available
  - gpu_available() — check if GPU server is reachable
"""

import os
import time
import shutil
import threading
from pathlib import Path

import config
import scanner_db

# =============================================================================
# State (thread-safe via GIL for simple booleans)
# =============================================================================
_nas_online = False
_gpu_online = False
_sync_pending = 0  # number of files waiting to sync


def nas_available() -> bool:
    """Return cached NAS availability status."""
    return _nas_online


def gpu_available() -> bool:
    """Return cached GPU server availability status."""
    return _gpu_online


def sync_pending() -> int:
    """Return number of local clips waiting to sync to NAS."""
    return _sync_pending


def get_clips_dir() -> str:
    """
    Return the directory to write clips to right now.
    If NAS is available, use CLIPS_DIR (NAS).
    Otherwise, use LOCAL_CLIPS_DIR (local SD card).
    """
    if _nas_online:
        return config.CLIPS_DIR
    return config.LOCAL_CLIPS_DIR


# =============================================================================
# NAS check logic
# =============================================================================
def _check_nas() -> bool:
    """
    Check if NAS is mounted and writable.
    Tests by checking if the mount point exists and is a mount (not just an empty dir).
    Then verifies write access with a temp file.
    """
    nas_dir = config.CLIPS_DIR
    try:
        # Check the parent mount point is actually mounted
        mount_point = config.NAS_MOUNT
        if not os.path.ismount(mount_point):
            return False

        # Verify we can write to the clips directory
        os.makedirs(nas_dir, exist_ok=True)
        test_file = os.path.join(nas_dir, ".nas_check")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except (OSError, IOError):
        return False


def _check_gpu() -> bool:
    """Check if GPU server is reachable."""
    if not config.GPU_SERVER_URL:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{config.GPU_SERVER_URL}/status",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# =============================================================================
# Sync worker
# =============================================================================
class NasSyncWorker:
    """
    Background thread that:
    1. Periodically checks NAS and GPU availability
    2. When NAS comes back online, moves any locally-stored clips to NAS
    3. Updates DB clip paths after successful move
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
        global _nas_online, _gpu_online, _sync_pending
        print("[nas-sync] Worker started.")

        # Ensure local clips dir exists
        Path(config.LOCAL_CLIPS_DIR).mkdir(parents=True, exist_ok=True)

        while not self._stop.is_set():
            # Check NAS
            was_online = _nas_online
            _nas_online = _check_nas()
            if _nas_online and not was_online:
                print("[nas-sync] NAS came online!")
            elif not _nas_online and was_online:
                print("[nas-sync] NAS went offline — using local storage.")

            # Check GPU
            _gpu_online = _check_gpu()

            # Sync local clips to NAS if NAS is back
            if _nas_online:
                moved = self._sync_local_to_nas()
                if moved > 0:
                    print(f"[nas-sync] Moved {moved} clip(s) to NAS.")

            # Count pending files
            _sync_pending = self._count_local_clips()

            self._stop.wait(config.NAS_CHECK_INTERVAL)

        print("[nas-sync] Worker stopped.")

    def _count_local_clips(self) -> int:
        """Count audio files in local clips dir (pending sync)."""
        local_dir = config.LOCAL_CLIPS_DIR
        if not os.path.exists(local_dir):
            return 0
        count = 0
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                if f.endswith((".wav", ".mp3")):
                    count += 1
        return count

    def _sync_local_to_nas(self) -> int:
        """
        Move clips from LOCAL_CLIPS_DIR to CLIPS_DIR (NAS), preserving
        date subfolder structure. Updates DB clip paths.
        Returns number of files moved.
        """
        local_dir = config.LOCAL_CLIPS_DIR
        nas_dir = config.CLIPS_DIR

        if not os.path.exists(local_dir):
            return 0

        moved = 0
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                if not filename.endswith((".wav", ".mp3")):
                    continue

                local_path = os.path.join(root, filename)
                # Preserve subfolder structure (e.g., 20260801/file.wav)
                rel_path = os.path.relpath(local_path, local_dir)
                nas_path = os.path.join(nas_dir, rel_path)

                try:
                    # Ensure target dir exists
                    os.makedirs(os.path.dirname(nas_path), exist_ok=True)

                    # Copy then delete (safer than move across filesystems)
                    shutil.copy2(local_path, nas_path)
                    os.remove(local_path)

                    # Update DB record: change clip path from local to NAS
                    self._update_clip_path(local_path, nas_path)
                    moved += 1

                except (OSError, IOError) as e:
                    # NAS probably went away mid-sync — stop and retry later
                    print(f"[nas-sync] Error moving {filename}: {e}")
                    break

        # Clean up empty date folders in local dir
        self._cleanup_empty_dirs(local_dir)
        return moved

    def _update_clip_path(self, old_path: str, new_path: str):
        """Update the clip path in the database."""
        try:
            with scanner_db.get_db() as conn:
                conn.execute(
                    "UPDATE transmissions SET clip = ? WHERE clip = ?",
                    (new_path, old_path),
                )
        except Exception:
            pass

    def _cleanup_empty_dirs(self, base_dir: str):
        """Remove empty subdirectories."""
        for root, dirs, files in os.walk(base_dir, topdown=False):
            if root == base_dir:
                continue
            if not os.listdir(root):
                try:
                    os.rmdir(root)
                except OSError:
                    pass
