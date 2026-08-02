"""
GPU Transcription Server — runs on your powerful PC to transcribe scanner audio.

This server polls the shared scanner_log.jsonl for records with
"transcribed": false, grabs the WAV clip from the NAS, transcribes it using
the local GPU (faster-whisper), and updates the JSONL record in place.

Architecture:
  - Polls the JSONL every few seconds for untranscribed items.
  - Loads faster-whisper with CUDA for fast GPU transcription.
  - Updates the JSONL atomically (read-modify-write with temp file).
  - Also exposes an HTTP API so the scanner logger can check status
    or force-queue specific items.

Requirements:
  - faster-whisper (pip install faster-whisper)
  - numpy
  - flask (for the status/control API)
  - Access to the NAS share where JSONL and clips are stored.

Run:  python gpu_transcribe_server.py
Stop: Ctrl+C

Configuration: edit the settings below or set environment variables.
"""

import os
import sys
import json
import time
import glob
import threading
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# JSONL_PATH kept for reference only (no longer written by GPU server).
# All reads/writes go through the Pi dashboard HTTP API now.
JSONL_PATH = r"\\d1\RadioScanner\scanner_log.jsonl"  # legacy, unused

# Base path where audio clips are stored (the 'clip' field in records is
# relative or absolute; we resolve against this if needed).
CLIPS_BASE = os.environ.get(
    "SCANNER_CLIPS_BASE",
    r"\\d1\RadioScanner\clips"
)

# Pi clips are stored on the NAS under a different subfolder.
# The Pi writes paths like "/mnt/nas/clips/file.wav" — we map
# /mnt/nas/ -> \\d1\RadioScanner\ for Windows access.
NAS_LINUX_PREFIX = "/mnt/nas/"
NAS_WINDOWS_PREFIX = r"\\d1\RadioScanner" + "\\"

# Whisper model settings (same as the scanner logger for consistency).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float32")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_SILENCE_RMS = float(os.environ.get("WHISPER_SILENCE_RMS", "0.0015"))

# Initial prompt for scanner domain vocabulary.
WHISPER_PROMPT = (
    "Show us en route. 10-4, copy. Signal 22, signal 30, 10-42. "
    "Adam Boy Charles David Edward Frank George Henry Ida John King Lincoln "
    "Mary Nora Ocean Paul Queen Robert Sam Tom Union Victor William X-ray Young Zebra. "
    "Fort Wayne, Allen County, Whitley County, DeKalb County, Noble County, "
    "Adams County, Wells County, Huntington County, Indiana. "
    "Coliseum, Coldwater, Lima Road, State Road 3, Interstate 69, US 30, "
    "Clinton, Calhoun, Jefferson, Washington, Lafayette, Stellhorn, Dupont, "
    "Maysville Road, Goshen Road, Bluffton Road, Decatur Road, "
    "Lutheran Hospital, Parkview, St. Joe Center. "
    "Copy that. Show me out. Show us en route. Show us on scene. "
    "Be advised. Negative. Affirmative. Roger. Clear. "
    "Dispatch, county, sheriff, deputy, unit, squad, engine, medic, "
    "responding, disregard, reference, complainant, suspect, subject, "
    "vehicle, plate, registration, driver's license, warrant. "
)

# Post-processing corrections (same as scanner config).
WHISPER_CORRECTIONS = {}

# How often to poll for new untranscribed items (seconds).
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))

# HTTP API port for status/control.
API_PORT = int(os.environ.get("API_PORT", "5555"))

# Maximum number of items to process per poll cycle.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))

# Pi dashboard URL for fetching fresh records (bypasses SMB cache)
PI_DASHBOARD_URL = os.environ.get("PI_DASHBOARD_URL", "http://pi3:8080")

# Cache Pi IP address — resolve once, keep forever (no TTL expiry)
_pi_ip_cache = None


def _get_pi_ip():
    """Get and cache Pi IP address from hostname. Never expires."""
    global _pi_ip_cache
    import socket
    
    if _pi_ip_cache is not None:
        return _pi_ip_cache
    
    try:
        ip = socket.gethostbyname("pi3")
        _pi_ip_cache = ip
        print(f"[gpu] Resolved pi3 to {ip} (cached permanently)")
        return ip
    except socket.gaierror as e:
        # Fallback to hardcoded IP if DNS fails completely
        _pi_ip_cache = "192.168.2.87"
        print(f"[gpu] DNS failed for pi3: {e}, using fallback {_pi_ip_cache}")
        return _pi_ip_cache


def _build_pi_url():
    """Build Pi dashboard URL, using cached IP after first resolution."""
    _get_pi_ip()  # Ensure IP is resolved and cached
    return PI_DASHBOARD_URL.replace("pi3", _pi_ip_cache)


# ---------------------------------------------------------------------------
# CUDA DLL registration (same as scanner transcriber.py)
# ---------------------------------------------------------------------------
def _register_cuda_dll_dirs():
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in list(getattr(nvidia, "__path__", [])):
        for bin_dir in glob.glob(os.path.join(base, "*", "bin")):
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except (OSError, AttributeError):
                    pass
                current = os.environ.get("PATH", "")
                if bin_dir not in current.split(os.pathsep):
                    os.environ["PATH"] = bin_dir + os.pathsep + current


_register_cuda_dll_dirs()


# ---------------------------------------------------------------------------
# Transcription engine
# ---------------------------------------------------------------------------
class GPUTranscriber:
    """Wrapper around faster-whisper for GPU transcription."""

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        from faster_whisper import WhisperModel
        print(f"[gpu] Loading model '{WHISPER_MODEL}' on {WHISPER_DEVICE} "
              f"({WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print("[gpu] Model loaded and ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe mono float32 16kHz audio. Returns text or empty string."""
        if audio.size == 0:
            return ""

        # Light energy gate.
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < WHISPER_SILENCE_RMS:
            return ""

        with self._lock:
            segments, _info = self.model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                beam_size=10,
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                initial_prompt=WHISPER_PROMPT or None,
            )

            good_parts = []
            for seg in segments:
                if seg.no_speech_prob > 0.5:
                    continue
                if seg.avg_logprob < -0.7:
                    continue
                good_parts.append(seg.text.strip())

        text = " ".join(good_parts).strip()

        # Apply corrections.
        if text and WHISPER_CORRECTIONS:
            import re
            for wrong, right in WHISPER_CORRECTIONS.items():
                text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

        return text


# ---------------------------------------------------------------------------
# JSONL read/update utilities
# ---------------------------------------------------------------------------

def load_untranscribed_records(limit=BATCH_SIZE):
    """Fetch untranscribed records from the Pi dashboard via HTTP.

    This bypasses Windows SMB oplock caching which causes the GPU server
    to see stale JSONL content when reading the file directly.
    """
    import urllib.request
    try:
        url = f"{_build_pi_url()}/api/untranscribed?limit={limit}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("records", [])
    except Exception as e:
        print(f"[gpu] Error fetching untranscribed from Pi: {e}")
        return []


def load_pi_transcribed_records(limit=100):
    """Fetch Pi-transcribed records via HTTP (for re-transcription)."""
    import urllib.request
    try:
        url = f"{_build_pi_url()}/api/pi_transcribed?limit={limit}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("records", [])
    except Exception as e:
        print(f"[gpu] Error fetching pi_transcribed from Pi: {e}")
        return []


def update_jsonl_record(record_id: str, updates: dict) -> bool:
    """Post a transcription result to the Pi dashboard via HTTP.

    Replaces the old JSONL file manipulation. The Pi's SQLite DB is the
    single source of truth; we just POST the result to it.
    """
    import urllib.request
    payload = {
        "id": record_id,
        "text": updates.get("text", ""),
        "transcribed_by": updates.get("transcribed_by", "gpu"),
    }
    try:
        url = f"{_build_pi_url()}/api/transcribe_result"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[gpu] Error posting result for {record_id}: {e}")
        return False


def batch_update_jsonl(updates_map: dict) -> bool:
    """Post multiple transcription results to the Pi dashboard.

    Since the batch endpoint may not be available, we use individual POSTs.
    Returns True if at least one record was updated successfully.
    """
    if not updates_map:
        return False
    
    success_count = 0
    for record_id, updates in updates_map.items():
        if update_jsonl_record(record_id, updates):
            success_count += 1
    
    if success_count > 0:
        print(f"[gpu] Updated {success_count}/{len(updates_map)} records")
    return success_count > 0


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(clip_path: str) -> np.ndarray:
    """Load a WAV or MP3 file and return mono float32 samples at 16kHz."""
    import wave

    # Map Pi's Linux NAS paths to Windows UNC paths.
    if clip_path and clip_path.startswith(NAS_LINUX_PREFIX):
        relative = clip_path[len(NAS_LINUX_PREFIX):]
        clip_path = NAS_WINDOWS_PREFIX + relative.replace("/", "\\")

    # Resolve the clip path: if it's a relative path like "clips/...",
    # resolve against CLIPS_BASE parent.
    if clip_path and not os.path.isabs(clip_path) and not clip_path.startswith("\\\\"):
        base_parent = str(Path(CLIPS_BASE).parent)
        resolved = os.path.join(base_parent, clip_path)
        if os.path.exists(resolved):
            clip_path = resolved
        elif os.path.exists(os.path.join(CLIPS_BASE, os.path.basename(clip_path))):
            clip_path = os.path.join(CLIPS_BASE, os.path.basename(clip_path))

    if not clip_path or not os.path.exists(clip_path):
        # Try .mp3 version if .wav was converted
        if clip_path and clip_path.lower().endswith(".wav"):
            mp3_path = clip_path[:-4] + ".mp3"
            if os.path.exists(mp3_path):
                clip_path = mp3_path
            else:
                return np.zeros(0, dtype=np.float32)
        else:
            return np.zeros(0, dtype=np.float32)

    # MP3: decode with av (PyAV) which faster-whisper already depends on
    if clip_path.lower().endswith(".mp3"):
        try:
            import av
            from av.audio.resampler import AudioResampler
            resampler = AudioResampler(format="flt", layout="mono", rate=16000)
            parts = []
            with av.open(clip_path) as container:
                if not container.streams.audio:
                    return np.zeros(0, dtype=np.float32)
                for frame in container.decode(audio=0):
                    for rf in resampler.resample(frame):
                        parts.append(rf.to_ndarray().reshape(-1))
                for rf in resampler.resample(None):
                    parts.append(rf.to_ndarray().reshape(-1))
            if not parts:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts).astype(np.float32)
        except Exception as e:
            print(f"[gpu] Error reading MP3 '{clip_path}': {e}")
            return np.zeros(0, dtype=np.float32)

    # WAV: use standard library
    try:
        with wave.open(clip_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        print(f"[gpu] Error reading WAV '{clip_path}': {e}")
        return np.zeros(0, dtype=np.float32)

    # Convert to float32.
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        print(f"[gpu] Unsupported sample width {sample_width} in '{clip_path}'")
        return np.zeros(0, dtype=np.float32)

    # Take first channel if stereo.
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    # Resample to 16kHz if needed.
    if framerate != 16000:
        duration = len(samples) / framerate
        new_len = int(duration * 16000)
        if new_len <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)

    return samples.astype(np.float32)


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------
class TranscriptionWorker:
    """Two-thread worker: one for fresh items (high priority), one for
    re-transcription (low priority, batched). They share the GPU via a lock."""

    def __init__(self, transcriber: GPUTranscriber):
        self.transcriber = transcriber
        self._stop = threading.Event()
        self._fresh_thread = None
        self._retrans_thread = None
        self._failed_ids = {}
        self._stats = {
            "total_transcribed": 0,
            "total_retranscribed": 0,
            "total_errors": 0,
            "last_transcription": None,
            "running": False,
        }

    @property
    def stats(self):
        return dict(self._stats)

    def start(self):
        self._stats["running"] = True
        self._fresh_thread = threading.Thread(target=self._run_fresh, daemon=True, name="fresh-worker")
        self._retrans_thread = threading.Thread(target=self._run_retrans, daemon=True, name="retrans-worker")
        self._fresh_thread.start()
        self._retrans_thread.start()

    def stop(self):
        self._stop.set()
        if self._fresh_thread:
            self._fresh_thread.join(timeout=30)
        if self._retrans_thread:
            self._retrans_thread.join(timeout=30)
        self._stats["running"] = False

    # === Thread 1: Fresh untranscribed items (high priority, polls every 3s) ===
    def _run_fresh(self):
        print(f"[gpu:fresh] Worker started. Polling every {POLL_INTERVAL}s...")
        while not self._stop.is_set():
            try:
                records = load_untranscribed_records(limit=BATCH_SIZE)
                if records:
                    print(f"[gpu:fresh] Found {len(records)} untranscribed item(s).")
                    for record in records:
                        if self._stop.is_set():
                            break
                        self._transcribe_record(record)
            except Exception as e:
                print(f"[gpu:fresh] Error: {e}")
                traceback.print_exc()
            self._stop.wait(POLL_INTERVAL)
        print("[gpu:fresh] Stopped.")

    # === Thread 2: Re-transcribe Pi items (low priority, batched, polls every 10s) ===
    def _run_retrans(self):
        print(f"[gpu:retrans] Worker started. Polling every 10s...")
        time.sleep(5)  # let fresh worker get a head start
        while not self._stop.is_set():
            try:
                self._retranscribe_batch()
            except Exception as e:
                print(f"[gpu:retrans] Error: {e}")
                traceback.print_exc()
            self._stop.wait(10)
        print("[gpu:retrans] Stopped.")

    def _retranscribe_batch(self):
        pi_records = load_pi_transcribed_records(limit=100)
        if not pi_records:
            return
        print(f"[gpu:retrans] Found {len(pi_records)} Pi-transcribed item(s) to improve.")
        updates_map = {}
        for record in pi_records:
            if self._stop.is_set():
                break
            record_id = record["id"]
            clip_path = record.get("clip")
            old_text = record.get("text", "")
            name = record.get("name", "?")
            time_str = record.get("time", "?")

            if not clip_path:
                updates_map[record_id] = {"transcribed_by": "gpu"}
                continue

            audio = load_audio(clip_path)
            if audio.size == 0:
                updates_map[record_id] = {"transcribed_by": "gpu"}
                continue

            # Transcribe (no lock — CTranslate2 is thread-safe)
            try:
                text = self.transcriber.transcribe(audio)
            except Exception as e:
                print(f"[gpu:retrans] Error for {record_id}: {e}")
                updates_map[record_id] = {"transcribed_by": "gpu"}
                continue

            updates_map[record_id] = {"text": text, "transcribed_by": "gpu"}
            self._stats["total_retranscribed"] += 1
            self._stats["last_transcription"] = datetime.now().isoformat()
            changed = " [IMPROVED]" if text != old_text else ""
            display_text = text[:70] + "..." if len(text) > 70 else text
            print(f"[gpu:retrans] {time_str} | {name} -> {display_text or '(no speech)'}{changed}")

        if updates_map:
            success = batch_update_jsonl(updates_map)
            if success:
                print(f"[gpu:retrans] Batch-updated {len(updates_map)} record(s).")
            else:
                print(f"[gpu:retrans] Batch write FAILED.")

    def _transcribe_record(self, record: dict):
        record_id = record["id"]
        clip_path = record.get("clip")
        name = record.get("name", "?")
        time_str = record.get("time", "?")

        # Skip records that have failed too many times
        if self._failed_ids.get(record_id, 0) >= 3:
            return

        if not clip_path:
            # No audio clip — mark as transcribed with empty text.
            update_jsonl_record(record_id, {
                "text": "",
                "transcribed": True,
            })
            return

        # Load audio (retry a few times — NAS may not have synced yet).
        audio = np.zeros(0, dtype=np.float32)
        for attempt in range(4):
            audio = load_audio(clip_path)
            if audio.size > 0:
                break
            # Wait for NAS sync
            time.sleep(2)

        if audio.size == 0:
            print(f"[gpu:fresh] No audio for {record_id} ({clip_path}), will retry later.")
            self._failed_ids[record_id] = self._failed_ids.get(record_id, 0) + 1
            self._stats["total_errors"] += 1
            return

        # Transcribe fresh item immediately (no lock — CTranslate2 is thread-safe).
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as e:
            print(f"[gpu:fresh] Transcription error for {record_id}: {e}")
            self._stats["total_errors"] += 1
            return

        # Update via HTTP API.
        updates = {
            "text": text,
            "transcribed": True,
            "transcribed_by": "gpu",
        }
        success = update_jsonl_record(record_id, updates)

        if success:
            self._stats["total_transcribed"] += 1
            self._stats["last_transcription"] = datetime.now().isoformat()
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"[gpu:fresh] {time_str} | {name} -> {display_text or '(no speech)'}")
        else:
            # Record was already transcribed by the local worker (race won).
            pass


# ---------------------------------------------------------------------------
# Flask API for status and control
# ---------------------------------------------------------------------------
def create_api(worker: TranscriptionWorker):
    from flask import Flask, jsonify, request
    api = Flask(__name__)

    @api.route("/status")
    def status():
        return jsonify({
            "service": "gpu-transcribe-server",
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "poll_interval": POLL_INTERVAL,
            **worker.stats,
        })

    @api.route("/transcribe", methods=["POST"])
    def transcribe_clip():
        """Transcribe a specific clip by record ID or clip path.

        POST JSON: {"id": "record_id"} or {"clip": "path/to/file.wav"}
        Returns: {"text": "transcribed text", "success": true}
        """
        data = request.get_json(force=True)
        record_id = data.get("id")
        clip_path = data.get("clip")

        if clip_path:
            audio = load_audio(clip_path)
            if audio.size == 0:
                return jsonify({"success": False, "error": "Could not load audio"})
            try:
                text = worker.transcriber.transcribe(audio)
                # If record_id provided, update JSONL too.
                if record_id:
                    update_jsonl_record(record_id, {
                        "text": text,
                        "transcribed": True,
                    })
                return jsonify({"success": True, "text": text})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})

        if record_id:
            # Find the record in JSONL and transcribe it.
            records = load_untranscribed_records(limit=500)
            target = next((r for r in records if r["id"] == record_id), None)
            if not target:
                return jsonify({"success": False, "error": "Record not found or already transcribed"})
            worker._transcribe_record(target)
            return jsonify({"success": True})

        return jsonify({"success": False, "error": "Provide 'id' or 'clip'"})

    @api.route("/pending")
    def pending():
        """Return count and list of untranscribed records."""
        records = load_untranscribed_records(limit=100)
        return jsonify({
            "count": len(records),
            "records": [
                {"id": r["id"], "time": r.get("time"), "name": r.get("name"),
                 "clip": r.get("clip")}
                for r in records
            ]
        })

    return api


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  GPU Transcription Server")
    print("=" * 60)
    print(f"  Pi API: {PI_DASHBOARD_URL}")
    print(f"  Clips:  {CLIPS_BASE}")
    print(f"  Model:   {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
    print(f"  Poll:    every {POLL_INTERVAL}s")
    print(f"  API:     http://0.0.0.0:{API_PORT}")
    print("=" * 60)

    # Load the model.
    transcriber = GPUTranscriber()
    transcriber.load()

    # Start the background worker.
    worker = TranscriptionWorker(transcriber)
    worker.start()

    # Start the API server.
    api = create_api(worker)
    try:
        api.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[gpu] Shutting down...")
        worker.stop()
        print("[gpu] Done.")


if __name__ == "__main__":
    main()
