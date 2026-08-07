"""
Unified GPU Server — transcription + daily summary pipeline.

Single process that handles:
  Priority 1: Fresh transcriptions (untranscribed clips from Pi)
  Priority 2: Re-transcriptions (upgrade Pi's whisper-tiny to GPU large-v3)
  Priority 3: Write daily transcription logs (once all re-transcribed for a day)
  Priority 4: Generate daily LLM summaries (once transcription log exists)

Whisper (faster-whisper large-v3) stays loaded on GPU permanently.
Ollama (llama3.1:8b) runs as a separate process and manages its own VRAM.
Summarization only runs when transcription queues are empty.

Requirements:
  - faster-whisper (pip install faster-whisper)
  - numpy, requests, flask
  - Ollama running locally (ollama serve)
  - Access to Pi dashboard (http://pi3:8080)
  - Access to NAS (\\\\d1\\RadioScanner)

Run:  python gpu_server.py
Stop: Ctrl+C
"""

import os
import sys
import json
import time
import glob
import wave
import socket
import threading
import traceback
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np
import requests

# ===========================================================================
# Configuration
# ===========================================================================
PI_URL = os.environ.get("PI_URL", "http://pi3:8080")
CLIPS_BASE = os.environ.get("SCANNER_CLIPS_BASE", r"\\d1\RadioScanner\clips")
NAS_LINUX_PREFIX = "/mnt/nas/"
NAS_WINDOWS_PREFIX = r"\\d1\RadioScanner" + "\\"

# Output directories on NAS
SUMMARIES_DIR = os.environ.get("SUMMARIES_DIR", r"\\d1\RadioScanner\summaries")
TRANSCRIBED_DIR = os.environ.get("TRANSCRIBED_DIR", r"\\d1\RadioScanner\transcribed")

# Whisper
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float32")
WHISPER_LANGUAGE = "en"
WHISPER_SILENCE_RMS = 0.0015
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

# Ollama (for summarization)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Timing
POLL_INTERVAL = 3        # seconds between fresh transcription polls
RETRANS_INTERVAL = 0     # no delay between re-transcription batches
PIPELINE_CHECK = 60      # seconds between day-completion checks
BATCH_SIZE = 10          # max fresh items per poll cycle
RETRANS_BATCH = 20       # re-transcription items to fetch at once

# Flask API
API_PORT = int(os.environ.get("API_PORT", "5555"))

# Resolved Pi IP (cached)
_pi_ip = None


def _resolve_pi():
    """Resolve Pi hostname once."""
    global _pi_ip
    if _pi_ip:
        return _pi_ip
    try:
        _pi_ip = socket.gethostbyname("pi3")
        print(f"[init] Resolved pi3 -> {_pi_ip}")
    except socket.gaierror:
        _pi_ip = "192.168.2.87"
        print(f"[init] DNS failed, using fallback {_pi_ip}")
    return _pi_ip


def _pi_url():
    """Get Pi URL with resolved IP."""
    _resolve_pi()
    return PI_URL.replace("pi3", _pi_ip)


# ===========================================================================
# CUDA DLL registration (Windows)
# ===========================================================================
def _register_cuda_dlls():
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
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()


# ===========================================================================
# Whisper Transcription Engine
# ===========================================================================
class Transcriber:
    """faster-whisper GPU transcriber."""

    def __init__(self):
        self.model = None

    def load(self):
        from faster_whisper import WhisperModel
        print(f"[whisper] Loading {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                  compute_type=WHISPER_COMPUTE_TYPE)
        print("[whisper] Ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 mono 16kHz audio. Returns text or empty."""
        if audio.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < WHISPER_SILENCE_RMS:
            return ""

        segments, _ = self.model.transcribe(
            audio, language=WHISPER_LANGUAGE, beam_size=10,
            vad_filter=False, condition_on_previous_text=False,
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            initial_prompt=WHISPER_PROMPT,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.5 or seg.avg_logprob < -0.7:
                continue
            parts.append(seg.text.strip())
        return " ".join(parts).strip()


# ===========================================================================
# Audio Loading
# ===========================================================================
def load_audio(clip_path: str) -> np.ndarray:
    """Load WAV/MP3, return mono float32 at 16kHz."""
    # Map Pi Linux path to Windows UNC
    if clip_path and clip_path.startswith(NAS_LINUX_PREFIX):
        clip_path = NAS_WINDOWS_PREFIX + clip_path[len(NAS_LINUX_PREFIX):].replace("/", "\\")

    # Resolve relative paths
    if clip_path and not os.path.isabs(clip_path) and not clip_path.startswith("\\\\"):
        resolved = os.path.join(str(Path(CLIPS_BASE).parent), clip_path)
        if os.path.exists(resolved):
            clip_path = resolved

    if not clip_path or not os.path.exists(clip_path):
        if clip_path and clip_path.lower().endswith(".wav"):
            mp3 = clip_path[:-4] + ".mp3"
            if os.path.exists(mp3):
                clip_path = mp3
            else:
                return np.zeros(0, dtype=np.float32)
        else:
            return np.zeros(0, dtype=np.float32)

    # MP3
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
            return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"[audio] MP3 error '{clip_path}': {e}")
            return np.zeros(0, dtype=np.float32)

    # WAV
    try:
        with wave.open(clip_path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"[audio] WAV error '{clip_path}': {e}")
        return np.zeros(0, dtype=np.float32)

    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return np.zeros(0, dtype=np.float32)

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]
    if fr != 16000:
        dur = len(samples) / fr
        new_len = int(dur * 16000)
        if new_len <= 0:
            return np.zeros(0, dtype=np.float32)
        samples = np.interp(
            np.linspace(0, dur, new_len, endpoint=False),
            np.linspace(0, dur, len(samples), endpoint=False),
            samples,
        ).astype(np.float32)
    return samples


# ===========================================================================
# Pi API Communication
# ===========================================================================
def pi_get(endpoint, params=None, timeout=15):
    """GET from Pi dashboard API."""
    try:
        r = requests.get(f"{_pi_url()}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[pi] GET {endpoint} failed: {e}")
        return None


def pi_post(endpoint, data, timeout=15):
    """POST to Pi dashboard API."""
    try:
        r = requests.post(f"{_pi_url()}{endpoint}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[pi] POST {endpoint} failed: {e}")
        return None


def fetch_untranscribed(limit=BATCH_SIZE):
    """P1: Get fresh untranscribed records."""
    data = pi_get("/api/untranscribed", {"limit": limit})
    return data.get("records", []) if data else []


def fetch_pi_transcribed(limit=1):
    """P2: Get records transcribed by Pi (for GPU re-transcription)."""
    data = pi_get("/api/pi_transcribed", {"limit": limit})
    return data.get("records", []) if data else []


def post_result(record_id, text, transcribed_by="gpu"):
    """Post transcription result back to Pi."""
    return pi_post("/api/transcribe_result", {
        "id": record_id, "text": text, "transcribed_by": transcribed_by,
    })


def fetch_day_records(target_date: date):
    """Fetch all transcribed records for a date (for completion check + log writing)."""
    data = pi_get("/api/day_transmissions", {
        "date": target_date.isoformat(), "hide_blank": "0",
    }, timeout=30)
    return data.get("records", []) if data else []


# ===========================================================================
# Day Completion Detection & File Writing
# ===========================================================================
def _transcribed_file(d: date) -> str:
    return os.path.join(TRANSCRIBED_DIR, f"transcribed_{d}.txt")


def _summary_file(d: date) -> str:
    return os.path.join(SUMMARIES_DIR, f"summary_{d}.md")


def is_day_transcribed(d: date) -> bool:
    """Check if the transcribed log file already exists for this date."""
    return os.path.exists(_transcribed_file(d))


def is_day_summarized(d: date) -> bool:
    """Check if the summary file already exists for this date."""
    return os.path.exists(_summary_file(d))


def is_day_fully_gpu_transcribed(records: list[dict]) -> bool:
    """Check if ALL records for a day have been processed by GPU.
    
    A record counts as 'done' if:
      - transcribed_by == 'gpu', OR
      - text is empty/blank (silence clips that neither Pi nor GPU could transcribe)
    """
    if not records:
        return False
    for r in records:
        tb = r.get("transcribed_by", "")
        text = r.get("text", "").strip()
        # GPU-transcribed: done
        if tb == "gpu":
            continue
        # Empty text with no transcriber: silence/blank clip, count as done
        if not text and not tb:
            continue
        # Pi-transcribed with actual text: needs GPU re-transcription
        return False
    return True


def write_transcription_log(target_date: date, records: list[dict]) -> str:
    """Write grouped transcription log to NAS."""
    os.makedirs(TRANSCRIBED_DIR, exist_ok=True)
    filepath = _transcribed_file(target_date)

    # Group by channel
    grouped = defaultdict(list)
    for r in records:
        system = r.get("system", "").strip()
        group = r.get("group", "").strip()
        channel = r.get("channel", "").strip()
        parts = [p for p in [system, group, channel] if p]
        key = " > ".join(parts) if parts else "(Unknown)"
        grouped[key].append(r)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Scanner Transcription Log — {target_date}\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Total: {len(records)} transmissions across {len(grouped)} channels\n")
        f.write(f"{'=' * 70}\n\n")

        for ch_key in sorted(grouped.keys(), key=lambda k: -len(grouped[k])):
            txs = grouped[ch_key]
            f.write(f"[{ch_key}] ({len(txs)})\n")
            f.write(f"{'-' * 70}\n")
            for r in txs:
                t = r.get("time", "")
                if "T" in t:
                    t = t.split("T")[1][:8]
                text = r.get("text", "").strip().replace("\n", " ")
                f.write(f"  {t}  {text}\n")
            f.write("\n")

    return filepath


# ===========================================================================
# Summarization (calls Ollama)
# ===========================================================================
SUMMARY_SYSTEM_PROMPT = """You are an analyst producing a detailed event log from police/fire/EMS radio scanner transmissions.

You receive a batch of transcribed radio transmissions from one channel. Signal codes are decoded for you in brackets — always use the decoded plain-English meaning, never leave raw signal numbers.

Your job: identify each distinct EVENT (incident, call for service, traffic stop, medical run, etc.) and write a detailed narrative entry for it.

For each event, include ALL available details:
- What happened (use decoded signal meaning, not the code number)
- Location (address, intersection, landmark)
- Involved parties (names, descriptions, unit numbers)
- Vehicles (color, make, model, plate)
- Outcome/disposition (arrest, transport, cleared, etc.)
- Any other specifics mentioned (DOB, warrants, weapons, injuries)

Rules:
- Translate ALL signal/10-codes to their meaning.
- Group related transmissions into one event entry.
- Use 24-hour time format (HH:MM).
- Skip pure routine status changes unless they contain incident details.
- Write in short, factual narrative style. No filler words.
- Output ONLY the event list, nothing else."""


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call Ollama for summarization."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama not running at {OLLAMA_URL}. Start with: ollama serve")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _build_channel_prompt(channel_key: str, transmissions: list[dict]) -> str:
    """Build LLM prompt for one channel's transmissions."""
    lines = [f"Channel: {channel_key}", f"Transmissions ({len(transmissions)} total):", "---"]
    for t in transmissions:
        ts = t.get("time", "")
        if "T" in ts:
            ts = ts.split("T")[1][:8]
        text = t.get("text", "")
        dt = t.get("decoded_text", {}) or {}
        annot = []
        if dt.get("codes"):
            for c in dt["codes"]:
                annot.append(f"[{c['code']}={c['meaning']}]")
        if dt.get("plates"):
            annot.append(f"[Plate: {', '.join(dt['plates'])}]")
        annot_str = " " + " ".join(annot) if annot else ""
        lines.append(f"[{ts}] {text}{annot_str}")
    lines.append("---\n")
    lines.append("Write each event as:\n")
    lines.append("**HH:MM–HH:MM | <Event Type>**")
    lines.append("<Location if known>")
    lines.append("<Detailed narrative: what happened, who, vehicles/plates, outcome>\n")
    lines.append("Skip pure status updates. Decode all signal codes to plain English.")
    return "\n".join(lines)


def generate_daily_summary(target_date: date, records: list[dict]) -> str:
    """Generate full daily summary using Ollama. Returns filepath written."""
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    # Filter out blank records for summarization
    active = [r for r in records if r.get("text", "").strip()
              and r["text"] not in ("(no speech)", "[BLANK_AUDIO]", "")]

    # Group by channel
    grouped = defaultdict(list)
    for r in active:
        system = r.get("system", "").strip()
        group = r.get("group", "").strip()
        channel = r.get("channel", "").strip()
        parts = [p for p in [system, group, channel] if p]
        key = " > ".join(parts) if parts else "(Unknown)"
        grouped[key].append(r)

    # Filter low-activity channels
    grouped = {k: v for k, v in grouped.items() if len(v) >= 2}

    print(f"[summary] {len(active)} active transmissions across {len(grouped)} channels")

    # Summarize each channel
    channel_summaries = {}
    total = len(grouped)
    for idx, (ch, txs) in enumerate(sorted(grouped.items(), key=lambda x: -len(x[1])), 1):
        print(f"[summary]   [{idx}/{total}] {ch} ({len(txs)} tx)...")
        try:
            # Chunk large channels
            chunk_size = 80
            if len(txs) <= chunk_size:
                prompt = _build_channel_prompt(ch, txs)
                channel_summaries[ch] = _call_ollama(prompt, SUMMARY_SYSTEM_PROMPT)
            else:
                parts = []
                for i in range(0, len(txs), chunk_size):
                    chunk = txs[i:i + chunk_size]
                    prompt = _build_channel_prompt(f"{ch} (part {i//chunk_size+1})", chunk)
                    parts.append(_call_ollama(prompt, SUMMARY_SYSTEM_PROMPT))
                    time.sleep(1)
                channel_summaries[ch] = "\n\n".join(parts)
        except Exception as e:
            print(f"[summary]   ERROR: {e}")
            channel_summaries[ch] = f"(Summary failed: {e})"

    # Assemble report
    filepath = _summary_file(target_date)
    _write_summary_report(filepath, target_date, grouped, channel_summaries, active)
    return filepath


def _write_summary_report(filepath, target_date, grouped, channel_summaries, records):
    """Write the final markdown summary report."""
    # Simple entity extraction for quick reference
    plates, names, addresses, phones = [], [], [], []
    for r in records:
        dt = r.get("decoded_text", {}) or {}
        t = r.get("time", "")
        ts = t.split("T")[1][:5] if "T" in t else t
        ch = r.get("channel", "") or r.get("group", "")
        ctx = r.get("text", "")[:60].replace("|", "/")
        for p in dt.get("plates", []):
            plates.append(f"| {ts} | {ch[:25]} | **{p}** | {ctx} |")
        for p in dt.get("phones", []):
            phones.append(f"| {ts} | {ch[:25]} | {p} | {ctx} |")

    day_name = target_date.strftime("%A")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Scanner Daily Summary — {target_date} ({day_name})\n\n")
        f.write(f"**Total transmissions:** {len(records)}  \n")
        f.write(f"**Active channels:** {len(grouped)}  \n")
        f.write(f"**Plates detected:** {len(plates)}  \n\n---\n\n")
        f.write("## Events\n\n")

        # Busiest channels first
        for ch in sorted(channel_summaries.keys(), key=lambda k: -len(grouped.get(k, []))):
            summary = channel_summaries[ch]
            count = len(grouped.get(ch, []))
            f.write(f"### {ch}\n*{count} transmissions*\n\n{summary}\n\n---\n\n")

        # Quick Reference
        f.write("## Quick Reference\n\n")
        f.write("### License Plates\n\n")
        if plates:
            f.write("| Time | Channel | Plate | Context |\n")
            f.write("|------|---------|-------|---------||\n")
            seen = set()
            for line in plates:
                if line not in seen:
                    seen.add(line)
                    f.write(line + "\n")
            f.write("\n")
        else:
            f.write("*No plates detected.*\n\n")

        if phones:
            f.write("### Phone Numbers\n\n")
            f.write("| Time | Channel | Phone | Context |\n")
            f.write("|------|---------|-------|---------||\n")
            seen = set()
            for line in phones:
                if line not in seen:
                    seen.add(line)
                    f.write(line + "\n")
            f.write("\n")

        f.write("---\n*Generated by GPU Server Pipeline*\n")


# ===========================================================================
# Pipeline Controller
# ===========================================================================
class Pipeline:
    """
    Priority-based pipeline controller.
    
    Runs in the main worker thread alongside transcription.
    Checks for day-completion opportunities during idle periods.
    """

    def __init__(self):
        self._last_check = 0
        self._days_in_progress = set()  # days we've already attempted

    def check(self):
        """
        Called during idle periods (no transcription work).
        Checks if any past days need transcribed logs or summaries.
        """
        now = time.time()
        if now - self._last_check < PIPELINE_CHECK:
            return
        self._last_check = now

        # Check yesterday and the day before (recent days most likely to complete)
        today = date.today()
        empty_streak = 0
        for days_ago in range(1, 366):  # check last year
            d = today - timedelta(days=days_ago)
            # Skip days we already have both files for
            if is_day_transcribed(d) and is_day_summarized(d):
                empty_streak = 0
                continue
            result = self._process_day(d)
            if result == "no_data":
                empty_streak += 1
                if empty_streak >= 3:
                    break  # no more historical data
            else:
                empty_streak = 0

    def _process_day(self, d: date):
        """Check and process a single day through the pipeline. Returns status string."""
        # Step 1: Is transcribed log already written?
        if not is_day_transcribed(d):
            # Check if the day is fully GPU-transcribed
            records = fetch_day_records(d)
            if not records:
                return "no_data"

            if is_day_fully_gpu_transcribed(records):
                print(f"[pipeline] Day {d}: all {len(records)} records GPU-transcribed. Writing log...")
                path = write_transcription_log(d, records)
                print(f"[pipeline] Written: {path}")
            else:
                # Count remaining
                remaining = sum(1 for r in records if r.get("transcribed_by") != "gpu"
                                and r.get("text", "").strip() and r.get("transcribed_by") != "")
                if d not in self._days_in_progress:
                    print(f"[pipeline] Day {d}: {remaining}/{len(records)} still need GPU transcription")
                    self._days_in_progress.add(d)
                return "incomplete"

        # Step 2: Transcribed log exists. Does summary exist?
        if not is_day_summarized(d):
            # Verify Ollama is running before attempting summarization
            try:
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                if r.status_code != 200:
                    raise ConnectionError()
            except Exception:
                if d not in self._days_in_progress:
                    print(f"[pipeline] Day {d}: ready for summary but Ollama not available. Will retry.")
                    self._days_in_progress.add(d)
                return "waiting_ollama"

            print(f"[pipeline] Day {d}: transcription log exists, generating summary...")
            records = fetch_day_records(d)
            if records:
                try:
                    path = generate_daily_summary(d, records)
                    print(f"[pipeline] Summary written: {path}")
                except Exception as e:
                    print(f"[pipeline] Summary FAILED for {d}: {e}")
                    traceback.print_exc()

        return "done"


# ===========================================================================
# Main Worker Loop
# ===========================================================================
class Worker:
    """
    Unified worker with priority scheduling:
      P1: Fresh transcriptions (every 3s)
      P2: Re-transcriptions (when P1 queue empty)
      P3+P4: Day completion pipeline (when P1+P2 idle)
    """

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber
        self.pipeline = Pipeline()
        self._stop = threading.Event()
        self._thread = None
        self._failed = {}
        self.stats = {
            "fresh_transcribed": 0,
            "retranscribed": 0,
            "errors": 0,
            "days_logged": 0,
            "days_summarized": 0,
            "last_activity": None,
            "state": "idle",
        }

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="worker")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)

    def _run(self):
        print("[worker] Started. Priority: fresh > re-transcribe > pipeline")
        while not self._stop.is_set():
            try:
                # P1: Fresh transcriptions
                did_fresh = self._do_fresh()

                if did_fresh:
                    continue  # Loop back immediately for more fresh work

                # P2: Re-transcription (one item)
                did_retrans = self._do_retranscribe()

                if not did_retrans:
                    # P3+P4: Pipeline checks (day completion + summarization)
                    self.stats["state"] = "pipeline"
                    self.pipeline.check()
                    self.stats["state"] = "idle"
                    # Longer sleep when truly idle
                    self._stop.wait(POLL_INTERVAL)
                else:
                    # Brief pause between retrans batches
                    self._stop.wait(RETRANS_INTERVAL)

            except Exception as e:
                print(f"[worker] Error: {e}")
                traceback.print_exc()
                self._stop.wait(10)

    def _do_fresh(self) -> bool:
        """Process fresh untranscribed items. Returns True if work was done."""
        records = fetch_untranscribed(limit=BATCH_SIZE)
        if not records:
            return False

        self.stats["state"] = "transcribing (fresh)"
        print(f"[P1:fresh] {len(records)} item(s)")

        for record in records:
            if self._stop.is_set():
                break
            self._transcribe_one(record, "P1:fresh")

        return True

    def _do_retranscribe(self) -> bool:
        """Re-transcribe a batch of Pi-transcribed items. Returns True if work was done."""
        records = fetch_pi_transcribed(limit=RETRANS_BATCH)
        if not records:
            return False

        self.stats["state"] = "transcribing (retrans)"

        for idx, record in enumerate(records):
            if self._stop.is_set():
                break

            rid = record["id"]
            clip = record.get("clip", "")
            old_text = record.get("text", "")
            ts = record.get("time", "?")

            if not clip:
                post_result(rid, old_text, "gpu")
                self.stats["retranscribed"] += 1
                continue

            audio = load_audio(clip)
            if audio.size == 0:
                post_result(rid, old_text, "gpu")
                self.stats["retranscribed"] += 1
                continue

            try:
                text = self.transcriber.transcribe(audio)
            except Exception as e:
                print(f"[P2:retrans] Error {rid}: {e}")
                post_result(rid, old_text, "gpu")
                self.stats["retranscribed"] += 1
                continue

            post_result(rid, text, "gpu")
            self.stats["retranscribed"] += 1
            self.stats["last_activity"] = datetime.now().isoformat()

            changed = " *" if text != old_text else ""
            disp = (text[:70] + "...") if len(text) > 70 else (text or "(silence)")
            print(f"[P2:retrans] {ts} -> {disp}{changed}")

        return True

    def _transcribe_one(self, record: dict, tag: str):
        """Transcribe a single fresh record."""
        rid = record["id"]
        clip = record.get("clip", "")
        ts = record.get("time", "?")
        name = record.get("name", "?")

        if self._failed.get(rid, 0) >= 3:
            return

        if not clip:
            post_result(rid, "", "gpu")
            return

        # Load audio (retry for NAS sync)
        audio = np.zeros(0, dtype=np.float32)
        for _ in range(4):
            audio = load_audio(clip)
            if audio.size > 0:
                break
            time.sleep(2)

        if audio.size == 0:
            self._failed[rid] = self._failed.get(rid, 0) + 1
            self.stats["errors"] += 1
            return

        try:
            text = self.transcriber.transcribe(audio)
        except Exception as e:
            print(f"[{tag}] Error {rid}: {e}")
            self.stats["errors"] += 1
            return

        result = post_result(rid, text, "gpu")
        if result:
            self.stats["fresh_transcribed"] += 1
            self.stats["last_activity"] = datetime.now().isoformat()
            disp = (text[:70] + "...") if len(text) > 70 else (text or "(silence)")
            print(f"[{tag}] {ts} | {name} -> {disp}")


# ===========================================================================
# Flask Status API
# ===========================================================================
def create_api(worker: Worker):
    from flask import Flask, jsonify
    api = Flask(__name__)

    @api.route("/status")
    def status():
        return jsonify({
            "service": "gpu-server-unified",
            "whisper_model": WHISPER_MODEL,
            "whisper_device": WHISPER_DEVICE,
            "ollama_model": OLLAMA_MODEL,
            "poll_interval": POLL_INTERVAL,
            **worker.stats,
        })

    @api.route("/pipeline")
    def pipeline_status():
        """Show status of daily files."""
        today = date.today()
        days = []
        for i in range(30):
            d = today - timedelta(days=i)
            days.append({
                "date": str(d),
                "transcribed_log": is_day_transcribed(d),
                "summary": is_day_summarized(d),
            })
        return jsonify({"days": days})

    return api


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("  GPU Server — Unified Transcription + Summary Pipeline")
    print("=" * 60)
    print(f"  Pi:       {PI_URL}")
    print(f"  Clips:    {CLIPS_BASE}")
    print(f"  Whisper:  {WHISPER_MODEL} ({WHISPER_COMPUTE_TYPE}) on {WHISPER_DEVICE}")
    print(f"  Ollama:   {OLLAMA_MODEL} at {OLLAMA_URL}")
    print(f"  Output:   {SUMMARIES_DIR}")
    print(f"            {TRANSCRIBED_DIR}")
    print(f"  API:      http://0.0.0.0:{API_PORT}")
    print(f"  Priority: P1=fresh, P2=retrans, P3=log, P4=summary")
    print("=" * 60)

    # Load Whisper
    transcriber = Transcriber()
    transcriber.load()

    # Start worker
    worker = Worker(transcriber)
    worker.start()

    # Start Flask API
    api = create_api(worker)
    try:
        api.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[main] Shutting down...")
        worker.stop()
        print("[main] Done.")


if __name__ == "__main__":
    main()
