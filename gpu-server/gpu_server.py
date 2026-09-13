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
    # Unit designators (most common misrecognition source)
    "Medic 1, Medic 2, Medic 3, Medic 4, Medic 5, Medic 7, Medic 8, Medic 9, "
    "Medic 11, Medic 12, Medic 15, Medic 21, Medic 25, Medic 35, Medic 45, "
    "Medic 71, Medic 81, Medic 94, Medic 95, Medic 108, Medic 115, Medic 135, Medic 195. "
    "Engine 1, Engine 2, Engine 3, Engine 4, Engine 5, Engine 7, Engine 8, Engine 9, "
    "Engine 11, Engine 12, Engine 15, Engine 181. "
    "Ladder 1, Ladder 2, Ladder 3, Ladder 7. "
    "Squad 1, Squad 2, Squad 3, Squad 4, Squad 5. "
    "Battalion 1, Battalion 2, Battalion 3. "
    "Unit 1, Unit 2, Unit 3, Unit 4, Unit 5, Unit 6, Unit 7, Unit 8. "
    "Rescue 1, Rescue 2, Rescue 3. "
    "6 is en route. 20 is en route. 35 is en route. 45 is en route. "
    "6, clear. 20, clear. 35, clear. Show me en route. Show me on scene. "
    "Dispatch, county dispatch, city dispatch. "
    "Show us en route. Show us on scene. Show me out. Show us clear. "
    "10-4, copy. 10-8, in service. 10-42, end of shift. 10-76, en route. "
    "Signal 22, signal 30, signal 40, signal 43, signal 46, signal 50, signal 75. "
    "Copy that. Be advised. Negative. Affirmative. Roger. Clear. "
    "Emergency run. Stat transfer. Priority 1. Code 3. "
    "Parkview Hospital, Lutheran Hospital, Dupont Hospital. "
    "Parkview North, Parkview South, Parkview Randallia, Parkview Whitley. "
    "Patient, chest pain, difficulty breathing, cardiac arrest, unresponsive. "
    "EMS, paramedic, EMT, ambulance, first responders. "
    "Structure fire, working fire, mutual aid. "
    "Sheriff, deputy, trooper, officer, sergeant. "
    "Traffic stop, vehicle pursuit, suspect, subject, complainant. "
    "Vehicle, plate, registration, driver's license, warrant. "
    "Adam Boy Charles David Edward Frank George Henry Ida John King Lincoln "
    "Mary Nora Ocean Paul Queen Robert Sam Tom Union Victor William X-ray Young Zebra. "
    "Fort Wayne, Allen County, Whitley County, DeKalb County, Noble County, "
    "Adams County, Wells County, Huntington County, LaGrange County, Steuben County, Indiana. "
    "Coliseum, Coldwater, Lima Road, State Road 3, State Road 9, "
    "Interstate 69, Interstate 469, US 30, US 33. "
    "Clinton, Calhoun, Jefferson, Washington, Lafayette, Stellhorn, Dupont, "
    "Maysville Road, Goshen Road, Bluffton Road, Decatur Road. "
    "Auburn, Garrett, Kendallville, Ligonier, Columbia City, Bluffton, Decatur. "
)

# Post-processing corrections for consistent Whisper errors on scanner audio.
# Applied after transcription. Keys are regex patterns (case-insensitive).
WHISPER_CORRECTIONS = {
    r"\bmay i agree\b": "Medic 3",
    r"\bmedic free\b": "Medic 3",
    r"\bmagic (\d)": r"Medic \1",
    r"\bengine for\b": "Engine 4",
    r"\bengine won\b": "Engine 1",
    r"\bunit won\b": "Unit 1",
    r"\bunit too\b": "Unit 2",
    r"\bunit to\b": "Unit 2",
    r"\bsquad won\b": "Squad 1",
    r"\bsquad too\b": "Squad 2",
    r"\bladder won\b": "Ladder 1",
    r"\bbattalion won\b": "Battalion 1",
    r"\bshow us in route\b": "show us en route",
    r"\bin route\b": "en route",
    r"\bsingle (\d+)": r"signal \1",
    r"\b10 for\b": "10-4",
    r"\b10 42\b": "10-42",
    r"\b10 8\b": "10-8",
    r"\bpark view\b": "Parkview",
    r"\bcold water\b": "Coldwater",
    r"\bwhite cheese and rum\b": "20 is en route",
    # "one" as unit number (when followed by status/action verbs)
    r"\bone (have in custody|is en route|is on scene|is clear|is 10-8|show me|, clear|, en route|, on scene)\b": r"1 \1",
    # Whisper hallucinations (outputs memorized training data on bad audio)
    r"(?i)closed captioning.*": "",
    r"(?i)subtitles by the amara\.org community": "",
    r"(?i)thanks for watching": "",
    r"(?i)please subscribe": "",
    r"(?i)thank you for watching": "",
}

# Ollama (for summarization)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Timing
POLL_INTERVAL = 5        # seconds between fresh transcription polls
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
            if seg.no_speech_prob > 0.5 or seg.avg_logprob < -1.0:
                continue
            parts.append(seg.text.strip())
        text = " ".join(parts).strip()

        # Apply post-processing corrections
        if text and WHISPER_CORRECTIONS:
            import re
            for pattern, replacement in WHISPER_CORRECTIONS.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text


# ===========================================================================
# Audio Loading
# ===========================================================================
_CLIP_CACHE_DIR = os.path.join(os.environ.get("TEMP", r"C:\Temp"), "scanner_clips")


def _fetch_clip_from_pi(clip_path: str) -> str:
    """
    Fetch a clip from the Pi via HTTP when it's stored locally on the Pi
    (not on NAS). Downloads to a temp cache directory.
    Returns local path to the downloaded file, or None on failure.
    """
    import urllib.parse
    os.makedirs(_CLIP_CACHE_DIR, exist_ok=True)

    # Build URL: Pi dashboard serves clips at /audio/<path>
    encoded = urllib.parse.quote(clip_path, safe="")
    url = f"{_pi_url()}/audio/{encoded}"

    # Use filename as cache key
    filename = os.path.basename(clip_path)
    local_path = os.path.join(_CLIP_CACHE_DIR, filename)

    # Skip if already cached
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
        return local_path

    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 100:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
    except Exception:
        pass

    return None


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
                # Try fetching from Pi via HTTP
                clip_path = _fetch_clip_from_pi(clip_path) or ""
                if not clip_path:
                    return np.zeros(0, dtype=np.float32)
        elif clip_path:
            # Not on local/NAS — try Pi HTTP
            clip_path = _fetch_clip_from_pi(clip_path) or ""
            if not clip_path:
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
      - text is empty/blank (silence clips that neither Pi nor GPU could transcribe), OR
      - text is an error message (audio not found, etc.) that can't be re-transcribed
    """
    # Error texts that indicate a permanently un-transcribable record
    ERROR_TEXTS = {"(audio not found)", "(no speech)", "[BLANK_AUDIO]", ""}

    if not records:
        return False
    for r in records:
        tb = r.get("transcribed_by", "")
        text = r.get("text", "").strip()
        # GPU-transcribed: done
        if tb == "gpu":
            continue
        # Empty/error text with no transcriber: can't be improved, count as done
        if not tb and (not text or text in ERROR_TEXTS):
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


# --- Name/address regex extraction (addresses aren't in decoded_text) -------
_NAME_PATTERNS = [
    r"registered\s+(?:to|owner)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:RP|complainant|reporting party)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:subject|suspect|driver|passenger)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:resident|homeowner|victim)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"lives?\s+(?:at|with)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
]

_ADDRESS_PATTERNS = [
    r"(?:at|address\s+(?:is|of)?|location\s*:?|responding\s+to|en\s+route\s+to|"
    r"headed\s+to|going\s+to)\s+"
    r"(\d+\s+(?:block\s+(?:of\s+)?)?(?:(?:North|South|East|West|N|S|E|W)\.?\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r"(?:\s+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
    r"Court|Ct|Way|Place|Pl|Circle|Cir|Parkway|Pkwy|Trail|Terrace|Pike)\.?)?)",
    r",\s+(\d+\s+(?:(?:North|South|East|West|N|S|E|W)\.?\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
    r"Court|Ct|Way|Place|Pl|Circle|Cir|Parkway|Pkwy|Trail|Terrace|Pike)\.?)",
    r"(?:at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]

_NOT_NAMES = {
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry",
    "Ida", "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul",
    "Queen", "Robert", "Sam", "Tom", "Union", "Victor", "William", "Young",
    "Zebra", "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
    "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar",
    "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Whiskey",
    "Yankee", "Zulu", "Signal", "Copy", "Clear", "Roger", "Dispatch",
    "County", "Allen", "Noble", "Fort", "Wayne", "Indiana",
    "North", "South", "East", "West",
}


def _extract_names(text: str) -> list:
    """Extract person names from a transcript via regex."""
    if not text:
        return []
    import re
    out = []
    for pattern in _NAME_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            name = " ".join(w.capitalize() for w in m.group(1).strip().split())
            parts = name.split()
            if len(parts) < 2:
                continue
            if parts[0] in _NOT_NAMES and parts[1] in _NOT_NAMES:
                continue
            out.append(name)
    return list(dict.fromkeys(out))  # dedupe, preserve order


def _extract_addresses(text: str) -> list:
    """Extract street addresses from a transcript via regex."""
    if not text:
        return []
    import re
    out = []
    for pattern in _ADDRESS_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            addr = m.group(1).strip()
            if len(addr) > 8:
                out.append(addr)
    return list(dict.fromkeys(out))


def _write_summary_report(filepath, target_date, grouped, channel_summaries, records):
    """Write the final markdown summary report."""
    # Entity extraction for quick reference:
    #  - plates/phones from decoded_text (already decoded)
    #  - names/addresses via regex from transcript text
    plates, names, addresses, phones = [], [], [], []
    for r in records:
        dt = r.get("decoded_text", {}) or {}
        t = r.get("time", "")
        ts = t.split("T")[1][:5] if "T" in t else t
        ch = r.get("channel", "") or r.get("group", "")
        text = r.get("text", "")
        ctx = text[:60].replace("|", "/")
        for p in dt.get("plates", []):
            plates.append(f"| {ts} | {ch[:25]} | **{p}** | {ctx} |")
        for p in dt.get("phones", []):
            phones.append(f"| {ts} | {ch[:25]} | {p} | {ctx} |")
        for nm in _extract_names(text):
            names.append(f"| {ts} | {ch[:25]} | **{nm}** | {ctx} |")
        for addr in _extract_addresses(text):
            addresses.append(f"| {ts} | {ch[:25]} | {addr} | {ctx} |")

    def _write_table(f, title, rows, header):
        f.write(f"### {title}\n\n")
        if rows:
            f.write(header + "\n")
            # Separator: one "---" per column (columns = pipes - 1)
            ncols = header.count("|") - 1
            f.write("|" + "------|" * ncols + "\n")
            seen = set()
            for line in rows:
                if line not in seen:
                    seen.add(line)
                    f.write(line + "\n")
            f.write("\n")
        else:
            f.write(f"*None detected.*\n\n")

    day_name = target_date.strftime("%A")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Scanner Daily Summary — {target_date} ({day_name})\n\n")
        f.write(f"**Total transmissions:** {len(records)}  \n")
        f.write(f"**Active channels:** {len(grouped)}  \n")
        f.write(f"**Plates detected:** {len(plates)}  \n")
        f.write(f"**Names detected:** {len(names)}  \n")
        f.write(f"**Addresses detected:** {len(addresses)}  \n\n---\n\n")
        f.write("## Events\n\n")

        # Busiest channels first
        for ch in sorted(channel_summaries.keys(), key=lambda k: -len(grouped.get(k, []))):
            summary = channel_summaries[ch]
            count = len(grouped.get(ch, []))
            f.write(f"### {ch}\n*{count} transmissions*\n\n{summary}\n\n---\n\n")

        # Quick Reference
        f.write("## Quick Reference\n\n")
        _write_table(f, "License Plates", plates, "| Time | Channel | Plate | Context |")
        _write_table(f, "Names Mentioned", names, "| Time | Channel | Name | Context |")
        _write_table(f, "Addresses Mentioned", addresses, "| Time | Channel | Address | Context |")
        _write_table(f, "Phone Numbers", phones, "| Time | Channel | Phone | Context |")

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

    def check(self, allow_summary: bool = True):
        """
        Checks if any past days need transcribed logs or summaries.
        
        allow_summary: if False, only writes transcription logs (cheap) and
        skips Ollama summarization (expensive, competes with Whisper for GPU).
        Set False when the re-transcription queue is still busy.
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
            result = self._process_day(d, allow_summary=allow_summary)
            if result == "no_data":
                empty_streak += 1
                if empty_streak >= 3:
                    break  # no more historical data
            else:
                empty_streak = 0

    def _process_day(self, d: date, allow_summary: bool = True):
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
            # Defer summarization when retrans is busy (Ollama competes with Whisper)
            if not allow_summary:
                return "log_only"
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

                # P3+P4: Pipeline checks run on a timer regardless of retrans backlog.
                # (Previously starved when retrans queue was large — pipeline never ran.)
                # When retrans is busy, only write transcription logs (cheap); defer
                # Ollama summaries until retrans is idle to avoid GPU contention.
                self.pipeline.check(allow_summary=not did_retrans)

                if not did_retrans:
                    self._stop.wait(POLL_INTERVAL)
                else:
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
        count_before = self.stats["fresh_transcribed"] + self.stats["errors"]

        print(f"[P1:fresh] {len(records)} item(s)")

        for record in records:
            if self._stop.is_set():
                break
            self._transcribe_one(record, "P1:fresh")

        count_after = self.stats["fresh_transcribed"] + self.stats["errors"]
        # Only report "work done" if we actually processed something
        # (prevents tight loop when all items are in retry-wait state)
        return count_after > count_before

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

        if self._failed.get(rid, 0) >= 2:
            # Permanently failed — mark as done so it leaves the queue
            post_result(rid, "(audio not found)", "gpu")
            print(f"[{tag}] {rid} failed 2x, marking as (audio not found)")
            del self._failed[rid]
            return

        if not clip:
            post_result(rid, "", "gpu")
            return

        # Load audio (single attempt + Pi HTTP fallback, no long retries)
        audio = load_audio(clip)

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
