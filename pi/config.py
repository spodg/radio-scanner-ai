"""
Pi Scanner Configuration.

Edit this file to match your hardware setup. The defaults work for
a basic Pi-only installation with a BCD436HP and USB sound card.

DEPLOYMENT MODES:
  "local" - Clips stored on Pi, transcribe locally (simplest, no extra hardware)
  "gpu"   - Clips on Pi, GPU server handles transcription (fast, needs PC with GPU)
  "nas"   - Clips on NAS, GPU server transcribes (full setup, resilient storage)
"""

# =============================================================================
# DEPLOYMENT MODE
# =============================================================================
DEPLOYMENT_MODE = "local"  # "local", "gpu", or "nas"

# =============================================================================
# SCANNER HARDWARE (BCD436HP via USB serial)
# =============================================================================
# Run `ls /dev/ttyACM* /dev/ttyUSB*` to find your serial device.
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
POLL_INTERVAL = 0.3  # seconds between serial polls

# =============================================================================
# AUDIO CAPTURE (USB sound card)
# =============================================================================
# Run `arecord -l` to find your audio device.
# The Sabrent AU-MMSA typically shows as card 1 or 2.
AUDIO_DEVICE = "plughw:1,0"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Pre-buffer: capture audio from BEFORE squelch opens (catches start of speech)
AUDIO_PREBUFFER_SEC = 0.5

# =============================================================================
# TRANSMISSION DETECTION
# =============================================================================
# Ignore transmissions shorter than this (filters squelch blips)
MIN_TRANSMISSION_SEC = 1.5

# Silence-based splitting: break long transmissions on gaps
SILENCE_SPLIT_SEC = 1.0
SILENCE_SPLIT_RMS = 0.003

# Energy gate: skip clips below this RMS (dead silence)
WHISPER_SILENCE_RMS = 0.003

# =============================================================================
# LOCAL TRANSCRIPTION (runs in separate pi-transcriber process)
# =============================================================================
# tiny.en is the only model that fits in Pi 3's 1GB RAM.
# For Pi 4 (2GB+), you can try "base.en" for better accuracy.
WHISPER_MODEL = "tiny.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "en"

# Domain vocabulary hint (helps accuracy for scanner traffic)
WHISPER_PROMPT = (
    "Medic 1, Medic 2, Medic 3, Medic 4, Medic 5, Medic 7, Medic 9, "
    "Medic 11, Medic 12, Medic 35, Medic 45, Medic 71, Medic 81, Medic 94, Medic 195. "
    "Engine 1, Engine 2, Engine 3, Engine 4, Engine 5, Engine 11, Engine 181. "
    "Ladder 1, Ladder 2, Ladder 3. Squad 1, Squad 2, Squad 3. "
    "Battalion 1, Battalion 2. Unit 1, Unit 2, Unit 3, Unit 4, Unit 5. "
    "Dispatch, county dispatch. "
    "Show us en route. Show us on scene. Show me out. Show us clear. "
    "10-4, copy. 10-8, in service. 10-42, end of shift. "
    "Signal 22, signal 30, signal 40, signal 43, signal 50. "
    "Copy that. Be advised. Negative. Affirmative. Roger. Clear. "
    "Parkview Hospital, Lutheran Hospital. "
    "EMS, paramedic, ambulance. Sheriff, deputy, trooper. "
    "Traffic stop, suspect, subject, complainant. "
    "Vehicle, plate, registration, driver's license, warrant. "
    "Adam Boy Charles David Edward Frank George Henry Ida John King Lincoln "
    "Mary Nora Ocean Paul Queen Robert Sam Tom Union Victor William X-ray Young Zebra. "
    "Fort Wayne, Allen County, DeKalb County, Noble County, Indiana. "
)

# =============================================================================
# STORAGE
# =============================================================================
# Primary clip storage (NAS when available).
# For "local" mode: "/home/pi/scanner/clips"
# For "nas" mode:   "/mnt/nas/clips" (mount your NAS share first)
CLIPS_DIR = "/mnt/nas/clips"

# Local fallback when NAS is unavailable. Clips written here get synced
# to CLIPS_DIR automatically once NAS comes back online.
LOCAL_CLIPS_DIR = "/home/pi/scanner/clips"

# How often to check NAS availability (seconds)
NAS_CHECK_INTERVAL = 15

# Human-readable log file
LOG_FILE = "/home/pi/scanner/scanner_log.txt"

# SQLite database (always local on Pi for fast access)
DB_PATH = "/home/pi/scanner/scanner.db"

# =============================================================================
# GPU SERVER (optional — set DEPLOYMENT_MODE = "gpu" or "nas" to use)
# =============================================================================
# IP or hostname of the machine running gpu_server.py
# Used for status display on dashboard and auto-discovery.
GPU_SERVER_URL = "http://192.168.2.36:5555"

# How often Pi checks if GPU server is online (seconds)
GPU_CHECK_INTERVAL = 30

# =============================================================================
# NAS (optional — set DEPLOYMENT_MODE = "nas" to use)
# =============================================================================
# NAS share mount point (set up in /etc/fstab or ensure_nas.sh)
NAS_MOUNT = "/mnt/nas"
# NAS clips directory (set CLIPS_DIR to this when using NAS)
# NAS_CLIPS_DIR = "/mnt/nas/clips"

# =============================================================================
# DASHBOARD
# =============================================================================
DASHBOARD_PORT = 8080
DASHBOARD_HOST = "0.0.0.0"  # Listen on all interfaces

# =============================================================================
# ADVANCED / WATCHDOG
# =============================================================================
# Stuck channel watchdog: force channel change after this many seconds
MAX_HOLD_SEC = 60

# Max clip duration (truncate audio to prevent CPU overload during transcription)
MAX_CLIP_DURATION_SEC = 30.0
