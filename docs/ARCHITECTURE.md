# Architecture

## System Overview

The Radio Scanner AI system is built around three independent processes that communicate
through a shared SQLite database. This separation ensures the critical audio capture
path is never blocked by CPU-intensive transcription.

## Data Flow

```
BCD436HP Scanner
     │
     ├── USB Serial (/dev/ttyACM0) ──► pi_scanner.py
     │       • GLG polling (squelch detection)
     │       • STS polling (LCD screen mirroring)
     │       • Stuck channel watchdog (60s max)
     │
     └── Audio (3.5mm headphone) ──► USB Sound Card ──► pi_scanner.py
             • ALSA capture (16kHz mono)
             • Pre-buffer (0.5s before squelch)
             • Silence-based splitting
             • Tone/DTMF/FSK analysis
             • WAV file save
             • SQLite record insert

SQLite Database (scanner.db)
     │
     ├── pi_transcriber.py reads untranscribed records
     │       • Polls every 3s
     │       • Whisper tiny.en (local CPU)
     │       • Yields to GPU when online
     │       • Updates text + decoded_text
     │
     ├── dashboard.py serves web UI + API
     │       • Flask (threaded mode)
     │       • Real-time AJAX updates
     │       • Audio file serving
     │       • Inline text decoding on GPU results
     │
     └── GPU Server (remote) polls via HTTP API
             • Fetches untranscribed records
             • Reads clips from NAS/network
             • Whisper large-v3 on CUDA
             • Posts results back to Pi API
```

## SQLite Schema

```sql
CREATE TABLE transmissions (
    id TEXT PRIMARY KEY,           -- 12-char hex UUID
    time TEXT NOT NULL,            -- ISO timestamp (2026-07-18T14:30:22)
    frequency TEXT DEFAULT '',     -- TGID or frequency
    name TEXT DEFAULT '',          -- Display name (site / group / channel)
    system TEXT DEFAULT '',        -- Radio system name
    grp TEXT DEFAULT '',           -- Group/department name
    channel TEXT DEFAULT '',       -- Channel name
    duration_sec REAL DEFAULT 0,   -- Clip length in seconds
    text TEXT DEFAULT '',          -- Transcribed text (or "Transcribing...")
    transcribed INTEGER DEFAULT 0, -- 0=pending, 1=done
    transcribed_by TEXT DEFAULT '',-- "gpu" or "" (Pi local)
    clip TEXT DEFAULT '',          -- Path to audio file
    source TEXT DEFAULT '',        -- "pi_capture" or "pi_reingest"
    decoded TEXT DEFAULT '{}',     -- JSON: audio decoders (DTMF, tones, morse)
    decoded_text TEXT DEFAULT '{}',-- JSON: text decoders (codes, plates, phones)
    tags TEXT DEFAULT '{}',        -- JSON: scanner metadata
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
```

## Process Details

### pi_scanner.py (Critical Path)

**Purpose**: Capture transmissions. Must never block or crash.

- Polls BCD436HP via serial every 300ms (GLG command)
- Detects squelch open/close transitions
- Records audio via ALSA (pre-buffered)
- Splits long transmissions on silence gaps
- Runs fast audio analyzers (numpy-based, <50ms):
  - DTMF digit detection (Goertzel algorithm)
  - Steady tone detection (dispatch pages)
  - FSK/data burst detection (spectral flatness)
  - Morse code detection
- Saves WAV clip to disk
- Inserts placeholder record in SQLite ("Transcribing...")
- Stuck channel watchdog: sends KEY,L/O,P after 60s on same frequency
- Cleanup worker: WAV→MP3 conversion, orphan re-ingestion

**Resource usage**: ~25MB RAM, <5% CPU (no Whisper loaded)

### pi_transcriber.py (Background Worker)

**Purpose**: Transcribe audio when GPU is offline.

- Separate process with own systemd service
- Polls SQLite for `transcribed=0` records every 3s
- Checks GPU server availability every 30s
- If GPU online: idles (GPU handles it faster)
- If GPU offline: transcribes locally with Whisper tiny.en
- Marks "Transcribing now" during active processing
- Runs text decoders after transcription (codes, plates, phones)
- Capped at 400MB RAM, Nice 15 (expendable)

**Resource usage**: ~200MB RAM (Whisper model), 2-3 CPU cores when active

### dashboard.py (Web Server)

**Purpose**: Serve the web UI and REST API.

- Flask with `threaded=True` (parallel request handling)
- Serves main HTML page (server-rendered, AJAX-refreshed)
- API endpoints for GPU server communication
- Inline text decoding when GPU posts results
- Audio file serving with MP3 fallback
- Scanner LCD screen mirroring via status.json
- No-cache headers to prevent stale content

**Resource usage**: ~50MB RAM, <5% CPU

### gpu_transcribe_server.py (Optional, on PC)

**Purpose**: Fast transcription using GPU.

- Polls Pi's `/api/untranscribed` for pending records
- Reads audio clips from NAS (Windows UNC paths)
- Transcribes with faster-whisper large-v3 on CUDA
- Posts results to Pi's `/api/transcribe_result`
- Also re-transcribes Pi-transcribed records (better quality)
- IP caching to handle DNS issues
- Batch processing (10 items per cycle)

**Resource usage**: ~6GB VRAM, minimal CPU

## Deduplication

Records are deduplicated at display time by timestamp. If two records share the
exact same second-level timestamp, only the first (by ID) is shown. This handles:
- WAV+MP3 duplicate captures
- Re-ingested orphan files
- Race conditions between Pi and GPU transcription

## Stuck Channel Watchdog

The scanner poller tracks how long the BCD436HP stays on one active frequency.
If duration exceeds `MAX_HOLD_SEC` (default 60s):
1. Ends the current audio capture (saves what was recorded)
2. Sends `KEY,L/O,P` serial command (temporary lockout on scanner)
3. Scanner resumes scanning other frequencies
4. Lockout resets on scanner power cycle

## Audio Clip Lifecycle

1. Scanner detects squelch open → begin audio capture
2. Squelch closes → end capture, split on silence
3. Each segment saved as WAV (capped at 30s max)
4. WAV filename encodes: `YYYYMMDD_HHMMSS_N_System___Group___Channel___Freq.wav`
5. Placeholder inserted in DB with "Transcribing..."
6. Transcriber processes it (local or GPU)
7. Cleanup worker converts WAV→MP3 (ffmpeg, 32kbps mono)
8. WAV deleted after successful MP3 conversion

## Code Decoder System

The `codes.py` module uses agency profiles to decode radio codes:

1. `pick_profile(channel_name)` selects the code table based on channel
2. `decode_text(text, profile)` extracts 10-codes and signal codes
3. Only decodes when "signal" or "10-" appears in text (no bare number guessing)

Profiles included:
- Allen County (signal codes 1-154)
- Fort Wayne City PD (different meanings for same numbers)
- Indiana State Police
- Generic Indiana fallback (for surrounding counties)

## File Path Conventions

Audio clips are stored with date subfolders:
```
/home/pi/scanner/clips/20260718/20260718_143022_1_Indiana_Project_Hoosier___Noble_County__57____Sheriff__Dispatch___12092.mp3
```

Format: `YYYYMMDD/YYYYMMDD_HHMMSS_N_System___Group___Channel___Freq.ext`

Triple-underscore (`___`) separates metadata fields. This allows the re-ingest
worker to recover all metadata from just the filename if the DB record is lost.
