# Radio Scanner AI

A 24/7 radio scanner system that captures, transcribes, and analyzes police/fire/EMS
transmissions using AI. Built around a Uniden BCD436HP digital scanner connected to a
Raspberry Pi, with optional GPU acceleration for fast transcription.

## What It Does

- **Captures** radio transmissions via BCD436HP scanner (serial + audio)
- **Transcribes** speech to text using OpenAI Whisper (locally on Pi or via GPU server)
- **Decodes** police 10-codes, signal codes, license plates, phone numbers, DTMF tones
- **Displays** everything on a real-time web dashboard accessible from any device
- **Records** all audio clips as MP3 for playback and archival

## Hardware Requirements

| Component | Purpose | Cost |
|-----------|---------|------|
| Raspberry Pi 3B+ or newer | Capture + dashboard | $35-75 |
| Uniden BCD436HP scanner | Receives radio signals | ~$400 |
| USB sound card (Sabrent AU-MMSA) | Captures scanner audio | ~$8 |
| 3.5mm audio cable | Scanner headphone → sound card mic | ~$5 |
| USB cable (scanner to Pi) | Serial control of scanner | included |
| MicroSD card (32GB+) | Pi OS + clips storage | ~$10 |

**Optional for faster transcription:**
- PC with NVIDIA GPU (any GTX 1060+ or RTX card)

## Quick Start (Pi-Only Setup)

```bash
# On your Raspberry Pi:
git clone https://github.com/YOUR_USERNAME/radio-scanner-ai.git
cd radio-scanner-ai/pi
chmod +x install.sh
./install.sh

# Edit config (set your audio device and serial port):
nano ~/scanner/config.py

# Start:
sudo systemctl start pi-scanner pi-dashboard pi-transcriber

# Open dashboard:
# http://<pi-ip>:8080
```

## Deployment Modes

### 1. Pi-Only (Default)
Everything runs on the Pi. Transcription is slow (~15x realtime with tiny.en model)
but fully standalone — no network, no extra hardware needed.

### 2. Pi + GPU Server
Pi captures audio, a PC with NVIDIA GPU transcribes 50x faster using `large-v3` model.
Set `GPU_SERVER_URL` in config.py and run `gpu_transcribe_server.py` on your PC.

```bash
# On your PC (Windows/Linux with NVIDIA GPU):
cd gpu-server
pip install -r requirements.txt
python gpu_transcribe_server.py
```

### 3. Pi + NAS + GPU Server
Full setup with network-attached storage for clip archival.
Set `DEPLOYMENT_MODE = "nas"` and configure your NAS mount.

## Dashboard

The web dashboard shows all transmissions in real-time:

- Live scanner LCD display mirror
- Searchable transmission list with text, decoded badges, and audio playback
- Filterable by System, Group, Channel, Frequency
- Decoded column shows 10-codes, signal codes, license plates, phone numbers
- Toggle visibility of decoder types (DTMF, CW, FSK, Tones, Codes, Plates, Phones)
- Pagination with adjustable page size
- CPU/RAM/Temperature monitoring

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Raspberry Pi                     │
│                                                   │
│  pi_scanner.py ─── Serial ──── BCD436HP Scanner  │
│       │                                           │
│       ├── Audio capture (ALSA)                    │
│       ├── WAV save + SQLite insert                │
│       ├── Tone/DTMF/FSK analysis                  │
│       └── Stuck channel watchdog                  │
│                                                   │
│  pi_transcriber.py (separate process)             │
│       └── Whisper tiny.en (local fallback)        │
│                                                   │
│  dashboard.py (Flask web server)                  │
│       ├── Real-time web UI                        │
│       ├── Audio playback                          │
│       ├── REST API for GPU server                 │
│       └── Text decoder (codes, plates, phones)    │
│                                                   │
│  scanner.db (SQLite)                              │
│       └── All transmissions + metadata            │
└───────────────────────┬─────────────────────────-─┘
                        │ HTTP API
                        ▼
┌───────────────────────────────────────────────────┐
│           GPU Server (optional, on PC)             │
│                                                    │
│  gpu_transcribe_server.py                          │
│       ├── Polls Pi for untranscribed records       │
│       ├── Reads audio clips (local/NAS)            │
│       ├── Whisper large-v3 on CUDA (fast)          │
│       └── Posts results back to Pi API             │
└────────────────────────────────────────────────────┘
```

## Process Separation

The system uses three independent processes on the Pi for reliability:

| Service | Purpose | Priority | Memory |
|---------|---------|----------|--------|
| `pi-scanner` | Audio capture + serial control | Normal | ~25MB |
| `pi-transcriber` | Local Whisper (fallback) | Nice 15 | ~200MB |
| `pi-dashboard` | Web UI + API | Normal | ~50MB |

If the transcriber crashes or runs out of memory, the scanner keeps capturing.
Systemd automatically restarts any failed service.

## Configuration

All settings are in `pi/config.py`. Key settings:

```python
SERIAL_PORT = "/dev/ttyACM0"     # Scanner USB serial
AUDIO_DEVICE = "plughw:1,0"      # USB sound card (run arecord -l)
CLIPS_DIR = "/home/pi/scanner/clips"
GPU_SERVER_URL = ""               # Empty = Pi-only mode
MAX_HOLD_SEC = 60                 # Watchdog: force channel change after 60s
MAX_CLIP_DURATION_SEC = 30.0      # Cap clips to prevent CPU overload
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard web UI |
| `/api/table` | GET | Table data (AJAX refresh) |
| `/api/status` | GET | Scanner status + LCD screen |
| `/api/sysinfo` | GET | CPU, RAM, temperature |
| `/api/filter_options` | GET | Cascading dropdown options |
| `/api/untranscribed` | GET | Records awaiting transcription |
| `/api/pi_transcribed` | GET | Records for GPU re-transcription |
| `/api/transcribe_result` | POST | GPU posts transcription result |
| `/api/batch_transcribe_result` | POST | GPU posts batch results |
| `/audio/<path>` | GET | Serve audio file for playback |

## Decoded Data

The system automatically decodes:

- **10-codes**: Indiana 10-code system (10-4, 10-42, etc.)
- **Signal codes**: Allen County / Fort Wayne / ISP signal numbers
- **License plates**: Phonetic alphabet spelling (Adam Boy Charles...)
- **Phone numbers**: Detected in transcribed text
- **DTMF tones**: Touch-tone digit sequences (PTT-IDs)
- **Morse/CW**: Morse code detection
- **Tones**: Alert/page tones (two-tone, single-tone)

Code tables are customizable in `codes.py` for your local agencies.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

Quick checks:
```bash
# Is the scanner connected?
ls /dev/ttyACM* /dev/ttyUSB*

# Is audio working?
arecord -D plughw:1,0 -f S16_LE -r 16000 -d 5 test.wav && aplay test.wav

# Service status:
systemctl status pi-scanner pi-dashboard pi-transcriber

# Logs:
journalctl -u pi-scanner --no-pager -f
```

## License

MIT License. See [LICENSE](LICENSE).

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — GPU inference
- [pywhispercpp](https://github.com/aAbstract/pywhispercpp) — Pi-native Whisper
- [Uniden](https://www.uniden.com/) — BCD436HP scanner
- [RadioReference](https://www.radioreference.com/) — frequency database
