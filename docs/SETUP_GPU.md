# GPU Server Setup

The GPU server is optional. It provides ~50x faster transcription using
a PC with an NVIDIA GPU. When running, the Pi's local transcriber idles
and the GPU handles all speech-to-text processing.

## Requirements

- Windows 10/11 or Linux with NVIDIA GPU
- GTX 1060 (6GB) or better (RTX recommended)
- Python 3.10+
- NVIDIA drivers + CUDA toolkit
- Network access to the Pi

## Installation

### Windows

```cmd
cd gpu-server
pip install -r requirements.txt
```

### Linux

```bash
cd gpu-server
pip install -r requirements.txt
```

### CUDA Verification

```python
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA toolkit from https://developer.nvidia.com/cuda-downloads

## Configuration

Edit the top of `gpu_transcribe_server.py`:

```python
# Pi dashboard URL (the GPU server fetches work from here)
PI_DASHBOARD_URL = "http://pi3:8080"  # or use IP: "http://192.168.1.50:8080"

# Model settings
WHISPER_MODEL = "large-v3"          # Best accuracy
WHISPER_COMPUTE_TYPE = "float32"    # Use "float16" for RTX cards, "float32" for GTX
```

### Model Selection

| Model | VRAM | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `large-v3` | 5-7GB | 1x realtime | Best | GTX 1080+ / RTX cards |
| `medium` | 3-4GB | 2x realtime | Good | GTX 1060 6GB |
| `small` | 2GB | 4x realtime | OK | GTX 1060 3GB |

### Compute Type

| Type | VRAM | Speed | Accuracy | Cards |
|------|------|-------|----------|-------|
| `float32` | Most | Slower | Best | GTX (Pascal) |
| `float16` | Less | Faster | Good | RTX (Turing+) |
| `int8_float16` | Least | Fastest | OK | Any |

For GTX 1080 Ti: use `float32` (Pascal has no native FP16 acceleration).
For RTX 3060+: use `float16` for best speed/quality balance.

## Running

### Windows
Double-click `start.bat` or:
```cmd
python gpu_transcribe_server.py
```

### Linux
```bash
python3 gpu_transcribe_server.py
```

## How It Works

1. GPU server polls Pi's `/api/untranscribed` every 3 seconds
2. For each pending record, reads the audio clip from the network path
3. Transcribes with faster-whisper (CUDA)
4. Posts result to Pi's `/api/transcribe_result`
5. Also re-transcribes Pi-transcribed records for better quality

## Network Path Mapping

The GPU server needs to read audio files. The Pi stores clips at paths like
`/mnt/nas/clips/20260718/file.mp3`. On Windows, these map to UNC paths:

```python
# In gpu_transcribe_server.py:
NAS_LINUX_PREFIX = "/mnt/nas/"
NAS_WINDOWS_PREFIX = r"\\NAS_HOSTNAME\share\"
```

For Pi-only setups (no NAS), you'll need to share the Pi's clips folder via
Samba or configure the GPU server to fetch clips via HTTP.

## Verifying Connection

The GPU server prints status on startup:
```
[gpu] Resolved pi3 to 192.168.1.50 (cached permanently)
================================================================
  GPU Transcription Server
================================================================
  Pi API: http://pi3:8080
  Model:  large-v3 on cuda (float32)
  Poll:   every 3s
  API:    http://0.0.0.0:5555
================================================================
[gpu] Model loaded and ready.
[gpu:fresh] Found 5 untranscribed item(s).
```

## Troubleshooting

**"getaddrinfo failed"** — DNS can't resolve Pi hostname.
Fix: Use IP address directly in `PI_DASHBOARD_URL`.

**"Connection refused"** — Pi dashboard isn't running.
Fix: `ssh pi@<ip> "sudo systemctl restart pi-dashboard"`

**"Could not load audio"** — NAS path mapping is wrong.
Fix: Check `NAS_LINUX_PREFIX` and `NAS_WINDOWS_PREFIX` in config.

**GPU not detected** — CUDA not installed or wrong driver.
Fix: `nvidia-smi` should show your GPU. Install CUDA toolkit if missing.
