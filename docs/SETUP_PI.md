# Pi Setup Guide

## Hardware Connections

```
BCD436HP Scanner
  ├── USB cable ──────────► Pi USB port (serial control)
  └── 3.5mm headphone ──► USB Sound Card (mic/line input) ──► Pi USB port
```

1. Connect scanner's **USB port** to Pi (provides serial control)
2. Connect scanner's **headphone jack** to USB sound card's **mic input** (pink)
3. Plug USB sound card into another Pi USB port
4. Set scanner volume to ~50% (too loud = clipping, too quiet = silence)

## Software Installation

### Prerequisites
- Raspberry Pi 3B+ or newer (1GB+ RAM)
- Raspberry Pi OS Lite (no desktop needed — saves ~300MB RAM)
- Network connection (for initial setup)

### Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/radio-scanner-ai.git
cd radio-scanner-ai/pi

# Run installer
chmod +x install.sh
./install.sh
```

### Configure

```bash
nano ~/scanner/config.py
```

You need to set two things:

**1. Audio device** — find yours with:
```bash
arecord -l
```
Look for your USB sound card (usually card 1 or 2):
```
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
```
Set `AUDIO_DEVICE = "plughw:1,0"` (use your card number)

**2. Serial port** — find yours with:
```bash
ls /dev/ttyACM* /dev/ttyUSB*
```
Usually `/dev/ttyACM0` for the BCD436HP.

### Test Audio

```bash
# Record 5 seconds and play back
source ~/scanner/venv/bin/activate
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav
aplay /tmp/test.wav
```

You should hear whatever was playing on the scanner.

### Test Serial

```bash
source ~/scanner/venv/bin/activate
python3 -c "
from scanner_serial import ScannerSerial
s = ScannerSerial('/dev/ttyACM0')
s.open()
print('Model:', s.get_model())
print('State:', s.glance())
s.close()
"
```

Should print the scanner model and current state.

### Start Services

```bash
sudo systemctl start pi-scanner pi-dashboard pi-transcriber
```

### Access Dashboard

Open a browser to: `http://<pi-ip-address>:8080`

Find your Pi's IP with: `hostname -I`

## Scanner Setup (BCD436HP)

The scanner must be in **database scan mode** (not Close Call or Search):

1. Power on the scanner
2. Enter your zip code when prompted
3. Press SCAN to begin scanning
4. Enable desired service types (Menu → Service Types)
5. Connect USB cable to Pi

The Pi communicates via serial at 115200 baud. No special scanner configuration
is needed beyond enabling the USB connection.

## Running Headless (No Desktop)

For best performance, run Pi OS Lite (no GUI) or disable the desktop:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

This saves ~300MB RAM and ~10% CPU.

## Disable Unnecessary Services

```bash
# PackageKit (background package updates)
sudo systemctl disable packagekit
sudo systemctl stop packagekit

# Bluetooth (if not needed)
sudo systemctl disable bluetooth
```

## Service Management

```bash
# Check status
systemctl status pi-scanner pi-dashboard pi-transcriber

# View logs
journalctl -u pi-scanner -f          # Live scanner log
journalctl -u pi-transcriber -f      # Transcription log
journalctl -u pi-dashboard -f        # Dashboard log

# Restart
sudo systemctl restart pi-scanner
sudo systemctl restart pi-transcriber
sudo systemctl restart pi-dashboard

# Stop everything
sudo systemctl stop pi-scanner pi-transcriber pi-dashboard
```

## Storage Management

Audio clips accumulate over time (~600MB per 30,000 transmissions at 32kbps MP3).

To check disk usage:
```bash
du -sh ~/scanner/clips/
df -h /
```

To delete old clips (keep last 7 days):
```bash
find ~/scanner/clips/ -name "*.mp3" -mtime +7 -delete
```

## Upgrading

```bash
cd radio-scanner-ai
git pull
cp pi/*.py ~/scanner/
sudo systemctl restart pi-scanner pi-transcriber pi-dashboard
```
