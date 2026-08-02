# Troubleshooting

## Pi Not Responding / SSH Timeout

**Cause**: CPU overloaded from transcription.

**Fix**: The pi-transcriber service has resource limits (CPUQuota=200%, MemoryMax=400M).
If it exceeds memory, systemd kills and restarts it. Wait 15 seconds.

**Prevention**: This shouldn't happen with the process separation. If it does:
```bash
# From another machine:
ssh pi@<ip> "echo <password> | sudo -S systemctl stop pi-transcriber"
```

## No Audio Recorded

```bash
# Check if sound card is detected
arecord -l

# Test recording
arecord -D plughw:1,0 -f S16_LE -r 16000 -d 5 /tmp/test.wav
aplay /tmp/test.wav

# Check scanner volume (should be ~50%)
# Check cable connection (headphone → mic input)
```

## Scanner Not Detected (Serial)

```bash
# Check USB connection
ls /dev/ttyACM* /dev/ttyUSB*

# Test serial
python3 -c "
import serial
s = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
s.write(b'MDL\r')
print(s.read_until(b'\r'))
s.close()
"
```

If no `/dev/ttyACM*` appears, try a different USB port or cable.

## Dashboard Not Loading

```bash
# Check service
systemctl status pi-dashboard

# Check port
curl http://localhost:8080/api/status

# Restart
sudo systemctl restart pi-dashboard
```

## "Transcribing..." Stuck Forever

**Cause**: Pi transcriber crashed or GPU server is offline.

```bash
# Check transcriber
systemctl status pi-transcriber

# Check pending count
python3 -c "import scanner_db; print(scanner_db.get_pending_count())"

# Restart transcriber
sudo systemctl restart pi-transcriber
```

## GPU Server Can't Connect to Pi

**"getaddrinfo failed"**: Use IP address instead of hostname.
```python
PI_DASHBOARD_URL = "http://192.168.1.50:8080"  # Use Pi's actual IP
```

**"Connection refused"**: Dashboard isn't running on Pi.
```bash
ssh pi@<ip> "sudo systemctl restart pi-dashboard"
```

## Audio Playback Delay

**Cause**: Flask single-threaded blocking (fixed in latest version with `threaded=True`).

If still slow, check NAS connectivity:
```bash
time dd if=/mnt/nas/clips/SOMEFILE.mp3 of=/dev/null bs=64k
```
Should be <100ms for a 10KB file.

## High CPU / RAM Usage

```bash
# Check what's using resources
ps aux --sort=-%cpu | head -10
free -m

# The transcriber should be Nice 15 (low priority)
# Scanner should use <30MB
# If transcriber exceeds 400MB, systemd auto-restarts it
```

## Duplicate Records on Dashboard

**Cause**: Fixed in latest version with timestamp-based deduplication.

If you see duplicates, they may be from before the fix:
```bash
cd ~/scanner
python3 -c "
import scanner_db, sqlite3
conn = sqlite3.connect('scanner.db')
cur = conn.execute('''
    DELETE FROM transmissions WHERE id IN (
        SELECT id FROM transmissions t1
        WHERE EXISTS (
            SELECT 1 FROM transmissions t2
            WHERE t2.time = t1.time AND t2.id < t1.id
        )
    )
''')
print(f'Removed {cur.rowcount} duplicates')
conn.commit()
"
```

## Missing Metadata (Empty System/Group/Channel)

**Cause**: Records from `pi_reingest` (orphan files recovered after crash).

The re-ingest worker now parses metadata from filenames. For old records:
```bash
# Re-run the metadata backfill
python3 tools/batch_decode.py
```
