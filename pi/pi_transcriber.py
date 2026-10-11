"""
Pi Local Transcriber — standalone process for Whisper transcription.

Polls the SQLite database for untranscribed records, transcribes them using
whisper.cpp (tiny.en on CPU), and updates the DB. Runs independently of the
scanner capture process so it never affects audio recording.

If the GPU server is online, this process idles (GPU handles transcription
much faster). Only transcribes locally when GPU is unreachable.

Run:  python3 pi_transcriber.py
Stop: Ctrl+C or systemd stop
"""

import os
import sys
import time
import json
import signal
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import scanner_db

# Limit CPU priority so transcriber never starves the scanner
try:
    os.nice(19)  # Lowest priority
except OSError:
    pass

# Text decoders
from codes import decode_for
from phonetic import decode_plates
from phone import detect_phones

POLL_INTERVAL = 3  # seconds between checking for new items
GPU_CHECK_INTERVAL = getattr(config, 'GPU_CHECK_INTERVAL', 10)
GPU_SERVER_URL = getattr(config, 'GPU_SERVER_URL', '')

# --- Backpressure / capacity management ---
# The Pi 3B+ transcribes ~4-8 clips/min with tiny.en, but busy scanner traffic
# can arrive at 30+/min. Without limits the backlog grows unbounded and Whisper
# grinds the CPU forever, wedging the whole Pi. These limits keep local
# transcription within the Pi's real capacity and let it degrade gracefully.

# Sleep this long between clips so Whisper never monopolizes the CPU.
PER_CLIP_SLEEP = 1.0

# If the backlog is larger than this, the Pi has fallen too far behind to ever
# catch up locally. Abandon the oldest clips (mark them so they stop clogging
# the queue) and only transcribe the freshest. When the GPU server comes back,
# it re-transcribes everything properly anyway.
MAX_LOCAL_BACKLOG = 150

# Only transcribe clips newer than this locally. Older clips are left for the
# GPU server (which can catch up fast). Marks stale ones as skipped.
LOCAL_FRESHNESS_HOURS = 2

_stop = False


def _signal_handler(sig, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _run_text_decoders(text, channel_name):
    """Run text-based decoders on a transcript."""
    if not text:
        return {}
    results = {}
    try:
        plates = decode_plates(text)
        if plates:
            results["plates"] = [p["plate"] for p in plates]
    except Exception:
        pass
    try:
        phones = detect_phones(text)
        if phones:
            results["phones"] = [p["phone"] for p in phones]
    except Exception:
        pass
    try:
        profile, codes = decode_for(text, channel_name)
        if codes:
            results["codes"] = [{"code": c["code"], "meaning": c["meaning"]} for c in codes]
            if profile:
                results["code_profile"] = profile.name
    except Exception:
        pass
    return results


def check_gpu_online():
    """Check if GPU server is reachable."""
    if not GPU_SERVER_URL:
        return False
    try:
        url = GPU_SERVER_URL + "/status"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_pending_records(limit=5):
    """Get untranscribed records from DB, newest first."""
    with scanner_db.get_db() as conn:
        rows = conn.execute("""
            SELECT id, clip, channel, time, duration_sec
            FROM transmissions
            WHERE transcribed = 0 AND clip != '' AND clip IS NOT NULL
            ORDER BY time DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_pending_count():
    with scanner_db.get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM transmissions WHERE transcribed = 0 "
            "AND clip != '' AND clip IS NOT NULL"
        ).fetchone()[0]


def shed_backlog():
    """When the Pi has fallen too far behind, abandon old pending clips so the
    queue cannot grow unbounded. Marks them transcribed with empty text and a
    'skipped_local' marker — the GPU server (transcribed_by != 'gpu' check) will
    still re-transcribe them later when it is online. Returns number shed."""
    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.localtime(time.time() - LOCAL_FRESHNESS_HOURS * 3600),
    )
    with scanner_db.get_db() as conn:
        # Only shed if backlog is genuinely large
        n = conn.execute(
            "SELECT COUNT(*) FROM transmissions WHERE transcribed = 0 "
            "AND clip != '' AND clip IS NOT NULL"
        ).fetchone()[0]
        if n <= MAX_LOCAL_BACKLOG:
            return 0
        # Mark everything older than the freshness window as skipped-local.
        # transcribed_by stays '' so GPU re-transcription still picks it up.
        cur = conn.execute(
            """UPDATE transmissions
               SET transcribed = 1, text = ''
               WHERE transcribed = 0 AND clip != '' AND clip IS NOT NULL
                 AND time < ?""",
            (cutoff,),
        )
        return cur.rowcount


def transcribe_record(record_id, clip_path, channel, duration, model):
    """Transcribe a single record."""
    # Check if GPU already handled it
    with scanner_db.get_db() as conn:
        row = conn.execute(
            "SELECT transcribed_by FROM transmissions WHERE id = ?",
            (record_id,)
        ).fetchone()
        if row and row["transcribed_by"] == "gpu":
            return

    # Mark as actively transcribing
    scanner_db.update_transmission(record_id, {"text": "Transcribing now"})

    # Give GPU 3 seconds to potentially finish first
    time.sleep(3)

    # Check again
    with scanner_db.get_db() as conn:
        row = conn.execute(
            "SELECT transcribed_by, text FROM transmissions WHERE id = ?",
            (record_id,)
        ).fetchone()
        if row and row["transcribed_by"] == "gpu":
            if row["text"] == "Transcribing now":
                scanner_db.update_transmission(record_id, {"text": ""})
            return

    # Transcribe
    text = ""
    try:
        if not os.path.exists(clip_path):
            # Try mp3 version
            if clip_path.endswith('.wav'):
                mp3 = clip_path[:-4] + '.mp3'
                if os.path.exists(mp3):
                    clip_path = mp3
                else:
                    scanner_db.update_transmission(record_id, {
                        "text": "(audio not found)", "transcribed": True
                    })
                    return
            else:
                scanner_db.update_transmission(record_id, {
                    "text": "(audio not found)", "transcribed": True
                })
                return

        result = model.transcribe(clip_path, language="en")
        parts = [seg.text.strip() for seg in result if seg.text.strip()]
        text = " ".join(parts)
    except Exception as e:
        print(f"[transcriber] Error transcribing {record_id}: {e}")

    # Final GPU check
    with scanner_db.get_db() as conn:
        row = conn.execute(
            "SELECT transcribed_by, text FROM transmissions WHERE id = ?",
            (record_id,)
        ).fetchone()
        if row and row["transcribed_by"] == "gpu":
            if row["text"] == "Transcribing now":
                scanner_db.update_transmission(record_id, {"text": ""})
            return

    speech = text if text else ""

    # Run text decoders
    text_decoded = _run_text_decoders(speech, channel)

    updates = {
        "text": speech,
        "transcribed": True,
    }
    if text_decoded:
        updates["decoded_text"] = text_decoded

    scanner_db.update_transmission(record_id, updates)

    display = speech[:70] + "..." if len(speech) > 70 else (speech or "(no speech)")
    print(f"[transcriber] {record_id[:8]} -> {display}")


def main():
    print("=" * 50)
    print("  Pi Local Transcriber (standalone)")
    print("=" * 50)
    print(f"  Model:  {config.WHISPER_MODEL}")
    print(f"  GPU:    {GPU_SERVER_URL or '(none configured)'}")
    print(f"  Poll:   every {POLL_INTERVAL}s")
    print("=" * 50)

    model = None  # Lazy-load: only when needed
    gpu_online = False
    last_gpu_check = 0
    last_shed_check = 0
    idle_since = time.time()

    while not _stop:
        now = time.time()

        # Periodically check GPU status
        if now - last_gpu_check >= GPU_CHECK_INTERVAL:
            gpu_online = check_gpu_online()
            last_gpu_check = now

        # If GPU is online, unload model to free RAM and idle.
        # The GPU handles all transcription; the Pi does nothing.
        if gpu_online:
            if model is not None:
                print("[transcriber] GPU online, unloading model to free RAM")
                del model
                model = None
                import gc; gc.collect()
            time.sleep(POLL_INTERVAL)
            continue

        # GPU is OFFLINE — the Pi must transcribe locally, but it cannot keep up
        # with busy traffic. Apply backpressure: shed old backlog periodically so
        # the queue can't grow without bound and peg the CPU forever.
        if now - last_shed_check >= 30:
            shed = shed_backlog()
            last_shed_check = now
            if shed:
                print(f"[transcriber] Backlog too large — shed {shed} old clip(s) "
                      f"(will be transcribed by GPU when it returns)")

        # Fetch a small batch of the freshest pending clips
        records = get_pending_records(limit=3)
        if not records:
            if model is not None and (now - idle_since) > 600:
                print("[transcriber] Idle 10min, unloading model to free RAM")
                del model
                model = None
                import gc; gc.collect()
            time.sleep(POLL_INTERVAL)
            continue

        # Load model on demand (single thread — we are pinned to 2 cores by
        # systemd AllowedCPUs, and a single thread leaves one of those free).
        if model is None:
            from pywhispercpp.model import Model
            print("[transcriber] Loading Whisper model (1 thread)...")
            model = Model(config.WHISPER_MODEL, n_threads=1)
            print("[transcriber] Model ready.")

        idle_since = now

        for rec in records:
            if _stop:
                break
            # Re-check GPU before each clip so we yield promptly when it returns
            if time.time() - last_gpu_check >= GPU_CHECK_INTERVAL:
                gpu_online = check_gpu_online()
                last_gpu_check = time.time()
                if gpu_online:
                    print("[transcriber] GPU came online, yielding")
                    break

            transcribe_record(
                rec["id"], rec["clip"], rec["channel"],
                rec["duration_sec"], model
            )
            # Mandatory pause between clips — guarantees the CPU is never
            # monopolized, even with a deep backlog.
            if not _stop:
                time.sleep(PER_CLIP_SLEEP)

    print("[transcriber] Stopped.")


if __name__ == "__main__":
    scanner_db.init_db()
    main()
