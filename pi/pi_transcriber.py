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

# Text decoders
from codes import decode_for
from phonetic import decode_plates
from phone import detect_phones

POLL_INTERVAL = 3  # seconds between checking for new items
GPU_CHECK_INTERVAL = getattr(config, 'GPU_CHECK_INTERVAL', 30)
GPU_SERVER_URL = getattr(config, 'GPU_SERVER_URL', '')

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
    from pywhispercpp.model import Model

    print("=" * 50)
    print("  Pi Local Transcriber (standalone)")
    print("=" * 50)
    print(f"  Model:  {config.WHISPER_MODEL}")
    print(f"  GPU:    {GPU_SERVER_URL or '(none configured)'}")
    print(f"  Poll:   every {POLL_INTERVAL}s")
    print("=" * 50)

    print("[transcriber] Loading Whisper model...")
    model = Model(config.WHISPER_MODEL, n_threads=3)
    print("[transcriber] Model ready.")

    gpu_online = False
    last_gpu_check = 0

    while not _stop:
        # Periodically check GPU status
        now = time.time()
        if now - last_gpu_check >= GPU_CHECK_INTERVAL:
            gpu_online = check_gpu_online()
            last_gpu_check = now
            if gpu_online:
                # GPU is handling transcription, idle
                time.sleep(POLL_INTERVAL)
                continue

        if gpu_online:
            time.sleep(POLL_INTERVAL)
            continue

        # Get pending records
        records = get_pending_records(limit=5)
        if not records:
            time.sleep(POLL_INTERVAL)
            continue

        for rec in records:
            if _stop:
                break
            # Re-check GPU before each item
            if now - last_gpu_check >= GPU_CHECK_INTERVAL:
                gpu_online = check_gpu_online()
                last_gpu_check = time.time()
                if gpu_online:
                    print("[transcriber] GPU came online, yielding")
                    break

            transcribe_record(
                rec["id"], rec["clip"], rec["channel"],
                rec["duration_sec"], model
            )

    print("[transcriber] Stopped.")


if __name__ == "__main__":
    scanner_db.init_db()
    main()
