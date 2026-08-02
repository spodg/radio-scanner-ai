"""
Pi Scanner Station — standalone 24/7 capture + transcription daemon.

Runs on Raspberry Pi 3B+ with:
  - BCD436HP via USB serial (GLG polling for squelch + metadata)
  - Sabrent AU-MMSA USB sound card (scanner headphone -> mic input)
  - On-device Whisper transcription (tiny.en, slow but fully standalone)
  - Remote GPU transcription: writes placeholder immediately, GPU server
    on powerful PC picks up untranscribed items over the network.

Architecture:
  - Main thread: serial polling (lightweight, real-time)
  - Audio thread: continuous ALSA capture via arecord
  - Transcription worker: background queue, processes clips one by one
  - GPU server (on PC): polls JSONL for untranscribed items and processes
    them ~50x faster than the Pi can locally.

New workflow: transmissions appear on the dashboard IMMEDIATELY with
"Transcribing..." and get updated when either local or GPU finishes first.

Run:  python3 pi_scanner.py
Stop: Ctrl+C or systemd stop
"""

import os
import sys
import json
import time
import uuid
import wave
import threading
import datetime as dt
from pathlib import Path

import numpy as np

import config
from scanner_serial import ScannerSerial, ScannerPoller, ReceptionState
from audio_capture import ALSAAudioCapture
from tones import analyze as analyze_tones
from morse import decode as morse_decode
from fsk_decode import decode_fsk, format_results as format_fsk
from codes import decode_for
from phonetic import decode_plates
from phone import detect_phones, format_phones
import scanner_db


def _run_text_decoders(text, channel_name):
    """Run text-based decoders on a transcript. Returns a dict of findings."""
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


def _generate_record_id():
    """Generate a short unique ID for each transmission record."""
    return uuid.uuid4().hex[:12]


class PiScannerStation:
    def __init__(self):
        self.scanner = ScannerSerial(config.SERIAL_PORT, config.SERIAL_BAUD)
        self.audio = ALSAAudioCapture(
            device=config.AUDIO_DEVICE,
            sample_rate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            prebuffer_sec=config.AUDIO_PREBUFFER_SEC,
        )
        self._active_capture_id = None
        self._active_start_time = None
        self._active_system_name = ""
        self._stop = threading.Event()
        self._transcriber = None

        Path(config.CLIPS_DIR).mkdir(parents=True, exist_ok=True)

    def _write_status(self, state, channel):
        """Write live status for the dashboard to read."""
        # Also grab the LCD screen via STS
        screen_lines = self._get_screen()

        status = {
            "state": state,
            "channel": channel,
            "queue": scanner_db.get_pending_count(),
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "screen": screen_lines,
        }
        try:
            with open("/home/pi/scanner/status.json", "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    def _on_poll(self, state):
        """Called every poll cycle — update the screen display."""
        screen_lines = self._get_screen()
        st = "receiving" if state.active else "scanning"
        status = {
            "state": st,
            "channel": state.display_name if state.active else "",
            "queue": scanner_db.get_pending_count(),
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "screen": screen_lines,
        }
        try:
            with open("/home/pi/scanner/status.json", "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    def _get_screen(self):
        """Grab the LCD screen content via STS command."""
        try:
            sts = self.scanner.command("STS")
            return self._parse_sts(sts)
        except Exception:
            return []

    @staticmethod
    def _parse_sts(sts_resp):
        """Parse the STS response into a structured screen display.

        Returns a dict with:
          - header: list of 2 lines (status bar)
          - sections: list of 3 sections, each a list of 3 lines
          - footer: string (TGID/Tag line)

        This matches the BCD436HP's physical LCD layout:
          [header: status indicators + date/time]
          [section 1: system name - 3 lines]
          [section 2: group/dept - 3 lines]
          [section 3: channel name - 3 lines]
          [footer: TGID + tag info]
        """
        EMPTY_SCREEN = {
            "header": ["", ""],
            "sections": [["", "", ""], ["", "", ""], ["", "", ""]],
            "footer": ""
        }

        if not sts_resp or not sts_resp.startswith("STS"):
            return EMPTY_SCREEN

        # Split on the underscore separator (24 underscores)
        raw_sections = sts_resp.split("________________________")

        # First part contains the STS prefix + header lines
        # Remaining parts are the content sections + footer
        if len(raw_sections) < 2:
            return EMPTY_SCREEN

        # Parse header (first raw section has STS prefix + F0/S0 lines)
        header_raw = raw_sections[0]
        header_lines = []
        # Extract F0 and S0 lines
        for part in header_raw.split(","):
            part = part.strip()
            if part.startswith("F0:"):
                header_lines.append(part[3:].strip())
            elif part.startswith("S0:"):
                header_lines.append(part[3:].strip())
        while len(header_lines) < 2:
            header_lines.append("")
        header_lines = header_lines[:2]

        # Parse content sections (each has lines separated by commas)
        content_sections = raw_sections[1:]

        def parse_section(raw):
            """Extract up to 3 meaningful lines from a raw section."""
            lines = []
            for part in raw.split(","):
                part = part.strip()
                # Skip empty parts and format codes
                if not part:
                    continue
                if part.startswith("F0:") or part.startswith("S0:"):
                    part = part[3:].strip()
                if part:
                    lines.append(part)
            return lines

        # We expect 3 main sections (system, group, channel) + footer
        sections = []
        footer = ""

        for i, raw_sec in enumerate(content_sections):
            parsed = parse_section(raw_sec)
            if i < 3:
                # Pad/truncate to exactly 3 lines
                while len(parsed) < 3:
                    parsed.append("")
                sections.append(parsed[:3])
            else:
                # Everything after section 3 is footer
                footer_parts = [p for p in parsed if p]
                if footer_parts:
                    footer = "  ".join(footer_parts)

        # Ensure we always have exactly 3 sections
        while len(sections) < 3:
            sections.append(["", "", ""])

        return {
            "header": header_lines,
            "sections": sections[:3],
            "footer": footer
        }

    def _on_start(self, state: ReceptionState):
        self._active_start_time = dt.datetime.now()
        self._active_capture_id = self.audio.begin_capture()
        # Grab the real system name from STS (GLG only reports the site name)
        screen = self._get_screen()
        if isinstance(screen, dict) and screen.get("sections"):
            sec1 = screen["sections"][0]
            self._active_system_name = sec1[0] if sec1 and sec1[0] else state.system_name
        else:
            self._active_system_name = state.system_name
        self._write_status("receiving", state.display_name)

    def _on_stop(self, state: ReceptionState, duration: float):
        import sys
        cid = self._active_capture_id
        start = self._active_start_time
        self._active_capture_id = None
        self._active_start_time = None
        self._write_status("scanning", "")
        if cid is None:
            print("[_on_stop] no capture id", file=sys.stderr, flush=True)
            return

        audio = self.audio.end_capture(cid)
        if audio.size == 0:
            print("[_on_stop] empty audio", file=sys.stderr, flush=True)
            return

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < config.WHISPER_SILENCE_RMS:
            print(f"[_on_stop] silent rms={rms:.5f}", file=sys.stderr, flush=True)
            return

        print(f"[_on_stop] processing {state.display_name} dur={duration:.1f}s rms={rms:.4f} samples={audio.size}", file=sys.stderr, flush=True)

        # Skip dead silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < config.WHISPER_SILENCE_RMS:
            return

        sr = config.AUDIO_SAMPLE_RATE

        # Split on silence and immediately write each segment to JSONL
        segments = self._split_on_silence(audio, sr)
        for i, (seg, offset) in enumerate(segments):
            seg_time = start + dt.timedelta(seconds=offset)
            seg_dur = len(seg) / sr
            if seg_dur < config.MIN_TRANSMISSION_SEC:
                continue
            # Cap max duration to 30s to prevent CPU overload during transcription
            max_dur = 30.0
            if seg_dur > max_dur:
                seg = seg[:int(max_dur * sr)]
                seg_dur = max_dur

            # Save WAV clip IMMEDIATELY
            ts_str = seg_time.strftime("%Y%m%d_%H%M%S")
            date_folder = seg_time.strftime("%Y%m%d")
            suffix = f"_{i}" if len(segments) > 1 else ""
            # Encode full metadata in filename: Site___Group___Channel___Freq
            def _sanitize(s, maxlen=50):
                return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:maxlen]
            name_parts = [
                _sanitize(self._active_system_name, 30),
                _sanitize(state.group_name, 30),
                _sanitize(state.channel_name, 50),
                _sanitize(state.frequency, 10),
            ]
            safe_name = "___".join(p for p in name_parts if p)
            base = f"{ts_str}{suffix}_{safe_name}"
            clip_dir = os.path.join(config.CLIPS_DIR, date_folder)
            os.makedirs(clip_dir, exist_ok=True)
            wav_path = os.path.join(clip_dir, f"{base}.wav")

            pcm16 = (np.clip(seg, -1, 1) * 32767).astype("<i2")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())

            # Run audio-based decoders (fast, just numpy — no transcription needed)
            decoded = {}
            try:
                sig = analyze_tones(seg, sr, silence_rms=config.WHISPER_SILENCE_RMS)
                if sig.get("dtmf") and len(sig["dtmf"]) >= 2:
                    dtmf = sig["dtmf"]
                    # Filter false positives:
                    # - >50% are A/B/C/D = voice harmonics
                    # - all same digit = voice sustaining one formant
                    rare = sum(1 for c in dtmf if c in "ABCD")
                    if rare <= len(dtmf) * 0.5 and len(set(dtmf)) > 1:
                        decoded["dtmf"] = dtmf
                if sig.get("tones"):
                    # Store raw tones list (already filtered to >= 0.5s by _steady_tones)
                    decoded["tones"] = sig["tones"]
                if sig.get("data_burst"):
                    decoded["data_burst"] = True
            except Exception:
                pass
            try:
                m = morse_decode(seg, sr)
                if m and m.get("text") and m.get("confidence", 0) > 0.7 and len(m["text"]) >= 5:
                    # Filter false positives: reject if only E and T (simplest patterns)
                    unique = set(m["text"].replace(" ", ""))
                    if not unique.issubset({"E", "T"}):
                        decoded["morse"] = m["text"]
            except Exception:
                pass
            try:
                fsk = decode_fsk(seg, sr)
                if fsk:
                    decoded["fsk"] = format_fsk(fsk)
            except Exception:
                pass

            # Write placeholder to DB IMMEDIATELY
            record_id = _generate_record_id()
            placeholder = {
                "id": record_id,
                "time": seg_time.isoformat(timespec="seconds"),
                "frequency": state.frequency,
                "name": state.display_name,
                "system": self._active_system_name,
                "group": state.group_name,
                "channel": state.channel_name,
                "duration_sec": round(seg_dur, 1),
                "text": "Transcribing...",
                "transcribed": False,
                "decoded": decoded,
                "clip": wav_path,
                "source": "pi_capture",
                "tags": state.tags,
            }
            scanner_db.insert_transmission(placeholder)

            print(f"{seg_time:%Y-%m-%d %H:%M:%S} | [Transcribing...] | {state.display_name} | {state.frequency} | {seg_dur:.1f}s")

    def _cleanup_worker(self):
        """Background worker that converts transcribed WAV clips to MP3
        and runs text decoders on GPU-transcribed items.

        Scans the JSONL for records with transcribed=True and clip ending in
        .wav. Converts to MP3 via ffmpeg, updates the JSONL clip path, and
        deletes the WAV. Also runs text decoders on items transcribed by GPU.
        Runs every 30 seconds.
        """
        import subprocess as sp
        time.sleep(15)  # initial delay to let things settle
        # Lower priority — cleanup is not time-critical
        import os as _os
        try:
            _os.nice(15)
        except OSError:
            pass
        print("[cleanup] WAV-to-MP3 + text decoder worker started.")

        while not self._stop.is_set():
            try:
                self._run_text_decoder_pass()
            except Exception as e:
                print(f"[cleanup] decoder error: {e}")
            try:
                self._convert_wavs_to_mp3(sp)
            except Exception as e:
                print(f"[cleanup] mp3 error: {e}")
            self._stop.wait(30)

    def _run_text_decoder_pass(self):
        """Find GPU-transcribed records without text decoding and decode them."""
        with scanner_db.get_db() as conn:
            rows = conn.execute("""
                SELECT id, text, name, channel, decoded_text FROM transmissions
                WHERE transcribed = 1
                  AND text != '' AND text != '(no speech)' AND text != '(audio not found)'
                  AND (decoded_text = '{}' OR decoded_text = '')
                LIMIT 200
            """).fetchall()

        for row in rows:
            # Use channel name for code profile matching (more accurate than display name)
            channel = row["channel"] if row["channel"] else row["name"]
            text_decoded = _run_text_decoders(row["text"], channel)
            scanner_db.update_transmission(row["id"], {
                "decoded_text": text_decoded or {}
            })

    def _convert_wavs_to_mp3(self, sp):
        """Find transcribed WAV clips and convert them to MP3."""
        with scanner_db.get_db() as conn:
            rows = conn.execute("""
                SELECT id, clip FROM transmissions
                WHERE transcribed = 1 AND clip LIKE '%.wav'
                LIMIT 20
            """).fetchall()

        for row in rows:
            clip = row["clip"]
            if not clip or not os.path.exists(clip):
                continue
            mp3_path = clip[:-4] + ".mp3"
            try:
                result = sp.run(
                    ["ffmpeg", "-y", "-i", clip, "-ac", "1", "-ar", "16000",
                     "-b:a", "32k", "-loglevel", "error", mp3_path],
                    capture_output=True, timeout=60
                )
                if result.returncode == 0 and os.path.exists(mp3_path):
                    os.remove(clip)
                    scanner_db.update_transmission(row["id"], {"clip": mp3_path})
            except Exception:
                pass

        # Re-ingest orphan WAV files (saved but not in DB — lost during restart)
        import glob as _glob
        try:
            with scanner_db.get_db() as conn:
                existing_clips = set(
                    r[0] for r in conn.execute("SELECT clip FROM transmissions").fetchall()
                )
                # Build set of known basenames (handles path format differences)
                known_basenames = set()
                for c in existing_clips:
                    if c:
                        base = os.path.basename(c)
                        known_basenames.add(base)
                        # Also add .wav variant of .mp3 files
                        if base.endswith(".mp3"):
                            known_basenames.add(base[:-4] + ".wav")

            now = time.time()
            reingested = 0
            for wav in sorted(_glob.glob(os.path.join(config.CLIPS_DIR, "**", "*.wav"), recursive=True)):
                basename = os.path.basename(wav)
                if basename in known_basenames:
                    continue
                # Skip if MP3 version already exists (conversion in progress or done)
                mp3_version = wav[:-4] + ".mp3"
                if os.path.exists(mp3_version):
                    continue
                # Wait 5 minutes before re-ingesting to avoid races
                age = now - os.path.getmtime(wav)
                if age < 300:
                    continue
                import wave as _wave
                try:
                    with _wave.open(wav, "rb") as wf:
                        dur = wf.getnframes() / wf.getframerate() if wf.getframerate() > 0 else 0
                    if dur < 1.5:
                        os.remove(wav)
                        continue
                except Exception:
                    os.remove(wav)
                    continue

                record_id = uuid.uuid4().hex[:12]
                fname = os.path.basename(wav)
                try:
                    ts = dt.datetime.strptime(fname[:15], "%Y%m%d_%H%M%S")
                    time_str = ts.isoformat(timespec="seconds")
                except Exception:
                    time_str = dt.datetime.now().isoformat(timespec="seconds")

                # Parse metadata from filename: YYYYMMDD_HHMMSS_N_System___Group___Channel___Freq.wav
                import re as _re
                fname_noext = fname.rsplit('.', 1)[0]
                m = _re.match(r'\d{8}_\d{6}(?:_\d+)?_(.*)', fname_noext)
                raw_name = m.group(1) if m else fname_noext
                parts = _re.split(r'_{3,}', raw_name)
                parts = [p.replace('_', ' ').strip() for p in parts if p.strip()]
                # Parts: [System, Group, Channel, Freq] or fewer
                r_system = parts[0] if len(parts) >= 1 else ""
                r_group = parts[1] if len(parts) >= 2 else ""
                r_channel = parts[2] if len(parts) >= 3 else ""
                r_freq = parts[3] if len(parts) >= 4 else ""
                display_name = ' / '.join(parts[:3]) if parts else raw_name.replace('_', ' ')[:60]
                # Default system if we only got site name
                if r_system and not r_system.startswith("Allen") and not r_system.startswith("Indiana"):
                    r_system = "Indiana Project Hoosier"
                elif "Public Safety" in r_system:
                    r_system = "Allen County P25"

                scanner_db.insert_transmission({
                    "id": record_id,
                    "time": time_str,
                    "frequency": r_freq,
                    "name": display_name,
                    "system": r_system,
                    "group": r_group,
                    "channel": r_channel,
                    "duration_sec": round(dur, 1),
                    "text": "Transcribing...",
                    "transcribed": False,
                    "clip": wav,
                    "source": "pi_reingest",
                })
                reingested += 1

                # Add to transcription queue immediately
                mock_state = ReceptionState(
                    active=False, frequency=r_freq,
                    system_name=r_system, group_name=r_group,
                    channel_name=r_channel,
                )
                try:
                    seg_time = dt.datetime.fromisoformat(time_str)
                except Exception:
                    seg_time = dt.datetime.now()
                self._queue.put((record_id, wav, mock_state, seg_time, dur))
                self._update_queue_count()

            if reingested:
                print(f"[cleanup] Re-ingested {reingested} orphan WAV(s)")
        except Exception:
            pass

    def _split_on_silence(self, audio, sr):
        gap_sec = config.SILENCE_SPLIT_SEC
        silence_rms = config.SILENCE_SPLIT_RMS
        min_seg_sec = config.MIN_TRANSMISSION_SEC

        frame_len = int(0.02 * sr)
        hop = frame_len
        n_frames = len(audio) // hop
        if n_frames < 2:
            return [(audio, 0.0)]

        is_silent = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            seg = audio[i * hop:(i + 1) * hop]
            is_silent[i] = np.sqrt(np.mean(seg ** 2)) < silence_rms

        gap_frames = int(gap_sec / 0.02)
        splits = []
        run_start = None
        for i in range(n_frames):
            if is_silent[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and (i - run_start) >= gap_frames:
                    splits.append((run_start + i) // 2)
                run_start = None

        if not splits:
            return [(audio, 0.0)]

        segments = []
        prev = 0
        for sp in splits:
            sample_pos = sp * hop
            seg = audio[prev:sample_pos]
            if len(seg) / sr >= min_seg_sec:
                segments.append((seg, prev / sr))
            prev = sample_pos
        seg = audio[prev:]
        if len(seg) / sr >= min_seg_sec:
            segments.append((seg, prev / sr))

        return segments if segments else [(audio, 0.0)]

    def run(self):
        # Initialize SQLite database
        scanner_db.init_db()

        print("=" * 50)
        print("  Pi Scanner Station (standalone)")
        print("=" * 50)
        print(f"  Serial:  {config.SERIAL_PORT}")
        print(f"  Audio:   {config.AUDIO_DEVICE}")
        print(f"  Model:   {config.WHISPER_MODEL} ({config.WHISPER_COMPUTE_TYPE})")
        print(f"  Clips:   {config.CLIPS_DIR}")
        print(f"  Log:     {config.LOG_FILE}")
        print()

        try:
            self.scanner.open()
            model = self.scanner.get_model()
            print(f"  Scanner: {model}")
        except Exception as e:
            print(f"ERROR: Cannot open serial port: {e}")
            print("  Run: ls /dev/ttyUSB* /dev/ttyACM*")
            return

        self.audio.start()
        print("  Audio capture started.")

        # Start WAV-to-MP3 cleanup worker
        cleanup = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup.start()

        poller = ScannerPoller(
            self.scanner,
            interval=config.POLL_INTERVAL,
            on_start=self._on_start,
            on_stop=self._on_stop,
            min_duration=config.MIN_TRANSMISSION_SEC,
        )
        poller.on_poll = self._on_poll
        poller.start()
        print("\n  Capturing + transcribing. Press Ctrl+C to stop.\n")

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")

        self._stop.set()
        poller.stop()
        poller.join(timeout=2)
        self.audio.stop()
        self.scanner.close()
        print("Done.")


if __name__ == "__main__":
    station = PiScannerStation()
    station.run()
