"""
Audio signal classifier + tone decoder for scanner clips.

The line-in audio we capture contains more than speech. This module looks at
the captured samples and classifies each transmission as one of:

    "speech"   - human voice (let Whisper transcribe it)
    "tone"     - sequential pure tones (e.g. two-tone / Quick Call II paging,
                 alert tones) -> we report the tone frequencies and, when they
                 match a known two-tone format, a best-effort note
    "dtmf"     - touch-tone digits -> decoded to the actual digit string
    "data"     - digital burst (MDC-1200 PTT-ID, POCSAG/Flex paging, FleetSync,
                 FFSK telemetry). We DETECT and label these, but we do NOT try
                 to demodulate the payload here - that's a real demod job best
                 handled by multimon-ng (see README). Labeling them stops them
                 from being logged as garbled "speech".
    "silent"   - dead-air carrier
    "noise"    - something with energy we couldn't classify

Why not fully decode data bursts? They carry real digital payloads (unit IDs,
capcodes, messages) that require proper FSK demodulation and error correction.
Doing that accurately from clipped, single-channel, 16 kHz line-in audio is
unreliable; the honest move is to detect them and hand the audio to a proven
decoder if the user wants payloads.
"""

import numpy as np

# --- DTMF -------------------------------------------------------------------
DTMF_LOW = [697, 770, 852, 941]
DTMF_HIGH = [1209, 1336, 1477, 1633]
DTMF_MAP = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3", (697, 1633): "A",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6", (770, 1633): "B",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9", (852, 1633): "C",
    (941, 1209): "*", (941, 1336): "0", (941, 1477): "#", (941, 1633): "D",
}


def _goertzel(samples, sr, target):
    """Goertzel power at a single target frequency (fast single-bin DFT)."""
    n = len(samples)
    k = int(0.5 + (n * target) / sr)
    w = (2.0 * np.pi * k) / n
    cw = np.cos(w)
    coeff = 2.0 * cw
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power


def _mean_spectral_flatness(audio, sr, win_s=0.025, hop_s=0.0125, min_rms=0.01):
    """
    Mean spectral flatness (geometric mean / arithmetic mean of the magnitude
    spectrum) over active frames, in the >200 Hz band. Tonal/data signals score
    higher here; voice with formant structure scores lower. Range ~0..1.
    """
    win = max(16, int(win_s * sr))
    hop = max(8, int(hop_s * sr))
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = freqs > 200
    vals = []
    for s in range(0, len(audio) - win, hop):
        seg = audio[s:s + win]
        if np.sqrt(np.mean(seg ** 2)) < min_rms:
            continue
        spec = np.abs(np.fft.rfft(seg * hann))[band]
        if spec.size == 0:
            continue
        gm = np.exp(np.mean(np.log(spec + 1e-12)))
        am = np.mean(spec) + 1e-12
        vals.append(gm / am)
    return float(np.mean(vals)) if vals else 0.0


def _dom_freq_series(audio, sr, win_s=0.04, hop_s=0.02, min_rms=0.01):
    """Per-window dominant frequency (Hz) and per-window rms."""
    win = max(8, int(win_s * sr))
    hop = max(4, int(hop_s * sr))
    doms, rmss = [], []
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    fmask = freqs > 200
    for s in range(0, len(audio) - win, hop):
        seg = audio[s:s + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        rmss.append(rms)
        if rms < min_rms:
            doms.append(None)
            continue
        spec = np.abs(np.fft.rfft(seg * hann))
        spec[~fmask] = 0
        doms.append(float(freqs[np.argmax(spec)]))
    return doms, rmss


def _dtmf_window(seg, sr):
    """
    Detect a DTMF digit in one window, with an energy-concentration guard so
    broadband data bursts don't masquerade as touch-tones. Returns digit or None.
    """
    total = float(np.sum(seg ** 2)) + 1e-9
    low_p = sorted(((_goertzel(seg, sr, f), f) for f in DTMF_LOW), reverse=True)
    high_p = sorted(((_goertzel(seg, sr, f), f) for f in DTMF_HIGH), reverse=True)
    # Each group's strongest must clearly dominate the rest of its group.
    if low_p[0][0] < 6 * (low_p[1][0] + 1e-9):
        return None
    if high_p[0][0] < 6 * (high_p[1][0] + 1e-9):
        return None
    # The two DTMF tones together must hold most of the window's energy.
    # (Data bursts spread energy across many bins, so this ratio stays low.)
    concentration = (low_p[0][0] + high_p[0][0]) / total
    if concentration < 0.30:
        return None
    return DTMF_MAP.get((low_p[0][1], high_p[0][1]))


def _detect_dtmf(audio, sr):
    """Return a DTMF digit string if the clip is touch-tones, else ''."""
    win = int(0.04 * sr)
    hop = int(0.02 * sr)
    digits = []
    last = None
    run = 0
    for s in range(0, len(audio) - win, hop):
        seg = audio[s:s + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.05:
            last = None
            run = 0
            continue
        digit = _dtmf_window(seg, sr)
        if digit is None:
            last = None
            run = 0
            continue
        if digit == last:
            run += 1
        else:
            run = 1
        # Require a digit to persist ~>=2 windows (>=~60ms) before accepting,
        # so single spurious matches in noise are ignored.
        if run == 2:
            digits.append(digit)
        last = digit
    return "".join(digits)


def _steady_tones(doms, hop_s=0.02, tol=25, min_dur=0.50):
    """
    Collapse the dominant-frequency series into steady tones lasting at least
    min_dur seconds. Returns list of (freq_hz, duration_s).
    """
    tones = []
    cur = []
    for d in doms:
        if d is None:
            if cur:
                tones.append(cur); cur = []
            continue
        if not cur or abs(d - np.mean([c for c in cur])) <= tol:
            cur.append(d)
        else:
            tones.append(cur); cur = [d]
    if cur:
        tones.append(cur)
    out = []
    for grp in tones:
        dur = len(grp) * hop_s
        if dur >= min_dur:
            out.append((round(float(np.mean(grp)), 1), round(dur, 2)))
    return out


def classify(audio: np.ndarray, sr: int, silence_rms: float = 0.0015):
    """Deprecated thin wrapper kept for compatibility. Use analyze()."""
    return analyze(audio, sr, silence_rms)


VOICE_EXPECTED_KEYWORDS = [
    "air traffic", "atc", "center", "approach", "departure", "tower",
    "ground control", "clearance", "unicom", "ctaf", "atis", "airport",
    "aviation", "flight", "tracon", "artcc", "rcag", "air route",
    "marine", "coast guard",
]


def channel_voice_expected(channel_name: str) -> bool:
    """
    True for channels that carry voice and no FSK paging data, but whose audio
    can be weak/noisy enough to mimic data (e.g. AM aviation/ATC). On these we
    bias toward transcription.
    """
    if not channel_name:
        return False
    n = channel_name.lower()
    return any(kw in n for kw in VOICE_EXPECTED_KEYWORDS)


def analyze(audio: np.ndarray, sr: int, silence_rms: float = 0.0015,
            voice_expected: bool = False):
    """
    Run EVERY detector on the clip and report whatever produced output. We do
    NOT try to guess a single "kind" (voice vs data vs tone) - that guessing was
    unreliable. Instead the caller always also runs speech-to-text (unless the
    clip is silent) and shows any detector result that is meaningful.

    `voice_expected` is accepted for signature compatibility but no longer gates
    anything (we always attempt transcription on non-silent audio).

    Returns dict:
      {
        "is_silent": bool,             # truly no energy -> skip everything
        "dtmf": "171",                 # DTMF digits found (or "")
        "tones": [(freq_hz, dur_s)],   # steady tones found
        "data_burst": bool,            # looks like an FSK/data burst
        "detail": "...",               # human summary of non-voice signals found
        "switch_rate": float,
        "flatness": float,
      }
    """
    out = {"is_silent": True, "dtmf": "", "tones": [], "data_burst": False,
           "detail": "", "switch_rate": 0.0, "flatness": 0.0}
    if audio is None or audio.size == 0:
        return out

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if peak < 0.02 or rms < silence_rms:
        return out  # silent

    hop_s = 0.02
    doms, rmss = _dom_freq_series(audio, sr, hop_s=hop_s,
                                  min_rms=max(0.01, silence_rms))
    voiced = [d for d in doms if d is not None]
    if not voiced:
        return out  # silent

    out["is_silent"] = False

    arr = np.array(voiced)
    switches = int(np.sum(np.abs(np.diff(arr)) > 60))
    switch_rate = switches / max(1, len(arr))
    flatness = _mean_spectral_flatness(audio, sr, min_rms=max(0.01, silence_rms))
    out["switch_rate"] = round(switch_rate, 2)
    out["flatness"] = round(flatness, 3)

    # Run every detector unconditionally.
    out["dtmf"] = _detect_dtmf(audio, sr)
    out["tones"] = _steady_tones(doms, hop_s=hop_s)

    # A data-burst flag is informational only (it no longer suppresses speech).
    out["data_burst"] = (switch_rate > 0.45 and flatness >= 0.12
                         and not out["dtmf"] and not voice_expected)

    # Human-readable summary of the non-voice signals we found.
    bits = []
    if out["dtmf"]:
        bits.append(f"DTMF: {out['dtmf']}")
    if out["tones"]:
        shown = ", ".join(f"{f:.0f}Hz/{d:.2f}s" for f, d in out["tones"][:4])
        note = _two_tone_note(out["tones"])
        bits.append("tones: " + shown + (f" ({note})" if note else ""))
    if out["data_burst"]:
        bits.append(f"data/FSK burst (switch_rate={switch_rate:.2f})")
    out["detail"] = "; ".join(bits)
    return out


def _two_tone_note(tones):
    """Best-effort label for a 2-tone sequence (Quick Call II style)."""
    long_tones = [(f, d) for f, d in tones if d >= 0.2]
    if len(long_tones) == 2:
        (f1, d1), (f2, d2) = long_tones
        return (f"possible two-tone page: A={f1:.0f}Hz B={f2:.0f}Hz")
    return ""
