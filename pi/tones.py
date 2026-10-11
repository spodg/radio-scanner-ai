"""
Steady alert-tone detector for scanner clips.

Reports sustained pure tones (e.g. two-tone / Quick Call II paging and alert
tones) found in a captured clip, with a best-effort note when a pair looks like
a two-tone page. DTMF and FSK/data-burst decoding were removed: on 16 kHz
line-in voice audio they only produced false positives from voice formants
(use multimon-ng if real DTMF/paging payload decode is ever needed).
"""

import numpy as np

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
    Detect steady alert tones (e.g. two-tone / Quick Call II paging) in the
    clip. The caller always also runs speech-to-text on non-silent audio, so
    this only surfaces the non-voice tone info.

    `voice_expected` is accepted for signature compatibility but unused.

    Returns dict:
      {
        "is_silent": bool,             # truly no energy -> skip everything
        "tones": [(freq_hz, dur_s)],   # steady tones found
        "detail": "...",               # human summary of tones found
      }
    """
    out = {"is_silent": True, "tones": [], "detail": ""}
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
    out["tones"] = _steady_tones(doms, hop_s=hop_s)

    if out["tones"]:
        shown = ", ".join(f"{f:.0f}Hz/{d:.2f}s" for f, d in out["tones"][:4])
        note = _two_tone_note(out["tones"])
        out["detail"] = "tones: " + shown + (f" ({note})" if note else "")
    return out


def _two_tone_note(tones):
    """Best-effort label for a 2-tone sequence (Quick Call II style)."""
    long_tones = [(f, d) for f, d in tones if d >= 0.2]
    if len(long_tones) == 2:
        (f1, d1), (f2, d2) = long_tones
        return (f"possible two-tone page: A={f1:.0f}Hz B={f2:.0f}Hz")
    return ""
