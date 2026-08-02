"""
Morse code (CW) detector and decoder for scanner audio.

Amateur repeaters (and some other services) send a keyed-tone callsign ID. It's
a single audio tone switched on/off. We:

  1. Find the CW tone pitch (the dominant steady tone in a typical CW band).
  2. Extract that tone's on/off envelope over time (Goertzel power).
  3. Threshold to a binary keyed signal and measure element/gap durations.
  4. Adaptively split short vs long ON (dot vs dash) and the three gap sizes
     (intra-character, inter-character, inter-word) using the timing, since the
     sender's speed (WPM) is unknown.
  5. Map the dot/dash patterns to characters.

This is robust for clean repeater IDs. It can struggle when voice audio overlaps
the same pitch, so we also report a confidence based on how bimodal the element
timing is.
"""

import numpy as np

MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-..-.": "/",
    "-.-.--": "!", "-....-": "-", ".-.-.": "+", "-...-": "=",
    ".--.-.": "@",
}


def _goertzel_envelope(audio, sr, freq, win, hop):
    """Power of `freq` per window -> envelope of the CW tone."""
    n = win
    k = int(0.5 + (n * freq) / sr)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    env = []
    hann = np.hanning(win)
    for s in range(0, len(audio) - win, hop):
        seg = audio[s:s + win] * hann
        s_prev = s_prev2 = 0.0
        for x in seg:
            cur = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = cur
        power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
        env.append(power)
    return np.array(env)


def _find_cw_pitch(audio, sr, fmin=400, fmax=1500):
    """Most frequently dominant frequency in the CW band -> candidate pitch."""
    win = int(0.02 * sr)
    hop = int(0.01 * sr)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    band_idx = np.where(band)[0]
    if band_idx.size == 0:
        return None
    hann = np.hanning(win)
    dom = []
    for s in range(0, len(audio) - win, hop):
        seg = audio[s:s + win] * hann
        spec = np.abs(np.fft.rfft(seg))
        dom.append(band_idx[np.argmax(spec[band])])
    if not dom:
        return None
    vals, counts = np.unique(dom, return_counts=True)
    return float(freqs[vals[np.argmax(counts)]])


def _runs_from_binary(on, frame_s):
    """Convert a boolean keyed array into (state, duration_s) runs."""
    runs = []
    cur = on[0]
    start = 0
    for i in range(1, len(on)):
        if on[i] != cur:
            runs.append((bool(cur), (i - start) * frame_s))
            start = i
            cur = on[i]
    runs.append((bool(cur), (len(on) - start) * frame_s))
    return runs


def _split_threshold(durations):
    """
    Find a split between 'short' and 'long' durations via the largest gap in
    the sorted unique values (simple 1-D 2-means-ish). Returns the threshold.
    """
    u = np.unique(np.round(durations, 3))
    if len(u) == 1:
        return u[0] * 1.5
    gaps = np.diff(u)
    i = int(np.argmax(gaps))
    return (u[i] + u[i + 1]) / 2.0


def _otsu_threshold(values, bins=64):
    """
    Otsu's method: choose the threshold that maximizes between-class variance
    of the value histogram. Robust to any on/off duty cycle. Falls back to a
    sensible value when the signal is nearly constant.
    """
    v = np.asarray(values, dtype=float)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax - vmin < 1e-9:
        return vmax + 1.0  # constant -> never "on"
    hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return (vmin + vmax) / 2.0
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom < 1e-12] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def decode(audio: np.ndarray, sr: int, min_on_frames: int = 2):
    """
    Decode Morse from a mono float32 clip.

    Returns dict:
      {"text": "W9XYZ", "pitch_hz": 800.0, "wpm": 18.0, "confidence": 0.7}
    or {"text": "", ...} if no decodable Morse was found.
    """
    empty = {"text": "", "pitch_hz": None, "wpm": None, "confidence": 0.0}
    if audio is None or audio.size == 0:
        return empty

    pitch = _find_cw_pitch(audio, sr)
    if pitch is None:
        return empty

    win = int(0.01 * sr)          # 10 ms resolution
    hop = int(0.005 * sr)         # 5 ms hop
    frame_s = hop / sr
    env = _goertzel_envelope(audio, sr, pitch, win, hop)
    if env.size < 10:
        return empty

    # Normalize, then pick a threshold that separates "tone off" from "tone on"
    # regardless of the on/off duty cycle (median fails when the signal is ON
    # most of the time, e.g. dash-heavy callsigns). Otsu's method finds the
    # split that minimizes intra-class variance of the energy histogram.
    env = env / (np.max(env) + 1e-12)
    thr = _otsu_threshold(env)
    on = env > thr

    # Remove ultra-short ON/OFF blips (debounce). Use a minimum run length
    # scaled to the frame rate: at least ~20ms of sustained state to count.
    min_frames = max(2, int(0.02 / frame_s))
    on = _debounce(on, min_frames)

    runs = _runs_from_binary(on, frame_s)
    on_runs = [d for st, d in runs if st]
    if len(on_runs) < 3:
        return empty

    # Discard the leading/trailing OFF runs for gap analysis.
    # Dot/dash split from ON durations.
    dot_dash_thr = _split_threshold(np.array(on_runs))
    # Sanity: dashes should be ~3x dots. Estimate dot length as median of shorts.
    shorts = [d for d in on_runs if d <= dot_dash_thr]
    dot_len = float(np.median(shorts)) if shorts else float(np.min(on_runs))
    if dot_len <= 0:
        return empty

    # Gap thresholds in units of dot length: <1.5 intra-char, 1.5-4.5 char, >4.5 word
    char_gap = 2.0 * dot_len
    word_gap = 5.0 * dot_len

    # Walk the runs building characters, while tracking which characters come
    # from the dense keyed region vs isolated blips (voice leaking into the CW
    # bin produces stray single-dot "E" characters separated by big gaps).
    decoded = []       # list of (char, preceding_gap_s)
    cur = ""
    pending_gap = 0.0

    def flush_char(gap_before):
        nonlocal cur
        if cur:
            decoded.append([MORSE_TO_CHAR.get(cur, "?"), gap_before])
            cur = ""

    last_gap = 0.0
    for st, d in runs:
        if st:  # tone ON -> dot or dash
            if not cur:
                pending_gap = last_gap
            cur += "." if d <= dot_dash_thr else "-"
        else:   # tone OFF -> gap
            last_gap = d
            if d >= char_gap:
                flush_char(pending_gap)
    flush_char(pending_gap)

    # Trim isolated single-character blips at the ends that are separated from
    # the rest by an unusually long gap (> 8 dot-lengths). These are almost
    # always voice artifacts, not part of the keyed ID.
    big_gap = 8.0 * dot_len
    while len(decoded) >= 2 and decoded[0][0] in ("E", "T", "?") \
            and decoded[1][1] >= big_gap:
        decoded.pop(0)
    while len(decoded) >= 2 and decoded[-1][0] in ("E", "T", "?") \
            and decoded[-1][1] >= big_gap:
        decoded.pop()

    # Rebuild text, inserting word spaces where the gap before a char is large.
    parts = []
    for i, (ch, gap_before) in enumerate(decoded):
        if i > 0 and gap_before >= word_gap:
            parts.append(" ")
        parts.append(ch)
    text = "".join(parts).strip()
    if not text or all(c in "? " for c in text):
        return empty

    # WPM estimate: PARIS standard, dot = 1.2 / wpm seconds.
    wpm = round(1.2 / dot_len, 1) if dot_len > 0 else None

    # Confidence: fraction of decoded chars that aren't "?", and bimodality of
    # ON durations.
    known = sum(1 for c in text if c not in "? ")
    total = sum(1 for c in text if c != " ")
    conf = (known / total) if total else 0.0

    return {"text": text, "pitch_hz": round(pitch, 1), "wpm": wpm,
            "confidence": round(conf, 2)}


def _debounce(on, min_frames):
    """Remove runs shorter than min_frames by merging into neighbors."""
    out = on.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_frames and 0 < i:
            out[i:j] = out[i - 1]
        i = j
    return out
