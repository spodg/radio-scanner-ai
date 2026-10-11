"""Shared text-transcript decoders (plates, phones, radio codes).

Single source of truth for the decoding logic that was previously duplicated
in pi_scanner.py, pi_transcriber.py and dashboard.py.
"""

try:
    from codes import decode_for
    from phonetic import decode_plates
    from phone import detect_phones
    _DECODERS_AVAILABLE = True
except ImportError:
    _DECODERS_AVAILABLE = False


def run_text_decoders(text, channel_name):
    """Run text-based decoders on a transcript. Returns a dict of findings."""
    if not _DECODERS_AVAILABLE or not text:
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
        # Only surface confident matches; "low" confidence is almost always
        # transcript digit-salad (7-digit runs with no area code).
        kept = [p["phone"] for p in phones if p.get("confidence") in ("high", "medium")]
        if kept:
            results["phones"] = kept
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
