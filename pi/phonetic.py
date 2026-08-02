"""
Phonetic-alphabet plate/ID decoder.

Officers spell letters on the air with a phonetic alphabet (APCO "Adam, Boy,
Charles..." and/or NATO "Alpha, Bravo, Charlie..."), mixed with spoken digits.
This turns a transcript fragment like:

    "9-7-2, Edward, Robert, Gideon"   ->  "972ERG"

into the compact plate/identifier string.

SCOPE: this only converts spoken characters into a string (a readability
transform of audio we already transcribed). It does NOT look up vehicle
registration, owners, or VINs from a plate - that data is restricted under the
federal Driver's Privacy Protection Act and is not lawfully available to the
public. See README for the VIN-decode (NHTSA vPIC) path, which is the legitimate
direction: a VIN you already have -> make/model/year, never plate -> identity.
"""

import re

# Map spoken words to a single character. Includes APCO, NATO, and common
# agency variants (e.g. G = George / Gideon / golf).
PHONETIC = {
    # A
    "adam": "A", "alpha": "A",
    # B
    "boy": "B", "bravo": "B", "baker": "B",
    # C
    "charles": "C", "charlie": "C",
    # D
    "david": "D", "delta": "D",
    # E
    "edward": "E", "echo": "E", "easy": "E",
    # F
    "frank": "F", "foxtrot": "F", "fox": "F",
    # G
    "george": "G", "gideon": "G", "golf": "G",
    # H
    "henry": "H", "hotel": "H",
    # I
    "ida": "I", "india": "I",
    # J
    "john": "J", "juliet": "J", "juliett": "J",
    # K
    "king": "K", "kilo": "K",
    # L
    "lincoln": "L", "lima": "L",
    # M
    "mary": "M", "mike": "M",
    # N
    "nora": "N", "november": "N", "nancy": "N",
    # O
    "ocean": "O", "oscar": "O",
    # P
    "paul": "P", "papa": "P",
    # Q
    "queen": "Q", "quebec": "Q",
    # R
    "robert": "R", "romeo": "R",
    # S
    "sam": "S", "sierra": "S",
    # T
    "tom": "T", "tango": "T", "town": "T",   # "town" is common Whisper error for "tom"
    # U
    "union": "U", "uniform": "U",
    # V
    "victor": "V",
    # W
    "william": "W", "whiskey": "W",
    # X
    "xray": "X", "x-ray": "X",
    # Y
    "young": "Y", "yankee": "Y",
    # Z
    "zebra": "Z", "zulu": "Z",
}

# Words that are clearly phonetic-alphabet (used to gate plate detection so we
# don't treat ordinary speech like "two units" as a plate). These are the words
# unlikely to appear as normal English in this context.
STRONG_PHONETIC = {
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "juliett", "kilo", "lima", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "x-ray", "yankee", "zulu",
    "adam", "boy", "charles", "edward", "gideon", "ida", "lincoln", "nora",
    "ocean", "paul", "queen", "robert", "zebra", "baker", "town",
}

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "niner": "9",
}


def _token_to_char(tok: str):
    """Return (char, is_strong_phonetic) for a token, or (None, False)."""
    t = tok.lower().strip(".,")
    if t.isdigit() and len(t) == 1:
        return t, False
    if t in DIGIT_WORDS:
        return DIGIT_WORDS[t], False
    if t in PHONETIC:
        return PHONETIC[t], (t in STRONG_PHONETIC)
    # Single letter (e.g. "T", "P", "T.", "p.") — valid plate character but
    # not a strong phonetic indicator on its own.
    if len(t) == 1 and t.isalpha():
        return t.upper(), False
    return None, False


def decode_plates(text: str, min_len: int = 4, max_len: int = 8):
    """
    Find phonetic plate/ID strings in a transcript.

    Returns a list of dicts: {"plate": "972ERG", "spoken": "9-7-2, Edward, ..."}.

    Heuristics to avoid false positives from ordinary speech:
      - Consider maximal runs of consecutive mappable tokens (digits + phonetic
        words), allowing comma/hyphen/space and the filler word "and" between.
      - Accept a run only if it is min_len..max_len characters AND contains at
        least two STRONG phonetic words, OR is a single tightly-spelled run with
        >=2 strong phonetic words. Pure number runs are ignored (too ambiguous).
    """
    if not text:
        return []

    # Tokenize into words, digit-runs, and separators.
    raw = re.findall(r"[A-Za-z]+|\d+|[-,/]| +", text)
    tokens = [r for r in raw if r.strip() not in ("", "")]

    results = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i].strip()
        # skip separators/fillers
        if tok in ("-", ",", "/", "") or tok.lower() == "and":
            i += 1
            continue

        # Try to grow a run starting here.
        run_chars = []
        strong = 0
        j = i
        last_mapped = i - 1
        while j < n:
            t = tokens[j].strip()
            if t in ("-", ",", "/", "") or t.lower() == "and":
                j += 1
                continue
            # multi-digit number token: expand to individual digits
            if t.isdigit():
                for d in t:
                    run_chars.append(d)
                last_mapped = j
                j += 1
                continue
            ch, is_strong = _token_to_char(t)
            if ch is None:
                break
            run_chars.append(ch)
            if is_strong:
                strong += 1
            last_mapped = j
            j += 1

        plate = "".join(run_chars)
        if (min_len <= len(plate) <= max_len) and strong >= 2:
            spoken = " ".join(tokens[i:last_mapped + 1]).strip()
            spoken = re.sub(r"\s+", " ", spoken)
            results.append({"plate": plate, "spoken": spoken})
            i = last_mapped + 1
        else:
            i += 1

    # Also detect raw plate patterns (e.g. "299 ERM") with context clues
    raw_plates = _detect_raw_plates(text)
    seen = {r["plate"] for r in results}
    for rp in raw_plates:
        if rp["plate"] not in seen:
            results.append(rp)

    return results


def decode_plate_string(text: str) -> str:
    """Convenience: return the first decoded plate, or ''."""
    plates = decode_plates(text)
    return plates[0]["plate"] if plates else ""


def _detect_raw_plates(text: str):
    """
    Detect plate-like patterns directly from text (not phonetic-spelled).
    Matches common US formats: 3 digits + 2-3 letters, or 2-3 letters + 4 digits, etc.
    Only matches when preceded by plate-context words.
    """
    if not text:
        return []
    
    # Context words that strongly suggest a plate is being read
    context_re = re.compile(
        r'(?:plate|tag|registration|blackout|tag number|10-28|10-27|'
        r'registered to|registered owner|vehicle reg)',
        re.IGNORECASE
    )
    
    if not context_re.search(text):
        return []
    
    # Common plate patterns: NNN LLL, NNN LLLL, LLL NNNN, etc.
    # Must be 5-7 chars total with mix of letters and digits
    plate_re = re.compile(
        r'\b(\d{2,4}\s?[A-Z]{2,4})\b|\b([A-Z]{2,4}\s?\d{2,4})\b',
        re.IGNORECASE
    )
    
    # Exclude common false positive words
    exclude = {'WEST', 'EAST', 'NORTH', 'SOUTH', 'ROAD', 'STREET', 'DRIVE',
               'CARD', 'JUST', 'THAT', 'THIS', 'WITH', 'FROM', 'BLUE', 'SHOW',
               'JOHN', 'NORA', 'ADAM', 'KING'}
    
    results = []
    for m in plate_re.finditer(text):
        plate = (m.group(1) or m.group(2)).replace(" ", "").upper()
        if 5 <= len(plate) <= 7:
            has_alpha = any(c.isalpha() for c in plate)
            has_digit = any(c.isdigit() for c in plate)
            # Check it's not a common word
            alpha_part = ''.join(c for c in plate if c.isalpha())
            if has_alpha and has_digit and alpha_part not in exclude:
                results.append({"plate": plate, "spoken": m.group(0)})
    
    return results
