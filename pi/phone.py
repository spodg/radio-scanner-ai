"""
Phone number detector for scanner transcripts.

Officers read phone numbers digit-by-digit on the air, e.g.:
  "1-4-1-4-2-6-0-2-8-5-1-4-1-4"  -> (260) 285-1414

This module finds runs of digits in the transcript that match US phone number
patterns and formats them readably. It recognizes:
  - 10-digit (area + number): 2602851414 -> (260) 285-1414
  - 11-digit (1 + area + number): 12602851414 -> (260) 285-1414
  - 7-digit (local number, assume local area code): 2851414 -> 285-1414

Local area codes for the Fort Wayne / NE Indiana area are used to validate
and disambiguate (so a run of 10 digits starting with 260 is confidently a
phone number, not a random string).
"""

import re

# Area codes for the Fort Wayne / NE Indiana region. These are used to
# boost confidence that a digit run is a phone number (vs. a case number,
# address, or other numeric data officers read digit-by-digit).
LOCAL_AREA_CODES = {"260", "463"}

# All valid US area codes start with 2-9 (first digit) and the second digit
# is 0-9 (no restriction since 1995 when interchangeable NPA was introduced).
# We'll accept any 3-digit sequence where the first digit is 2-9 as a
# plausible area code.
def _valid_area_code(s):
    return len(s) == 3 and s[0] in "23456789"


def _looks_like_date(digits):
    """
    Check if a 7-8 digit string looks like a date rather than a phone number.
    Officers read dates digit-by-digit too: 1-31-2026 -> "1312026" (7 digits)
    or 01-31-2026 -> "01312026" (8 digits).
    """
    d = digits
    # Try M/DD/YYYY (7 digits): d[0]=month, d[1:3]=day, d[3:7]=year
    if len(d) == 7:
        m, dd, yyyy = d[0:1], d[1:3], d[3:7]
        if _valid_date_parts(m, dd, yyyy):
            return True
        # Try MM/D/YYYY: d[0:2]=month, d[2:3]=day, d[3:7]=year
        m, dd, yyyy = d[0:2], d[2:3], d[3:7]
        if _valid_date_parts(m, dd, yyyy):
            return True
        # Try M/DD/YY + ambiguous: not worth it, 7-digit local phones are low
        # confidence anyway so just check if last 4 digits are a plausible year
        yyyy = d[3:7]
        if yyyy.startswith("20") or yyyy.startswith("19"):
            return True
    if len(d) == 8:
        # MM/DD/YYYY
        m, dd, yyyy = d[0:2], d[2:4], d[4:8]
        if _valid_date_parts(m, dd, yyyy):
            return True
    return False


def _valid_date_parts(m, dd, yyyy):
    """Check if month/day/year strings form a plausible date."""
    try:
        mi, di, yi = int(m), int(dd), int(yyyy)
        return 1 <= mi <= 12 and 1 <= di <= 31 and 1900 <= yi <= 2099
    except ValueError:
        return False


def _format_phone(digits):
    """Format a 10-digit US phone number as (xxx) xxx-xxxx."""
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def detect_phones(text: str) -> list:
    """
    Find phone numbers in a transcript. Returns list of dicts:
      {"phone": "(260) 285-1414", "raw": "12602851414", "confidence": "high"}

    Works on both digit strings ("2602851414") and separated digits
    ("2-6-0-2-8-5-1-4-1-4" or "two six zero two eight five one four one four").
    """
    if not text:
        return []

    # Normalize: replace digit-word spellings with actual digits.
    norm = _words_to_digits(text)

    # Extract all digit characters, preserving their positions.
    # First, find all maximal runs of digits (possibly separated by hyphens,
    # spaces, or commas which officers use between digits).
    # Pattern: a sequence of digits separated by [-,. ] with at least 7 digits total.
    # e.g. "1-4-1-4-2-6-0-2-8-5-1-4-1-4" or "260 285 1414"
    pattern = r'\d(?:[\s\-,./]*\d){6,13}'
    results = []
    seen = set()

    for m in re.finditer(pattern, norm):
        raw = m.group(0)
        digits = re.sub(r'[^\d]', '', raw)

        if digits in seen:
            continue

        phone = None
        confidence = "low"

        if len(digits) == 11 and digits[0] == "1":
            # 1 + area code + number
            area = digits[1:4]
            if _valid_area_code(area):
                phone = _format_phone(digits[1:])
                confidence = "high" if area in LOCAL_AREA_CODES else "medium"

        elif len(digits) == 10:
            area = digits[:3]
            if _valid_area_code(area):
                phone = _format_phone(digits)
                confidence = "high" if area in LOCAL_AREA_CODES else "medium"

        elif len(digits) == 7:
            # Local number without area code — but first check if it looks
            # like a date (MMDDYYY, MDDYYYY, etc.). Common false positive.
            if _looks_like_date(digits):
                continue
            phone = f"{digits[:3]}-{digits[3:]}"
            confidence = "low"

        # Longer runs: officers sometimes repeat part of the number for
        # confirmation (e.g. "1-4-1-4-2-6-0-2-8-5-1-4-1-4" where "1414" is
        # repeated). Try to find a valid 10 or 11 digit phone within the run.
        if not phone and len(digits) > 11:
            # Scan for a local area code starting position
            for start in range(len(digits) - 9):
                chunk = digits[start:]
                # Try 11-digit (1+area+number)
                if len(chunk) >= 11 and chunk[0] == "1" and _valid_area_code(chunk[1:4]):
                    candidate = chunk[1:11]
                    if candidate[:3] in LOCAL_AREA_CODES:
                        phone = _format_phone(candidate)
                        confidence = "high"
                        break
                # Try 10-digit (area+number)
                if len(chunk) >= 10 and _valid_area_code(chunk[:3]):
                    candidate = chunk[:10]
                    if candidate[:3] in LOCAL_AREA_CODES:
                        phone = _format_phone(candidate)
                        confidence = "high"
                        break

        if phone:
            seen.add(digits)
            results.append({"phone": phone, "raw": digits, "confidence": confidence})

    return results


def format_phones(phones: list) -> str:
    """Format detected phones for the log line."""
    if not phones:
        return ""
    return "phone: " + ", ".join(p["phone"] for p in phones)


# Digit word mapping (same as in codes.py but standalone here)
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "niner": "9",
}


def _words_to_digits(text: str) -> str:
    """Replace spelled-out single digits with their numeral."""
    result = text
    for word, digit in _DIGIT_WORDS.items():
        result = re.sub(r'\b' + word + r'\b', digit, result, flags=re.IGNORECASE)
    return result
