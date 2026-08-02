"""
Police radio code decoder for the Fort Wayne / northeast Indiana area
(zip 46835 = Allen County).

WHY AGENCY PROFILES?
Codes are NOT universal. Even neighboring agencies disagree on the same number
(e.g. Allen County signal 8 = "In service", but Fort Wayne City PD signal 8 =
"Out of State"; Indiana State Police use a different scheme entirely). The
scanner here also hears multiple counties (Adams, Noble, etc.), so we route
each transmission to the right code table based on its GLG name
(system / group / channel).

decode_for() picks a profile from the channel name, then decodes the
transcript against that profile. If we don't have a specific table for an
agency, we fall back to the Allen County / Indiana scheme and mark the result
approximate.

Sources: publicly published Indiana radio-code references (police-codes.com,
RadioReference). Treat decoded text as a best-effort hint; codes drift over
time and vary by agency.
"""

import re
from dataclasses import dataclass, field


# =============================================================================
# Code tables
# =============================================================================

# --- Allen County / Fort Wayne signals (bare-number "signal" style) ----------
ALLEN_SIGNALS = {
    1: "Meet", 2: "Disregard", 3: "Telephone", 4: "Courthouse",
    5: "Garage", 6: "Communications", 7: "Out of service", 8: "In service",
    9: "On the scene", 10: "Give location", 11: "Pick up prisoner",
    12: "County jail", 13: "Ambulance", 14: "Warrant check",
    15: "Assist party", 16: "Pick up items", 17: "Read motor number",
    18: "Work traffic", 19: "Escort", 20: "Problem unknown", 21: "Adult",
    22: "Juvenile", 23: "Vandalism", 24: "Lost/missing person",
    25: "Auto accident", 26: "Auto accident w/ injury", 27: "Reckless driving",
    28: "DUI", 29: "Special event", 30: "Traffic stop", 31: "Breathalyzer",
    32: "Death investigation", 33: "Disabled vehicle", 35: "Parked vehicle",
    36: "Parked car occupied", 37: "Parking violation", 38: "Lost/stolen plate",
    39: "Vehicle theft", 40: "Alarm (intrusion)", 41: "Man down",
    42: "Intoxicated subject", 43: "Disturbance", 44: "Nuisance (smoke/noise)",
    45: "Neighborhood disturbance", 46: "Domestic disturbance",
    47: "Removal of articles", 48: "Suspicious person", 49: "Prowler",
    50: "Burglary", 51: "Theft", 52: "Shoplifter", 53: "Armed robbery",
    55: "Fight", 56: "Molesting", 57: "Robbery (strong arm)", 58: "Shooting",
    59: "Cutting/stabbing", 60: "Homicide", 61: "Rape", 62: "Party armed",
    63: "Attempt to contact", 64: "Party not seen", 65: "Suicide",
    66: "Demented person", 67: "Indecent exposure", 68: "Protect evidence",
    69: "Wires down", 70: "Fire", 71: "Explosion", 72: "Failure to pay",
    73: "Walk away", 74: "Unruly crowd", 75: "Pursuit", 76: "Road block",
    77: "Labor dispute", 78: "Communicable disease", 79: "Confidential info",
    80: "Civil investigation", 81: "Change channel", 82: "Improper procedure",
    83: "Advise disposition", 84: "Urgent", 85: "At once",
    86: "Municipal property involved", 87: "Begin tour of duty",
    88: "End tour of duty", 89: "EOD", 91: "Blood run", 92: "Drowning",
    93: "Aircraft accident", 94: "Out of jurisdiction", 95: "In jurisdiction",
    96: "Battery", 97: "Found items", 98: "Vice investigation",
    99: "Narcotics investigation", 100: "Hold all but emergency traffic",
    101: "Open door", 102: "Threats", 103: "Unwanted party", 104: "Kidnapping",
    105: "Are you OK?", 106: "Response OK?", 107: "On the air",
    108: "Follow-up", 109: "Illegal dumping", 110: "Known burglar",
    111: "Animal investigation", 112: "Riot", 120: "Grabill",
    121: "Woodburn", 122: "Huntertown", 154: "Officer in trouble",
}

# --- Fort Wayne CITY Police signals (note differences vs Allen County!) -------
FWPD_SIGNALS = {
    1: "Meet", 2: "Disregard", 3: "Transmit", 4: "Police station",
    5: "Police garage", 6: "Communications", 7: "In service",
    8: "Out of state", 9: "Arrived on scene", 10: "Give location",
    11: "Pick up prisoner", 12: "County jail", 13: "Ambulance",
    14: "Warrant clerk", 15: "Assist party", 16: "Make pick up",
    17: "Read motor number", 18: "Work traffic", 19: "Escort",
    20: "Problem unknown", 21: "Adults", 22: "Juveniles", 23: "Vandalism",
    24: "Lost child", 25: "Auto accident", 26: "Auto accident w/ injury",
    27: "Reckless driving", 28: "Drunk driver", 29: "Drag racing",
    30: "Traffic stopped", 31: "Breathalyzer", 32: "Street box alarm",
    33: "Vehicle disabled", 34: "Traffic hazard", 35: "Parked car (empty)",
    36: "Parked car (occupied)", 37: "Illegal parking", 38: "Stealing vehicle",
    39: "Stolen vehicle", 40: "Alarm (bank, etc.)", 41: "Man down",
    42: "Drunk", 43: "Disturbance", 45: "Neighborhood trouble",
    46: "Family trouble", 47: "Removal of articles", 48: "Suspicious person",
    49: "Prowler", 50: "Break in", 51: "Theft", 52: "Shoplifter",
    53: "Robbery (armed)", 54: "Robbery in progress", 55: "Fight",
    56: "Molesting", 57: "Robbery (strong arm)", 58: "Shooting",
    59: "Stabbing/cutting", 60: "Homicide", 61: "Rape", 62: "Party armed",
    63: "Attempt to contact", 64: "Party not seen", 65: "Suicide",
    66: "Demented person", 67: "Indecent exposure", 68: "Protect evidence",
    69: "Wires down", 70: "Fire", 71: "Explosion", 72: "Failure to pay",
    73: "Walk away", 74: "Unruly crowd", 75: "Riot", 76: "Road block",
    77: "Strike duty", 78: "Dog bite", 79: "Confidential information",
    80: "Give call letters", 81: "Change channel", 82: "Improper procedure",
    83: "Advise disposition", 84: "Urgent", 85: "At once",
    86: "City property involved", 87: "On duty", 88: "Off duty",
    89: "Bomb threat", 100: "Hold all but emergency traffic",
}

# --- Indiana State Police status codes (spoken as "signal N") -----------------
ISP_SIGNALS = {
    1: "Call your district", 2: "Call general HQ", 3: "Call state police radio",
    4: "Report to general HQ", 5: "Report to your district", 6: "Call by telephone",
    7: "Emergency (serious situation)", 8: "Meet", 9: "Disregard",
    10: "Rush (quick action)", 11: "Information confidential",
    12: "Reply via other than radio", 13: "Army convoy", 14: "Plain clothes detail",
    16: "Aircraft accident", 19: "Truck check", 20: "Car wash",
    21: "Car service", 22: "Car repair", 23: "Speeding car",
    24: "Vehicle/occupants detained", 27: "Stopping car", 28: "Bank detail",
    31: "Traffic congestion", 32: "Check records for wanted", 33: "Known burglar",
    34: "Possible mental", 40: "Records indicate subject wanted",
    41: "Lie detector available?", 42: "Drunkometer available?",
    46: "In pursuit", 47: "Escort", 48: "Visitors/officials present",
    60: "Narcotics/dangerous drugs involved", 100: "Emergency: hold all traffic",
}

# --- Indiana 10-codes (shared across most Indiana agencies) -------------------
INDIANA_TEN_CODES = {
    "10-0": "Caution", "10-1": "Unable to copy", "10-2": "Signal good",
    "10-3": "Stop transmitting", "10-4": "Acknowledged / OK", "10-5": "Relay",
    "10-6": "Busy, stand by", "10-7": "Out of service", "10-8": "In service",
    "10-9": "Repeat", "10-10": "Fight in progress", "10-11": "Dog case",
    "10-12": "Stand by", "10-13": "Weather/road report", "10-14": "Prowler report",
    "10-15": "Civil disturbance", "10-16": "Domestic problem",
    "10-17": "Meet complainant", "10-18": "Complete assignment quickly",
    "10-19": "Return to ...", "10-20": "Location", "10-21": "Call by phone",
    "10-22": "Disregard", "10-23": "Arrived at scene", "10-24": "Assignment complete",
    "10-25": "Report in person", "10-26": "Detaining subject",
    "10-27": "Driver's license info", "10-28": "Vehicle registration info",
    "10-29": "Check record for wanted", "10-30": "Illegal use of radio",
    "10-31": "Crime in progress", "10-32": "Man with gun", "10-33": "Emergency",
    "10-34": "Riot", "10-35": "Major crime alert", "10-36": "Correct time",
    "10-37": "Investigate suspicious vehicle", "10-38": "Stopping suspicious vehicle",
    "10-39": "Urgent, lights & siren", "10-40": "Silent run, no lights/siren",
    "10-41": "Beginning tour of duty", "10-42": "Ending tour of duty",
    "10-43": "Information", "10-44": "Request permission to leave patrol",
    "10-45": "Animal carcass in road", "10-46": "Assist motorist",
    "10-47": "Emergency road repairs", "10-48": "Traffic control",
    "10-49": "Traffic light out", "10-50": "Accident (F/PI/PD)",
    "10-51": "Wrecker needed", "10-52": "Ambulance needed", "10-53": "Road blocked",
    "10-54": "Livestock on highway", "10-55": "Intoxicated driver",
    "10-56": "Intoxicated pedestrian", "10-57": "Hit & run", "10-58": "Direct traffic",
    "10-59": "Convoy/escort", "10-60": "Squad in vicinity", "10-62": "Reply to message",
    "10-63": "Prepare to copy", "10-66": "Message cancellation", "10-70": "Fire alarm",
    "10-71": "Advise nature of fire", "10-76": "En route", "10-77": "ETA",
    "10-78": "Need assistance", "10-79": "Notify coroner", "10-80": "Chase in progress",
    "10-89": "Bomb threat", "10-90": "Bank alarm", "10-91": "Pick up prisoner/subject",
    "10-95": "Subject in custody", "10-96": "Mental subject", "10-97": "Test signal",
    "10-98": "Jail break", "10-99": "Records indicate wanted",
}


# =============================================================================
# Agency profiles + routing
# =============================================================================

@dataclass
class Profile:
    name: str                       # human label, shown in logs
    signals: dict                   # number -> meaning
    ten_codes: dict                 # "10-x" -> meaning
    uses_bare_signals: bool         # decode bare numbers as signals?
    keywords: list = field(default_factory=list)  # lowercase substrings to match
    approximate: bool = False       # True when we're guessing the table


# Order matters: more specific agencies are checked first.
PROFILES = [
    Profile(
        name="Indiana State Police",
        signals=ISP_SIGNALS,
        ten_codes=INDIANA_TEN_CODES,
        uses_bare_signals=False,    # ISP leans on 10-codes, not bare signals
        keywords=["state police", "isp", "district 22", "indiana state"],
    ),
    Profile(
        name="Fort Wayne City PD",
        signals=FWPD_SIGNALS,
        ten_codes=INDIANA_TEN_CODES,
        uses_bare_signals=True,
        keywords=["fort wayne police", "fwpd", "fw police", "city police",
                  "fort wayne pd"],
    ),
    Profile(
        name="Allen County",
        signals=ALLEN_SIGNALS,
        ten_codes=INDIANA_TEN_CODES,
        uses_bare_signals=True,
        keywords=["allen county", "allen co", "acpd", "acso", "new haven",
                  "woodburn", "grabill", "huntertown", "allen "],
    ),
]

# Fallback for agencies we don't have a specific table for (e.g. surrounding
# counties like Adams, Noble, DeKalb, Whitley). We use the Allen County signal
# scheme + Indiana 10-codes as a best guess and flag it approximate. This is
# ONLY used for channels that look like law enforcement (see pick_profile).
DEFAULT_PROFILE = Profile(
    name="Indiana (generic)",
    signals=ALLEN_SIGNALS,
    ten_codes=INDIANA_TEN_CODES,
    uses_bare_signals=True,
    keywords=[],
    approximate=True,
)

# Used for channels that are NOT law enforcement (aviation/ATC, railroad,
# hospitals, schools, aircraft, Red Cross, business, etc.). Police signal/
# 10-code decoding does not apply to these, so we decode nothing rather than
# mistranslate (e.g. an ATC "9-5-4-2" frequency is not "signal 9, signal 5...").
NO_CODE_PROFILE = Profile(
    name="(no codes)",
    signals={},
    ten_codes={},
    uses_bare_signals=False,
    keywords=[],
    approximate=False,
)

# Substrings that indicate a law-enforcement channel. Only these get the
# generic police-code fallback when no specific agency profile matches.
LAW_ENFORCEMENT_KEYWORDS = [
    "police", "sheriff", "law enforcement", "trooper", "state police",
    "marshal", "constable", "deputy", "patrol", "pd ", " pd", "p.d.",
    "public safety", "corrections", "swat",
    "ems", "medic", "ambulance", "fire", "rescue", "dispatch",
    "emergency", "paramedic", "engine", "ladder", "battalion",
]

# Substrings that clearly indicate a NON-police channel. Checked so we never
# apply police codes to these even if some odd word overlaps.
NON_POLICE_KEYWORDS = [
    "air route", "traffic control", "rcag", "artcc", "approach", "tower",
    "ground", "departure", "atis", "unicom", "aircraft", "aviation",
    "airport", "sector", "railroad", "rail ", "amtrak",
    "school", "red cross", "weather",
    "noaa", "marine", "coast guard", "business", "taxi", "tow", "utility",
    "public works", "transit", "citilink", "roads", "highway dept",
]


def _kw_matches(keyword: str, name: str) -> bool:
    """
    Match a keyword against a name on word boundaries, so short keywords like
    "isp" don't match inside unrelated words (e.g. "d-isp-atch"). Keywords that
    intentionally end in a space (e.g. "allen ") are matched as substrings to
    preserve that prefix behavior.
    """
    if keyword.endswith(" "):
        return keyword in name
    return re.search(r"\b" + re.escape(keyword) + r"\b", name) is not None


def pick_profile(channel_name: str) -> Profile:
    """
    Choose the best code profile for a GLG channel/system/group name.

    Routing logic:
      1. A specific agency profile (ISP / Fort Wayne City / Allen County) if a
         keyword matches.
      2. NO_CODE_PROFILE if the channel is clearly non-police (aviation, rail,
         hospital, school, etc.) -> we decode nothing.
      3. The generic Indiana police fallback ONLY if the channel looks like law
         enforcement.
      4. Otherwise NO_CODE_PROFILE (don't guess police codes for unknown,
         non-law-enforcement traffic).
    """
    if not channel_name:
        return NO_CODE_PROFILE
    name = channel_name.lower()

    # 1) Clearly non-police channels (aviation, rail, hospital, school, etc.)
    #    short-circuit to no decoding, even if a geographic keyword like
    #    "allen county" also appears (e.g. "NW Allen County Schools").
    if any(kw in name for kw in NON_POLICE_KEYWORDS):
        return NO_CODE_PROFILE

    # 2) Specific agency match (ISP / Fort Wayne City / Allen County).
    for prof in PROFILES:
        if any(_kw_matches(kw, name) for kw in prof.keywords):
            return prof

    # 3) Looks like law enforcement -> generic police fallback (approximate).
    if any(kw in name for kw in LAW_ENFORCEMENT_KEYWORDS):
        return DEFAULT_PROFILE

    # 4) Unknown and not obviously police -> decode nothing.
    return NO_CODE_PROFILE


# =============================================================================
# Transcript parsing
# =============================================================================

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_digits(text: str) -> str:
    """
    Replace spelled-out numbers (0-159) with digit strings, so the matchers
    work whether Whisper wrote "ten forty-two" or "10-42". Number-internal
    hyphens ("forty-two") and a "hundred" multiplier are handled. Crucially we
    DON'T introduce spaces around existing digit hyphens, so "10-42" stays a
    tight token for the 10-code matcher.
    """
    s = text.lower()
    # Split into number-words, digit runs, hyphens, and other punctuation.
    tokens = re.findall(r"[a-z]+|\d+|-|[^\w\s-]+|\s+", s)

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not tok.isalpha():
            out.append(tok)
            i += 1
            continue

        val = None
        consumed = 1
        if tok in _TENS:
            val = _TENS[tok]
            # look ahead past an optional hyphen/space for a ones word
            j = i + 1
            while j < n and (tokens[j] == "-" or tokens[j].isspace()):
                j += 1
            if j < n and tokens[j] in _ONES and _ONES[tokens[j]] < 10:
                val += _ONES[tokens[j]]
                consumed = (j - i) + 1
        elif tok in _ONES:
            val = _ONES[tok]

        if val is not None:
            out.append(str(val))
            i += consumed
        else:
            out.append(tok)
            i += 1

    return "".join(out)


# Words that precede a number and indicate it's a unit ID, not a signal code.
# e.g. "Medic 93", "Engine 72", "Unit 14", "Squad 5" — the number is an ID.
_UNIT_PREFIX_RE = re.compile(
    r'(?:medic|engine|ladder|truck|squad|unit|battalion|chief|car|'
    r'rescue|ambulance|tanker|pumper|station|deputy|trooper|officer|'
    r'patrol|adam|lincoln|david|king|county|shell|block|lot|'
    r'room|zone|floor|level|code|route|highway|interstate)\s*$',
    re.IGNORECASE)

# 10-code: "10-42", "10 42", "1042". Tolerant of spaces/hyphen between parts.
_TEN_RE = re.compile(r"\b10\s*-?\s*(\d{1,2})\b")
# Explicit "signal 22" / "sig 22" / "signal-22".
_SIGNAL_RE = re.compile(r"\b(?:signal|sig)\s*[-#]?\s*(\d{1,3})\b")
# Any bare number.
_BARE_NUM_RE = re.compile(r"\b(\d{1,3})\b")

def decode_text(text: str, profile: Profile = None):
    """
    Decode a transcript against a profile. Returns a list of dicts:
        {"code": "10-42", "type": "10-code", "meaning": "Ending tour of duty"}
        {"code": "signal 22", "type": "signal", "meaning": "Juvenile"}
    """
    if not text:
        return []
    if profile is None:
        profile = DEFAULT_PROFILE

    norm = _words_to_digits(text)
    results = []
    seen = set()
    consumed_spans = []   # char ranges used by 10-codes, so we don't re-read them

    # 1) 10-codes first (anchored on the leading "10").
    for m in _TEN_RE.finditer(norm):
        num = int(m.group(1))
        code = f"10-{num}"
        if code in profile.ten_codes:
            if ("ten", num) not in seen:
                seen.add(("ten", num))
                results.append({"code": code, "type": "10-code",
                                "meaning": profile.ten_codes[code]})
            consumed_spans.append(m.span())

    def in_consumed(pos):
        return any(a <= pos < b for a, b in consumed_spans)

    # 2) Explicit "signal N" / "sig N" — only decode when the word is present.
    for m in _SIGNAL_RE.finditer(norm):
        num = int(m.group(1))
        if num in profile.signals and ("sig", num) not in seen:
            seen.add(("sig", num))
            results.append({"code": f"signal {num}", "type": "signal",
                            "meaning": profile.signals[num]})

    return results


def decode_to_string(text: str, profile: Profile = None) -> str:
    """One-liner: 'signal 9=On the scene; 10-42=Ending tour of duty'."""
    codes = decode_text(text, profile)
    if not codes:
        return ""
    return "; ".join(f"{c['code']}={c['meaning']}" for c in codes)


def decode_for(text: str, channel_name: str):
    """
    Convenience: pick the profile from the channel name, decode, and return
    (profile, codes_list). This is what the logger calls.
    """
    profile = pick_profile(channel_name)
    return profile, decode_text(text, profile)
