"""
Entity extraction from transmissions and LLM summaries.

Extracts: license plates, names, addresses, phone numbers.
Combines already-decoded entities (from decoded_text field) with
regex-based extraction for names/addresses that aren't currently decoded.
"""

import re
from datetime import datetime


# =============================================================================
# Regex patterns for names and addresses in radio readbacks
# =============================================================================

# Common patterns for name readback:
#   "registered to John Smith"
#   "registered owner John Smith"
#   "RP is Jane Doe"
#   "complainant Jane Doe"
#   "subject John Smith"
#   "suspect is John Smith"
#   "driver John Smith"
NAME_PATTERNS = [
    r"registered\s+(?:to|owner)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:RP|complainant|reporting party)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:subject|suspect|driver|passenger)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:resident|homeowner|victim)\s+(?:is\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
    r"(?:last|first)\s+(?:name|of)\s+([A-Z][a-z]{2,})",
    r"lives?\s+(?:at|with)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})",
]

# Common patterns for address readback:
#   "at 1422 East State Boulevard"
#   "address is 500 West Main Street"
#   "responding to 2200 Maplecrest"
#   "location 1500 block of Coldwater"
ADDRESS_PATTERNS = [
    # "at 1422 East State Boulevard", "address is 500 West Main Street"
    r"(?:at|address\s+(?:is|of)?|location\s*:?|responding\s+to|en\s+route\s+to|"
    r"headed\s+to|going\s+to)\s+"
    r"(\d+\s+(?:block\s+(?:of\s+)?)?(?:(?:North|South|East|West|N|S|E|W)\.?\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
    r"(?:\s+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
    r"Court|Ct|Way|Place|Pl|Circle|Cir|Parkway|Pkwy|Trail|Terrace|Pike)\.?)?)",
    # Standalone address after comma: ", 1422 East State Blvd"
    r",\s+(\d+\s+(?:(?:North|South|East|West|N|S|E|W)\.?\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|"
    r"Court|Ct|Way|Place|Pl|Circle|Cir|Parkway|Pkwy|Trail|Terrace|Pike)\.?)",
    # Intersections: "at Coldwater and Dupont"
    r"(?:at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]

# Words that are NOT names (filter false positives from phonetic alphabet, etc.)
NOT_NAMES = {
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry",
    "Ida", "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul",
    "Queen", "Robert", "Sam", "Tom", "Union", "Victor", "William", "Young",
    "Zebra", "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
    "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar",
    "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Whiskey",
    "Yankee", "Zulu", "Signal", "Copy", "Clear", "Roger", "Dispatch",
    "County", "Allen", "Noble", "Fort", "Wayne", "Indiana",
    "North", "South", "East", "West",
}


def _is_likely_name(name: str) -> bool:
    """Filter out phonetic alphabet words and common false positives."""
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    # If first word is a phonetic alphabet word and second is too, skip
    if parts[0] in NOT_NAMES and (len(parts) < 2 or parts[1] in NOT_NAMES):
        return False
    # Single phonetic words paired with a real-looking last name are OK
    return True


def extract_names_from_text(text: str) -> list[str]:
    """Extract person names from a transcript using regex."""
    names = []
    for pattern in NAME_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1).strip()
            # Title-case it for consistency
            name = " ".join(w.capitalize() for w in name.split())
            if _is_likely_name(name):
                names.append(name)
    return list(set(names))


def extract_addresses_from_text(text: str) -> list[str]:
    """Extract street addresses from a transcript using regex."""
    addresses = []
    for pattern in ADDRESS_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            addr = match.group(1).strip()
            if len(addr) > 8:  # Skip very short matches
                addresses.append(addr)
    return list(set(addresses))


def extract_entities_from_records(records: list[dict]) -> dict:
    """
    Extract all entities from a list of transmission records.
    
    Combines:
      - Already-decoded plates/phones from decoded_text field
      - Regex-extracted names and addresses from transcript text
    
    Returns: {
        "plates": [{"time": ..., "plate": ..., "channel": ..., "context": ...}, ...],
        "names": [{"time": ..., "name": ..., "channel": ..., "context": ...}, ...],
        "addresses": [{"time": ..., "address": ..., "channel": ..., "context": ...}, ...],
        "phones": [{"time": ..., "phone": ..., "channel": ..., "context": ...}, ...],
    }
    """
    plates = []
    names = []
    addresses = []
    phones = []

    for r in records:
        time_str = r.get("time", "")
        if "T" in time_str:
            time_short = time_str.split("T")[1][:5]  # HH:MM
        else:
            time_short = time_str

        channel = r.get("channel", "") or r.get("group", "") or r.get("system", "")
        text = r.get("text", "")
        decoded_text = r.get("decoded_text", {}) or {}

        # Context: first 80 chars of the transcript
        context = text[:80].strip() if text else ""

        # Plates from decoded_text
        for plate in decoded_text.get("plates", []):
            plates.append({
                "time": time_short,
                "plate": plate,
                "channel": channel,
                "context": context,
            })

        # Phones from decoded_text
        for phone in decoded_text.get("phones", []):
            phones.append({
                "time": time_short,
                "phone": phone,
                "channel": channel,
                "context": context,
            })

        # Names from regex
        for name in extract_names_from_text(text):
            names.append({
                "time": time_short,
                "name": name,
                "channel": channel,
                "context": context,
            })

        # Addresses from regex
        for addr in extract_addresses_from_text(text):
            addresses.append({
                "time": time_short,
                "address": addr,
                "channel": channel,
                "context": context,
            })

    return {
        "plates": plates,
        "names": names,
        "addresses": addresses,
        "phones": phones,
    }
