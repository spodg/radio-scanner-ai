#!/usr/bin/env python3
"""Re-run text decoders on all transcribed records.

Use this after updating codes.py, phonetic.py, or phone.py to refresh
the decoded_text column for all existing records.
"""
import sqlite3
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pi'))

from codes import decode_for
from phonetic import decode_plates
from phone import detect_phones

DB_PATH = os.environ.get("SCANNER_DB", "/home/pi/scanner/scanner.db")


def run_text_decoders(text, channel_name):
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


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, text, channel FROM transmissions
        WHERE transcribed = 1
          AND text != '' AND text != '(no speech)' AND text != '(audio not found)'
    """).fetchall()

    print(f"Re-decoding {len(rows)} records...")

    updated = 0
    for i, row in enumerate(rows):
        decoded = run_text_decoders(row['text'], row['channel'])
        conn.execute("UPDATE transmissions SET decoded_text = ? WHERE id = ?",
                     (json.dumps(decoded), row['id']))
        if decoded:
            updated += 1
        if (i + 1) % 1000 == 0:
            conn.commit()
            print(f"  {i+1}/{len(rows)} processed, {updated} with decoded content...")

    conn.commit()
    print(f"Done! {updated} of {len(rows)} records have decoded content.")


if __name__ == "__main__":
    main()
