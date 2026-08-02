#!/usr/bin/env python3
"""Migrate data from old JSONL format to SQLite database.

Usage: python migrate_jsonl.py /path/to/scanner_log.jsonl
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pi'))
import scanner_db


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_jsonl.py /path/to/scanner_log.jsonl")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    scanner_db.init_db()
    count = scanner_db.migrate_from_jsonl(jsonl_path)
    print(f"Migration complete: {count} records imported.")


if __name__ == "__main__":
    main()
