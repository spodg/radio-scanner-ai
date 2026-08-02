#!/usr/bin/env python3
"""Database statistics and health check."""
import sqlite3
import os
import sys

DB_PATH = os.environ.get("SCANNER_DB", "/home/pi/scanner/scanner.db")

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    print("Set SCANNER_DB environment variable or run from ~/scanner/")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)

total = conn.execute("SELECT COUNT(*) FROM transmissions").fetchone()[0]
transcribed = conn.execute("SELECT COUNT(*) FROM transmissions WHERE transcribed = 1").fetchone()[0]
pending = conn.execute("SELECT COUNT(*) FROM transmissions WHERE transcribed = 0").fetchone()[0]
by_gpu = conn.execute("SELECT COUNT(*) FROM transmissions WHERE transcribed_by = 'gpu'").fetchone()[0]

db_size = os.path.getsize(DB_PATH)

print(f"Database: {DB_PATH}")
print(f"Size: {db_size / 1024 / 1024:.1f} MB")
print(f"Total records: {total:,}")
print(f"Transcribed: {transcribed:,} ({by_gpu:,} by GPU)")
print(f"Pending: {pending}")
print()

# Records by day (last 7)
rows = conn.execute("""
    SELECT substr(time, 1, 10) as day, COUNT(*) as cnt
    FROM transmissions GROUP BY day ORDER BY day DESC LIMIT 7
""").fetchall()
print("Last 7 days:")
for day, cnt in reversed(rows):
    print(f"  {day}: {cnt:,}")

# Top systems
print("\nTop systems:")
rows = conn.execute("""
    SELECT system, COUNT(*) as cnt FROM transmissions
    WHERE system != '' GROUP BY system ORDER BY cnt DESC LIMIT 5
""").fetchall()
for sys_name, cnt in rows:
    print(f"  {sys_name}: {cnt:,}")
