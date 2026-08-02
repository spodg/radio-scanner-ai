"""
SQLite database layer for the Pi Scanner.

Single source of truth for all transmission records. Replaces the JSONL file.
DB lives locally on the Pi at /home/pi/scanner/scanner.db for fast atomic
access. Dashboard reads directly, GPU server accesses via HTTP API.

Uses WAL mode for concurrent readers + single writer without blocking.
"""

import os
import json
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = "/home/pi/scanner/scanner.db"

_local = threading.local()


def _get_conn():
    """Get a thread-local connection (reused within the same thread)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, timeout=30)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=10000")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


@contextmanager
def get_db():
    """Context manager for DB access."""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Create tables and indexes if they don't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transmissions (
                id TEXT PRIMARY KEY,
                time TEXT NOT NULL,
                frequency TEXT DEFAULT '',
                name TEXT DEFAULT '',
                system TEXT DEFAULT '',
                grp TEXT DEFAULT '',
                channel TEXT DEFAULT '',
                duration_sec REAL DEFAULT 0,
                text TEXT DEFAULT '',
                transcribed INTEGER DEFAULT 0,
                transcribed_by TEXT DEFAULT '',
                clip TEXT DEFAULT '',
                source TEXT DEFAULT '',
                decoded TEXT DEFAULT '{}',
                decoded_text TEXT DEFAULT '{}',
                tags TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        # Indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time ON transmissions(time DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcribed
            ON transmissions(transcribed) WHERE transcribed = 0
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcribed_by
            ON transmissions(transcribed_by) WHERE transcribed = 1 AND transcribed_by != 'gpu'
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_system ON transmissions(system)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_grp ON transmissions(grp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel ON transmissions(channel)
        """)


def insert_transmission(record: dict):
    """Insert a new transmission record (placeholder or complete)."""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO transmissions
            (id, time, frequency, name, system, grp, channel, duration_sec,
             text, transcribed, transcribed_by, clip, source, decoded, decoded_text, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("id", ""),
            record.get("time", ""),
            record.get("frequency", ""),
            record.get("name", ""),
            record.get("system", ""),
            record.get("group", record.get("grp", "")),
            record.get("channel", ""),
            record.get("duration_sec", 0),
            record.get("text", ""),
            1 if record.get("transcribed") else 0,
            record.get("transcribed_by", ""),
            record.get("clip", ""),
            record.get("source", ""),
            json.dumps(record.get("decoded", {})),
            json.dumps(record.get("decoded_text", {})),
            json.dumps(record.get("tags", {})),
        ))


def update_transmission(record_id: str, updates: dict):
    """Update specific fields of a transmission by ID."""
    if not updates:
        return False
    # Build SET clause dynamically
    field_map = {
        "text": "text",
        "transcribed": "transcribed",
        "transcribed_by": "transcribed_by",
        "clip": "clip",
        "decoded": "decoded",
        "decoded_text": "decoded_text",
    }
    sets = []
    values = []
    for key, col in field_map.items():
        if key in updates:
            val = updates[key]
            if key == "transcribed":
                val = 1 if val else 0
            elif key in ("decoded", "decoded_text"):
                val = json.dumps(val) if isinstance(val, dict) else val
            sets.append(f"{col} = ?")
            values.append(val)

    if not sets:
        return False

    values.append(record_id)
    with get_db() as conn:
        cursor = conn.execute(
            f"UPDATE transmissions SET {', '.join(sets)} WHERE id = ?",
            values
        )
        return cursor.rowcount > 0


def get_untranscribed(limit=10):
    """Get records awaiting transcription."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM transmissions
            WHERE transcribed = 0
            ORDER BY time ASC, id ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_pi_transcribed(limit=100):
    """Get records transcribed by Pi but not GPU (for re-transcription)."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM transmissions
            WHERE transcribed = 1
              AND transcribed_by != 'gpu'
              AND text != ''
              AND text != '(no speech)'
              AND text != '(audio not found)'
              AND clip != ''
            ORDER BY time ASC, id ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_transmissions(hours=24, system="", group="", channel="", freq="",
                      query="", hide_blank=False, limit=500, offset=0):
    """Get filtered transmissions for the dashboard."""
    conditions = []
    params = []

    if hours < 9999:
        conditions.append("time >= datetime('now', 'localtime', ?)")
        params.append(f"-{hours} hours")

    if system:
        conditions.append("system = ?")
        params.append(system)
    if group:
        conditions.append("grp = ?")
        params.append(group)
    if channel:
        conditions.append("channel = ?")
        params.append(channel)
    if freq:
        conditions.append("frequency = ?")
        params.append(freq)
    if hide_blank:
        conditions.append("text NOT IN ('', '(no speech)', '[BLANK_AUDIO]')")
    if query:
        conditions.append("(text LIKE ? OR name LIKE ? OR channel LIKE ?)")
        q = f"%{query}%"
        params.extend([q, q, q])

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_db() as conn:
        rows = conn.execute(f"""
            SELECT * FROM transmissions
            {where}
            ORDER BY time DESC, id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()
        
        # Deduplicate records with identical timestamp (same transmission saved as MP3+WAV
        # or transcribed separately by Pi and GPU with slightly different text)
        seen = set()
        deduplicated = []
        for row in rows:
            key = row["time"]
            if key not in seen:
                seen.add(key)
                deduplicated.append(row)
        
        return [_row_to_dict(r) for r in deduplicated]


def get_total_count():
    """Get total number of records."""
    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) FROM transmissions").fetchone()[0]


def get_filtered_count(hours=24, system="", group="", channel="", freq="",
                       query="", hide_blank=False):
    """Get count of records matching current filters."""
    conditions = []
    params = []

    if hours < 9999:
        conditions.append("time >= datetime('now', 'localtime', ?)")
        params.append(f"-{hours} hours")
    if system:
        conditions.append("system = ?")
        params.append(system)
    if group:
        conditions.append("grp = ?")
        params.append(group)
    if channel:
        conditions.append("channel = ?")
        params.append(channel)
    if freq:
        conditions.append("frequency = ?")
        params.append(freq)
    if hide_blank:
        conditions.append("text NOT IN ('', '(no speech)', '[BLANK_AUDIO]')")
    if query:
        conditions.append("(text LIKE ? OR name LIKE ? OR channel LIKE ?)")
        q = f"%{query}%"
        params.extend([q, q, q])

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_db() as conn:
        return conn.execute(
            f"SELECT COUNT(*) FROM transmissions {where}", params
        ).fetchone()[0]


def get_filter_options(hours=24, system="", group="", channel="", freq=""):
    """Get valid dropdown options given current selections (cascading).
    
    Hierarchy: System > Group > Channel > Freq
    Each dropdown only shows options valid for upstream selections.
    Returns options with counts: [{"value": "X", "count": N}, ...]
    """
    base_conditions = []
    params = []

    if hours < 9999:
        base_conditions.append("time >= datetime('now', 'localtime', ?)")
        params.append(f"-{hours} hours")

    def query_values(field, upstream_filters):
        """Query distinct values with counts, filtered by upstream selections."""
        conds = list(base_conditions)
        p = list(params)
        for col, val in upstream_filters:
            if val:
                conds.append(f"{col} = ?")
                p.append(val)
        where = "WHERE " + " AND ".join(conds) if conds else ""
        with get_db() as conn:
            rows = conn.execute(
                f"SELECT {field}, COUNT(*) as cnt FROM transmissions {where} GROUP BY {field} ORDER BY {field}",
                p
            ).fetchall()
            return [{"value": r[0], "count": r[1]} for r in rows if r[0]]

    return {
        # Systems: no upstream filter
        "systems": query_values("system", []),
        # Groups: filtered by system
        "groups": query_values("grp", [("system", system)]),
        # Channels: filtered by system + group
        "channels": query_values("channel", [("system", system), ("grp", group)]),
        # Freqs: filtered by system + group + channel
        "freqs": query_values("frequency", [("system", system), ("grp", group), ("channel", channel)]),
    }


def get_pending_count():
    """Count of untranscribed records."""
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM transmissions WHERE transcribed = 0"
        ).fetchone()[0]


def _row_to_dict(row):
    """Convert a sqlite3.Row to a plain dict with JSON fields parsed."""
    d = dict(row)
    # Parse JSON fields
    for field in ("decoded", "decoded_text", "tags"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = {}
    # Convert transcribed from int to bool
    d["transcribed"] = bool(d.get("transcribed"))
    # Rename 'grp' back to 'group' for API compatibility
    d["group"] = d.pop("grp", "")
    return d


def migrate_from_jsonl(jsonl_path):
    """Import records from a JSONL file into the SQLite DB."""
    if not os.path.exists(jsonl_path):
        print(f"JSONL file not found: {jsonl_path}")
        return 0
    count = 0
    with get_db() as conn:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("id") or not r.get("time"):
                    continue
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO transmissions
                        (id, time, frequency, name, system, grp, channel,
                         duration_sec, text, transcribed, transcribed_by,
                         clip, source, decoded, decoded_text, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        r.get("id", ""),
                        r.get("time", ""),
                        r.get("frequency", ""),
                        r.get("name", ""),
                        r.get("system", ""),
                        r.get("group", ""),
                        r.get("channel", ""),
                        r.get("duration_sec", 0),
                        r.get("text", ""),
                        1 if r.get("transcribed") else 0,
                        r.get("transcribed_by", ""),
                        r.get("clip", ""),
                        r.get("source", ""),
                        json.dumps(r.get("decoded", {})),
                        json.dumps(r.get("decoded_text", {})),
                        json.dumps(r.get("tags", {})),
                    ))
                    count += 1
                except sqlite3.IntegrityError:
                    pass
    print(f"Migrated {count} records from JSONL to SQLite")
    return count


if __name__ == "__main__":
    """Run standalone to initialize DB and optionally migrate JSONL."""
    import sys
    init_db()
    print(f"Database initialized at {DB_PATH}")
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
        migrate_from_jsonl(jsonl_path)
    else:
        print("Usage: python scanner_db.py [path_to_jsonl] to migrate data")
