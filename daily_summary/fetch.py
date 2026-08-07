"""
Fetch a full day of transcribed transmissions from the Pi dashboard API.
"""

import requests
import socket
from datetime import date, timedelta

import config


# Cache resolved IP to avoid repeated DNS lookups (Pi hostname can be flaky).
_resolved_url = None


def _resolve_pi_url():
    """Resolve Pi hostname once and cache it."""
    global _resolved_url
    if _resolved_url:
        return _resolved_url

    url = config.PI_URL
    # Try to resolve hostname to IP for reliability
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or 80
        ip = socket.gethostbyname(hostname)
        _resolved_url = f"http://{ip}:{port}"
        print(f"  Resolved {hostname} -> {ip}")
    except Exception:
        _resolved_url = url

    return _resolved_url


def fetch_day(target_date: date) -> list[dict]:
    """
    Fetch all transcribed transmissions for a specific date from the Pi.

    Returns a list of transmission dicts sorted by time ascending.
    Each dict has: id, time, frequency, name, system, group, channel,
                   duration_sec, text, decoded, decoded_text, tags, etc.
    """
    base_url = _resolve_pi_url()
    endpoint = f"{base_url}/api/day_transmissions"

    params = {
        "date": target_date.isoformat(),
        "hide_blank": "1",
    }

    print(f"  Fetching transmissions for {target_date} from {endpoint}...")

    try:
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError as e:
        print(f"  ERROR: Cannot connect to Pi at {base_url}: {e}")
        raise SystemExit(1)
    except requests.HTTPError as e:
        print(f"  ERROR: Pi returned error: {e}")
        raise SystemExit(1)

    data = resp.json()
    records = data.get("records", [])
    print(f"  Got {len(records)} transcribed transmissions for {target_date}")

    return records
