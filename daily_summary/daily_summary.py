"""
Scanner Daily Summary Server.

Fetches all transcribed transmissions for a given day from the Pi scanner
dashboard API, groups them by channel, generates event summaries using a
local LLM (Ollama) or free cloud API (Groq), extracts entities (plates,
names, addresses), and writes a Markdown summary report.

Usage:
    python daily_summary.py                   # Summarize yesterday
    python daily_summary.py today             # Summarize today (so far)
    python daily_summary.py 2026-07-30        # Summarize a specific date
    python daily_summary.py --last 3          # Summarize last 3 days

Requirements:
    pip install requests

LLM setup (pick one):
    Option A - Ollama (local, free, private):
        1. Install Ollama: https://ollama.ai
        2. Pull model: ollama pull llama3.1:8b
        3. Set LLM_PROVIDER=ollama in config.py (default)

    Option B - Groq (cloud, free tier, fast):
        1. Get API key: https://console.groq.com/keys
        2. Set GROQ_API_KEY in config.py or environment
        3. Set LLM_PROVIDER=groq in config.py

Pi setup:
    The Pi dashboard must have the /api/day_transmissions endpoint.
    Update radio-scanner-ai/pi/dashboard.py with the new endpoint and restart.
"""

import sys
import time
from datetime import date, timedelta

import config
from fetch import fetch_day
from summarize import group_by_channel, generate_channel_summaries
from report import format_report, write_report, write_transcription_log


def run_summary(target_date: date):
    """Run the full summary pipeline for a single date."""
    print(f"\n{'='*60}")
    print(f"  Scanner Daily Summary — {target_date}")
    print(f"  LLM: {config.LLM_PROVIDER} ({getattr(config, f'{config.LLM_PROVIDER.upper()}_MODEL', '?')})")
    print(f"{'='*60}\n")

    # 1. Fetch transmissions
    print("[1/4] Fetching transmissions from Pi...")
    records = fetch_day(target_date)
    if not records:
        print("  No transcribed transmissions found for this date.")
        print("  (Are all transmissions still queued for transcription?)")
        return

    # 2. Group by channel
    print("[2/4] Grouping by channel...")
    grouped = group_by_channel(records)
    print(f"  {len(grouped)} active channels (min {config.MIN_TRANSMISSIONS_PER_CHANNEL} transmissions each)")
    for ch, txs in sorted(grouped.items(), key=lambda x: -len(x[1])):
        print(f"    {ch}: {len(txs)} transmissions")

    # 3. Summarize each channel via LLM
    print(f"\n[3/4] Generating summaries via {config.LLM_PROVIDER}...")
    start_time = time.time()

    def progress(idx, total, channel, count):
        elapsed = time.time() - start_time
        print(f"  [{idx}/{total}] {channel} ({count} tx)...", end="", flush=True)

    channel_summaries = generate_channel_summaries(grouped, progress_callback=None)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # 4. Generate report and transcription log
    print("[4/4] Writing report and transcription log...")
    report_content = format_report(target_date, grouped, channel_summaries, records)
    filepath = write_report(target_date, report_content)
    print(f"  Summary:      {filepath}")

    log_path = write_transcription_log(target_date, records)
    print(f"  Transcribed:  {log_path}")

    print(f"\n  Done! {len(records)} transmissions -> {len(grouped)} channels")


def main():
    """Parse CLI arguments and run."""
    args = sys.argv[1:]

    if not args or args[0] == "yesterday":
        # Default: summarize yesterday (full day)
        dates = [date.today() - timedelta(days=1)]
    elif args[0] == "today":
        dates = [date.today()]
    elif args[0] == "--last":
        # Summarize last N days
        n = int(args[1]) if len(args) > 1 else 3
        dates = [date.today() - timedelta(days=i) for i in range(n, 0, -1)]
    elif args[0] == "--help" or args[0] == "-h":
        print(__doc__)
        return
    else:
        # Assume YYYY-MM-DD
        try:
            dates = [date.fromisoformat(args[0])]
        except ValueError:
            print(f"ERROR: Invalid date format '{args[0]}'. Use YYYY-MM-DD.")
            print("Usage: python daily_summary.py [yesterday|today|YYYY-MM-DD|--last N]")
            sys.exit(1)

    for target_date in dates:
        run_summary(target_date)


if __name__ == "__main__":
    main()
