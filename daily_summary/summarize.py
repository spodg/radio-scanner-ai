"""
Core summarization logic.

Groups transmissions by channel, clusters into events by time proximity,
sends chunks to LLM for narrative summarization, and assembles the final output.
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict

import config
from llm import call_llm, build_channel_prompt, SYSTEM_PROMPT


def group_by_channel(records: list[dict]) -> dict[str, list[dict]]:
    """
    Group records by (system, group, channel) into a dict keyed by display name.
    Filters out channels with too few transmissions.
    """
    grouped = defaultdict(list)
    for r in records:
        system = r.get("system", "").strip()
        group = r.get("group", "").strip()
        channel = r.get("channel", "").strip()

        # Build a readable key
        parts = [p for p in [system, group, channel] if p]
        key = " > ".join(parts) if parts else "(Unknown)"
        grouped[key].append(r)

    # Filter out low-activity channels
    min_tx = config.MIN_TRANSMISSIONS_PER_CHANNEL
    return {k: v for k, v in grouped.items() if len(v) >= min_tx}


def cluster_events(records: list[dict], gap_sec: int = None) -> list[list[dict]]:
    """
    Cluster consecutive transmissions into events based on time gaps.
    Records on the same channel within gap_sec seconds are part of the same event.
    """
    if gap_sec is None:
        gap_sec = config.EVENT_GAP_SEC

    if not records:
        return []

    events = []
    current_event = [records[0]]

    for i in range(1, len(records)):
        prev_time = _parse_time(records[i - 1].get("time", ""))
        curr_time = _parse_time(records[i].get("time", ""))

        if prev_time and curr_time:
            gap = (curr_time - prev_time).total_seconds()
            if gap > gap_sec:
                events.append(current_event)
                current_event = []
        current_event.append(records[i])

    if current_event:
        events.append(current_event)

    return events


def _parse_time(time_str: str):
    """Parse ISO time string."""
    try:
        return datetime.fromisoformat(time_str)
    except (ValueError, TypeError):
        return None


def summarize_channel(channel_key: str, records: list[dict]) -> str:
    """
    Summarize all transmissions for one channel using LLM.
    
    For large channels, splits into chunks and summarizes each.
    Returns the raw LLM summary text.
    """
    chunk_size = config.CHUNK_SIZE
    
    if len(records) <= chunk_size:
        # Single chunk — send all at once
        prompt = build_channel_prompt(channel_key, records)
        return call_llm(prompt, SYSTEM_PROMPT)
    
    # Multiple chunks needed
    summaries = []
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        chunk_label = f"{channel_key} (part {i // chunk_size + 1})"
        prompt = build_channel_prompt(chunk_label, chunk)
        summary = call_llm(prompt, SYSTEM_PROMPT)
        summaries.append(summary)
        # Small delay between chunks to avoid rate limits
        time.sleep(1)

    return "\n\n".join(summaries)


def generate_channel_summaries(grouped: dict[str, list[dict]], 
                                progress_callback=None) -> dict[str, str]:
    """
    Generate LLM summaries for all channels.
    
    Returns: {channel_key: summary_text, ...}
    """
    summaries = {}
    total = len(grouped)

    for idx, (channel_key, records) in enumerate(sorted(grouped.items()), 1):
        if progress_callback:
            progress_callback(idx, total, channel_key, len(records))
        else:
            print(f"  [{idx}/{total}] {channel_key} ({len(records)} transmissions)...")

        try:
            summary = summarize_channel(channel_key, records)
            summaries[channel_key] = summary
        except Exception as e:
            print(f"    ERROR summarizing {channel_key}: {e}")
            summaries[channel_key] = f"(Summary failed: {e})"

        # Rate limit spacing
        if config.LLM_PROVIDER.lower() == "groq":
            time.sleep(2)  # Groq free tier: ~30 req/min

    return summaries
