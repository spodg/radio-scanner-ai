"""
LLM integration for daily summary generation.

Supports:
  - Ollama (local, free, private) — requires `ollama pull llama3.1:8b`
  - Groq (cloud, free tier, fast) — requires API key from console.groq.com
"""

import json
import time
import requests

import config


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call local Ollama instance."""
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 4096,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {config.OLLAMA_URL}. "
            f"Is Ollama running? Start it with: ollama serve"
        )
    except requests.HTTPError as e:
        if "model" in str(e).lower() or resp.status_code == 404:
            raise RuntimeError(
                f"Model '{config.OLLAMA_MODEL}' not found. "
                f"Pull it with: ollama pull {config.OLLAMA_MODEL}"
            )
        raise


def _call_groq(prompt: str, system_prompt: str = "") -> str:
    """Call Groq free API."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and set it in config.py or as an environment variable."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code == 429:
            # Rate limited — wait and retry once
            wait = int(resp.headers.get("retry-after", 10))
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.ConnectionError:
        raise RuntimeError("Cannot reach Groq API. Check internet connection.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq API error: {e} — {resp.text[:200]}")


def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Call the configured LLM provider."""
    provider = config.LLM_PROVIDER.lower()
    if provider == "ollama":
        return _call_ollama(prompt, system_prompt)
    elif provider == "groq":
        return _call_groq(prompt, system_prompt)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}. Use 'ollama' or 'groq'.")


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an analyst producing a detailed event log from police/fire/EMS radio scanner transmissions.

You receive a batch of transcribed radio transmissions from one channel. Signal codes are decoded for you in brackets — always use the decoded plain-English meaning, never leave raw signal numbers.

Your job: identify each distinct EVENT (incident, call for service, traffic stop, medical run, etc.) and write a detailed narrative entry for it.

For each event, include ALL available details:
- What happened (use decoded signal meaning, not the code number)
- Location (address, intersection, landmark)
- Involved parties (names, descriptions, unit numbers)
- Vehicles (color, make, model, plate)
- Outcome/disposition (arrest, transport, cleared, etc.)
- Any other specifics mentioned (DOB, warrants, weapons, injuries)

Rules:
- Translate ALL signal/10-codes to their meaning. Example: "Signal 30" with [Signal 30=Traffic stop] → write "Traffic stop", NOT "Signal 30".
- Group related transmissions into one event entry.
- Use 24-hour time format (HH:MM).
- Routine status changes (10-8 in service, 10-42 end of shift) with no incident — skip them entirely or combine into one "Routine" line.
- Write in short, factual narrative style. No filler words.
- Output ONLY the event list, nothing else."""


def build_channel_prompt(channel_key: str, transmissions: list[dict]) -> str:
    """Build the LLM prompt for a batch of transmissions on one channel."""
    lines = []
    lines.append(f"Channel: {channel_key}")
    lines.append(f"Transmissions ({len(transmissions)} total):")
    lines.append("---")

    for t in transmissions:
        time_str = t.get("time", "")
        if "T" in time_str:
            time_str = time_str.split("T")[1][:8]  # HH:MM:SS

        text = t.get("text", "")
        decoded_text = t.get("decoded_text", {}) or {}

        # Annotate decoded info inline so LLM sees plain meanings
        annotations = []
        if decoded_text.get("codes"):
            for c in decoded_text["codes"]:
                annotations.append(f"[{c['code']}={c['meaning']}]")
        if decoded_text.get("plates"):
            annotations.append(f"[Plate: {', '.join(decoded_text['plates'])}]")
        if decoded_text.get("phones"):
            annotations.append(f"[Phone: {', '.join(decoded_text['phones'])}]")

        annotation_str = " " + " ".join(annotations) if annotations else ""
        lines.append(f"[{time_str}] {text}{annotation_str}")

    lines.append("---")
    lines.append("")
    lines.append("Write each event as:")
    lines.append("")
    lines.append("**HH:MM–HH:MM | <Event Type>**")
    lines.append("<Location if known>")
    lines.append("<Detailed narrative: what happened, who was involved, vehicles/plates, outcome>")
    lines.append("")
    lines.append("If an event is a single transmission, use just the one time: **HH:MM | <Type>**")
    lines.append("")
    lines.append("Skip pure status updates (in-service, end-of-shift) unless they contain incident details.")
    lines.append("Always decode signal codes to plain English — never output raw code numbers.")

    return "\n".join(lines)
