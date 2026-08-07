"""
Daily Summary Server Configuration.

Edit these settings to match your setup.
"""

import os

# =============================================================================
# PI DASHBOARD CONNECTION
# =============================================================================
# The Pi scanner dashboard URL (serves the /api/day_transmissions endpoint).
PI_URL = os.environ.get("PI_URL", "http://pi3:8080")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
# Provider: "ollama" (local) or "groq" (free cloud API)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# --- Ollama (local) ---
# Install: https://ollama.ai  then `ollama pull llama3.1:8b`
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# --- Groq (free cloud, fast) ---
# Get free API key: https://console.groq.com/keys
# Free tier: 30 req/min, 14400 req/day, 6000 tokens/min
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# =============================================================================
# OUTPUT
# =============================================================================
# Where to write daily summary files (Markdown reports).
OUTPUT_DIR = os.environ.get("SUMMARY_OUTPUT_DIR", r"\\d1\RadioScanner\summaries")

# Where to write raw transcription logs (plain text, one per day).
TRANSCRIBED_DIR = os.environ.get("TRANSCRIBED_OUTPUT_DIR", r"\\d1\RadioScanner\transcribed")

# =============================================================================
# SUMMARY PARAMETERS
# =============================================================================
# Maximum transmissions per LLM chunk (to stay within context window).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "80"))

# Minimum event gap in seconds — transmissions on the same channel separated by
# more than this are considered separate events.
EVENT_GAP_SEC = int(os.environ.get("EVENT_GAP_SEC", "180"))

# Skip channels with fewer than this many transmissions (reduces noise).
MIN_TRANSMISSIONS_PER_CHANNEL = int(os.environ.get("MIN_TRANSMISSIONS", "2"))
