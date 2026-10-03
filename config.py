"""Central configuration: every tunable value lives here (and only here)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real env vars still win

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ---- Models (override via environment variables) ----
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")

# First entry is the default in the UI (Gemini has a free tier).
PROVIDERS = {
    "Google Gemini (free tier)": "google",
    "OpenAI": "openai",
    "Anthropic Claude": "anthropic",
}
API_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Embedding provider: "google" or "openai". Empty = auto (Google if its key exists).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()

# Embedding batching (keeps free-tier rate limits happy)
EMBED_BATCH_SIZE = 50
EMBED_MAX_RETRIES = 3

# ---- Chunking / retrieval defaults ----
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4

# Below this cosine similarity the best chunk is considered unrelated and the
# LLM is not even called (saves cost and prevents hallucination).
MIN_SIMILARITY = 0.15

# ---- Files ----
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 25

# ---- Chat memory ----
MAX_HISTORY_MESSAGES = 6      # messages used to rewrite follow-up questions
MAX_HISTORY_CHARS = 500       # per-message truncation
