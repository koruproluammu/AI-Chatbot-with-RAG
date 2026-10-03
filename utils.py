"""Shared helpers: custom exceptions, logging, filenames and formatting."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------- exceptions
class RAGChatbotError(Exception):
    """Base class. Messages are always safe to show to the user."""


class MissingAPIKeyError(RAGChatbotError):
    pass


class UnsupportedFileError(RAGChatbotError):
    pass


class EmptyDocumentError(RAGChatbotError):
    pass


class DocumentLoadError(RAGChatbotError):
    pass


class VectorStoreError(RAGChatbotError):
    pass


class EmbeddingError(RAGChatbotError):
    pass


class LLMError(RAGChatbotError):
    pass


# ------------------------------------------------------------------- helpers
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def sanitize_filename(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded name."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return base or "document"


def save_upload(filename: str, data: bytes, data_dir: Path) -> Path:
    """Store raw uploaded bytes (never executed) so the index can be rebuilt."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / sanitize_filename(filename)
    path.write_bytes(data)
    return path


def format_source(metadata: Mapping[str, Any]) -> str:
    """'file.pdf — Page 5', or just 'file.txt' when no page exists."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    return f"{source} — Page {page}" if page is not None else str(source)


def similarity_label(score: float) -> str:
    """Human-friendly bucket for a cosine similarity score."""
    if score >= 0.5:
        return "High"
    if score >= 0.3:
        return "Medium"
    return "Low"
