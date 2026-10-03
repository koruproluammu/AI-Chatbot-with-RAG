"""Embedding model factory (Google Gemini or OpenAI)."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from src.config import (
    API_KEY_ENV,
    EMBEDDING_PROVIDER,
    GOOGLE_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
)
from src.utils import MissingAPIKeyError, RAGChatbotError


def resolve_embedding_provider() -> str:
    """EMBEDDING_PROVIDER if set, else Google when its key exists, else OpenAI."""
    if EMBEDDING_PROVIDER:
        if EMBEDDING_PROVIDER not in ("google", "openai"):
            raise RAGChatbotError("EMBEDDING_PROVIDER must be 'google' or 'openai'.")
        return EMBEDDING_PROVIDER
    return "google" if os.getenv(API_KEY_ENV["google"]) else "openai"


def embedding_model_name(provider: str) -> str:
    return GOOGLE_EMBEDDING_MODEL if provider == "google" else OPENAI_EMBEDDING_MODEL


def get_embeddings() -> Embeddings:
    """Return an embedding model. Keys are read from the environment only."""
    provider = resolve_embedding_provider()
    key_name = API_KEY_ENV[provider]
    if not os.getenv(key_name):
        raise MissingAPIKeyError(f"{key_name} is missing. Add it to your .env file.")

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=f"models/{GOOGLE_EMBEDDING_MODEL}")

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
