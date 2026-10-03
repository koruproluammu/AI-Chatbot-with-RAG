"""Chat-model factory supporting Google Gemini, OpenAI and Anthropic Claude."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import (
    ANTHROPIC_CHAT_MODEL,
    API_KEY_ENV,
    GOOGLE_CHAT_MODEL,
    OPENAI_CHAT_MODEL,
)
from src.utils import MissingAPIKeyError, RAGChatbotError


def has_api_key(provider: str) -> bool:
    return bool(os.getenv(API_KEY_ENV.get(provider, "")))


def get_llm(provider: str = "google", temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model. Temperature 0 keeps answers grounded and repeatable."""
    provider = provider.lower()
    if provider not in API_KEY_ENV:
        raise RAGChatbotError(f"Unknown LLM provider: {provider}")
    if not has_api_key(provider):
        raise MissingAPIKeyError(
            f"{API_KEY_ENV[provider]} is missing. Add it to your .env file."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=ANTHROPIC_CHAT_MODEL, temperature=temperature, max_tokens=1024)
