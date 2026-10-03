"""Similarity search over the FAISS index with normalised similarity scores."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import DEFAULT_TOP_K
from src.utils import RAGChatbotError, VectorStoreError


@dataclass
class RetrievedChunk:
    document: Document
    similarity: float  # approx. cosine similarity in [0, 1]


def distance_to_similarity(l2_distance: float) -> float:
    """Convert FAISS L2 distance to cosine similarity.

    OpenAI embeddings are unit-length, so cos = 1 - d^2 / 2.
    """
    return max(0.0, min(1.0, 1.0 - (l2_distance ** 2) / 2.0))


class Retriever:
    def __init__(self, store: FAISS, top_k: int = DEFAULT_TOP_K) -> None:
        if top_k < 1:
            raise RAGChatbotError("top_k must be at least 1.")
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise RAGChatbotError("Please enter a question.")
        try:
            results = self.store.similarity_search_with_score(query, k=self.top_k)
        except Exception as exc:
            raise VectorStoreError(
                f"Document search failed ({type(exc).__name__})."
            ) from exc
        return [RetrievedChunk(doc, distance_to_similarity(float(s))) for doc, s in results]
