"""Split documents into overlapping chunks suitable for embedding."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.utils import RAGChatbotError


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Chunk documents; metadata (source/page/type) is inherited by each chunk."""
    if chunk_size <= 0:
        raise RAGChatbotError("Chunk size must be greater than 0.")
    if not 0 <= chunk_overlap < chunk_size:
        raise RAGChatbotError("Chunk overlap must be >= 0 and smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
