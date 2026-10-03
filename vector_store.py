"""FAISS vector store: build, persist, load, extend, clear and rebuild."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import (
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    SUPPORTED_EXTENSIONS,
    VECTORSTORE_DIR,
)
from src.document_loader import load_document
from src.embeddings import embedding_model_name, get_embeddings, resolve_embedding_provider
from src.text_splitter import split_documents
from src.utils import (
    EmbeddingError,
    EmptyDocumentError,
    MissingAPIKeyError,
    RAGChatbotError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)
_INDEX_NAME = "index"
_META_FILE = "embedding_meta.json"


class VectorStoreManager:
    """Owns the FAISS index so embeddings are computed only when documents change."""

    def __init__(
        self, index_dir: Path = VECTORSTORE_DIR, embeddings: Embeddings | None = None
    ) -> None:
        self.index_dir = Path(index_dir)
        self._embeddings = embeddings
        self._injected = embeddings is not None  # tests pass fake embeddings
        self.store: FAISS | None = None

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    # -- embedding-model bookkeeping: vectors from different models can't be mixed
    def _current_signature(self) -> dict | None:
        if self._injected:
            return None  # injected embeddings (tests): skip the check
        provider = resolve_embedding_provider()
        return {"provider": provider, "model": embedding_model_name(provider)}

    def _write_meta(self) -> None:
        sig = self._current_signature()
        if sig:
            (self.index_dir / _META_FILE).write_text(json.dumps(sig))

    def _check_meta(self) -> None:
        sig, path = self._current_signature(), self.index_dir / _META_FILE
        if sig and path.exists() and json.loads(path.read_text()) != sig:
            raise VectorStoreError(
                "This index was built with a different embedding model. "
                "Click 'Clear vector store' and process your documents again."
            )

    def exists(self) -> bool:
        return (self.index_dir / f"{_INDEX_NAME}.faiss").exists() and (
            self.index_dir / f"{_INDEX_NAME}.pkl"
        ).exists()

    def load(self) -> bool:
        """Load a saved index. Returns False if none exists."""
        if not self.exists():
            return False
        self._check_meta()
        try:
            # The pickle is our own file written by save(), so this is safe.
            self.store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                index_name=_INDEX_NAME,
                allow_dangerous_deserialization=True,
                normalize_L2=True,
            )
            return True
        except MissingAPIKeyError:
            raise
        except Exception as exc:
            logger.exception("Failed to load vector store")
            raise VectorStoreError(
                "The saved vector store could not be loaded. Clear it and re-process."
            ) from exc

    def save(self) -> None:
        if self.store is None:
            raise VectorStoreError("Nothing to save: the vector store is empty.")
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.index_dir), index_name=_INDEX_NAME)
            self._write_meta()
        except Exception as exc:
            logger.exception("Failed to save vector store")
            raise VectorStoreError("Could not save the vector store to disk.") from exc

    def _embed_batch(self, batch: list[Document]) -> None:
        """Embed one batch into the store, retrying on transient/rate-limit errors."""
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                if self.store is None:
                    # normalize_L2 makes distances -> cosine similarity for any model
                    self.store = FAISS.from_documents(
                        batch, self.embeddings, normalize_L2=True
                    )
                else:
                    self.store.add_documents(batch)
                return
            except MissingAPIKeyError:
                raise
            except Exception as exc:
                if attempt == EMBED_MAX_RETRIES:
                    logger.exception("Embedding failed")
                    raise EmbeddingError(
                        f"Embedding failed ({type(exc).__name__}). Check your API key, "
                        "free-tier quota (wait a minute and retry) and network."
                    ) from exc
                time.sleep(5 * attempt)

    def _ingest(self, chunks: list[Document]) -> None:
        if not chunks:
            raise EmptyDocumentError("There are no text chunks to index.")
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            self._embed_batch(chunks[i : i + EMBED_BATCH_SIZE])
        self.save()

    def build(self, chunks: list[Document]) -> None:
        """Create a brand-new index from chunks (calls the embedding API)."""
        self.store = None
        self._ingest(chunks)

    def add_documents(self, chunks: list[Document]) -> None:
        """Embed new chunks and append them to the existing index."""
        self._ingest(chunks)

    def indexed_sources(self) -> list[str]:
        """Filenames currently present in the index."""
        if self.store is None:
            return []
        names = {d.metadata.get("source") for d in self.store.docstore._dict.values()}
        return sorted(n for n in names if n)

    def count(self) -> int:
        return 0 if self.store is None else self.store.index.ntotal

    def clear(self) -> None:
        """Delete the index from memory and disk (uploaded files are kept)."""
        self.store = None
        for name in (f"{_INDEX_NAME}.faiss", f"{_INDEX_NAME}.pkl", _META_FILE):
            (self.index_dir / name).unlink(missing_ok=True)

    def rebuild_from_data(
        self,
        data_dir: Path = DATA_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Re-chunk and re-embed every stored file (e.g. after changing chunk size)."""
        files = [
            p for p in sorted(Path(data_dir).iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        chunks: list[Document] = []
        for path in files:
            chunks.extend(split_documents(load_document(path), chunk_size, chunk_overlap))
        self.clear()
        self.build(chunks)
        return len(chunks)
