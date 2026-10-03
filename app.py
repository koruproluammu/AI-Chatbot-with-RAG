"""Streamlit front-end for the AI Chatbot with RAG."""

from __future__ import annotations

import logging

import streamlit as st

from src.config import (
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    PROVIDERS,
    SUPPORTED_EXTENSIONS,
)
from src.document_loader import load_document
from src.llm import get_llm, has_api_key
from src.rag_chain import RAGChain, RAGResponse
from src.retriever import Retriever
from src.text_splitter import split_documents
from src.utils import RAGChatbotError, save_upload, setup_logging
from src.vector_store import VectorStoreManager

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Chatbot with RAG", page_icon="🤖", layout="wide")


# ------------------------------------------------------------ session state
def init_state() -> None:
    st.session_state.setdefault("messages", [])
    if "vsm" not in st.session_state:
        st.session_state.vsm = VectorStoreManager()
        try:
            st.session_state.vsm.load()  # reuse saved index, no re-embedding
        except RAGChatbotError as exc:
            st.session_state.startup_error = str(exc)


def render_assistant_extras(meta: dict) -> None:
    """Sources + retrieval indicator under an assistant message."""
    if meta.get("sources"):
        with st.expander("📚 Sources", expanded=True):
            for i, src in enumerate(meta["sources"], start=1):
                st.markdown(f"{i}. {src}")
    if meta.get("label"):
        st.caption(
            f"Retrieval similarity: **{meta['label']}** ({meta['score']:.2f}). "
            "This measures how close the best chunk is to your question — "
            "it is *not* a guarantee the answer is factually correct."
        )
    if meta.get("chunks"):
        with st.expander("🔎 Retrieved passages"):
            for c in meta["chunks"]:
                st.markdown(f"**{c['source']}** — similarity {c['score']:.2f}")
                st.text(c["text"][:600])


def response_to_meta(resp: RAGResponse) -> dict:
    from src.utils import format_source

    return {
        "sources": resp.sources,
        "label": resp.retrieval_label,
        "score": resp.top_similarity,
        "chunks": [
            {
                "source": format_source(c.document.metadata),
                "score": c.similarity,
                "text": c.document.page_content,
            }
            for c in resp.chunks
        ],
    }


# ------------------------------------------------------------------ actions
def process_documents(uploads, chunk_size: int, chunk_overlap: int) -> None:
    vsm: VectorStoreManager = st.session_state.vsm
    already = set(vsm.indexed_sources())
    all_chunks, skipped, failed = [], [], []

    progress = st.sidebar.progress(0.0, text="Reading documents…")
    for i, up in enumerate(uploads, start=1):
        try:
            path = save_upload(up.name, up.getvalue(), DATA_DIR)
            if path.name in already:
                skipped.append(path.name)
            else:
                docs = load_document(path)
                all_chunks.extend(split_documents(docs, chunk_size, chunk_overlap))
        except RAGChatbotError as exc:
            failed.append(str(exc))
        progress.progress(i / len(uploads), text=f"Processed {i}/{len(uploads)}")

    for msg in failed:
        st.sidebar.error(msg)
    if skipped:
        st.sidebar.info(f"Already indexed (skipped): {', '.join(skipped)}")
    if not all_chunks:
        progress.empty()
        return

    try:
        with st.sidebar.status("Creating embeddings…", expanded=False):
            vsm.add_documents(all_chunks)
        st.sidebar.success(f"Indexed {len(all_chunks)} new chunks.")
    except RAGChatbotError as exc:
        st.sidebar.error(str(exc))
    progress.empty()


def build_chain(provider: str, top_k: int) -> RAGChain:
    vsm: VectorStoreManager = st.session_state.vsm
    if vsm.store is None:
        raise RAGChatbotError("Please upload and process documents first.")
    return RAGChain(Retriever(vsm.store, top_k), get_llm(provider))


# ------------------------------------------------------------------ sidebar
def sidebar() -> tuple[str, int, int, int]:
    sb = st.sidebar
    sb.header("⚙️ Configuration")

    uploads = sb.file_uploader(
        "1. Upload documents",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    provider_label = sb.selectbox("2. LLM provider", list(PROVIDERS))
    provider = PROVIDERS[provider_label]
    top_k = sb.slider("3. Retrieved chunks (top_k)", 1, 10, DEFAULT_TOP_K)
    chunk_size = sb.slider("4. Chunk size", 200, 3000, DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = sb.slider("5. Chunk overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=25)
    sb.caption("Chunk settings apply to documents processed from now on.")

    if sb.button("6. Process documents", type="primary", use_container_width=True):
        if not uploads:
            sb.warning("Upload at least one document first.")
        else:
            process_documents(uploads, chunk_size, chunk_overlap)

    if sb.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if sb.button("🗑️ Clear vector store", use_container_width=True):
        st.session_state.vsm.clear()
        st.session_state.messages = []
        sb.success("Vector store cleared.")

    sb.divider()
    vsm: VectorStoreManager = st.session_state.vsm
    sb.markdown("**Status**")
    sb.write(f"{'✅' if has_api_key('google') else '❌'} Google Gemini key")
    sb.write(f"{'✅' if has_api_key('openai') else '❌'} OpenAI key")
    sb.write(f"{'✅' if has_api_key('anthropic') else '❌'} Anthropic key")
    sb.write(f"📄 Documents indexed: {len(vsm.indexed_sources())}")
    sb.write(f"🧩 Chunks in index: {vsm.count()}")
    for name in vsm.indexed_sources():
        sb.caption(f"• {name}")
    return provider, top_k, chunk_size, chunk_overlap


# --------------------------------------------------------------------- main
def main() -> None:
    init_state()
    provider, top_k, _, _ = sidebar()

    st.title("🤖 AI Chatbot with RAG")
    st.caption("Ask questions about your documents using Retrieval Augmented Generation.")
    if err := st.session_state.pop("startup_error", None):
        st.error(err)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_assistant_extras(msg.get("meta", {}))

    question = st.chat_input("Ask a question about your documents…")
    if question is None:
        return
    if not question.strip():
        st.warning("Please type a question.")
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chain = build_chain(provider, top_k)
            with st.spinner("Searching documents and thinking…"):
                resp = chain.ask(question, st.session_state.messages)
            st.markdown(resp.answer)
            meta = response_to_meta(resp)
            render_assistant_extras(meta)
            st.session_state.messages += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": resp.answer, "meta": meta},
            ]
        except RAGChatbotError as exc:
            st.error(str(exc))
        except Exception:  # last-resort guard: never leak internals/keys
            logger.exception("Unexpected error")
            st.error("Something went wrong. Please try again.")


main()
