"""The RAG pipeline: rewrite follow-up -> retrieve -> build context -> LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import MAX_HISTORY_CHARS, MAX_HISTORY_MESSAGES, MIN_SIMILARITY
from src.retriever import RetrievedChunk, Retriever
from src.utils import LLMError, RAGChatbotError, format_source, similarity_label

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."
NOT_ENOUGH_MESSAGE = "I couldn't find enough information about this in the uploaded documents."

RAG_PROMPT_TEMPLATE = """You are a document-based question answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
'I couldn't find this information in the uploaded documents.'

Do not invent facts.
Do not use unsupported information.
Keep the answer clear and concise.

Context:
{context}

Question:
{question}"""

CONDENSE_PROMPT_TEMPLATE = """Given the conversation below and a follow-up question, rewrite \
the follow-up as a single standalone question that can be understood without the conversation \
(replace pronouns like "it" or "that" with the concept they refer to). If it is already \
standalone, return it unchanged. Return ONLY the question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
CONDENSE_PROMPT = ChatPromptTemplate.from_template(CONDENSE_PROMPT_TEMPLATE)


@dataclass
class RAGResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
    answered_from_documents: bool = False
    top_similarity: float = 0.0

    @property
    def retrieval_label(self) -> str:
        return similarity_label(self.top_similarity)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Join chunks into one context block, each tagged with its source."""
    return "\n\n---\n\n".join(
        f"[{i}] ({format_source(c.document.metadata)})\n{c.document.page_content}"
        for i, c in enumerate(chunks, start=1)
    )


def format_history(history: list[dict]) -> str:
    """Compact the most recent turns so history never swamps the context."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:MAX_HISTORY_CHARS]}")
    return "\n".join(lines)


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        label = format_source(c.document.metadata)
        if label not in seen:
            seen.append(label)
    return seen


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.min_similarity = min_similarity
        self._answer_chain = RAG_PROMPT | llm | StrOutputParser()
        self._condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def _standalone_question(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        try:
            rewritten = self._condense_chain.invoke(
                {"history": format_history(history), "question": question}
            ).strip()
            return rewritten or question
        except Exception:
            logger.exception("Question rewriting failed; using original question")
            return question

    def ask(self, question: str, history: list[dict] | None = None) -> RAGResponse:
        if not question or not question.strip():
            raise RAGChatbotError("Please enter a question.")
        question = question.strip()
        standalone = self._standalone_question(question, history or [])

        chunks = self.retriever.retrieve(standalone)
        top = max((c.similarity for c in chunks), default=0.0)

        # Guard 1: nothing relevant retrieved -> don't even call the LLM.
        if not chunks or top < self.min_similarity:
            return RAGResponse(NOT_ENOUGH_MESSAGE, [], chunks, standalone, False, top)

        try:
            answer = self._answer_chain.invoke(
                {"context": format_context(chunks), "question": standalone}
            ).strip()
        except Exception as exc:
            logger.exception("LLM call failed")
            raise LLMError(
                f"The language model request failed ({type(exc).__name__}). "
                "Check your API key, quota and network connection."
            ) from exc

        # Guard 2: the model itself said the context lacks the answer.
        if answer.lower().startswith("i couldn't find"):
            return RAGResponse(answer, [], chunks, standalone, False, top)

        return RAGResponse(answer, unique_sources(chunks), chunks, standalone, True, top)
