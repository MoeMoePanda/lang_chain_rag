"""LCEL conversational RAG chain."""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from hdb_rag.prompts import ANSWER_PROMPT, REWRITE_PROMPT


class ChainInput(TypedDict):
    question: str
    chat_history: list[BaseMessage]


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "")
        url = d.metadata.get("source_url", "")
        parts.append(f"[{i}] {title} ({url})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _with_retrieval_ranks(docs: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "retrieval_rank": i},
        )
        for i, doc in enumerate(docs, start=1)
    ]


def build_chain(
    *,
    retriever: BaseRetriever,
    answer_llm: BaseChatModel,
    fast_llm: BaseChatModel,
):
    """Returns a runnable that takes ChainInput and yields {answer, context}."""
    rewrite = REWRITE_PROMPT | fast_llm | StrOutputParser()

    standalone = RunnableBranch(
        (lambda x: not x["chat_history"], lambda x: x["question"]),
        rewrite,
    )

    return (
        RunnablePassthrough.assign(standalone_question=standalone)
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: _with_retrieval_ranks(retriever.invoke(x["standalone_question"]))
            ),
        )
        | RunnablePassthrough.assign(
            context_text=RunnableLambda(lambda x: _format_context(x["context"])),
        )
        | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(
                    lambda x: {
                        "context": x["context_text"],
                        "chat_history": x["chat_history"],
                        "question": x["question"],
                    }
                )
                | ANSWER_PROMPT
                | answer_llm
                | StrOutputParser()
            ),
        )
    )


def dedupe_sources(docs: list[Document]) -> list[dict]:
    """Return unique source descriptors in retrieval order."""
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        url = d.metadata.get("source_url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({
            "url": url,
            "title": d.metadata.get("title", url),
            "category": d.metadata.get("category"),
            "doc_type": d.metadata.get("doc_type"),
        })
    return out
