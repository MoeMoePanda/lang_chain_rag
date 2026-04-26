"""Wrap a retriever in ContextualCompressionRetriever with LLM-listwise rerank."""
from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

_RERANK_SYSTEM_TEMPLATE = """You are reranking retrieved HDB documents for grounded question answering.

Prefer documents that can directly answer the query. Pages or sections whose
title, URL, or content names the specific grant, scheme, acronym, condition, or
policy in the query are more relevant than broad overview or process pages.
Acronyms and expanded forms are equivalent, e.g. EHG means Enhanced CPF Housing
Grant and PHG means Proximity Housing Grant.

Down-rank generic buying-process, HFE-letter, next-step, and overview pages
unless the user explicitly asks about those processes. Keep specific grant or
scheme pages high enough to be cited even if a general page also mentions the
same words.

{context}

Sort all Documents by their relevance to the Query.
"""


class _RankDocuments(BaseModel):
    """Rank the documents by their relevance to the user question."""

    ranked_document_ids: list[int] = Field(
        ...,
        description="Document IDs sorted from most to least relevant to the user question.",
    )


def build_rerank_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _RERANK_SYSTEM_TEMPLATE),
        ("human", "{query}"),
    ])


def _get_prompt_input(input_: dict) -> dict[str, str]:
    context_parts = []
    for index, doc in enumerate(input_["documents"]):
        title = str(doc.metadata.get("title") or "Untitled")
        section = str(doc.metadata.get("section_title") or title)
        url = str(doc.metadata.get("source_url") or "")
        context_parts.append(
            "\n".join([
                f"Document ID: {index}",
                f"Title: {title}",
                f"Section: {section}",
                f"URL: {url}",
                "Content:",
                f"```{doc.page_content}```",
            ])
        )

    document_range = "empty list"
    if input_["documents"]:
        document_range = f"Document ID: 0, ..., Document ID: {len(input_['documents']) - 1}"
    context_parts.append(f"Documents = [{document_range}]")
    return {"query": input_["query"], "context": "\n\n".join(context_parts)}


def _parse_ranking(results: dict) -> list[Document]:
    docs = results["documents"]
    ranked_ids = results["ranking"].ranked_document_ids
    ranked: list[Document] = []
    seen: set[int] = set()
    for doc_id in ranked_ids:
        if doc_id in seen or doc_id < 0 or doc_id >= len(docs):
            continue
        ranked.append(docs[doc_id])
        seen.add(doc_id)

    ranked.extend(doc for i, doc in enumerate(docs) if i not in seen)
    return ranked


def wrap_with_compression(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    top_n: int,
) -> BaseRetriever:
    reranker = RunnablePassthrough.assign(
        ranking=RunnableLambda(_get_prompt_input)
        | build_rerank_prompt()
        | llm.with_structured_output(_RankDocuments),
    ) | RunnableLambda(_parse_ranking)

    return ContextualCompressionRetriever(
        base_compressor=LLMListwiseRerank(reranker=reranker, top_n=top_n),
        base_retriever=retriever,
    )
