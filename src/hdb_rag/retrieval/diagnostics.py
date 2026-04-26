"""Lightweight retrieval diagnostics for eval reporting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


def _doc_ref(doc: Document, *, rank: int) -> dict[str, Any]:
    metadata = doc.metadata
    return {
        "rank": rank,
        "chunk_id": metadata.get("chunk_id"),
        "chunk_index": metadata.get("chunk_index"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "section_title": metadata.get("section_title"),
        "page_number": metadata.get("page_number"),
        "retrieval_rank": metadata.get("retrieval_rank"),
    }


@dataclass
class RetrievalDiagnostics:
    """Mutable per-query retrieval trace used only by eval runs."""

    generated_queries: list[str] = field(default_factory=list)
    retrieved_by_query: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.generated_queries = []
        self.retrieved_by_query = []

    def record_generated_queries(self, queries: list[str]) -> None:
        self.generated_queries = list(queries)

    def record_query_results(self, query: str, docs: list[Document]) -> None:
        self.retrieved_by_query.append({
            "query": query,
            "results": [_doc_ref(doc, rank=i) for i, doc in enumerate(docs, start=1)],
        })

    def to_dict(
        self,
        *,
        original_query: str,
        standalone_query: str | None,
        final_docs: list[Document],
    ) -> dict[str, Any]:
        return {
            "original_query": original_query,
            "standalone_query": standalone_query,
            "generated_queries": self.generated_queries,
            "retrieved_by_query": self.retrieved_by_query,
            "final_reranked": [
                _doc_ref(doc, rank=i) for i, doc in enumerate(final_docs, start=1)
            ],
        }
