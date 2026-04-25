"""Assemble the configured retriever stack based on mode."""
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from hdb_rag import config
from hdb_rag.retrieval.bm25 import build_bm25_retriever
from hdb_rag.retrieval.compression import wrap_with_compression
from hdb_rag.retrieval.hybrid import build_hybrid_retriever
from hdb_rag.retrieval.multi_query import wrap_with_multi_query

Mode = Literal["fast", "best"]


def _all_chunks(store: VectorStore) -> list[Document]:
    """Pull every chunk from the store so BM25 can index them.

    Both Chroma and PGVector lack a clean public iterator, so we use the
    backend-specific shortcut where possible and fall back to a wide
    similarity search.
    """
    # Chroma path
    if hasattr(store, "_collection"):
        rec = store._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(rec["documents"], rec["metadatas"])
        ]
    # PGVector path: SQLAlchemy ORM exposes the EmbeddingStore model
    if hasattr(store, "EmbeddingStore") and hasattr(store, "session_maker"):
        with store.session_maker() as sess:
            rows = sess.query(store.EmbeddingStore).all()
            return [
                Document(page_content=r.document, metadata=r.cmetadata or {})
                for r in rows
            ]
    # Generic fallback
    return store.similarity_search("a", k=10_000)


def build_retriever(*, mode: Mode, store: VectorStore, fast_llm: BaseChatModel) -> BaseRetriever:
    if mode == "fast":
        return store.as_retriever(search_kwargs={"k": config.RETRIEVAL["rerank_top_n"]})

    chunks = _all_chunks(store)
    bm25 = build_bm25_retriever(chunks, k=config.RETRIEVAL["ensemble_top_k"])
    vector = store.as_retriever(search_kwargs={"k": config.RETRIEVAL["ensemble_top_k"]})
    hybrid = build_hybrid_retriever(
        bm25,
        vector,
        bm25_weight=config.RETRIEVAL["bm25_weight"],
        vector_weight=config.RETRIEVAL["vector_weight"],
    )
    multi = wrap_with_multi_query(hybrid, fast_llm)
    return wrap_with_compression(multi, fast_llm, top_n=config.RETRIEVAL["rerank_top_n"])
