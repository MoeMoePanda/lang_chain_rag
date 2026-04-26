"""Assemble the configured retriever stack based on mode."""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from hdb_rag import config
from hdb_rag.chunk_cache import load_chunks_jsonl
from hdb_rag.retrieval.diagnostics import RetrievalDiagnostics
from hdb_rag.retrieval.bm25 import build_bm25_retriever
from hdb_rag.retrieval.compression import wrap_with_compression
from hdb_rag.retrieval.hybrid import build_hybrid_retriever
from hdb_rag.retrieval.multi_query import wrap_with_multi_query

Mode = Literal["fast", "best"]


def _load_bm25_chunks():
    try:
        return load_chunks_jsonl(config.CHUNKS_JSONL)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Chunk cache missing at {config.CHUNKS_JSONL}. "
            "Run `make ingest` before using best retrieval mode."
        ) from e


def build_retriever(
    *,
    mode: Mode,
    store: VectorStore,
    fast_llm: BaseChatModel,
    diagnostics: RetrievalDiagnostics | None = None,
) -> BaseRetriever:
    if mode == "fast":
        return store.as_retriever(search_kwargs={"k": config.RETRIEVAL["rerank_top_n"]})

    chunks = _load_bm25_chunks()
    bm25 = build_bm25_retriever(chunks, k=config.RETRIEVAL["ensemble_top_k"])
    vector = store.as_retriever(search_kwargs={"k": config.RETRIEVAL["ensemble_top_k"]})
    hybrid = build_hybrid_retriever(
        bm25,
        vector,
        bm25_weight=config.RETRIEVAL["bm25_weight"],
        vector_weight=config.RETRIEVAL["vector_weight"],
    )
    multi = wrap_with_multi_query(
        hybrid,
        fast_llm,
        variants=config.RETRIEVAL["multi_query_variants"],
        diagnostics=diagnostics,
    )
    return wrap_with_compression(multi, fast_llm, top_n=config.RETRIEVAL["rerank_top_n"])
