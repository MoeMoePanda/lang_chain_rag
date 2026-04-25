"""Ensemble (BM25 + vector) retriever using Reciprocal Rank Fusion."""
from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever


def build_hybrid_retriever(
    bm25: BaseRetriever,
    vector: BaseRetriever,
    *,
    bm25_weight: float,
    vector_weight: float,
) -> BaseRetriever:
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[bm25_weight, vector_weight],
    )
