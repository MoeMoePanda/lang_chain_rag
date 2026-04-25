"""BM25 retriever built from the same chunks as the vector store."""
from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def build_bm25_retriever(docs: list[Document], *, k: int) -> BaseRetriever:
    r = BM25Retriever.from_documents(docs)
    r.k = k
    return r
