"""Wrap a retriever in ContextualCompressionRetriever with LLM-listwise rerank."""
from __future__ import annotations

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever


def wrap_with_compression(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    top_n: int,
) -> BaseRetriever:
    return ContextualCompressionRetriever(
        base_compressor=LLMListwiseRerank.from_llm(llm=llm, top_n=top_n),
        base_retriever=retriever,
    )
