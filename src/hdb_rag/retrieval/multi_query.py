"""Wrap a retriever in MultiQueryRetriever for paraphrase expansion."""
from __future__ import annotations

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever


def wrap_with_multi_query(retriever: BaseRetriever, llm: BaseChatModel) -> BaseRetriever:
    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        include_original=True,
    )
