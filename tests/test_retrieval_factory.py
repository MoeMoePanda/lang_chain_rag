import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from hdb_rag import config
from hdb_rag.chunk_cache import write_chunks_jsonl
from hdb_rag.retrieval.diagnostics import RetrievalDiagnostics
from hdb_rag.retrieval.factory import build_retriever


class StaticRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


class DummyStore:
    def as_retriever(self, search_kwargs):
        return StaticRetriever(docs=[
            Document(page_content="Vector hit.", metadata={"source_url": "vector"})
        ])


def test_best_retriever_requires_chunk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHUNKS_JSONL", tmp_path / "missing.jsonl")

    with pytest.raises(RuntimeError, match="make ingest"):
        build_retriever(mode="best", store=object(), fast_llm=object())


def test_best_retriever_builds_from_chunk_cache(tmp_path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(
        [
            Document(
                page_content="The minimum occupation period is 5 years.",
                metadata={"source_url": "cache", "chunk_id": "cache-1"},
            )
        ],
        chunks_path,
    )
    monkeypatch.setattr(config, "CHUNKS_JSONL", chunks_path)
    monkeypatch.setattr(
        "hdb_rag.retrieval.factory.wrap_with_compression",
        lambda retriever, llm, top_n: retriever,
    )

    diagnostics = RetrievalDiagnostics()
    retriever = build_retriever(
        mode="best",
        store=DummyStore(),
        fast_llm=FakeListChatModel(responses=[
            "MOP\nminimum occupation period\nHow long must I live in the flat before selling?"
        ]),
        diagnostics=diagnostics,
    )
    hits = retriever.invoke("MOP")

    assert hits
    assert diagnostics.generated_queries == [
        "MOP",
        "minimum occupation period",
        "How long must I live in the flat before selling?",
        "MOP",
    ]
    assert [entry["query"] for entry in diagnostics.retrieved_by_query] == [
        "MOP",
        "minimum occupation period",
        "How long must I live in the flat before selling?",
        "MOP",
    ]
