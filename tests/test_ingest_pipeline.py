import pytest
from langchain_core.documents import Document

from hdb_rag import config
from hdb_rag.chunk_cache import load_chunks_jsonl
from hdb_rag.ingest import pipeline


def _patch_basic_ingest(monkeypatch, tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    ingested_at_path = tmp_path / "ingested_at.txt"
    monkeypatch.setattr(config, "CHUNKS_JSONL", chunks_path)
    monkeypatch.setattr(config, "INGESTED_AT", ingested_at_path)
    monkeypatch.setattr(
        pipeline,
        "_load_sources",
        lambda path: [
            {
                "url": "https://hdb.test/page",
                "title": "Eligibility",
                "category": "buying",
                "type": "html",
            }
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "_doc_from_source",
        lambda source, ingested_at, gate: [
            Document(
                page_content="Eligibility\nYou must satisfy the conditions before applying.",
                metadata={
                    "source_url": source["url"],
                    "title": source["title"],
                    "category": source["category"],
                    "doc_type": source["type"],
                    "page_number": None,
                    "ingested_at": ingested_at,
                },
            )
        ],
    )
    return chunks_path, ingested_at_path


def test_run_ingest_chunks_only_writes_chunks_without_embedding(monkeypatch, tmp_path):
    chunks_path, ingested_at_path = _patch_basic_ingest(monkeypatch, tmp_path)

    def fail_embedder():
        raise AssertionError("chunks-only should not build embeddings")

    monkeypatch.setattr(pipeline, "build_embedder", fail_embedder)

    pipeline.run_ingest(chunks_only=True)

    chunks = load_chunks_jsonl(chunks_path)
    assert chunks
    assert chunks[0].metadata["chunking_strategy"] == "section_recursive_v3"
    assert not ingested_at_path.exists()


def test_run_ingest_writes_chunks_before_embedding(monkeypatch, tmp_path):
    chunks_path, ingested_at_path = _patch_basic_ingest(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "build_embedder",
        lambda: (_ for _ in ()).throw(RuntimeError("missing key")),
    )

    with pytest.raises(RuntimeError, match="missing key"):
        pipeline.run_ingest()

    assert load_chunks_jsonl(chunks_path)
    assert not ingested_at_path.exists()
