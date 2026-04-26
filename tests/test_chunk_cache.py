from langchain_core.documents import Document

from hdb_rag.chunk_cache import load_chunks_jsonl, write_chunks_jsonl


def test_chunk_cache_round_trips_documents(tmp_path):
    chunks = [
        Document(
            page_content="Eligibility\nMOP is 5 years.",
            metadata={
                "source_url": "https://hdb/a",
                "title": "A",
                "chunk_id": "chunk-a",
                "retrieval_rank": 1,
            },
        ),
        Document(
            page_content="Income Ceiling\nIncome rules apply.",
            metadata={"source_url": "https://hdb/b", "title": "B", "chunk_id": "chunk-b"},
        ),
    ]
    path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(chunks, path)
    loaded = load_chunks_jsonl(path)

    assert [doc.page_content for doc in loaded] == [doc.page_content for doc in chunks]
    assert [doc.metadata for doc in loaded] == [doc.metadata for doc in chunks]


def test_chunk_cache_rejects_invalid_records(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"metadata": {}}\n')

    try:
        load_chunks_jsonl(path)
    except ValueError as e:
        assert "Invalid chunk record" in str(e)
    else:
        raise AssertionError("expected invalid chunk record to raise")
