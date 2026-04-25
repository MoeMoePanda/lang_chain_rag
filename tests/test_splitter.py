from langchain_core.documents import Document

from hdb_rag.ingest.splitter import chunk_documents


def test_chunking_preserves_metadata():
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    doc = Document(page_content=long_text, metadata={"source_url": "https://x", "title": "T"})
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source_url"] == "https://x"
        assert c.metadata["title"] == "T"
        assert "chunk_index" in c.metadata
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
