from langchain_core.documents import Document

from hdb_rag.ingest.splitter import chunk_documents


def test_chunking_preserves_metadata():
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    doc = Document(
        page_content=long_text,
        metadata={
            "source_url": "https://x",
            "title": "T",
            "category": "buying",
            "doc_type": "html",
            "page_number": None,
            "ingested_at": "2026-04-25T00:00:00+00:00",
        },
    )
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source_url"] == "https://x"
        assert c.metadata["title"] == "T"
        assert "chunk_index" in c.metadata
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunking_adds_rich_stable_metadata():
    text = "\n".join([
        "Eligibility",
        "You must be a Singapore Citizen aged 21 or above.",
        "You must form a family nucleus.",
        "",
        "Income Ceiling",
        "Your household income must be within the posted limit.",
    ])
    doc = Document(page_content=text, metadata={"source_url": "https://hdb/eligibility", "title": "BTO"})

    first = chunk_documents([doc], chunk_size=160, chunk_overlap=20)
    second = chunk_documents([doc], chunk_size=160, chunk_overlap=20)

    required = {
        "source_doc_id",
        "chunk_id",
        "chunk_index",
        "section_index",
        "section_title",
        "section_path",
        "chunk_start_char",
        "chunk_end_char",
        "chunk_char_count",
        "chunk_content_hash",
        "chunking_strategy",
        "chunk_size",
        "chunk_overlap",
    }
    assert first
    assert {chunk.metadata["chunk_id"] for chunk in first} == {
        chunk.metadata["chunk_id"] for chunk in second
    }
    assert len({chunk.metadata["chunk_id"] for chunk in first}) == len(first)
    for chunk in first:
        assert required <= set(chunk.metadata)
        assert chunk.metadata["chunking_strategy"] == "section_recursive_v1"
        assert chunk.metadata["chunk_size"] == 160
        assert chunk.metadata["chunk_overlap"] == 20


def test_chunking_detects_section_headings_and_preserves_them():
    text = "\n".join([
        "Eligibility",
        "You must satisfy the eligibility conditions before applying.",
        "",
        "Renovation Permit",
        "You need approval before starting renovation works.",
    ])
    doc = Document(page_content=text, metadata={"source_url": "https://hdb/rules", "title": "HDB Rules"})

    chunks = chunk_documents([doc], chunk_size=120, chunk_overlap=10)

    section_titles = [chunk.metadata["section_title"] for chunk in chunks]
    assert "Eligibility" in section_titles
    assert "Renovation Permit" in section_titles
    assert any(chunk.page_content.startswith("Eligibility") for chunk in chunks)
    assert any(chunk.page_content.startswith("Renovation Permit") for chunk in chunks)
    for chunk in chunks:
        assert chunk.metadata["section_path"].startswith("HDB Rules")


def test_chunking_falls_back_to_document_title_when_no_heading_found():
    text = "This is a normal paragraph. It has full sentences. There are no standalone headings."
    doc = Document(page_content=text, metadata={"source_url": "https://hdb/no-heading", "title": "Fallback"})

    chunks = chunk_documents([doc], chunk_size=80, chunk_overlap=10)

    assert chunks
    assert {chunk.metadata["section_title"] for chunk in chunks} == {"Fallback"}
    assert {chunk.metadata["section_path"] for chunk in chunks} == {"Fallback"}


def test_chunking_character_spans_are_consistent():
    text = "\n".join([
        "Eligibility",
        "You must satisfy the eligibility conditions before applying.",
        "Applicants must provide accurate information.",
    ])
    doc = Document(page_content=text, metadata={"source_url": "https://hdb/spans", "title": "Spans"})

    chunks = chunk_documents([doc], chunk_size=80, chunk_overlap=10)

    for chunk in chunks:
        start = chunk.metadata["chunk_start_char"]
        end = chunk.metadata["chunk_end_char"]
        assert text[start:end] == chunk.page_content
        assert chunk.metadata["chunk_char_count"] == len(chunk.page_content)
