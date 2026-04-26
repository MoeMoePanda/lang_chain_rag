from langchain_core.documents import Document

from hdb_rag.retrieval.bm25 import bm25_preprocess, build_bm25_retriever


def test_bm25_finds_exact_term():
    docs = [
        Document(page_content="The minimum occupation period (MOP) is 5 years.", metadata={"source_url": "a"}),
        Document(page_content="Renting out is allowed after MOP.", metadata={"source_url": "b"}),
        Document(page_content="Eligibility for BTO requires Singapore citizenship.", metadata={"source_url": "c"}),
    ]
    r = build_bm25_retriever(docs, k=2)
    hits = r.invoke("MOP")
    assert any("MOP" in h.page_content for h in hits)


def test_bm25_preprocess_lowercases_normalizes_hyphens_and_preserves_acronyms():
    tokens = bm25_preprocess("MOP for Build-To-Order and 2-room flats")

    assert "mop" in tokens
    assert "MOP" in tokens
    assert "build" in tokens
    assert "to" in tokens
    assert "order" in tokens
    assert "2" in tokens
    assert "room" in tokens
    assert "flats" in tokens


def test_bm25_expands_acronym_query_to_full_phrase():
    docs = [
        Document(page_content="The minimum occupation period is 5 years.", metadata={"source_url": "a"}),
        Document(page_content="Renovation permits are required for certain works.", metadata={"source_url": "b"}),
        Document(page_content="Pet ownership rules apply to approved dog breeds.", metadata={"source_url": "c"}),
    ]

    r = build_bm25_retriever(docs, k=1)
    hits = r.invoke("MOP")

    assert hits[0].metadata["source_url"] == "a"


def test_bm25_expands_full_phrase_query_to_acronym():
    docs = [
        Document(page_content="The EHG amount depends on average household income.", metadata={"source_url": "a"}),
        Document(page_content="Rental rules apply to whole flat rentals.", metadata={"source_url": "b"}),
        Document(page_content="Window works may require an approved contractor.", metadata={"source_url": "c"}),
    ]

    r = build_bm25_retriever(docs, k=1)
    hits = r.invoke("enhanced cpf housing grant")

    assert hits[0].metadata["source_url"] == "a"


def test_bm25_hyphen_normalization_matches_space_query():
    docs = [
        Document(page_content="Short-lease 2-room Flexi flats are available.", metadata={"source_url": "a"}),
        Document(page_content="Executive condominiums have different rules.", metadata={"source_url": "b"}),
        Document(page_content="Renovation noise should follow permitted hours.", metadata={"source_url": "c"}),
    ]

    r = build_bm25_retriever(docs, k=1)
    hits = r.invoke("short lease 2 room")

    assert hits[0].metadata["source_url"] == "a"
