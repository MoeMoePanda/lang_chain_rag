from langchain_core.documents import Document

from hdb_rag.retrieval.bm25 import build_bm25_retriever


def test_bm25_finds_exact_term():
    docs = [
        Document(page_content="The minimum occupation period (MOP) is 5 years.", metadata={"source_url": "a"}),
        Document(page_content="Renting out is allowed after MOP.", metadata={"source_url": "b"}),
        Document(page_content="Eligibility for BTO requires Singapore citizenship.", metadata={"source_url": "c"}),
    ]
    r = build_bm25_retriever(docs, k=2)
    hits = r.invoke("MOP")
    assert any("MOP" in h.page_content for h in hits)
