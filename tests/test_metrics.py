from langchain_core.documents import Document

from hdb_rag.eval.metrics import (
    citation_correct,
    refused,
    retrieval_at_k,
)


def test_retrieval_at_k_substring_match():
    docs = [
        Document(page_content="x", metadata={"source_url": "https://hdb/eligibility"}),
        Document(page_content="x", metadata={"source_url": "https://hdb/financing"}),
    ]
    assert retrieval_at_k(docs, must_contain=["eligibility"]) is True
    assert retrieval_at_k(docs, must_contain=["mop"]) is False


def test_citation_correct():
    sources = [{"url": "https://hdb/eligibility/single"}]
    assert citation_correct(sources, must_contain=["single"]) is True
    assert citation_correct(sources, must_contain=["mop"]) is False


def test_refused_detects_refusal_phrases():
    assert refused("I'm not sure based on what I've indexed — please check hdb.gov.sg") is True
    assert refused("This is unrelated to HDB; I can't help with that.") is True
    assert refused("The MOP is 5 years.") is False
