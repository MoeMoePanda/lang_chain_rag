from langchain_core.documents import Document

from hdb_rag.retrieval.diagnostics import RetrievalDiagnostics


def test_retrieval_diagnostics_records_queries_and_final_ranks():
    diagnostics = RetrievalDiagnostics()
    docs = [
        Document(
            page_content="MOP is 5 years.",
            metadata={
                "chunk_id": "c1",
                "chunk_index": 0,
                "source_url": "https://hdb/mop",
                "title": "MOP",
                "section_title": "Eligibility",
                "page_number": None,
                "retrieval_rank": 1,
            },
        )
    ]

    diagnostics.record_generated_queries(["MOP", "minimum occupation period"])
    diagnostics.record_query_results("MOP", docs)
    out = diagnostics.to_dict(
        original_query="How long is MOP?",
        standalone_query="How long is MOP?",
        final_docs=docs,
    )

    assert out["original_query"] == "How long is MOP?"
    assert out["generated_queries"] == ["MOP", "minimum occupation period"]
    assert out["retrieved_by_query"][0]["results"][0]["chunk_id"] == "c1"
    assert out["final_reranked"][0]["rank"] == 1
    assert out["final_reranked"][0]["retrieval_rank"] == 1
