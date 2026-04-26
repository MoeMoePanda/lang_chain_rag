from langchain_core.documents import Document

from hdb_rag.retrieval.compression import (
    _get_prompt_input,
    _parse_ranking,
    build_rerank_prompt,
)


def test_rerank_context_includes_source_metadata():
    docs = [
        Document(
            page_content="General resale process information.",
            metadata={
                "title": "Process for Buying a Resale Flat",
                "section_title": "Application for an HDB Flat Eligibility (HFE) Letter",
                "source_url": "https://www.hdb.gov.sg/process",
            },
        ),
        Document(
            page_content="Enhanced CPF Housing Grant eligibility details.",
            metadata={
                "title": "Enhanced CPF Housing Grant",
                "section_title": "Enhanced CPF Housing Grant",
                "source_url": "https://www.hdb.gov.sg/enhanced-cpf-housing-grant",
            },
        ),
    ]

    prompt_input = _get_prompt_input({"documents": docs, "query": "What is EHG?"})

    assert "Title: Enhanced CPF Housing Grant" in prompt_input["context"]
    assert "Section: Application for an HDB Flat Eligibility (HFE) Letter" in prompt_input["context"]
    assert "https://www.hdb.gov.sg/enhanced-cpf-housing-grant" in prompt_input["context"]


def test_rerank_prompt_prefers_specific_grant_pages():
    messages = build_rerank_prompt().format_messages(context="Documents = [empty list]", query="PHG")
    prompt_text = messages[0].content

    assert "specific grant" in prompt_text
    assert "Down-rank generic buying-process" in prompt_text
    assert "PHG means Proximity Housing Grant" in prompt_text


def test_parse_ranking_ignores_duplicate_and_invalid_ids_then_appends_missing_docs():
    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
        Document(page_content="C"),
    ]

    class Ranking:
        ranked_document_ids = [2, 2, 99, -1, 0]

    ranked = _parse_ranking({"documents": docs, "ranking": Ranking()})

    assert [doc.page_content for doc in ranked] == ["C", "A", "B"]
