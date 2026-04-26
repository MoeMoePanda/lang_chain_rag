from hdb_rag.retrieval.multi_query import build_multi_query_prompt


def test_multi_query_prompt_requests_three_specific_variants():
    prompt = build_multi_query_prompt(3)

    rendered = prompt.format(question="How long is MOP?")

    assert "exactly 3 alternative search queries" in rendered
    assert "literal/acronym-preserving query" in rendered
    assert "HDB terminology query" in rendered
    assert "natural-language paraphrase" in rendered
    assert "MOP" in rendered
    assert "no numbering, bullets, labels, quotes, or explanation" in rendered
