from hdb_rag.eval.runner import CaseResult, _saved_result_matches_case


def _result(**overrides):
    base = {
        "id": 1,
        "question": "What is EHG?",
        "category": "buying",
        "expected_behavior": "answer",
        "answer": "answer",
        "sources": [],
        "retrieved": [],
        "retrieval_pass": None,
        "faithfulness_pass": None,
        "relevance_pass": None,
        "citation_pass": None,
        "refusal_pass": True,
    }
    base.update(overrides)
    return CaseResult(**base)


def test_saved_result_matches_case_signature():
    case = {
        "id": 1,
        "question": "What is EHG?",
        "category": "buying",
        "expected_behavior": "answer",
    }

    assert _saved_result_matches_case(_result(), case) is True


def test_saved_result_mismatch_when_expected_behavior_changes():
    case = {
        "id": 1,
        "question": "What is EHG?",
        "category": "buying",
        "expected_behavior": "refuse",
    }

    assert _saved_result_matches_case(_result(), case) is False
