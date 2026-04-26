"""Run the eval set against the chain in a given mode.

Results are appended to `reports/eval_results_<mode>.jsonl` as each case
finishes, so a killed run can be resumed without re-running completed cases.
Delete the file to start fresh.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm

from hdb_rag import config
from hdb_rag.chain import build_chain, dedupe_sources
from hdb_rag.eval.metrics import (
    citation_correct,
    judge_faithfulness,
    judge_relevance,
    refused,
    retrieval_at_k,
)
from hdb_rag.retrieval.factory import build_retriever
from hdb_rag.stores import build_embedder, build_vector_store


@dataclass
class CaseResult:
    id: int
    question: str
    category: str
    expected_behavior: str
    answer: str
    sources: list[dict]
    retrieved: list[Document]
    retrieval_pass: bool | None
    faithfulness_pass: bool | None
    relevance_pass: bool | None
    citation_pass: bool | None
    refusal_pass: bool | None


def _load_cases() -> list[dict]:
    return yaml.safe_load(config.EVAL_SET_YAML.read_text())["cases"]


def _results_path(mode: str) -> Path:
    return config.REPORTS_DIR / f"eval_results_{mode}.jsonl"


def _result_to_dict(r: CaseResult) -> dict:
    """Serialize a CaseResult to a JSON-safe dict (drops `retrieved`)."""
    return {
        "id": r.id,
        "question": r.question,
        "category": r.category,
        "expected_behavior": r.expected_behavior,
        "answer": r.answer,
        "sources": r.sources,
        "retrieval_pass": r.retrieval_pass,
        "faithfulness_pass": r.faithfulness_pass,
        "relevance_pass": r.relevance_pass,
        "citation_pass": r.citation_pass,
        "refusal_pass": r.refusal_pass,
    }


def _dict_to_result(d: dict) -> CaseResult:
    return CaseResult(
        id=d["id"],
        question=d["question"],
        category=d["category"],
        expected_behavior=d["expected_behavior"],
        answer=d["answer"],
        sources=d["sources"],
        retrieved=[],  # not persisted; only used during scoring
        retrieval_pass=d.get("retrieval_pass"),
        faithfulness_pass=d.get("faithfulness_pass"),
        relevance_pass=d.get("relevance_pass"),
        citation_pass=d.get("citation_pass"),
        refusal_pass=d.get("refusal_pass"),
    )


def load_saved_results(mode: str) -> list[CaseResult]:
    """Load any previously-completed results for this mode from disk."""
    path = _results_path(mode)
    if not path.exists():
        return []
    out: list[CaseResult] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(_dict_to_result(json.loads(line)))
        except json.JSONDecodeError:
            # Tolerate a partially-written tail line from a kill-mid-write
            continue
    return out


def _append_result(r: CaseResult, mode: str) -> None:
    path = _results_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(_result_to_dict(r)) + "\n")


def run_mode(
    mode: str,
    *,
    answer_llm: BaseChatModel,
    fast_llm: BaseChatModel,
    judge_llm: BaseChatModel,
) -> list[CaseResult]:
    """Run all eval cases for `mode`, resuming any persisted progress.

    Already-completed cases (by id) are loaded from
    `reports/eval_results_<mode>.jsonl` and skipped. New results are appended
    to the same file as each case finishes, so a kill mid-run loses at most
    one case's worth of work.
    """
    saved = load_saved_results(mode)
    saved_by_id = {r.id: r for r in saved}

    cases = _load_cases()
    todo_count = sum(1 for c in cases if c["id"] not in saved_by_id)
    if saved:
        print(
            f"  resuming {mode}: {len(saved)}/{len(cases)} already done, "
            f"{todo_count} remaining"
        )

    # Lazy chain build — skip entirely if there's nothing to do
    chain = None
    if todo_count > 0:
        store = build_vector_store(build_embedder())
        retriever = build_retriever(mode=mode, store=store, fast_llm=fast_llm)
        chain = build_chain(retriever=retriever, answer_llm=answer_llm, fast_llm=fast_llm)

    results: list[CaseResult] = []
    for case in tqdm(cases, desc=f"eval ({mode})"):
        if case["id"] in saved_by_id:
            results.append(saved_by_id[case["id"]])
            continue

        try:
            out = chain.invoke({"question": case["question"], "chat_history": []})
            retrieved = out["context"]
            answer = out["answer"]
        except Exception as e:
            retrieved = []
            answer = f"(eval error: {e})"

        sources = dedupe_sources(retrieved)
        must = case.get("must_cite_url_contains", [])
        is_answer_case = case["expected_behavior"] == "answer"

        result = CaseResult(
            id=case["id"],
            question=case["question"],
            category=case["category"],
            expected_behavior=case["expected_behavior"],
            answer=answer,
            sources=sources,
            retrieved=retrieved,
            retrieval_pass=retrieval_at_k(retrieved, must_contain=must) if must else None,
            faithfulness_pass=(
                judge_faithfulness(answer, retrieved, judge_llm)
                if is_answer_case and retrieved
                else None
            ),
            relevance_pass=(
                judge_relevance(case["question"], answer, judge_llm)
                if is_answer_case
                else None
            ),
            citation_pass=citation_correct(sources, must_contain=must) if must else None,
            refusal_pass=(refused(answer) == (case["expected_behavior"] == "refuse")),
        )
        _append_result(result, mode)
        results.append(result)
    return results
