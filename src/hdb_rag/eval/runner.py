"""Run the eval set against the chain in a given mode."""
from __future__ import annotations

from dataclasses import dataclass

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


def run_mode(
    mode: str,
    *,
    answer_llm: BaseChatModel,
    fast_llm: BaseChatModel,
    judge_llm: BaseChatModel,
) -> list[CaseResult]:
    store = build_vector_store(build_embedder())
    retriever = build_retriever(mode=mode, store=store, fast_llm=fast_llm)
    chain = build_chain(retriever=retriever, answer_llm=answer_llm, fast_llm=fast_llm)
    results: list[CaseResult] = []

    for case in tqdm(_load_cases(), desc=f"eval ({mode})"):
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

        results.append(
            CaseResult(
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
        )
    return results
