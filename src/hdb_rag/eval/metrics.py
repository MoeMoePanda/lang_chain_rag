"""Eval metrics: deterministic + LLM-judge."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from hdb_rag.prompts import FAITHFULNESS_JUDGE_PROMPT, RELEVANCE_JUDGE_PROMPT

_REFUSAL_PHRASES = (
    "i'm not sure",
    "i'm sorry",
    "i am sorry",
    "please check hdb.gov.sg",
    "can't help",
    "cannot help",
    "can only provide information",
    "can only assist",
    "unrelated",
    "outside the scope",
    "not within",
    "i don't have",
)


def retrieval_at_k(retrieved: list[Document], *, must_contain: list[str]) -> bool:
    urls = [d.metadata.get("source_url", "").lower() for d in retrieved]
    return any(any(s.lower() in u for s in must_contain) for u in urls)


def citation_correct(sources: list[dict], *, must_contain: list[str]) -> bool:
    urls = [s.get("url", "").lower() for s in sources]
    return any(any(s.lower() in u for s in must_contain) for u in urls)


def refused(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in _REFUSAL_PHRASES)


def judge_faithfulness(answer: str, context: list[Document], llm: BaseChatModel) -> bool:
    text = "\n\n".join(d.page_content for d in context)
    chain = FAITHFULNESS_JUDGE_PROMPT | llm | StrOutputParser()
    out = chain.invoke({"context": text, "answer": answer}).strip().upper()
    return out.startswith("YES")


def judge_relevance(question: str, answer: str, llm: BaseChatModel) -> bool:
    chain = RELEVANCE_JUDGE_PROMPT | llm | StrOutputParser()
    out = chain.invoke({"question": question, "answer": answer}).strip().upper()
    return out.startswith("YES")
