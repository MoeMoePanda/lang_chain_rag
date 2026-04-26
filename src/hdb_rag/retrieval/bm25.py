"""BM25 retriever built from the same chunks as the vector store."""
from __future__ import annotations

import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_HYPHEN_RE = re.compile(r"[-\u2010-\u2015\u2212]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TERM_EXPANSIONS = {
    "bto": ("build", "to", "order"),
    "cpf": ("central", "provident", "fund"),
    "ec": ("executive", "condominium"),
    "ehg": ("enhanced", "cpf", "housing", "grant"),
    "eip": ("ethnic", "integration", "policy"),
    "hdb": ("housing", "development", "board"),
    "hfe": ("hdb", "flat", "eligibility"),
    "mop": ("minimum", "occupation", "period"),
    "phg": ("proximity", "housing", "grant"),
    "sbf": ("sale", "of", "balance", "flats"),
    "spr": ("singapore", "permanent", "resident"),
}


def _is_acronym(token: str) -> bool:
    return len(token) > 1 and token.isupper() and any(ch.isalpha() for ch in token)


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if len(phrase) > len(tokens):
        return False
    return any(tokens[i:i + len(phrase)] == list(phrase) for i in range(len(tokens) - len(phrase) + 1))


def bm25_preprocess(text: str) -> list[str]:
    """Normalize sparse-search tokens while preserving HDB acronym recall."""
    normalized = _HYPHEN_RE.sub(" ", text)
    raw_tokens = _TOKEN_RE.findall(normalized)
    lowercase_tokens = [token.lower() for token in raw_tokens]

    tokens: list[str] = []
    for raw, lowered in zip(raw_tokens, lowercase_tokens):
        tokens.append(lowered)
        if _is_acronym(raw):
            tokens.append(raw)
        if lowered in _TERM_EXPANSIONS:
            tokens.extend(_TERM_EXPANSIONS[lowered])

    for acronym, phrase in _TERM_EXPANSIONS.items():
        if _contains_phrase(lowercase_tokens, phrase):
            tokens.append(acronym)

    return tokens


def build_bm25_retriever(docs: list[Document], *, k: int) -> BaseRetriever:
    r = BM25Retriever.from_documents(docs, preprocess_func=bm25_preprocess)
    r.k = k
    return r
