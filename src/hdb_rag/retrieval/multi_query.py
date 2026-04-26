"""Wrap a retriever in MultiQueryRetriever for paraphrase expansion."""
from __future__ import annotations

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from hdb_rag.retrieval.diagnostics import RetrievalDiagnostics

_MULTI_QUERY_TEMPLATE = """You are generating search queries for an HDB rules RAG system.

Create exactly {variant_count} alternative search queries for the user's question.
Output one query per line, with no numbering, bullets, labels, quotes, or explanation.

For exactly 3 variants, use this order:
1. A literal/acronym-preserving query that keeps HDB acronyms and exact user wording where useful.
2. An HDB terminology query that uses or expands official terms such as MOP, BTO, HFE, EHG, CPF, EIP, SPR, resale flat, renting out, renovation permit.
3. A natural-language paraphrase that captures the user's intent in plain wording.

Question: {question}
"""


def build_multi_query_prompt(variant_count: int) -> PromptTemplate:
    return PromptTemplate.from_template(_MULTI_QUERY_TEMPLATE).partial(
        variant_count=str(variant_count)
    )


class RecordingMultiQueryRetriever(MultiQueryRetriever):
    """Multi-query retriever that records generated queries and per-query hits."""

    diagnostics: RetrievalDiagnostics | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        queries = self.generate_queries(query, run_manager)
        retrieval_queries = [*queries]
        if self.include_original:
            retrieval_queries.append(query)
        if self.diagnostics:
            self.diagnostics.record_generated_queries(retrieval_queries)
        documents = self.retrieve_documents(retrieval_queries, run_manager)
        return self.unique_union(documents)

    def retrieve_documents(
        self,
        queries: list[str],
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        documents: list[Document] = []
        for query in queries:
            docs = self.retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child()},
            )
            if self.diagnostics:
                self.diagnostics.record_query_results(query, docs)
            documents.extend(docs)
        return documents


def wrap_with_multi_query(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    variants: int,
    diagnostics: RetrievalDiagnostics | None = None,
) -> BaseRetriever:
    multi = RecordingMultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=build_multi_query_prompt(variants),
        include_original=True,
    )
    multi.diagnostics = diagnostics
    return multi
