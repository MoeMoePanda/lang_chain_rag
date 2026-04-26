from langchain_community.chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from hdb_rag.chain import build_chain, dedupe_sources


class StaticRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


def test_chain_first_turn_skips_rewrite_and_uses_retrieved_context():
    docs = [
        Document(page_content="MOP is 5 years.", metadata={"source_url": "u1", "title": "MOP"}),
    ]
    retriever = StaticRetriever(docs=docs)
    answer_llm = FakeListChatModel(responses=["The MOP is 5 years."])
    fast_llm = FakeListChatModel(responses=["should-not-be-called"])

    chain = build_chain(retriever=retriever, answer_llm=answer_llm, fast_llm=fast_llm)
    result = chain.invoke({"question": "How long is the MOP?", "chat_history": []})

    assert result["answer"] == "The MOP is 5 years."
    assert [d.page_content for d in result["context"]] == [d.page_content for d in docs]
    assert result["context"][0].metadata["retrieval_rank"] == 1


def test_dedupe_sources_preserves_order_and_dedupes():
    docs = [
        Document(page_content="x", metadata={"source_url": "u1", "title": "A"}),
        Document(page_content="y", metadata={"source_url": "u1", "title": "A"}),
        Document(page_content="z", metadata={"source_url": "u2", "title": "B"}),
    ]
    out = dedupe_sources(docs)
    assert [s["url"] for s in out] == ["u1", "u2"]
