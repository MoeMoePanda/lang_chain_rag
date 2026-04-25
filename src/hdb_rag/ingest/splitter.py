"""Document chunking with metadata preservation."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[Document], *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[Document] = []
    for doc in docs:
        sub_chunks = splitter.split_documents([doc])
        for i, c in enumerate(sub_chunks):
            c.metadata = {**doc.metadata, **c.metadata, "chunk_index": i}
            out.append(c)
    return out
