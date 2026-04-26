"""JSONL chunk cache shared by ingestion and retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _record_from_doc(doc: Document) -> dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def write_chunks_jsonl(chunks: list[Document], path: Path) -> None:
    """Write chunks to a deterministic JSONL cache for sparse retrieval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(_record_from_doc(chunk), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def load_chunks_jsonl(path: Path) -> list[Document]:
    """Load cached chunks written by `write_chunks_jsonl`."""
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in chunk cache {path} at line {line_no}") from e
            page_content = record.get("page_content")
            metadata = record.get("metadata", {})
            if not isinstance(page_content, str) or not isinstance(metadata, dict):
                raise ValueError(f"Invalid chunk record in {path} at line {line_no}")
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs
