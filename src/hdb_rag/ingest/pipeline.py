"""Ingestion orchestrator: sources.yaml → vector store."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.documents import Document
from tqdm import tqdm

from hdb_rag import config
from hdb_rag.ingest.loaders import (
    download_pdf,
    load_html_url,
    load_pdf,
)
from hdb_rag.ingest.splitter import chunk_documents
from hdb_rag.stores import build_embedder, build_vector_store


def _load_sources(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    return data.get("sources", [])


def _doc_from_source(source: dict, ingested_at: str) -> list[Document]:
    base_meta = {
        "source_url": source["url"],
        "title": source["title"],
        "category": source["category"],
        "doc_type": source["type"],
        "ingested_at": ingested_at,
    }
    if source["type"] == "html":
        text = load_html_url(source["url"], user_agent=config.DISCOVERY["user_agent"])
        return [Document(page_content=text, metadata={**base_meta, "page_number": None})]
    if source["type"] == "pdf":
        path = download_pdf(source["url"], config.PDF_CACHE, user_agent=config.DISCOVERY["user_agent"])
        pages = load_pdf(path)
        if not pages:
            print(f"  ⚠️  empty PDF (likely scanned image): {source['url']} — skipping")
            return []
        return [
            Document(page_content=text, metadata={**base_meta, "page_number": pg})
            for pg, text in pages
        ]
    raise ValueError(f"Unknown source type: {source['type']}")


def run_ingest() -> None:
    ingested_at = datetime.now(timezone.utc).isoformat()
    sources = _load_sources(config.SOURCES_YAML)
    print(f"Loaded {len(sources)} sources from {config.SOURCES_YAML}")

    docs: list[Document] = []
    for src in tqdm(sources, desc="loading"):
        try:
            docs.extend(_doc_from_source(src, ingested_at))
        except Exception as e:
            print(f"  ⚠️  failed: {src['url']} — {e}")

    print(f"Loaded {len(docs)} documents. Chunking…")
    chunks = chunk_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"Produced {len(chunks)} chunks. Embedding + writing to {config.VECTOR_STORE}…")

    embedder = build_embedder()
    store = build_vector_store(embedder, reset=True)
    BATCH = 100
    for i in tqdm(range(0, len(chunks), BATCH), desc="indexing"):
        store.add_documents(chunks[i:i + BATCH])

    config.INGESTED_AT.parent.mkdir(parents=True, exist_ok=True)
    config.INGESTED_AT.write_text(ingested_at)
    print(f"✅ ingest complete. Wrote {config.INGESTED_AT}")
