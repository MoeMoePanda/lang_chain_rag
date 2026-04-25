"""Vector store factory: Chroma (local) | pgvector (Supabase)."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from hdb_rag import config


def build_embedder() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment")
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def build_vector_store(embedder: Embeddings, *, reset: bool = False) -> VectorStore:
    """Return a VectorStore. If reset=True, the collection is wiped first."""
    if config.VECTOR_STORE == "chroma":
        from langchain_chroma import Chroma
        if reset:
            import shutil
            shutil.rmtree(config.CHROMA_PERSIST_DIR, ignore_errors=True)
        return Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
    if config.VECTOR_STORE == "pgvector":
        from langchain_postgres import PGVector
        if not config.SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL missing for pgvector backend")
        # SQLAlchemy defaults to psycopg2 for `postgresql://`; we ship psycopg (v3)
        # instead, so explicitly select it via the URL scheme.
        url = config.SUPABASE_DB_URL
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        elif url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        store = PGVector(
            embeddings=embedder,
            collection_name=config.COLLECTION_NAME,
            connection=url,
            use_jsonb=True,
        )
        if reset:
            store.delete_collection()
            store.create_collection()
        return store
    raise ValueError(f"Unknown VECTOR_STORE: {config.VECTOR_STORE}")
