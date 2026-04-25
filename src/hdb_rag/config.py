"""Single source of truth for all configurable knobs."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LLMs (Gemini API / Google AI Studio) ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gemma-4-31b-it")
FAST_MODEL: str = os.getenv("FAST_MODEL", "gemma-4-31b-it")

# --- Embeddings (OpenAI) ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536

# --- Vector store ---
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")  # "chroma" | "pgvector"
SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME: str = "hdb_rules"

# --- Discovery ---
DISCOVERY = {
    "start_urls": [
        "https://www.hdb.gov.sg/residential/buying-a-flat",
        "https://www.hdb.gov.sg/residential/living-in-an-hdb-flat",
    ],
    "scope_paths": [
        "/residential/buying-a-flat/",
        "/residential/living-in-an-hdb-flat/",
    ],
    "max_depth": 4,
    "max_urls": 200,
    "user_agent": "hdb-rag-portfolio/0.1 (github.com/MoeMoePanda/lang_chain_rag)",
    "request_delay_seconds": 1.0,
}

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# --- Retrieval ---
RETRIEVAL = {
    "bm25_weight": 0.4,
    "vector_weight": 0.6,
    "ensemble_top_k": 10,
    "multi_query_variants": 3,
    "rerank_top_n": 5,
}

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SOURCES_YAML = DATA_DIR / "sources.yaml"
EVAL_SET_YAML = DATA_DIR / "eval_set.yaml"
INGESTED_AT = DATA_DIR / "ingested_at.txt"
PDF_CACHE = DATA_DIR / "pdfs"
REPORTS_DIR = ROOT / "reports"
EVAL_REPORT = REPORTS_DIR / "eval-report.md"

# --- LangSmith ---
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
