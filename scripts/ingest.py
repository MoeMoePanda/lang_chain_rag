"""CLI: build the vector store from data/sources.yaml."""
from hdb_rag.ingest.pipeline import run_ingest


if __name__ == "__main__":
    run_ingest()
