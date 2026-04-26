"""CLI: build the vector store from data/sources.yaml."""
import argparse

from hdb_rag.ingest.pipeline import run_ingest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only ingest the first N sources (for smoke-testing).",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Load sources, write data/chunks.jsonl, and skip embeddings/vector-store writes.",
    )
    args = parser.parse_args()
    run_ingest(limit=args.limit, chunks_only=args.chunks_only)
