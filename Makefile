.PHONY: install discover ingest eval run test lint

install:
	uv sync --all-extras

discover:
	uv run python scripts/discover_sources.py

ingest:
	uv run python scripts/ingest.py

eval:
	uv run python scripts/eval.py

run:
	uv run streamlit run streamlit_app.py

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/ scripts/
