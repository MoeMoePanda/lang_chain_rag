"""Write the discovered URL list to data/sources.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml


def emit_sources_yaml(pages: list[dict], out_path: Path) -> None:
    """Write a stable, sorted YAML list of sources."""
    pages_sorted = sorted(pages, key=lambda p: p["url"])
    out = {
        "sources": [
            {
                "url": p["url"],
                "type": p["doc_type"],
                "category": p["category"],
                "title": p["title"],
            }
            for p in pages_sorted
        ]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
