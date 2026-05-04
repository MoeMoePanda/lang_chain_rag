"""CLI: fetch HDB sitemap.xml → write data/sources.yaml.

The site's sitemap.xml is authoritative (1,332 URLs). We filter by scope path
(buying-a-flat, selling-a-flat, managing-my-home), drop junk, and emit a sorted
YAML allowlist that subsequent ingestion runs against. The committed YAML is
the auditable source of truth for what the bot has been taught.
"""
from __future__ import annotations

from hdb_rag import config
from hdb_rag.discovery.emitter import emit_sources_yaml
from hdb_rag.discovery.sitemap import discover_from_sitemap


def main() -> None:
    print(f"Fetching sitemap: {config.DISCOVERY['sitemap_url']}")
    pages = discover_from_sitemap(
        sitemap_url=config.DISCOVERY["sitemap_url"],
        scope_paths=config.DISCOVERY["scope_paths"],
        user_agent=config.DISCOVERY["user_agent"],
        request_delay=config.DISCOVERY["request_delay_seconds"],
    )
    print(f"Found {len(pages)} in-scope pages.")
    by_cat: dict[str, int] = {}
    for p in pages:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")
    emit_sources_yaml(pages, config.SOURCES_YAML)
    print(f"Wrote {config.SOURCES_YAML}")


if __name__ == "__main__":
    main()
