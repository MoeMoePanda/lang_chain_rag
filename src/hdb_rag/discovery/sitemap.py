"""Sitemap-based source discovery.

Fetches the site's sitemap.xml, filters entries to in-scope paths, drops junk,
and emits dicts ready for the YAML emitter. No BFS crawling — the sitemap is
authoritative and one HTTP request is more polite than hundreds.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests

from hdb_rag.discovery.filters import is_junk_url
from hdb_rag.discovery.politeness import PolitenessGate

_NS_RE = re.compile(r"^\{[^}]+\}")  # strips the sitemap XML namespace from tags


def _categorize(url: str) -> str:
    path = urlparse(url).path
    if "/buying-a-flat/" in path:
        return "buying"
    if "/selling-a-flat/" in path:
        return "selling"
    if "/managing-my-home/" in path or "/renting-a-flat/" in path:
        return "living"
    return "uncategorized"


def _doc_type(url: str) -> str:
    return "pdf" if url.lower().endswith(".pdf") else "html"


def _in_scope(url: str, scope_paths: list[str]) -> bool:
    path = urlparse(url).path
    return any(path.startswith(s) for s in scope_paths)


def _title_from_url(url: str) -> str:
    """Derive a human-friendly title from the URL's last path segment."""
    path = urlparse(url).path.rstrip("/")
    if not path:
        return url
    slug = path.rsplit("/", 1)[-1]
    # strip file extensions
    if "." in slug:
        slug = slug.rsplit(".", 1)[0]
    return slug.replace("-", " ").replace("_", " ").strip().title() or slug


def _parse_sitemap(xml: str) -> list[str]:
    """Extract every <loc> entry from a sitemap XML document."""
    root = ET.fromstring(xml)
    locs: list[str] = []
    for el in root.iter():
        tag = _NS_RE.sub("", el.tag)
        if tag == "loc" and el.text and el.text.strip():
            locs.append(el.text.strip())
    return locs


def discover_from_sitemap(
    *,
    sitemap_url: str,
    scope_paths: list[str],
    user_agent: str,
    request_delay: float,
) -> list[dict]:
    """Fetch the sitemap, filter by scope + junk, return list of source dicts."""
    gate = PolitenessGate(user_agent=user_agent, request_delay=request_delay)

    if not gate.can_fetch(sitemap_url):
        raise RuntimeError(f"robots.txt disallows fetching {sitemap_url}")

    gate.wait_if_needed()
    resp = requests.get(sitemap_url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()

    pages: list[dict] = []
    for url in _parse_sitemap(resp.text):
        if not _in_scope(url, scope_paths):
            continue
        if is_junk_url(url):
            continue
        pages.append({
            "url": url,
            "title": _title_from_url(url),
            "doc_type": _doc_type(url),
            "category": _categorize(url),
        })
    return pages
