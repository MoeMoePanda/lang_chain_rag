"""Drop URLs that aren't useful HDB content."""
from __future__ import annotations

from urllib.parse import urlparse

JUNK_PATH_FRAGMENTS = (
    "/sitemap",
    "/contact",
    "/privacy",
    "/login",
    "/news-room",
    "/news",
    "/search",
    "/feedback",
    "/terms",
)
JUNK_SCHEMES = ("javascript", "mailto", "tel")


def is_junk_url(url: str) -> bool:
    """Return True if URL should be excluded from the source allowlist."""
    if not url:
        return True
    parsed = urlparse(url)
    if parsed.scheme in JUNK_SCHEMES:
        return True
    # Anchor-only links to the same page are not new content
    if parsed.fragment and not parsed.path:
        return True
    # Fragment-only re-references duplicate content
    if "#" in url and parsed.fragment:
        return True
    # Search/query pages
    if parsed.query and "search" in parsed.path.lower():
        return True
    path = parsed.path.lower()
    for frag in JUNK_PATH_FRAGMENTS:
        if frag in path:
            return True
    return False
