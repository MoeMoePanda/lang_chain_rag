"""robots.txt checks + per-host rate limiting.

We fetch robots.txt with `requests` (using our configured User-Agent) rather
than `RobotFileParser.read()` because some sites (e.g., hdb.gov.sg) front their
content with a WAF that 403s the default urllib User-Agent — even for
robots.txt itself.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


class PolitenessGate:
    """Caches per-host robots.txt and enforces a request delay between fetches."""

    def __init__(self, user_agent: str, request_delay: float):
        self.user_agent = user_agent
        self.request_delay = request_delay
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at: float = 0.0

    def _robots_for(self, url: str) -> RobotFileParser:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                resp = requests.get(
                    f"https://{host}/robots.txt",
                    headers={"User-Agent": self.user_agent},
                    timeout=10,
                )
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    # No / inaccessible robots.txt — conservative default: allow all
                    # (sitemap-driven discovery is already scope-limited.)
                    rp.allow_all = True
            except requests.RequestException:
                rp.allow_all = True
            self._robots[host] = rp
        return self._robots[host]

    def can_fetch(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.user_agent, url)

    def wait_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()
