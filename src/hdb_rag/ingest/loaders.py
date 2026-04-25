"""HTML and PDF loaders. Returns plain text; the caller wraps in Document."""
from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


STRIP_TAGS = ("nav", "footer", "script", "style", "header", "aside")


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_html_file(path: Path) -> str:
    return _clean_html(path.read_text(encoding="utf-8"))


def load_html_url(url: str, *, user_agent: str, timeout: int = 15) -> str:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return _clean_html(resp.text)


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_number, text) tuples (1-indexed)."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def download_pdf(url: str, dest_dir: Path, *, user_agent: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1].split("?", 1)[0] or "doc.pdf"
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    path = dest_dir / name
    if path.exists():
        return path
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path
