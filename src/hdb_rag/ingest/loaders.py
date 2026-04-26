"""HTML and PDF loaders. Returns plain text; the caller wraps in Document.

HDB's pages are Next.js (Sitecore-managed). The visible HTML body is a thin
shell; the actual rules content lives inside the `__NEXT_DATA__` JSON blob in
fields like `bodyContentVal`. We extract both visible text *and* that JSON
content to capture the page in full.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


STRIP_TAGS = ("nav", "footer", "script", "style", "header", "aside")
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.+?)</script>', re.DOTALL,
)
# Field keys in the __NEXT_DATA__ tree that hold substantive page content
_CONTENT_KEYS = (
    "bodyContentVal",
    "pageDescrition",  # HDB's typo, kept as-is
    "metaDescription",
    "descVal",
)


def _normalise_line_for_overlap(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).casefold()


def _nonempty_normalised_lines(text: str) -> list[str]:
    return [_normalise_line_for_overlap(line) for line in text.splitlines() if line.strip()]


def _dedupe_initial_repeated_line(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    if _normalise_line_for_overlap(lines[0]) != _normalise_line_for_overlap(lines[1]):
        return lines
    return [lines[0], *lines[2:]]


def _overlap_line_count(left: str, right: str) -> int:
    left_lines = _nonempty_normalised_lines(left)
    right_lines = _nonempty_normalised_lines(right)
    max_overlap = min(len(left_lines), len(right_lines))
    for count in range(max_overlap, 0, -1):
        if left_lines[-count:] == right_lines[:count]:
            return count
    return 0


def _drop_nonempty_prefix_lines(text: str, count: int) -> str:
    if count <= 0:
        return text

    remaining = count
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        cursor += len(raw_line)
        if raw_line.strip():
            remaining -= 1
            if remaining == 0:
                return text[cursor:].lstrip()
    return ""


def _merge_text_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left

    overlap = _overlap_line_count(left, right)
    right_tail = _drop_nonempty_prefix_lines(right, overlap)
    if not right_tail:
        return left.rstrip()
    return f"{left.rstrip()}\n\n{right_tail.lstrip()}"


def _strip_html(fragment: str) -> str:
    text = BeautifulSoup(fragment, "html.parser").get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _next_data_text(html: str) -> str:
    """Pull rich content out of a Next.js __NEXT_DATA__ JSON blob."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ""

    parts: list[str] = []
    seen: set[str] = set()  # de-dupe identical strings (Sitecore embeds a few)

    def push(s: str) -> None:
        if not s:
            return
        cleaned = _strip_html(s) if "<" in s else s.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        parts.append(cleaned)

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CONTENT_KEYS:
                    if isinstance(v, str):
                        push(v)
                    elif isinstance(v, dict) and isinstance(v.get("value"), str):
                        push(v["value"])
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return "\n\n".join(parts)


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = _dedupe_initial_repeated_line(lines)
    return "\n".join(lines)


def _clean_html(html: str) -> str:
    """Combine the visible body text with content extracted from __NEXT_DATA__."""
    visible = _visible_text(html)
    next_data = _next_data_text(html)
    if next_data and visible:
        return _merge_text_parts(visible, next_data)
    return next_data or visible


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
