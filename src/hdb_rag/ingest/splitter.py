"""Section-aware document chunking with rich metadata."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKING_STRATEGY = "section_recursive_v1"
_MAX_HEADING_CHARS = 90
_MAX_HEADING_WORDS = 12
_BULLET_RE = re.compile(r"^(\d+[.)]|[a-z][.)]|[-*\u2022])\s+")
_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]$")


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Section:
    index: int
    title: str
    start: int
    text: str


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _hash_text(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _source_doc_id(metadata: dict) -> str:
    source_url = str(metadata.get("source_url") or "")
    page_number = metadata.get("page_number")
    return _hash_text(f"{source_url}#{page_number}")


def _iter_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw)
        stripped = raw.strip()
        if not stripped:
            continue
        content_start = line_start + len(raw) - len(raw.lstrip())
        content_end = line_start + len(raw.rstrip())
        lines.append(_Line(text=stripped, start=content_start, end=content_end))
    return lines


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > _MAX_HEADING_CHARS:
        return False
    if len(stripped.split()) > _MAX_HEADING_WORDS:
        return False
    if _BULLET_RE.match(stripped):
        return False
    if _URL_RE.match(stripped):
        return False
    if _SENTENCE_END_RE.search(stripped):
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    return True


def _default_section_title(doc: Document) -> str:
    return str(doc.metadata.get("title") or "Untitled")


def _sections_for(doc: Document) -> list[_Section]:
    text = _normalize_text(doc.page_content)
    if not text:
        return []

    lines = _iter_lines(text)
    headings = [line for line in lines if _is_heading(line.text)]
    if not headings:
        return [_Section(index=0, title=_default_section_title(doc), start=0, text=text)]

    sections: list[_Section] = []
    first_heading = headings[0]
    if first_heading.start > 0 and text[:first_heading.start].strip():
        sections.append(
            _Section(
                index=len(sections),
                title=_default_section_title(doc),
                start=0,
                text=text[:first_heading.start].strip(),
            )
        )

    for i, heading in enumerate(headings):
        next_start = headings[i + 1].start if i + 1 < len(headings) else len(text)
        section_text = text[heading.start:next_start].strip()
        if not section_text:
            continue
        sections.append(
            _Section(
                index=len(sections),
                title=heading.text,
                start=heading.start,
                text=section_text,
            )
        )

    return sections


def _section_path(doc: Document, section_title: str) -> str:
    title = _default_section_title(doc)
    if not section_title or section_title == title:
        return title
    return f"{title} > {section_title}"


def _chunk_end_char(source_text: str, chunk_text: str, start: int) -> int:
    snippet = chunk_text.strip()
    if not snippet:
        return start
    found_at = source_text.find(snippet, start)
    if found_at != -1:
        return found_at + len(snippet)
    return start + len(snippet)


def _metadata_for_chunk(
    *,
    doc: Document,
    section: _Section,
    source_text: str,
    source_doc_id: str,
    chunk_text: str,
    section_start_index: int,
    chunk_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    chunk_start = section.start + section_start_index
    chunk_end = _chunk_end_char(source_text, chunk_text, chunk_start)
    content_hash = _hash_text(chunk_text)
    return {
        **doc.metadata,
        "source_doc_id": source_doc_id,
        "chunk_id": _hash_text(f"{source_doc_id}:{chunk_index}:{content_hash}"),
        "chunk_index": chunk_index,
        "section_index": section.index,
        "section_title": section.title,
        "section_path": _section_path(doc, section.title),
        "chunk_start_char": chunk_start,
        "chunk_end_char": chunk_end,
        "chunk_char_count": len(chunk_text),
        "chunk_content_hash": content_hash,
        "chunking_strategy": CHUNKING_STRATEGY,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def chunk_documents(docs: list[Document], *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    out: list[Document] = []
    for doc in docs:
        source_text = _normalize_text(doc.page_content)
        source_doc_id = _source_doc_id(doc.metadata)
        chunk_index = 0
        for section in _sections_for(doc):
            section_doc = Document(page_content=section.text, metadata=doc.metadata)
            sub_chunks = splitter.split_documents([section_doc])
            for chunk in sub_chunks:
                start_index = int(chunk.metadata.get("start_index", 0))
                chunk.metadata = _metadata_for_chunk(
                    doc=doc,
                    section=section,
                    source_text=source_text,
                    source_doc_id=source_doc_id,
                    chunk_text=chunk.page_content,
                    section_start_index=start_index,
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                out.append(chunk)
                chunk_index += 1
    return out
