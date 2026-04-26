"""Section-aware document chunking with rich metadata."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKING_STRATEGY = "section_recursive_v2"
_MAX_HEADING_CHARS = 90
_MAX_HEADING_WORDS = 12
_MIN_CHUNK_RATIO = 0.25
_MIN_CHUNK_CHARS_FLOOR = 200
_MIN_CHUNK_CHARS_CEILING = 350
_BULLET_RE = re.compile(r"^(\d+[.)]|[a-z][.)]|[-*\u2022])\s+")
_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_BOILERPLATE_HEADING_RE = re.compile(
    r"^(go to e-services|related topics|tools and resources|visit hdb flat portal)$",
    re.IGNORECASE,
)


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
    end: int
    text: str


@dataclass(frozen=True)
class _SectionGroup:
    start: int
    end: int
    text: str
    sections: tuple[_Section, ...]


@dataclass(frozen=True)
class _ChunkCandidate:
    start: int
    end: int
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
    if _BOILERPLATE_HEADING_RE.match(stripped):
        return False
    if _SENTENCE_END_RE.search(stripped):
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    return True


def _default_section_title(doc: Document) -> str:
    return str(doc.metadata.get("title") or "Untitled")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _make_section(
    *,
    index: int,
    title: str,
    source_text: str,
    start: int,
    end: int,
) -> _Section | None:
    start, end = _trim_span(source_text, start, end)
    if start >= end:
        return None
    return _Section(
        index=index,
        title=title,
        start=start,
        end=end,
        text=source_text[start:end],
    )


def _sections_for(doc: Document) -> list[_Section]:
    text = _normalize_text(doc.page_content)
    if not text:
        return []

    lines = _iter_lines(text)
    headings = [line for line in lines if _is_heading(line.text)]
    if not headings:
        section = _make_section(
            index=0,
            title=_default_section_title(doc),
            source_text=text,
            start=0,
            end=len(text),
        )
        return [section] if section else []

    sections: list[_Section] = []
    first_heading = headings[0]
    if first_heading.start > 0 and text[:first_heading.start].strip():
        section = _make_section(
            index=len(sections),
            title=_default_section_title(doc),
            source_text=text,
            start=0,
            end=first_heading.start,
        )
        if section:
            sections.append(section)

    for i, heading in enumerate(headings):
        next_start = headings[i + 1].start if i + 1 < len(headings) else len(text)
        section = _make_section(
            index=len(sections),
            title=heading.text,
            source_text=text,
            start=heading.start,
            end=next_start,
        )
        if section:
            sections.append(section)

    return sections


def _min_chunk_chars(chunk_size: int) -> int:
    return min(
        chunk_size,
        max(
            _MIN_CHUNK_CHARS_FLOOR,
            min(_MIN_CHUNK_CHARS_CEILING, int(chunk_size * _MIN_CHUNK_RATIO)),
        ),
    )


def _group_text(source_text: str, sections: list[_Section]) -> tuple[int, int, str]:
    start, end = _trim_span(source_text, sections[0].start, sections[-1].end)
    return start, end, source_text[start:end]


def _section_groups_for(
    *,
    doc: Document,
    source_text: str,
    chunk_size: int,
) -> list[_SectionGroup]:
    sections = _sections_for(doc)
    if not sections:
        return []

    min_chars = _min_chunk_chars(chunk_size)
    groups: list[_SectionGroup] = []
    pending: list[_Section] = []

    def flush() -> None:
        if not pending:
            return
        start, end, text = _group_text(source_text, pending)
        groups.append(_SectionGroup(start=start, end=end, text=text, sections=tuple(pending)))
        pending.clear()

    for section in sections:
        if not pending:
            pending.append(section)
            if len(section.text) >= chunk_size:
                flush()
            continue

        current_start, _, current_text = _group_text(source_text, pending)
        next_start, next_end = _trim_span(source_text, current_start, section.end)
        next_text_len = len(source_text[next_start:next_end])

        if len(current_text) >= min_chars and next_text_len > chunk_size:
            flush()
            pending.append(section)
            if len(section.text) >= chunk_size:
                flush()
            continue

        pending.append(section)
        if next_text_len >= chunk_size:
            flush()

    flush()
    return groups


def _section_path(doc: Document, section_title: str) -> str:
    title = _default_section_title(doc)
    if not section_title or section_title == title:
        return title
    return f"{title} > {section_title}"


def _chunk_span(source_text: str, chunk_text: str, start_hint: int) -> tuple[int, int]:
    snippet = chunk_text.strip()
    if not snippet:
        return start_hint, start_hint
    found_at = source_text.find(snippet, start_hint)
    if found_at != -1:
        return found_at, found_at + len(snippet)
    found_at = source_text.find(snippet)
    if found_at != -1:
        return found_at, found_at + len(snippet)
    return start_hint, start_hint + len(snippet)


def _merge_candidates(
    *,
    source_text: str,
    first: _ChunkCandidate,
    second: _ChunkCandidate,
) -> _ChunkCandidate:
    start, end = _trim_span(source_text, min(first.start, second.start), max(first.end, second.end))
    return _ChunkCandidate(start=start, end=end, text=source_text[start:end])


def _coalesce_small_chunks(
    *,
    candidates: list[_ChunkCandidate],
    source_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkCandidate]:
    if len(candidates) <= 1:
        return candidates

    min_chars = _min_chunk_chars(chunk_size)
    max_chars = chunk_size + max(chunk_overlap, 0)
    remaining = candidates[:]
    out: list[_ChunkCandidate] = []
    i = 0
    while i < len(remaining):
        candidate = remaining[i]
        if len(candidate.text) >= min_chars:
            out.append(candidate)
            i += 1
            continue

        if i + 1 < len(remaining):
            merged_with_next = _merge_candidates(
                source_text=source_text,
                first=candidate,
                second=remaining[i + 1],
            )
            if len(merged_with_next.text) <= max_chars:
                remaining[i + 1] = merged_with_next
                i += 1
                continue

        if out:
            merged_with_previous = _merge_candidates(
                source_text=source_text,
                first=out[-1],
                second=candidate,
            )
            if len(merged_with_previous.text) <= max_chars:
                out[-1] = merged_with_previous
                i += 1
                continue

        out.append(candidate)
        i += 1

    return out


def _section_for_chunk(sections: list[_Section], start: int, end: int) -> _Section:
    overlapping = [
        (min(end, section.end) - max(start, section.start), section)
        for section in sections
        if section.start < end and section.end > start
    ]
    if overlapping:
        return max(overlapping, key=lambda item: item[0])[1]

    previous = [section for section in sections if section.start <= start]
    if previous:
        return previous[-1]
    return sections[0]


def _metadata_for_chunk(
    *,
    doc: Document,
    section: _Section,
    source_doc_id: str,
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    chunk_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
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
        groups = _section_groups_for(doc=doc, source_text=source_text, chunk_size=chunk_size)
        sections = [section for group in groups for section in group.sections]
        candidates: list[_ChunkCandidate] = []
        for group in groups:
            group_doc = Document(page_content=group.text, metadata=doc.metadata)
            sub_chunks = splitter.split_documents([group_doc])
            for chunk in sub_chunks:
                start_index = int(chunk.metadata.get("start_index", 0))
                chunk_start, chunk_end = _chunk_span(
                    source_text=source_text,
                    chunk_text=chunk.page_content,
                    start_hint=group.start + start_index,
                )
                candidates.append(
                    _ChunkCandidate(
                        start=chunk_start,
                        end=chunk_end,
                        text=source_text[chunk_start:chunk_end],
                    )
                )

        candidates = _coalesce_small_chunks(
            candidates=candidates,
            source_text=source_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_index, candidate in enumerate(candidates):
            section = _section_for_chunk(sections, candidate.start, candidate.end)
            out.append(
                Document(
                    page_content=candidate.text,
                    metadata=_metadata_for_chunk(
                        doc=doc,
                        section=section,
                        source_doc_id=source_doc_id,
                        chunk_text=candidate.text,
                        chunk_start=candidate.start,
                        chunk_end=candidate.end,
                        chunk_index=chunk_index,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    ),
                )
            )
    return out
