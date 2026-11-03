"""Recursive character/markdown-header splitter (SPEC.md §5.4).

Target 400-600 tokens per chunk with 15% overlap. `section_path` is built by
tracking heading depth as the splitter walks the document, so a chunk
always knows which H1 > H2 > H3 it came from - this is what lets
`search_kb` return something a person (or the confidence gate) can
actually check against the source doc, not just a floating paragraph.

Token counting here is a whitespace-word approximation, not a real
tokenizer - good enough to hit the target chunk size band without adding a
tokenizer dependency; swap in a real tokenizer if exact token budgets
matter for a given model.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from l1agent.schemas import KBDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

TARGET_TOKENS = 500
MIN_TOKENS = 400
MAX_TOKENS = 600
OVERLAP_RATIO = 0.15


def _count_tokens(text: str) -> int:
    return len(text.split())


def _split_into_sections(text: str, doc_title: str) -> list[tuple[str, str]]:
    """Walk the document, returning [(section_path, content)] blocks."""
    lines = text.splitlines()
    stack: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []

    def flush():
        if current_lines and any(l.strip() for l in current_lines):
            path = " > ".join(stack) if stack else doc_title
            sections.append((path, list(current_lines)))
        current_lines.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
        else:
            current_lines.append(line)
    flush()

    if not sections:
        sections = [(doc_title, lines)]
    return [(path, "\n".join(content_lines).strip()) for path, content_lines in sections]


def _windowed_chunks(text: str) -> list[str]:
    """Split section text into ~TARGET_TOKENS-word windows with OVERLAP_RATIO overlap."""
    words = text.split()
    if not words:
        return []
    if len(words) <= MAX_TOKENS:
        return [text]

    step = max(1, int(TARGET_TOKENS * (1 - OVERLAP_RATIO)))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + TARGET_TOKENS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_document(
    product_id: str,
    source_uri: str,
    doc_title: str,
    text: str,
) -> list[KBDocument]:
    """Chunk one document's extracted text into KBDocument records."""
    sections = _split_into_sections(text, doc_title)
    now = datetime.now(timezone.utc)
    docs: list[KBDocument] = []
    chunk_index = 0
    for section_path, content in sections:
        if not content.strip():
            continue
        for chunk_text in _windowed_chunks(content):
            if not chunk_text.strip():
                continue
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_uri}#{chunk_index}"))
            docs.append(
                KBDocument(
                    doc_id=doc_id,
                    product_id=product_id,
                    source_uri=source_uri,
                    section_path=section_path,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=_count_tokens(chunk_text),
                    last_synced_at=now,
                )
            )
            chunk_index += 1
    return docs
