"""Local file connector - the M1 shipping connector (SPEC.md §5.2).

Handles local paths and the "uploaded files" source type from the brief:
.md/.txt read directly, .pdf via pypdf, .html via BeautifulSoup (nav/script
stripped), .csv (issue history) parsed with the column-mapping block on the
FileSourceRef, since real ticket exports rarely use the framework's own
field names.
"""
from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from l1agent.schemas import FileSourceRef, RawDocument, RawIssue, SourceRef

_DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".html", ".htm"}

_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
}


class FileKnowledgeConnector:
    def list_documents(self, source: SourceRef) -> Iterator[RawDocument]:
        assert isinstance(source, FileSourceRef)
        path = Path(source.path)
        paths: list[Path]
        if path.is_dir():
            paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in _DOC_EXTENSIONS)
        elif path.is_file():
            paths = [path]
        else:
            raise FileNotFoundError(f"kb_sources path does not exist: {path}")

        for p in paths:
            yield RawDocument(
                uri=str(p),
                title=p.stem,
                content_bytes=p.read_bytes(),
                content_type=_CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream"),
            )


class FileIssueConnector:
    def __init__(self):
        self._cursor: str | None = None

    def list_issues(self, source: SourceRef) -> Iterator[RawIssue]:
        assert isinstance(source, FileSourceRef)
        path = Path(source.path)
        summary_col = source.summary_col or "summary"
        body_col = source.body_col or "body"
        category_col = source.category_col
        resolution_col = source.resolution_col
        outcome_col = source.outcome_col

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                issue_id = (
                    row.get("issue_id")
                    or row.get("Id")
                    or row.get("ID")
                    or str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{reader.line_num}"))
                )
                category = row.get(category_col) if category_col else None
                resolution_body = row.get(resolution_col) if resolution_col else None
                outcome = row.get(outcome_col) if outcome_col else None
                if resolution_body and not outcome:
                    outcome = "accepted"
                yield RawIssue(
                    issue_id=str(issue_id),
                    summary=(row.get(summary_col) or "").strip(),
                    body=(row.get(body_col) or "").strip(),
                    category=(category or None),
                    resolution_body=(resolution_body or None),
                    outcome=(outcome or None),
                    source_channel="manual_import",
                    created_at=datetime.now(timezone.utc),
                )
        self._cursor = datetime.now(timezone.utc).isoformat()

    def sync_cursor(self) -> str | None:
        return self._cursor


def extract_text(doc: RawDocument) -> str:
    """Content-type-aware plain-text extraction, used by chunking.py."""
    if doc.content_type == "text/markdown" or doc.content_type == "text/plain":
        return doc.content_bytes.decode("utf-8", errors="replace")
    if doc.content_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(doc.content_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if doc.content_type == "text/html":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(doc.content_bytes.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["nav", "script", "style", "header", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    raise ValueError(f"unsupported content_type for text extraction: {doc.content_type}")
