"""URL/API connector - the "live source" ingestion path alongside local
files and (M5) wiki spaces. Fetches one resource per `UrlSourceRef` entry
(not a crawler); add multiple kb_sources/issue_sources entries for
multiple pages/endpoints.

Docs: one HTTP GET, `content_type` resolved from `format` or the
response's Content-Type header; downstream `extract_text()` in
file_connector.py handles the rest generically (no duplicated parsing).

Tickets: GETs a JSON API response, walks `json_records_path` (dotted keys,
e.g. "data.tickets") to the record list, maps fields via the same
summary_col/body_col/... convention FileSourceRef's CSV mapping uses, and
follows `next_page_path` for simple cursor-style pagination up to
`max_pages`.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Iterator

import httpx

from l1agent.schemas import RawDocument, RawIssue, SourceRef, UrlSourceRef

_FORMAT_CONTENT_TYPES = {
    "html": "text/html",
    "markdown": "text/markdown",
    "text": "text/plain",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "json": "application/json",
}


def _auth_headers(source: UrlSourceRef) -> dict[str, str]:
    if not source.auth_ref:
        return {}
    token = os.environ.get(source.auth_ref)
    if not token:
        raise RuntimeError(
            f"auth_ref {source.auth_ref!r} is not set in the environment (SourceRef.auth_ref "
            "names an env var holding a bearer token, never a literal credential)"
        )
    return {"Authorization": f"Bearer {token}"}


def _resolve_content_type(source: UrlSourceRef, response_content_type: str) -> str:
    if source.format != "auto":
        return _FORMAT_CONTENT_TYPES.get(source.format, "text/plain")
    base = response_content_type.split(";")[0].strip().lower()
    return base or "text/plain"


def _walk_path(obj, dotted_path: str | None):
    if not dotted_path:
        return obj
    cur = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class UrlKnowledgeConnector:
    def list_documents(self, source: SourceRef) -> Iterator[RawDocument]:
        assert isinstance(source, UrlSourceRef)
        resp = httpx.get(source.url, headers=_auth_headers(source), timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        content_type = _resolve_content_type(source, resp.headers.get("content-type", ""))
        yield RawDocument(uri=source.url, title=source.url, content_bytes=resp.content, content_type=content_type)


class UrlIssueConnector:
    def __init__(self):
        self._cursor: str | None = None

    def list_issues(self, source: SourceRef) -> Iterator[RawIssue]:
        assert isinstance(source, UrlSourceRef)
        summary_col = source.summary_col or "summary"
        body_col = source.body_col or "body"

        url: str | None = source.url
        pages = 0
        while url and pages < source.max_pages:
            resp = httpx.get(url, headers=_auth_headers(source), timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            payload = resp.json()
            records = _walk_path(payload, source.json_records_path) or []

            for i, rec in enumerate(records):
                issue_id = rec.get("id") or rec.get("issue_id") or str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{i}")
                )
                resolution_body = rec.get(source.resolution_col) if source.resolution_col else None
                outcome = rec.get(source.outcome_col) if source.outcome_col else None
                if resolution_body and not outcome:
                    outcome = "accepted"
                yield RawIssue(
                    issue_id=str(issue_id),
                    summary=str(rec.get(summary_col) or "").strip(),
                    body=str(rec.get(body_col) or "").strip(),
                    category=(rec.get(source.category_col) if source.category_col else None) or None,
                    resolution_body=resolution_body or None,
                    outcome=outcome or None,
                    source_channel="manual_import",
                    created_at=datetime.now(timezone.utc),
                )

            self._cursor = url
            url = _walk_path(payload, source.next_page_path) if source.next_page_path else None
            pages += 1

    def sync_cursor(self) -> str | None:
        return self._cursor
