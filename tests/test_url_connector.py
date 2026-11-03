from __future__ import annotations

import pytest

from l1agent.ingestion.connectors import url_connector as uc
from l1agent.schemas import UrlSourceRef


class _FakeResponse:
    def __init__(self, *, content: bytes = b"", json_body=None, headers=None):
        self.content = content
        self._json_body = json_body
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_body


def test_list_documents_fetches_one_resource_and_resolves_content_type(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        calls.append(url)
        return _FakeResponse(content=b"# Hello\n\nworld", headers={"content-type": "text/markdown; charset=utf-8"})

    monkeypatch.setattr(uc.httpx, "get", fake_get)

    source = UrlSourceRef(url="https://example.com/doc.md")
    docs = list(uc.UrlKnowledgeConnector().list_documents(source))

    assert len(docs) == 1
    assert docs[0].content_type == "text/markdown"
    assert docs[0].content_bytes == b"# Hello\n\nworld"
    assert calls == ["https://example.com/doc.md"]


def test_list_documents_format_override_wins_over_response_header(monkeypatch):
    monkeypatch.setattr(
        uc.httpx, "get",
        lambda url, headers=None, timeout=None, follow_redirects=None: _FakeResponse(
            content=b"<html><body>hi</body></html>", headers={"content-type": "application/octet-stream"}
        ),
    )
    source = UrlSourceRef(url="https://example.com/doc", format="html")
    docs = list(uc.UrlKnowledgeConnector().list_documents(source))
    assert docs[0].content_type == "text/html"


def test_list_issues_maps_fields_and_follows_pagination(monkeypatch):
    pages = {
        "https://api.example.com/tickets?page=1": {
            "data": {"tickets": [{"id": "1", "subject": "Refund", "desc": "help", "resolution": "done"}]},
            "next": "https://api.example.com/tickets?page=2",
        },
        "https://api.example.com/tickets?page=2": {
            "data": {"tickets": [{"id": "2", "subject": "Broken", "desc": "help2"}]},
            "next": None,
        },
    }

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        return _FakeResponse(json_body=pages[url])

    monkeypatch.setattr(uc.httpx, "get", fake_get)

    source = UrlSourceRef(
        url="https://api.example.com/tickets?page=1",
        json_records_path="data.tickets",
        next_page_path="next",
        summary_col="subject",
        body_col="desc",
        resolution_col="resolution",
    )
    issues = list(uc.UrlIssueConnector().list_issues(source))

    assert [i.issue_id for i in issues] == ["1", "2"]
    assert issues[0].summary == "Refund"
    assert issues[0].resolution_body == "done"
    assert issues[0].outcome == "accepted"  # inferred: resolution present, outcome_col unset
    assert issues[1].resolution_body is None


def test_list_issues_respects_max_pages(monkeypatch):
    call_count = {"n": 0}

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        call_count["n"] += 1
        return _FakeResponse(json_body={"tickets": [{"id": str(call_count["n"]), "summary": "s", "body": "b"}], "next": url})

    monkeypatch.setattr(uc.httpx, "get", fake_get)

    source = UrlSourceRef(url="https://api.example.com/t", json_records_path="tickets", next_page_path="next", max_pages=3)
    issues = list(uc.UrlIssueConnector().list_issues(source))

    assert call_count["n"] == 3
    assert len(issues) == 3


def test_auth_ref_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("SOME_TOKEN_ENV", raising=False)
    source = UrlSourceRef(url="https://example.com/doc", auth_ref="SOME_TOKEN_ENV")
    with pytest.raises(RuntimeError):
        list(uc.UrlKnowledgeConnector().list_documents(source))


def test_auth_ref_sends_bearer_header(monkeypatch):
    monkeypatch.setenv("SOME_TOKEN_ENV", "secret-token")
    seen_headers = {}

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        seen_headers.update(headers or {})
        return _FakeResponse(content=b"hi", headers={"content-type": "text/plain"})

    monkeypatch.setattr(uc.httpx, "get", fake_get)
    source = UrlSourceRef(url="https://example.com/doc", auth_ref="SOME_TOKEN_ENV")
    list(uc.UrlKnowledgeConnector().list_documents(source))
    assert seen_headers.get("Authorization") == "Bearer secret-token"
