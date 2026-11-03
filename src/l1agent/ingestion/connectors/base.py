"""Connector interfaces. See SPEC.md §5.1.

A connector registry resolves a `SourceRef` (tagged union: file | wiki) to
a concrete `KnowledgeConnector` / `IssueConnector` implementation.
"""
from __future__ import annotations

from typing import Iterator, Protocol

from l1agent.schemas import RawDocument, RawIssue, SourceRef


class KnowledgeConnector(Protocol):
    def list_documents(self, source: SourceRef) -> Iterator[RawDocument]: ...


class IssueConnector(Protocol):
    def list_issues(self, source: SourceRef) -> Iterator[RawIssue]: ...

    def sync_cursor(self) -> str | None:
        """Watermark for incremental sync (e.g. last-modified). None means
        the connector only supports full re-index (true of the file connector)."""
        ...


_KB_CONNECTORS: dict[str, KnowledgeConnector] = {}
_ISSUE_CONNECTORS: dict[str, IssueConnector] = {}


def register_kb_connector(source_type: str, connector: KnowledgeConnector) -> None:
    _KB_CONNECTORS[source_type] = connector


def register_issue_connector(source_type: str, connector: IssueConnector) -> None:
    _ISSUE_CONNECTORS[source_type] = connector


def _registry_key(source: SourceRef) -> str:
    # "wiki" is the only type that multiplexes to multiple named connectors
    # (confluence/notion/sharepoint); every other type is its own key.
    return source.connector if source.type == "wiki" else source.type


def resolve_kb_connector(source: SourceRef) -> KnowledgeConnector:
    key = _registry_key(source)
    try:
        return _KB_CONNECTORS[key]
    except KeyError as e:
        raise ValueError(f"no KnowledgeConnector registered for source type {key!r}") from e


def resolve_issue_connector(source: SourceRef) -> IssueConnector:
    key = _registry_key(source)
    try:
        return _ISSUE_CONNECTORS[key]
    except KeyError as e:
        raise ValueError(f"no IssueConnector registered for source type {key!r}") from e
