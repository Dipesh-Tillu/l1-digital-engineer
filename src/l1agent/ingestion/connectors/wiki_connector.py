"""Wiki connector - interface now, Confluence implementation in M5
(SPEC.md §5.3).

`list_documents` paginates the space; `sync_cursor` stores the space's
last-modified watermark so re-syncs are incremental, not full re-index.
`auth_ref` resolves to a secret name (env var / secrets store), never a
literal credential, in ProductProfile.
"""
from __future__ import annotations

from typing import Iterator

from l1agent.schemas import RawDocument, RawIssue, SourceRef, WikiSourceRef


class ConfluenceKnowledgeConnector:
    """Stub - implement in M5. See SPEC.md §14 for the open question on
    which wiki (Confluence/Notion/SharePoint) the pilot product actually uses."""

    def list_documents(self, source: SourceRef) -> Iterator[RawDocument]:
        assert isinstance(source, WikiSourceRef)
        raise NotImplementedError(
            "ConfluenceKnowledgeConnector is an M5 deliverable (SPEC.md §5.3, §13). "
            f"space_key={source.space_key!r} auth_ref={source.auth_ref!r}"
        )


class ConfluenceIssueConnector:
    def list_issues(self, source: SourceRef) -> Iterator[RawIssue]:
        raise NotImplementedError("Wiki spaces are not an issue source; see IssueConnector usage.")

    def sync_cursor(self) -> str | None:
        return None
