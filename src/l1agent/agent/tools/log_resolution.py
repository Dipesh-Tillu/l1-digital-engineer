"""Tool: log_resolution (SPEC.md §6.2, §1's feedback loop layer).

Input: {resolution}. Output: {ok, indexed}.

Always persists the resolution relationally. Once a resolution's outcome is
"accepted" (an L2 human confirmed it, or a later reconciliation step marks
an agent auto-resolution as held), it is additionally embedded and upserted
into the issues vector store as a new precedent case for future
`search_issues` calls - this is the feedback loop the blueprint describes.
A "pending" resolution (the common case right after `handle_issue` runs) is
logged but not yet indexed as precedent.

`finalize_resolution_outcome` (demo workflow addendum, spec §3/§5) is the
other half of that loop: the demo's accept/reject flow doesn't know the
final outcome at the moment a draft is logged (it's "pending" until the
requester clicks Accept/Reject), so it updates the already-persisted row's
outcome in place and - on acceptance - runs the same indexing tail as
above, instead of inserting a second row for the same attempt.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from l1agent.schemas import LogResolutionOutput, ResolutionOutcome, ResolutionRecord
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.relational import get_issue, insert_resolution, update_resolution_outcome
from l1agent.store.vector.base import VectorStore, issues_collection_name


def _index_as_precedent(
    resolution: ResolutionRecord,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> LogResolutionOutput:
    issue = get_issue(session, resolution.issue_id)
    if issue is None:
        return LogResolutionOutput(ok=True, indexed=False)

    text = f"{issue.summary}\n{issue.body}"
    vector = embedding_provider.embed_query(text)
    vector_store.upsert(
        issues_collection_name(issue.product_id),
        [issue.issue_id],
        [vector],
        [{
            "product_id": issue.product_id,
            "issue_id": issue.issue_id,
            "category": issue.category or "",
            "outcome": "accepted",
            "summary": issue.summary,
            "resolution_body": resolution.body,
        }],
        [text],
    )
    return LogResolutionOutput(ok=True, indexed=True)


def log_resolution(
    resolution: ResolutionRecord,
    session: Session,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> LogResolutionOutput:
    insert_resolution(session, resolution)

    if resolution.outcome != ResolutionOutcome.accepted or embedding_provider is None or vector_store is None:
        return LogResolutionOutput(ok=True, indexed=False)

    return _index_as_precedent(resolution, session, embedding_provider, vector_store)


def finalize_resolution_outcome(
    resolution_id: str,
    outcome: ResolutionOutcome,
    session: Session,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    rejection_reason: str | None = None,
) -> LogResolutionOutput:
    resolution = update_resolution_outcome(session, resolution_id, outcome, rejection_reason)
    if resolution is None:
        return LogResolutionOutput(ok=False, indexed=False)

    if outcome != ResolutionOutcome.accepted or embedding_provider is None or vector_store is None:
        return LogResolutionOutput(ok=True, indexed=False)

    return _index_as_precedent(resolution, session, embedding_provider, vector_store)
