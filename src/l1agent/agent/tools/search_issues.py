"""Tool: search_issues (SPEC.md §6.2).

Input: {query, top_k=5}. Output: {hits: [{issue_id, summary,
resolution_body, score, outcome}]}.
"""
from __future__ import annotations

from l1agent.schemas import IssueHit
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.vector.base import VectorStore, issues_collection_name


def search_issues(
    query: str,
    product_id: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 5,
) -> list[IssueHit]:
    vector = embedding_provider.embed_query(query)
    where = {"product_id": product_id}
    raw_hits = vector_store.query(issues_collection_name(product_id), vector, top_k=top_k, where=where)
    hits: list[IssueHit] = []
    for h in raw_hits:
        meta = h["metadata"]
        hits.append(
            IssueHit(
                issue_id=meta["issue_id"],
                summary=meta.get("summary", ""),
                resolution_body=meta.get("resolution_body", ""),
                score=h["score"],
                outcome=meta.get("outcome", "pending"),
            )
        )
    return hits
