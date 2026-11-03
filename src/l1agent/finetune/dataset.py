"""Builds fine-tuning training examples from a product's own already-
ingested history.

Every example is retrieval-augmented the same way a live `draft_response`
call is - persona + this turn's actually-retrieved passages in, the
historical human resolution out - so the fine-tuned model learns to answer
well *given* retrieved context, never to answer from memorized facts
alone. This is what keeps the confidence gate's citation check meaningful
even after fine-tuning: facts still have to come from retrieval at
inference time; fine-tuning only shapes persona/tone/response structure.

Historical resolutions don't carry machine-checkable citations (they're
just human-written text), so the `CITED_SOURCES: [...]` line appended to
each training target is a heuristic - the top-scoring retrieved hit(s) at
generation time - not a claim about what the original human actually used.
Its purpose is teaching the citation-emitting *format*; `draft_response`'s
own gate (via `gate/confidence.py`) is what verifies real citations at
inference time regardless of what the model learned here.
"""
from __future__ import annotations

import json

from l1agent.agent.options import render_persona
from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.draft_response import render_context_prompt
from l1agent.agent.tools.search_issues import search_issues
from l1agent.agent.tools.search_kb import search_kb
from l1agent.ingestion.connectors.base import resolve_issue_connector
from l1agent.schemas import IssueRecord, IssueStatus, ProductProfile, TrainingExample
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.vector.base import VectorStore


def build_training_examples(
    profile: ProductProfile,
    role: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    top_k_kb: int = 6,
    top_k_issues: int = 5,
) -> list[TrainingExample]:
    """Re-reads `profile.issue_sources` (the same connectors ingestion used)
    rather than querying the relational DB or vector store for historical
    text, since only the original source data carries the resolution body
    needed as a training target - the relational IssueRecord and vector
    store payloads don't persist it as a first-class field."""
    examples: list[TrainingExample] = []

    for source in profile.issue_sources:
        connector = resolve_issue_connector(source)
        for raw in connector.list_issues(source):
            if not raw.resolution_body or not raw.resolution_body.strip():
                continue  # only resolved historical tickets make usable training targets

            category = raw.category or classify_issue(raw.summary, raw.body, profile, embedding_provider).category
            if profile.tier_for(category).role != role:
                continue

            issue = IssueRecord(
                issue_id=raw.issue_id,
                product_id=profile.product_id,
                summary=raw.summary,
                body=raw.body,
                category=category,
                status=IssueStatus.resolved,
                source_channel=raw.source_channel,
                created_at=raw.created_at,
                updated_at=raw.created_at,
            )

            query = f"{issue.summary}\n{issue.body}"
            kb_hits = search_kb(
                query, profile.product_id, embedding_provider, vector_store,
                top_k=top_k_kb, category=category,
            )
            # Fetch one extra and drop the issue's own precedent to avoid
            # trivially teaching the model to copy its own answer.
            raw_issue_hits = search_issues(
                query, profile.product_id, embedding_provider, vector_store, top_k=top_k_issues + 1
            )
            issue_hits = [h for h in raw_issue_hits if h.issue_id != raw.issue_id][:top_k_issues]

            cited: list[str] = []
            if kb_hits:
                cited.append(kb_hits[0].doc_id)
            elif issue_hits:
                cited.append(issue_hits[0].issue_id)

            assistant = raw.resolution_body.strip()
            if cited:
                assistant += f"\n\nCITED_SOURCES: {json.dumps(cited)}"

            examples.append(
                TrainingExample(
                    system=render_persona(profile, role),
                    user=render_context_prompt(issue, kb_hits, issue_hits),
                    assistant=assistant,
                    role=role,
                    issue_id=raw.issue_id,
                )
            )

    return examples
