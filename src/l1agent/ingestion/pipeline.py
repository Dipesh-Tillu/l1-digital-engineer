"""Orchestrates connector -> chunk -> embed -> upsert (SPEC.md §5.5).

`run_ingestion(profile)`: connector -> chunk -> EmbeddingProvider.embed_batch()
(batched at 128) -> VectorStore.upsert(collection, ids, vectors, payloads) ->
mirror doc_id/source_uri/section_path into kb_document_meta. Issue ingestion
is the same shape against issues__{product_id}, additionally calling
classify_issue for any row missing a category.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from l1agent.ingestion.connectors.base import (
    resolve_issue_connector,
    resolve_kb_connector,
)
from l1agent.ingestion.connectors.file_connector import (
    FileIssueConnector,
    FileKnowledgeConnector,
    extract_text,
)
from l1agent.ingestion.connectors.base import register_issue_connector, register_kb_connector
from l1agent.ingestion.connectors.wiki_connector import (
    ConfluenceIssueConnector,
    ConfluenceKnowledgeConnector,
)
from l1agent.ingestion.connectors.url_connector import UrlIssueConnector, UrlKnowledgeConnector
from l1agent.ingestion.chunking import chunk_document
from l1agent.schemas import IssueRecord, IssueStatus, ProductProfile
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.relational import upsert_issue, upsert_kb_document_meta
from l1agent.store.vector.base import VectorStore, issues_collection_name, kb_collection_name

# Register the M1 file connectors on import so `run_ingestion` works out of the box.
register_kb_connector("file", FileKnowledgeConnector())
register_issue_connector("file", FileIssueConnector())
# Wiki is an interface-only stub until M5 (SPEC.md §5.3) - registered so a
# wiki-typed source resolves to a clear NotImplementedError instead of a KeyError.
register_kb_connector("confluence", ConfluenceKnowledgeConnector())
register_issue_connector("confluence", ConfluenceIssueConnector())
# URL/API connector: a live source alongside local files (SPEC.md-adjacent).
register_kb_connector("url", UrlKnowledgeConnector())
register_issue_connector("url", UrlIssueConnector())


@dataclass
class IngestionReport:
    kb_chunks_indexed: int
    issues_indexed: int


_FRONTMATTER_CATEGORY_RE = re.compile(r"^categor(?:y|ies):\s*(.+)\s*$", re.IGNORECASE)


def _extract_frontmatter_category(text: str) -> tuple[str | None, str]:
    """A leading `category: <taxonomy-path>` line lets a doc opt into
    category-scoped retrieval (§4.6's metadata.category); everything else
    stays product-wide (category=None, matched regardless of the caller's
    `category` filter)."""
    lines = text.splitlines()
    if lines:
        m = _FRONTMATTER_CATEGORY_RE.match(lines[0].strip())
        if m:
            return m.group(1).strip(), "\n".join(lines[1:]).lstrip("\n")
    return None, text


def ingest_kb(
    profile: ProductProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session: Session,
) -> int:
    collection = kb_collection_name(profile.product_id)
    total = 0
    for source in profile.kb_sources:
        connector = resolve_kb_connector(source)
        for raw_doc in connector.list_documents(source):
            text = extract_text(raw_doc)
            category, text = _extract_frontmatter_category(text)
            chunks = chunk_document(profile.product_id, raw_doc.uri, raw_doc.title, text)
            if not chunks:
                continue
            vectors = embedding_provider.embed_batch([c.chunk_text for c in chunks])
            ids = [c.doc_id for c in chunks]
            payloads = [
                {
                    "product_id": c.product_id,
                    "doc_id": c.doc_id,
                    "section_path": c.section_path,
                    "source_uri": c.source_uri,
                    "category": category or "",
                }
                for c in chunks
            ]
            documents = [c.chunk_text for c in chunks]
            vector_store.upsert(collection, ids, vectors, payloads, documents)
            for c in chunks:
                upsert_kb_document_meta(
                    session, c.doc_id, c.product_id, c.source_uri, c.section_path, c.last_synced_at
                )
            total += len(chunks)
    return total


def ingest_issues(
    profile: ProductProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session: Session,
    classify_fn,
) -> int:
    """`classify_fn(summary, body, profile) -> ClassifyIssueOutput`, called
    only for rows missing a category on intake (SPEC.md §5.5)."""
    collection = issues_collection_name(profile.product_id)
    total = 0
    for source in profile.issue_sources:
        connector = resolve_issue_connector(source)
        raw_issues = list(connector.list_issues(source))
        if not raw_issues:
            continue

        texts = [f"{ri.summary}\n{ri.body}" for ri in raw_issues]
        vectors = embedding_provider.embed_batch(texts)

        ids: list[str] = []
        payloads: list[dict] = []
        documents: list[str] = []
        for ri, text, vector in zip(raw_issues, texts, vectors):
            category = ri.category or classify_fn(ri.summary, ri.body, profile).category
            outcome = ri.outcome or "pending"
            status = IssueStatus.resolved if ri.resolution_body else IssueStatus.open

            issue = IssueRecord(
                issue_id=ri.issue_id,
                product_id=profile.product_id,
                summary=ri.summary,
                body=ri.body,
                category=category,
                status=status,
                source_channel=ri.source_channel,
                created_at=ri.created_at,
                updated_at=ri.created_at,
            )
            upsert_issue(session, issue)

            ids.append(ri.issue_id)
            payloads.append(
                {
                    "product_id": profile.product_id,
                    "issue_id": ri.issue_id,
                    "category": category,
                    "outcome": outcome,
                    "summary": ri.summary,
                    "resolution_body": ri.resolution_body or "",
                }
            )
            documents.append(text)
            total += 1

        vector_store.upsert(collection, ids, vectors, payloads, documents)
    return total


def run_ingestion(
    profile: ProductProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session: Session,
    classify_fn,
) -> IngestionReport:
    kb_count = ingest_kb(profile, embedding_provider, vector_store, session)
    issue_count = ingest_issues(profile, embedding_provider, vector_store, session, classify_fn)
    return IngestionReport(kb_chunks_indexed=kb_count, issues_indexed=issue_count)
