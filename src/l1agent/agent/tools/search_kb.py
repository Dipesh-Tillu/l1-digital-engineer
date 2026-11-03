"""Tool: search_kb (SPEC.md §6.2).

Input: {query, top_k=6, category?}. Output: {hits: [{doc_id, section_path,
text, score, source_uri}]}.

Exposed to the model during draft_response's inner agent loop (so it can
re-query if the first retrieval was thin) *and* called directly by
runner.py's deterministic orchestration - same underlying function either
way, per SPEC.md §6.3's note that only draft_response's prose generation
needs the model.
"""
from __future__ import annotations

from l1agent.schemas import KBHit
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.vector.base import VectorStore, kb_collection_name


def search_kb(
    query: str,
    product_id: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 6,
    category: str | None = None,
) -> list[KBHit]:
    vector = embedding_provider.embed_query(query)
    where: dict = {"product_id": product_id}
    if category:
        # category="" (untagged docs) always matches; a tagged doc must match exactly.
        where = {"$and": [{"product_id": product_id}, {"$or": [{"category": category}, {"category": ""}]}]}

    raw_hits = vector_store.query(kb_collection_name(product_id), vector, top_k=top_k, where=where)
    hits: list[KBHit] = []
    for h in raw_hits:
        meta = h["metadata"]
        hits.append(
            KBHit(
                doc_id=meta["doc_id"],
                section_path=meta.get("section_path", ""),
                text=meta.get("_document", ""),
                score=h["score"],
                source_uri=meta.get("source_uri", ""),
            )
        )
    return hits
