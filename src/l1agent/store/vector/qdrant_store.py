"""Production swap-in for ChromaVectorStore once a deployment needs
multi-node scale or managed hosting. Not built in M0-M4 (SPEC.md §2, §13) -
implement against the same VectorStore interface (`upsert`, `query`,
`delete_by_product`) when M5+ needs it; nothing above `store/` should need
to change.
"""
from __future__ import annotations

from l1agent.store.vector.base import VectorHit


class QdrantVectorStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "QdrantVectorStore is a production swap-in for ChromaVectorStore, "
            "not built in M0-M4. Implement against store.vector.base.VectorStore "
            "when a deployment needs multi-node scale or managed hosting."
        )

    def upsert(self, collection, ids, vectors, payloads, documents) -> None:
        raise NotImplementedError

    def query(self, collection, vector, top_k=6, where=None) -> list[VectorHit]:
        raise NotImplementedError

    def delete_by_product(self, collection, product_id) -> None:
        raise NotImplementedError
