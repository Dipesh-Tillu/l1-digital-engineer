"""Chroma-backed VectorStore. Local, embedded, on-disk - zero infra to stand
up, which is what makes the "plug and play against real product data"
prototype goal work on a laptop. See SPEC.md §2.
"""
from __future__ import annotations

import os

import chromadb

from l1agent.store.vector.base import VectorHit


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
        self._client = chromadb.PersistentClient(path=self.persist_dir)

    def _get_collection(self, collection: str):
        return self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        documents: list[str],
    ) -> None:
        if not ids:
            return
        col = self._get_collection(collection)
        col.upsert(ids=ids, embeddings=vectors, metadatas=payloads, documents=documents)

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 6,
        where: dict | None = None,
    ) -> list[VectorHit]:
        col = self._get_collection(collection)
        if col.count() == 0:
            return []
        result = col.query(
            query_embeddings=[vector],
            n_results=min(top_k, col.count()),
            where=where,
        )
        hits: list[VectorHit] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        for i, doc_id in enumerate(ids):
            distance = distances[i]
            # cosine distance -> similarity score in [0, 1] (approx: 1 - distance/2)
            score = max(0.0, 1.0 - distance / 2.0)
            meta = dict(metadatas[i] or {})
            meta["_document"] = documents[i]
            hits.append(VectorHit(id=doc_id, score=score, metadata=meta))
        return hits

    def delete_by_product(self, collection: str, product_id: str) -> None:
        try:
            col = self._get_collection(collection)
        except Exception:
            return
        col.delete(where={"product_id": product_id})
