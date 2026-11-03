"""VectorStore interface. See SPEC.md §2, §4.6.

One collection per product per index: `kb__{product_id}` and
`issues__{product_id}` - a hard namespace boundary so a query can never
cross products (see the cross-product-leakage risk in the blueprint, and
tests/test_second_product_cold_start.py).
"""
from __future__ import annotations

from typing import Protocol, TypedDict


class VectorHit(TypedDict):
    id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        documents: list[str],
    ) -> None: ...

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 6,
        where: dict | None = None,
    ) -> list[VectorHit]: ...

    def delete_by_product(self, collection: str, product_id: str) -> None: ...


def kb_collection_name(product_id: str) -> str:
    return f"kb__{product_id}"


def issues_collection_name(product_id: str) -> str:
    return f"issues__{product_id}"
