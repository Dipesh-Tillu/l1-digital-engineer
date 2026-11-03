"""Tool: classify_issue (SPEC.md §6.2, §6.3).

Input: {summary, body}. Output: {category, confidence}.

Implemented as a **plain function call, not a sub-agent turn**: one
embedding-similarity lookup against `issue_taxonomy` labels. Routing this
through another LLM call would add latency and cost without adding
accuracy, since it's a closed decision once the taxonomy exists.
"""
from __future__ import annotations

import math

from l1agent.schemas import ClassifyIssueOutput, ProductProfile
from l1agent.store.embeddings import EmbeddingProvider


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def classify_issue(
    summary: str,
    body: str,
    profile: ProductProfile,
    embedding_provider: EmbeddingProvider,
) -> ClassifyIssueOutput:
    if not profile.issue_taxonomy:
        raise ValueError(f"product {profile.product_id!r} has an empty issue_taxonomy")

    issue_vector = embedding_provider.embed_query(f"{summary}\n{body}")
    # `issue_taxonomy` entries are taxonomy paths like "billing/refunds" -
    # turn slashes into words so the label embeds closer to natural-language
    # issue text than a raw path string would.
    label_texts = [label.replace("/", " ").replace("_", " ") for label in profile.issue_taxonomy]
    label_vectors = embedding_provider.embed_batch(label_texts)

    scored = [
        (label, _cosine(issue_vector, vec))
        for label, vec in zip(profile.issue_taxonomy, label_vectors)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_label, best_score = scored[0]
    # cosine in [-1, 1] -> confidence in [0, 1]
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    return ClassifyIssueOutput(category=best_label, confidence=confidence)
