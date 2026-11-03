"""Runtime wiring shared by agent tools, the API, and the CLI.

Not named as a module in SPEC.md's repo layout, but needed as glue: the
spec's tools are plain functions/plain-function-tools bound to "the running
session's product" (§6.2's note on get_product_profile). `AppContext` is
that binding - which embedding provider, vector store, and DB session
factory a given call should use - so the swap-out interfaces
(EmbeddingProvider, VectorStore, DATABASE_URL) stay real interfaces instead
of hardcoded imports scattered through agent/tools/*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from l1agent.schemas import LLMTrace
from l1agent.store.embeddings import EmbeddingProvider, get_default_provider
from l1agent.store.relational import get_session_factory
from l1agent.store.vector.base import VectorStore
from l1agent.store.vector.chroma_store import ChromaVectorStore


@dataclass
class AppContext:
    embedding_provider: EmbeddingProvider = field(default_factory=get_default_provider)
    vector_store: VectorStore = field(default_factory=ChromaVectorStore)
    database_url: str | None = None

    def session_factory(self):
        return get_session_factory(self.database_url)


@dataclass
class TurnContext:
    """Per-ticket-handling-turn state, populated by runner.handle_issue as it
    calls search_kb/search_issues, then consumed by score_confidence so the
    gate checks citations against what was *actually* retrieved this turn -
    never trusted from the model's own claim (SPEC.md §7, citation gate)."""

    product_id: str
    category: str | None = None
    kb_hit_ids: set[str] = field(default_factory=set)
    issue_hit_ids: set[str] = field(default_factory=set)
    accepted_issue_hit_ids: set[str] = field(default_factory=set)
    scores_by_id: dict[str, float] = field(default_factory=dict)
    # Demo workflow addendum (per user request): the real LLM call(s)
    # draft_response made this turn - set by the real provider backends
    # (draft_response.py), left None by the fake agent_call tests inject.
    llm_trace: LLMTrace | None = None

    def record_kb_hits(self, hits) -> None:
        for h in hits:
            self.kb_hit_ids.add(h.doc_id)
            self.scores_by_id[h.doc_id] = max(self.scores_by_id.get(h.doc_id, 0.0), h.score)

    def record_issue_hits(self, hits) -> None:
        for h in hits:
            self.issue_hit_ids.add(h.issue_id)
            self.scores_by_id[h.issue_id] = max(self.scores_by_id.get(h.issue_id, 0.0), h.score)
            if h.outcome == "accepted":
                self.accepted_issue_hit_ids.add(h.issue_id)
