"""Evaluation harness (SPEC.md §10).

Replays a dataset of historical IssueRecords (held-out slice, never used for
indexing) through the same retrieval/draft/gate steps `handle_issue` uses,
without the side-effecting steps (`update_ticket`/`escalate_to_l2`/
`log_resolution` are simply not called - equivalent to stubbing them to
no-ops, and it keeps replay from mutating the DB it's evaluating against).

Metrics computed (§10.2):
- auto-resolution rate: share of cases the gate passed.
- draft/actual similarity: on passed cases, cosine similarity between the
  drafted body and the historical resolution body - a proxy for reopen
  risk, since true reopens don't exist in a replay.
- escalation precision proxy: share of gate-failed cases where the original
  human resolution looks like it took materially more than a single reply
  (reopened, or long) - a rough stand-in until live data exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.draft_response import AgentCallFn, draft_response
from l1agent.agent.tools.score_confidence import score_confidence
from l1agent.agent.tools.search_issues import search_issues
from l1agent.agent.tools.search_kb import search_kb
from l1agent.context import TurnContext
from l1agent.ingestion.connectors.file_connector import FileIssueConnector
from l1agent.schemas import FileSourceRef, IssueRecord, IssueStatus, ProductProfile
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.vector.base import VectorStore

COMPLEX_RESOLUTION_WORD_THRESHOLD = 120


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


@dataclass
class ReplayCase:
    issue: IssueRecord
    actual_resolution_body: str
    actual_outcome: str  # "accepted" | "reopened" | "pending"


@dataclass
class ReplayResult:
    issue_id: str
    category: str
    passed: bool
    score: float
    draft_actual_similarity: float | None
    actual_outcome: str
    actual_word_count: int


@dataclass
class CategoryMetrics:
    category: str
    total: int = 0
    auto_resolved: int = 0
    escalated: int = 0
    similarity_sum: float = 0.0
    similarity_count: int = 0
    escalation_correct: int = 0

    @property
    def auto_resolution_rate(self) -> float:
        return self.auto_resolved / self.total if self.total else 0.0

    @property
    def avg_draft_actual_similarity(self) -> float | None:
        return self.similarity_sum / self.similarity_count if self.similarity_count else None

    @property
    def escalation_precision_proxy(self) -> float | None:
        return self.escalation_correct / self.escalated if self.escalated else None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total": self.total,
            "auto_resolved": self.auto_resolved,
            "escalated": self.escalated,
            "auto_resolution_rate": self.auto_resolution_rate,
            "avg_draft_actual_similarity": self.avg_draft_actual_similarity,
            "escalation_precision_proxy": self.escalation_precision_proxy,
        }


@dataclass
class EvalReport:
    product_id: str
    total_cases: int
    auto_resolution_rate: float
    avg_draft_actual_similarity: float | None
    escalation_precision_proxy: float | None
    by_category: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "total_cases": self.total_cases,
            "auto_resolution_rate": self.auto_resolution_rate,
            "avg_draft_actual_similarity": self.avg_draft_actual_similarity,
            "escalation_precision_proxy": self.escalation_precision_proxy,
            "by_category": self.by_category,
        }


def load_replay_cases(profile: ProductProfile, dataset_path: str) -> list[ReplayCase]:
    template = next((s for s in profile.issue_sources if s.type == "file"), None)
    if template is None:
        raise ValueError(f"product {profile.product_id!r} has no file-type issue_sources to model the dataset's columns on")
    assert isinstance(template, FileSourceRef)
    source = template.model_copy(update={"path": dataset_path})

    connector = FileIssueConnector()
    cases: list[ReplayCase] = []
    for raw in connector.list_issues(source):
        issue = IssueRecord(
            issue_id=raw.issue_id,
            product_id=profile.product_id,
            summary=raw.summary,
            body=raw.body,
            category=raw.category,
            status=IssueStatus.resolved if raw.resolution_body else IssueStatus.open,
            source_channel=raw.source_channel,
            created_at=raw.created_at,
            updated_at=raw.created_at,
        )
        cases.append(
            ReplayCase(
                issue=issue,
                actual_resolution_body=raw.resolution_body or "",
                actual_outcome=raw.outcome or "pending",
            )
        )
    return cases


def replay_one(
    case: ReplayCase,
    profile: ProductProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    agent_call: AgentCallFn | None = None,
) -> ReplayResult:
    issue = case.issue.model_copy()
    if not issue.category:
        issue.category = classify_issue(issue.summary, issue.body, profile, embedding_provider).category

    query = f"{issue.summary}\n{issue.body}"
    kb_hits = search_kb(query, profile.product_id, embedding_provider, vector_store, category=issue.category)
    issue_hits = search_issues(query, profile.product_id, embedding_provider, vector_store)

    turn = TurnContext(product_id=profile.product_id, category=issue.category)
    turn.record_kb_hits(kb_hits)
    turn.record_issue_hits(issue_hits)

    draft = draft_response(issue, kb_hits, issue_hits, profile, turn, agent_call=agent_call)
    gate = score_confidence(draft.body, draft.cited_sources, issue.category, profile, turn)

    similarity = None
    if gate.passed and case.actual_resolution_body:
        draft_vec = embedding_provider.embed_query(draft.body)
        actual_vec = embedding_provider.embed_query(case.actual_resolution_body)
        similarity = _cosine(draft_vec, actual_vec)

    return ReplayResult(
        issue_id=issue.issue_id,
        category=issue.category or "unclassified",
        passed=gate.passed,
        score=gate.score,
        draft_actual_similarity=similarity,
        actual_outcome=case.actual_outcome,
        actual_word_count=len(case.actual_resolution_body.split()),
    )


def replay(
    profile: ProductProfile,
    dataset_path: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    agent_call: AgentCallFn | None = None,
) -> EvalReport:
    cases = load_replay_cases(profile, dataset_path)
    by_category: dict[str, CategoryMetrics] = {}
    overall = CategoryMetrics(category="*")

    for case in cases:
        result = replay_one(case, profile, embedding_provider, vector_store, agent_call=agent_call)
        metrics = by_category.setdefault(result.category, CategoryMetrics(category=result.category))
        for m in (metrics, overall):
            m.total += 1
            if result.passed:
                m.auto_resolved += 1
                if result.draft_actual_similarity is not None:
                    m.similarity_sum += result.draft_actual_similarity
                    m.similarity_count += 1
            else:
                m.escalated += 1
                looks_complex = (
                    result.actual_outcome == "reopened"
                    or result.actual_word_count > COMPLEX_RESOLUTION_WORD_THRESHOLD
                )
                if looks_complex:
                    m.escalation_correct += 1

    return EvalReport(
        product_id=profile.product_id,
        total_cases=overall.total,
        auto_resolution_rate=overall.auto_resolution_rate,
        avg_draft_actual_similarity=overall.avg_draft_actual_similarity,
        escalation_precision_proxy=overall.escalation_precision_proxy,
        by_category=[m.to_dict() for m in by_category.values()],
    )
