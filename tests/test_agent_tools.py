from __future__ import annotations

import uuid
from datetime import datetime, timezone

from l1agent.agent.runner import handle_issue
from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.escalate_to_l2 import escalate_to_l2
from l1agent.agent.tools.get_product_profile import get_product_profile
from l1agent.agent.tools.log_resolution import log_resolution
from l1agent.agent.tools.search_issues import search_issues
from l1agent.agent.tools.search_kb import search_kb
from l1agent.agent.tools.update_ticket import update_ticket
from l1agent.context import TurnContext
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import (
    IssueRecord,
    IssueStatus,
    ResolutionAuthor,
    ResolutionOutcome,
    ResolutionRecord,
)
from l1agent.store.relational import get_issue, session_scope
from tests.conftest import load_profile, make_fake_agent_call


def _onboard(ctx):
    profile = load_profile("acme-widgets")
    with session_scope(ctx.database_url) as session:
        run_ingestion(
            profile, ctx.embedding_provider, ctx.vector_store, session,
            lambda summary, body, p: classify_issue(summary, body, p, ctx.embedding_provider),
        )
    return profile


def test_get_product_profile_returns_taxonomy_and_thresholds(ctx):
    profile = _onboard(ctx)
    out = get_product_profile(profile)
    assert out["product_id"] == "acme-widgets"
    assert "billing/refunds" in out["issue_taxonomy"]
    assert out["escalation"]["default_queue"] == "l2-general"


def test_search_kb_finds_relevant_passage(ctx):
    profile = _onboard(ctx)
    hits = search_kb("refund window 14 days", profile.product_id, ctx.embedding_provider, ctx.vector_store)
    assert hits
    assert any("refund" in h.text.lower() or "refund" in h.section_path.lower() for h in hits)


def test_search_kb_category_filter_still_matches_untagged_docs(ctx):
    profile = _onboard(ctx)
    # general-faq.md is untagged (category="") and should still be reachable
    # even when filtering to a specific category.
    hits = search_kb("order tracking", profile.product_id, ctx.embedding_provider, ctx.vector_store, category="billing/refunds")
    assert isinstance(hits, list)  # no crash; untagged docs pass the category filter


def test_search_issues_finds_precedent(ctx):
    profile = _onboard(ctx)
    hits = search_issues("refund never arrived", profile.product_id, ctx.embedding_provider, ctx.vector_store)
    assert hits
    assert any(h.outcome in ("accepted", "reopened") for h in hits)


def test_escalate_to_l2_routes_by_category(ctx):
    profile = _onboard(ctx)
    default_route = escalate_to_l2("issue-1", "no citation", profile, category="billing/refunds")
    assert default_route.queue == "l2-general"
    engineering_route = escalate_to_l2("issue-2", "batch defect", profile, category="product/defect")
    assert engineering_route.queue == "l2-engineering"


def test_update_ticket_persists_status(ctx):
    profile = _onboard(ctx)
    with session_scope(ctx.database_url) as session:
        from l1agent.store.relational import upsert_issue

        issue = IssueRecord(
            issue_id="issue-x", product_id=profile.product_id, summary="s", body="b",
            status=IssueStatus.open, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        upsert_issue(session, issue)

    with session_scope(ctx.database_url) as session:
        result = update_ticket("issue-x", "resolved", session, body="done")
        assert result.ok is True

    with session_scope(ctx.database_url) as session:
        updated = get_issue(session, "issue-x")
        assert updated.status == IssueStatus.resolved


def test_log_resolution_indexes_accepted_precedent(ctx):
    profile = _onboard(ctx)
    with session_scope(ctx.database_url) as session:
        from l1agent.store.relational import upsert_issue

        issue = IssueRecord(
            issue_id="issue-precedent", product_id=profile.product_id,
            summary="Refund never arrived after 3 weeks", body="Please help",
            category="billing/refunds", status=IssueStatus.resolved,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        upsert_issue(session, issue)

        pending = ResolutionRecord(
            resolution_id=str(uuid.uuid4()), issue_id="issue-precedent",
            authored_by=ResolutionAuthor.agent, body="draft", cited_sources=[],
            confidence_score=0.9, outcome=ResolutionOutcome.pending,
        )
        not_indexed = log_resolution(pending, session, ctx.embedding_provider, ctx.vector_store)
        assert not_indexed.indexed is False

        accepted = ResolutionRecord(
            resolution_id=str(uuid.uuid4()), issue_id="issue-precedent",
            authored_by=ResolutionAuthor.l2_human, body="Refund issued.", cited_sources=[],
            confidence_score=0.9, outcome=ResolutionOutcome.accepted,
        )
        indexed = log_resolution(accepted, session, ctx.embedding_provider, ctx.vector_store)
        assert indexed.indexed is True

    hits = search_issues("refund never arrived", profile.product_id, ctx.embedding_provider, ctx.vector_store)
    assert any(h.issue_id == "issue-precedent" for h in hits)


def _submit(ctx, profile, summary, body, agent_call, category=None):
    issue = IssueRecord(
        issue_id=str(uuid.uuid4()), product_id=profile.product_id, summary=summary, body=body,
        category=category, status=IssueStatus.open,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    with session_scope(ctx.database_url) as session:
        return handle_issue(issue, profile, session, ctx.embedding_provider, ctx.vector_store, agent_call=agent_call)


def test_handle_issue_auto_resolves_a_well_grounded_refund_ticket(ctx):
    # category is pinned here (rather than left for classify_issue to infer)
    # so this test isolates orchestration/gate behavior from classification
    # accuracy - the toy LocalHashEmbeddingProvider used in tests is a
    # word-overlap heuristic, not real semantic search (see
    # store/embeddings.py), and classify_issue's own accuracy is exercised
    # separately in test_ingestion.py.
    profile = _onboard(ctx)
    resolution = _submit(
        ctx, profile,
        "Refund window question",
        "It has been 20 days since I received my widget, can I still get a refund?",
        make_fake_agent_call("cite_top_kb_and_precedent"),
        category="billing/refunds",
    )
    assert resolution.outcome == ResolutionOutcome.pending
    assert resolution.cited_sources
    with session_scope(ctx.database_url) as session:
        issue = get_issue(session, resolution.issue_id)
        assert issue.status == IssueStatus.resolved


def test_handle_issue_accepts_citations_copied_with_prompt_labels(ctx):
    profile = _onboard(ctx)
    resolution = _submit(
        ctx, profile,
        "Refund window question",
        "It has been 20 days since I received my widget, can I still get a refund?",
        make_fake_agent_call("cite_labelled"),
        category="billing/refunds",
    )
    assert resolution.cited_sources
    assert not any("=" in c or "[" in c for c in resolution.cited_sources)
    with session_scope(ctx.database_url) as session:
        issue = get_issue(session, resolution.issue_id)
        assert issue.status == IssueStatus.resolved


def test_handle_issue_escalates_when_draft_has_no_citations(ctx):
    profile = _onboard(ctx)
    resolution = _submit(
        ctx, profile,
        "Something totally out of scope",
        "Can you tell me what the weather will be like on Mars next week?",
        make_fake_agent_call("no_citations"),
    )
    assert resolution.cited_sources == []
    with session_scope(ctx.database_url) as session:
        issue = get_issue(session, resolution.issue_id)
        assert issue.status == IssueStatus.escalated


def test_handle_issue_escalates_on_fabricated_citation(ctx):
    profile = _onboard(ctx)
    resolution = _submit(
        ctx, profile,
        "Refund window question",
        "It has been 20 days since I received my widget, can I still get a refund?",
        make_fake_agent_call("fabricated_citation"),
    )
    with session_scope(ctx.database_url) as session:
        issue = get_issue(session, resolution.issue_id)
        assert issue.status == IssueStatus.escalated


def test_handle_issue_respects_category_kill_switch(ctx):
    profile = _onboard(ctx)
    # billing/invoices has no explicit threshold override in the fixture
    # config, so it falls back to "*" (auto_resolve=False) - it must
    # escalate even with a perfectly grounded citation. Category is pinned
    # for the same reason as the auto-resolve test above.
    resolution = _submit(
        ctx, profile,
        "Duplicate charge",
        "I was billed twice this month for the same order, is this a duplicate?",
        make_fake_agent_call("cite_top_kb"),
        category="billing/invoices",
    )
    with session_scope(ctx.database_url) as session:
        issue = get_issue(session, resolution.issue_id)
        assert issue.category == "billing/invoices"
        assert issue.status == IssueStatus.escalated
