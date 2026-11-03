"""Tests for the demo workflow addendum (l1-support-agent-demo-workflow-spec.md):
run_attempt/start_ticket/submit_feedback in agent/runner.py, and the new
GET /tickets, POST /tickets/{id}/run, POST /tickets/{id}/feedback routes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

import l1agent.agent.tools.draft_response as draft_response_module
import l1agent.api.deps as deps
from l1agent.agent.runner import run_attempt, start_ticket, submit_feedback
from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.search_issues import search_issues
from l1agent.api.main import app
from l1agent.context import AppContext
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import IssueRecord, IssueStatus, ResolutionOutcome
from l1agent.store.embeddings import LocalHashEmbeddingProvider
from l1agent.store.relational import get_issue, get_resolution, reset_engine_cache, session_scope
from l1agent.store.vector.chroma_store import ChromaVectorStore
from tests.conftest import REPO_ROOT, load_profile, make_fake_agent_call


def _onboard(ctx):
    profile = load_profile("acme-widgets")
    with session_scope(ctx.database_url) as session:
        run_ingestion(
            profile, ctx.embedding_provider, ctx.vector_store, session,
            lambda summary, body, p: classify_issue(summary, body, p, ctx.embedding_provider),
        )
    return profile


def _new_issue(product_id: str, summary: str, body: str, *, demo_mode: bool = True, category: str | None = None) -> IssueRecord:
    now = datetime.now(timezone.utc)
    return IssueRecord(
        issue_id=str(uuid.uuid4()), product_id=product_id, summary=summary, body=body,
        category=category, status=IssueStatus.new, created_at=now, updated_at=now, demo_mode=demo_mode,
    )


# --------------------------------------------------------------------------
# runner.py: run_attempt / start_ticket / submit_feedback
# --------------------------------------------------------------------------

def test_run_attempt_does_not_commit_an_outcome(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", category="billing/refunds")
    with session_scope(ctx.database_url) as session:
        resolution = run_attempt(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        assert resolution.outcome == ResolutionOutcome.pending
        assert resolution.attempt_number == 1
        assert resolution.gate_passed is True  # billing/refunds has auto_resolve=true in the fixture
        # run_attempt never touches issue status - still whatever it was created with.
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.new


def test_start_ticket_demo_mode_pauses_for_review_on_a_passing_gate(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", demo_mode=True, category="billing/refunds")
    with session_scope(ctx.database_url) as session:
        resolution = start_ticket(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        assert resolution.gate_passed is True
        assert resolution.outcome == ResolutionOutcome.pending  # not auto-closed
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.awaiting_review
        assert stored.attempt_count == 1


def test_start_ticket_production_mode_auto_resolves_on_a_passing_gate(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", demo_mode=False, category="billing/refunds")
    with session_scope(ctx.database_url) as session:
        start_ticket(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.resolved


def test_start_ticket_failing_gate_escalates_regardless_of_demo_mode(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Something totally out of scope", "What is the weather on Mars?", demo_mode=True)
    with session_scope(ctx.database_url) as session:
        resolution = start_ticket(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("no_citations"),
        )
        assert resolution.gate_passed is False
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.escalated
        assert stored.escalated_from_issue_id is None  # it's the original, not the linked L2 ticket

        # A real linked L2 ticket should now exist on the board.
        from l1agent.store.relational import list_issues
        linked = [i for i in list_issues(session) if i.escalated_from_issue_id == issue.issue_id]
        assert len(linked) == 1
        assert linked[0].status == IssueStatus.escalated


def test_submit_feedback_accept_closes_and_indexes_precedent(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", demo_mode=True, category="billing/refunds")
    with session_scope(ctx.database_url) as session:
        resolution = start_ticket(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        updated = submit_feedback(
            issue, resolution.resolution_id, True, None, profile, session,
            ctx.embedding_provider, ctx.vector_store,
        )
        assert updated.outcome == ResolutionOutcome.accepted
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.resolved

    # The feedback loop: this accepted resolution is now a precedent.
    hits = search_issues("refund window question", profile.product_id, ctx.embedding_provider, ctx.vector_store)
    assert any(h.issue_id == issue.issue_id for h in hits)


def test_submit_feedback_reject_reruns_with_extra_context_as_next_attempt(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", demo_mode=True, category="billing/refunds")
    with session_scope(ctx.database_url) as session:
        resolution = start_ticket(
            issue, profile, session, ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        first_id = resolution.resolution_id
        updated = submit_feedback(
            issue, first_id, False, "still not clear about partial refunds", profile, session,
            ctx.embedding_provider, ctx.vector_store,
            agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
        )
        assert updated.attempt_number == 2
        assert updated.resolution_id != first_id

        original = get_resolution(session, first_id)
        assert original.outcome == ResolutionOutcome.rejected
        assert original.rejection_reason == "still not clear about partial refunds"

        stored = get_issue(session, issue.issue_id)
        assert stored.attempt_count == 2


def test_submit_feedback_rejection_cap_forces_escalation_without_a_further_attempt(ctx):
    profile = _onboard(ctx)
    issue = _new_issue(profile.product_id, "Refund window question", "It has been 20 days, can I get a refund?", demo_mode=True, category="billing/refunds")
    fake = make_fake_agent_call("cite_top_kb_and_precedent")
    with session_scope(ctx.database_url) as session:
        resolution = start_ticket(issue, profile, session, ctx.embedding_provider, ctx.vector_store, agent_call=fake)
        # Reject attempt 1 -> attempt 2 runs.
        resolution = submit_feedback(issue, resolution.resolution_id, False, "no", profile, session, ctx.embedding_provider, ctx.vector_store, agent_call=fake)
        assert resolution.attempt_number == 2
        # Reject attempt 2 -> attempt 3 runs (default cap = 3 attempts total).
        resolution = submit_feedback(issue, resolution.resolution_id, False, "no", profile, session, ctx.embedding_provider, ctx.vector_store, agent_call=fake)
        assert resolution.attempt_number == 3
        # Reject attempt 3 -> cap exceeded, forced escalation, no attempt 4.
        resolution = submit_feedback(issue, resolution.resolution_id, False, "no", profile, session, ctx.embedding_provider, ctx.vector_store, agent_call=fake)
        assert resolution.attempt_number == 3  # unchanged - no 4th attempt was run
        stored = get_issue(session, issue.issue_id)
        assert stored.status == IssueStatus.escalated
        assert stored.attempt_count == 3


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    reset_engine_cache()
    test_ctx = AppContext(
        embedding_provider=LocalHashEmbeddingProvider(),
        vector_store=ChromaVectorStore(str(tmp_path / "chroma")),
        database_url=f"sqlite:///{tmp_path}/test.db",
    )
    monkeypatch.setattr(deps, "_app_context", test_ctx)

    fake = make_fake_agent_call("cite_top_kb_and_precedent")
    monkeypatch.setattr(
        draft_response_module,
        "_default_agent_call",
        lambda issue, kb_hits, issue_hits, profile, turn: fake(issue, kb_hits, issue_hits, profile, turn),
    )
    with TestClient(app) as c:
        yield c
    reset_engine_cache()


def _acme_profile_payload() -> dict:
    data = yaml.safe_load((REPO_ROOT / "config" / "products" / "acme-widgets.yaml").read_text())
    for key in ("kb_sources", "issue_sources"):
        for src in data.get(key, []):
            if src.get("type") == "file":
                src["path"] = str(REPO_ROOT / src["path"])
    return data


def test_demo_mode_create_is_creation_only(client):
    client.post("/products", json=_acme_profile_payload())
    resp = client.post(
        "/tickets",
        json={"product_id": "acme-widgets", "summary": "Refund window question",
              "body": "It has been 20 days, can I get a refund?", "demo_mode": True},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "new"
    assert body["resolution_id"] is None  # IssueRecord.resolution_id, not a ResolutionRecord - no attempt has run yet
    assert "confidence_score" not in body  # would be present if this were a ResolutionRecord


def test_run_then_feedback_accept_full_loop(client):
    client.post("/products", json=_acme_profile_payload())
    created = client.post(
        "/tickets",
        # Wording chosen so the real (unpinned) classify_issue lands on
        # billing/refunds, not billing/invoices - the fixture's kill-switch
        # category, which would escalate instead of reaching Awaiting Review.
        json={"product_id": "acme-widgets", "summary": "Refund request",
              "body": "I would like to request a refund for my recent order, please process my refund.",
              "demo_mode": True},
    ).json()
    issue_id = created["issue_id"]

    run_resp = client.post(f"/tickets/{issue_id}/run")
    assert run_resp.status_code == 200
    resolution = run_resp.json()
    assert "resolution_id" in resolution
    assert "gate_passed" in resolution

    detail = client.get(f"/tickets/{issue_id}").json()
    # billing/refunds auto_resolve=true in the fixture -> should be awaiting_review
    assert detail["issue"]["status"] == "awaiting_review"
    assert len(detail["attempts"]) == 1

    fb = client.post(f"/tickets/{issue_id}/feedback", json={"accepted": True})
    assert fb.status_code == 200
    assert fb.json()["issue"]["status"] == "resolved"
    assert fb.json()["resolution"]["outcome"] == "accepted"


def test_run_on_non_new_ticket_409s(client):
    client.post("/products", json=_acme_profile_payload())
    created = client.post(
        "/tickets",
        json={"product_id": "acme-widgets", "summary": "s", "body": "b", "demo_mode": True},
    ).json()
    issue_id = created["issue_id"]
    client.post(f"/tickets/{issue_id}/run")
    # Ticket is now awaiting_review or escalated - not New/Working anymore.
    resp = client.post(f"/tickets/{issue_id}/run")
    assert resp.status_code == 409


def test_feedback_before_run_409s(client):
    client.post("/products", json=_acme_profile_payload())
    created = client.post(
        "/tickets",
        json={"product_id": "acme-widgets", "summary": "s", "body": "b", "demo_mode": True},
    ).json()
    resp = client.post(f"/tickets/{created['issue_id']}/feedback", json={"accepted": True})
    assert resp.status_code == 409


def test_list_tickets_powers_the_board(client):
    # Onboarding ingests the fixture's own historical issue CSV (30 rows,
    # all already-resolved precedent), so the board is never empty -
    # filter by status="new" to isolate just what this test created.
    client.post("/products", json=_acme_profile_payload())
    client.post("/tickets", json={"product_id": "acme-widgets", "summary": "s1", "body": "b1", "demo_mode": True})
    client.post("/tickets", json={"product_id": "acme-widgets", "summary": "s2", "body": "b2", "demo_mode": True})
    resp = client.get("/tickets", params={"product_id": "acme-widgets", "status": "new"})
    assert resp.status_code == 200
    tickets = resp.json()
    assert len(tickets) == 2
    assert all(t["status"] == "new" for t in tickets)
    assert {t["summary"] for t in tickets} == {"s1", "s2"}

    resp2 = client.get("/tickets", params={"status": "does-not-exist"})
    assert resp2.json() == []
