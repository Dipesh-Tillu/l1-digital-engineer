"""Automated version of the blueprint's Phase 2 rollout check (SPEC.md §11,
§13 M4): onboard a second synthetic product with deliberately overlapping
vocabulary to the first (both mention "refund") and assert neither
retrieval crosses into the other's collection. Runs in CI from M2 onward.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from l1agent.agent.runner import handle_issue
from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.search_kb import search_kb
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import IssueRecord, IssueStatus
from l1agent.store.relational import session_scope
from tests.conftest import load_profile, make_fake_agent_call


def _onboard(ctx, name: str):
    profile = load_profile(name)
    with session_scope(ctx.database_url) as session:
        run_ingestion(
            profile, ctx.embedding_provider, ctx.vector_store, session,
            lambda summary, body, p: classify_issue(summary, body, p, ctx.embedding_provider),
        )
    return profile


def test_onboarding_second_product_requires_zero_code_changes(ctx):
    """Both products onboard through the exact same run_ingestion call with
    no product-specific branching - the acceptance bar from SPEC.md §13 M4."""
    acme = _onboard(ctx, "acme-widgets")
    beta = _onboard(ctx, "beta-gadgets")
    assert acme.product_id == "acme-widgets"
    assert beta.product_id == "beta-gadgets"


def test_kb_search_does_not_cross_product_boundaries(ctx):
    _onboard(ctx, "acme-widgets")
    _onboard(ctx, "beta-gadgets")

    acme_hits = search_kb("refund policy window", "acme-widgets", ctx.embedding_provider, ctx.vector_store, top_k=10)
    beta_hits = search_kb("refund policy window", "beta-gadgets", ctx.embedding_provider, ctx.vector_store, top_k=10)

    assert acme_hits, "acme-widgets should have refund content"
    assert beta_hits, "beta-gadgets should have refund content"
    assert all("acme-widgets" in h.source_uri for h in acme_hits)
    assert all("beta-gadgets" in h.source_uri for h in beta_hits)
    # the two products disagree on policy specifics - a leaked cross-product
    # hit would surface the wrong number here.
    acme_text = " ".join(h.text for h in acme_hits)
    beta_text = " ".join(h.text for h in beta_hits)
    assert "14 calendar days" in acme_text
    assert "14 calendar days" not in beta_text
    assert "30 calendar days" in beta_text
    assert "30 calendar days" not in acme_text


def test_ticket_resolution_only_cites_own_products_sources(ctx):
    acme = _onboard(ctx, "acme-widgets")
    beta = _onboard(ctx, "beta-gadgets")

    def submit(profile, summary, body):
        # category is pinned rather than left for classify_issue to infer:
        # the toy LocalHashEmbeddingProvider used in tests is a word-overlap
        # heuristic, not real semantic search (see store/embeddings.py), and
        # classification accuracy isn't what this test is checking -
        # namespace isolation is.
        issue = IssueRecord(
            issue_id=str(uuid.uuid4()), product_id=profile.product_id, summary=summary, body=body,
            category="billing/refunds", status=IssueStatus.open,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        with session_scope(ctx.database_url) as session:
            return handle_issue(
                issue, profile, session, ctx.embedding_provider, ctx.vector_store,
                agent_call=make_fake_agent_call("cite_top_kb_and_precedent"),
            )

    acme_resolution = submit(
        acme, "Refund question",
        "It has been 20 days since delivery, can I get a refund?",
    )
    beta_resolution = submit(
        beta, "Refund question",
        "It has been 20 days since delivery, can I get a refund?",
    )

    # Both cite something (fake agent always cites the top kb hit), but the
    # cited doc_ids must never overlap between products - that would mean
    # beta's answer was actually grounded in acme's (or vice versa) content.
    assert acme_resolution.cited_sources
    assert beta_resolution.cited_sources
    assert set(acme_resolution.cited_sources).isdisjoint(set(beta_resolution.cited_sources))
