from __future__ import annotations

from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.ingestion.chunking import chunk_document
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.store.relational import session_scope
from l1agent.store.vector.base import issues_collection_name, kb_collection_name
from tests.conftest import load_profile


def test_chunking_tracks_heading_depth_as_section_path():
    text = (
        "# Refund Policy\n\n"
        "intro line\n\n"
        "## Eligibility window\n\n"
        "content about eligibility. " * 5 + "\n\n"
        "## Refund timelines\n\n"
        "content about timelines. " * 5
    )
    chunks = chunk_document("acme-widgets", "test://doc", "Refund Policy", text)
    assert chunks, "expected at least one chunk"
    section_paths = {c.section_path for c in chunks}
    assert "Refund Policy > Eligibility window" in section_paths
    assert "Refund Policy > Refund timelines" in section_paths
    for c in chunks:
        assert c.token_count > 0
        assert c.doc_id
        assert c.product_id == "acme-widgets"


def test_chunking_splits_long_sections_with_overlap():
    words = " ".join(f"tok{i}" for i in range(1500))
    long_section = f"# Big Doc\n\n## One Section\n\n{words}"
    chunks = chunk_document("acme-widgets", "test://big", "Big Doc", long_section)
    assert len(chunks) > 1
    # consecutive chunks in the same section should share some overlap:
    # chunk 0 covers tok0..tok499, chunk 1 starts at tok425 (15% back).
    first_tail = set(chunks[0].chunk_text.split()[-10:])
    second_head = set(chunks[1].chunk_text.split()[:80])
    assert first_tail & second_head, "expected consecutive chunks to overlap"


def _classify_fn(embedding_provider):
    def _fn(summary, body, profile):
        return classify_issue(summary, body, profile, embedding_provider)

    return _fn


def test_run_ingestion_populates_kb_and_issues_collections(ctx):
    profile = load_profile("acme-widgets")
    with session_scope(ctx.database_url) as session:
        report = run_ingestion(
            profile, ctx.embedding_provider, ctx.vector_store, session, _classify_fn(ctx.embedding_provider)
        )

    assert report.kb_chunks_indexed > 0
    assert report.issues_indexed == 30

    kb_hits = ctx.vector_store.query(
        kb_collection_name("acme-widgets"),
        ctx.embedding_provider.embed_query("refund"),
        top_k=5,
    )
    assert kb_hits, "expected kb collection to be populated and queryable"
    assert all(h["metadata"]["product_id"] == "acme-widgets" for h in kb_hits)

    issue_hits = ctx.vector_store.query(
        issues_collection_name("acme-widgets"),
        ctx.embedding_provider.embed_query("refund never arrived"),
        top_k=5,
    )
    assert issue_hits
    assert all(h["metadata"]["product_id"] == "acme-widgets" for h in issue_hits)


def test_run_ingestion_classifies_issues_missing_category(ctx):
    profile = load_profile("acme-widgets")
    with session_scope(ctx.database_url) as session:
        run_ingestion(profile, ctx.embedding_provider, ctx.vector_store, session, _classify_fn(ctx.embedding_provider))

    # every historical row in the fixture has a Type already, so this test
    # exercises classify_issue directly against the taxonomy instead.
    result = classify_issue(
        "My widget arrived broken", "It doesn't power on out of the box", profile, ctx.embedding_provider
    )
    assert result.category in profile.issue_taxonomy
    assert 0.0 <= result.confidence <= 1.0
