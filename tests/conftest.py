from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from l1agent.context import AppContext
from l1agent.schemas import DraftResponseOutput, ProductProfile
from l1agent.store.embeddings import LocalHashEmbeddingProvider
from l1agent.store.relational import reset_engine_cache
from l1agent.store.vector.chroma_store import ChromaVectorStore

REPO_ROOT = Path(__file__).parent.parent


def load_profile(name: str) -> ProductProfile:
    """Load a product_profile YAML from config/products/, rewriting file
    source paths to be absolute so tests are independent of cwd."""
    data = yaml.safe_load((REPO_ROOT / "config" / "products" / f"{name}.yaml").read_text())
    for key in ("kb_sources", "issue_sources"):
        for src in data.get(key, []):
            if src.get("type") == "file":
                src["path"] = str(REPO_ROOT / src["path"])
    return ProductProfile.model_validate(data)


@pytest.fixture
def ctx(tmp_path) -> AppContext:
    """An isolated AppContext: local hash embeddings (no network/API key
    needed) + a Chroma store + SQLite DB scoped to this test's tmp_path."""
    reset_engine_cache()
    context = AppContext(
        embedding_provider=LocalHashEmbeddingProvider(),
        vector_store=ChromaVectorStore(str(tmp_path / "chroma")),
        database_url=f"sqlite:///{tmp_path}/test.db",
    )
    yield context
    reset_engine_cache()


def make_fake_agent_call(strategy: str = "cite_top_kb"):
    """A deterministic stand-in for the real Claude Agent SDK call, injected
    via `draft_response`'s `agent_call` seam so tests don't need
    ANTHROPIC_API_KEY. `strategy` picks a citation behavior so tests can
    exercise each confidence-gate rule."""

    def _call(issue, kb_hits, issue_hits, profile, turn):
        if strategy == "cite_top_kb":
            if not kb_hits:
                return DraftResponseOutput(body="Nothing relevant found; escalating.", cited_sources=[])
            top = max(kb_hits, key=lambda h: h.score)
            return DraftResponseOutput(
                body=f"Per {top.section_path}: {top.text[:300]}", cited_sources=[top.doc_id]
            )
        if strategy == "cite_top_kb_and_precedent":
            cited = []
            body_parts = []
            if kb_hits:
                top_kb = max(kb_hits, key=lambda h: h.score)
                cited.append(top_kb.doc_id)
                body_parts.append(top_kb.text[:300])
            accepted = [h for h in issue_hits if h.outcome == "accepted"]
            if accepted:
                top_issue = max(accepted, key=lambda h: h.score)
                cited.append(top_issue.issue_id)
                body_parts.append(top_issue.resolution_body[:300])
            return DraftResponseOutput(body=" ".join(body_parts) or "Not enough to go on.", cited_sources=cited)
        if strategy == "cite_labelled":
            # Cites the way real models do: copying render_context_prompt's
            # `[doc_id=...]` / `issue_id=...` labels rather than the bare ids.
            out = make_fake_agent_call("cite_top_kb_and_precedent")(issue, kb_hits, issue_hits, profile, turn)
            kb_ids = {h.doc_id for h in kb_hits}
            out.cited_sources = [
                f"[doc_id={c}]" if c in kb_ids else f"issue_id={c}" for c in out.cited_sources
            ]
            return out
        if strategy == "no_citations":
            return DraftResponseOutput(body="Not confident enough to answer; escalating.", cited_sources=[])
        if strategy == "fabricated_citation":
            return DraftResponseOutput(body="Fabricated citation.", cited_sources=["doc-that-does-not-exist"])
        raise ValueError(f"unknown fake agent_call strategy: {strategy}")

    return _call


@pytest.fixture
def fake_agent_call():
    return make_fake_agent_call("cite_top_kb")
