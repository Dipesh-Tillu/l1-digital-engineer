from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

import l1agent.agent.tools.draft_response as draft_response_module
import l1agent.api.deps as deps
from l1agent.api.main import app
from l1agent.context import AppContext
from l1agent.store.embeddings import LocalHashEmbeddingProvider
from l1agent.store.relational import reset_engine_cache
from l1agent.store.vector.chroma_store import ChromaVectorStore
from tests.conftest import REPO_ROOT, make_fake_agent_call


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


def test_create_product_runs_ingestion_and_rejects_duplicate(client):
    resp = client.post("/products", json=_acme_profile_payload())
    assert resp.status_code == 201
    assert resp.json() == {"product_id": "acme-widgets"}

    resp2 = client.post("/products", json=_acme_profile_payload())
    assert resp2.status_code == 409


def test_sync_reruns_ingestion(client):
    client.post("/products", json=_acme_profile_payload())
    resp = client.post("/products/acme-widgets/sync")
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_sync_unknown_product_404s(client):
    resp = client.post("/products/does-not-exist/sync")
    assert resp.status_code == 404


def test_ticket_submit_and_get(client):
    client.post("/products", json=_acme_profile_payload())
    resp = client.post(
        "/tickets",
        json={
            "product_id": "acme-widgets",
            "summary": "Refund window question",
            "body": "It has been 20 days since I received my widget, can I still get a refund?",
        },
    )
    assert resp.status_code == 201
    resolution = resp.json()
    assert "resolution_id" in resolution
    assert "issue_id" in resolution

    detail = client.get(f"/tickets/{resolution['issue_id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["issue"]["issue_id"] == resolution["issue_id"]
    assert body["resolution"]["resolution_id"] == resolution["resolution_id"]


def test_ticket_submit_missing_product_404s(client):
    resp = client.post("/tickets", json={"product_id": "nope", "summary": "s", "body": "b"})
    assert resp.status_code == 404


def test_get_ticket_missing_issue_404s(client):
    resp = client.get("/tickets/does-not-exist")
    assert resp.status_code == 404


def test_eval_replay_produces_report(client):
    client.post("/products", json=_acme_profile_payload())
    dataset_path = str(REPO_ROOT / "tests" / "fixtures" / "acme-widgets" / "issues.csv")
    resp = client.post("/eval/replay", json={"product_id": "acme-widgets", "dataset_path": dataset_path})
    assert resp.status_code == 200
    report = resp.json()
    assert report["product_id"] == "acme-widgets"
    assert report["total_cases"] == 30
    assert 0.0 <= report["auto_resolution_rate"] <= 1.0
    assert report["by_category"]
