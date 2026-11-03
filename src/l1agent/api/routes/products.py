"""POST /products, POST /products/{id}/sync (SPEC.md §8)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.api.deps import get_app_context, get_session, require_api_key
from l1agent.context import AppContext
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import ProductProfile
from l1agent.store.embeddings import get_provider_for
from l1agent.store.relational import get_product_profile, list_product_profiles, upsert_product_profile

router = APIRouter(prefix="/products", tags=["products"])


def _classify_fn(embedding_provider):
    def _fn(summary: str, body: str, profile: ProductProfile):
        return classify_issue(summary, body, profile, embedding_provider)

    return _fn


def _reject_wiki_sources(profile: ProductProfile) -> None:
    if any(s.type == "wiki" for s in profile.kb_sources) or any(s.type == "wiki" for s in profile.issue_sources):
        raise HTTPException(
            status_code=501,
            detail="wiki sources are an M5 deliverable (SPEC.md §5.3) - onboard with file sources for now",
        )


@router.get("")
def list_products(
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    """Demo workflow addendum (spec §6): powers the chat UI's product
    selector with real onboarded products only."""
    profiles = list_product_profiles(session)
    return [
        {"product_id": p.product_id, "display_name": p.display_name, "issue_taxonomy": p.issue_taxonomy}
        for p in profiles
    ]


@router.post("", status_code=201)
def create_product(
    profile: ProductProfile,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    if get_product_profile(session, profile.product_id) is not None:
        raise HTTPException(status_code=409, detail=f"product {profile.product_id!r} already exists")
    _reject_wiki_sources(profile)

    upsert_product_profile(session, profile)
    session.flush()
    embedding_provider = get_provider_for(profile.embedding)
    run_ingestion(profile, embedding_provider, ctx.vector_store, session, _classify_fn(embedding_provider))
    return {"product_id": profile.product_id}


@router.post("/{product_id}/sync", status_code=202)
def sync_product(
    product_id: str,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    profile = get_product_profile(session, product_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"product {product_id!r} not found")
    _reject_wiki_sources(profile)

    embedding_provider = get_provider_for(profile.embedding)
    run_ingestion(profile, embedding_provider, ctx.vector_store, session, _classify_fn(embedding_provider))
    return {"job_id": str(uuid.uuid4())}
