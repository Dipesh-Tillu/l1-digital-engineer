"""POST /eval/replay (SPEC.md §8, §10)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from l1agent.api.deps import get_app_context, get_session, require_api_key
from l1agent.context import AppContext
from l1agent.eval.replay import replay
from l1agent.store.embeddings import get_provider_for
from l1agent.store.relational import get_product_profile

router = APIRouter(prefix="/eval", tags=["eval"])


class ReplayRequest(BaseModel):
    product_id: str
    dataset_path: str


@router.post("/replay")
def replay_route(
    payload: ReplayRequest,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    profile = get_product_profile(session, payload.product_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"product {payload.product_id!r} not found")
    embedding_provider = get_provider_for(profile.embedding)
    report = replay(profile, payload.dataset_path, embedding_provider, ctx.vector_store)
    return report.to_dict()
