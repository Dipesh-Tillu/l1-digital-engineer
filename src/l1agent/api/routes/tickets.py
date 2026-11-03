"""POST /tickets, GET /tickets/{issue_id} (SPEC.md §8).

Demo workflow addendum (l1-support-agent-demo-workflow-spec.md §5) adds:
GET /tickets (board listing), POST /tickets/{id}/run (one retrieve ->
compose -> gate cycle), POST /tickets/{id}/feedback (accept/reject).

POST /tickets itself is extended, not replaced: `demo_mode` defaults to
False, which reproduces the *exact* original contract (create + run the
full pipeline synchronously, respond with the ResolutionRecord) - the
existing test (test_ticket_submit_and_get) exercises this path unchanged.
The demo chat UI always passes demo_mode=true, which instead only creates
the ticket (status=new) and returns the IssueRecord; the frontend then
calls /run itself so New -> Working is a visible, separate step on the
board (spec §1 steps 2-4) rather than something that flashes by inside one
network round trip.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from l1agent.agent.runner import DEFAULT_MAX_ATTEMPTS, handle_issue, start_ticket, submit_feedback
from l1agent.api.deps import get_app_context, get_session, require_api_key
from l1agent.context import AppContext
from l1agent.schemas import IssueRecord, IssueStatus, ResolutionRecord
from l1agent.store.embeddings import get_provider_for
from l1agent.store.relational import (
    get_issue,
    get_latest_resolution,
    get_product_profile,
    get_resolution,
    list_issues,
    list_resolutions_for_issue,
)

router = APIRouter(prefix="/tickets", tags=["tickets"])


class TicketCreate(BaseModel):
    product_id: str
    summary: str
    body: str
    source_channel: str = "manual_import"
    demo_mode: bool = False


class TicketDetail(BaseModel):
    issue: IssueRecord
    resolution: ResolutionRecord | None = None
    attempts: list[ResolutionRecord] = []


class FeedbackRequest(BaseModel):
    accepted: bool
    reason: str | None = None


class FeedbackResponse(BaseModel):
    issue: IssueRecord
    resolution: ResolutionRecord


def _new_issue(payload: TicketCreate, status: IssueStatus) -> IssueRecord:
    now = datetime.now(timezone.utc)
    return IssueRecord(
        issue_id=str(uuid.uuid4()),
        product_id=payload.product_id,
        summary=payload.summary,
        body=payload.body,
        status=status,
        source_channel=payload.source_channel,
        created_at=now,
        updated_at=now,
        demo_mode=payload.demo_mode,
    )


@router.post("", status_code=201)
def submit_ticket(
    payload: TicketCreate,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    profile = get_product_profile(session, payload.product_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"product {payload.product_id!r} not found")

    embedding_provider = get_provider_for(profile.embedding)

    if payload.demo_mode:
        # Create only (status=new) - the chat UI calls POST /tickets/{id}/run
        # right after, so "New" is a real, visible board state (spec §1).
        issue = _new_issue(payload, IssueStatus.new)
        from l1agent.store.relational import upsert_issue

        upsert_issue(session, issue)
        return issue

    # Original contract, byte-for-byte: create + run the full pipeline now.
    issue = _new_issue(payload, IssueStatus.open)
    return handle_issue(issue, profile, session, embedding_provider, ctx.vector_store)


@router.get("")
def list_tickets(
    product_id: str | None = None,
    status: str | None = None,
    source_channel: str | None = None,
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    return list_issues(session, product_id=product_id, status=status, source_channel=source_channel)


@router.get("/{issue_id}", response_model=TicketDetail)
def get_ticket(
    issue_id: str,
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    issue = get_issue(session, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail=f"ticket {issue_id!r} not found")
    resolution = get_latest_resolution(session, issue_id)
    attempts = list_resolutions_for_issue(session, issue_id)
    return TicketDetail(issue=issue, resolution=resolution, attempts=attempts)


@router.post("/{issue_id}/run")
def run_ticket(
    issue_id: str,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    issue = get_issue(session, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail=f"ticket {issue_id!r} not found")
    profile = get_product_profile(session, issue.product_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"product {issue.product_id!r} not found")
    if issue.status not in (IssueStatus.new, IssueStatus.working):
        raise HTTPException(
            status_code=409,
            detail=f"ticket {issue_id!r} is {issue.status.value!r}, not ready for a new attempt "
            "(New or Working only - use /feedback to react to an Awaiting Review draft)",
        )

    embedding_provider = get_provider_for(profile.embedding)
    return start_ticket(issue, profile, session, embedding_provider, ctx.vector_store)


@router.post("/{issue_id}/feedback", response_model=FeedbackResponse)
def ticket_feedback(
    issue_id: str,
    payload: FeedbackRequest,
    ctx: AppContext = Depends(get_app_context),
    session: Session = Depends(get_session),
    _auth: None = Depends(require_api_key),
):
    issue = get_issue(session, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail=f"ticket {issue_id!r} not found")
    profile = get_product_profile(session, issue.product_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"product {issue.product_id!r} not found")
    if issue.status != IssueStatus.awaiting_review:
        raise HTTPException(
            status_code=409,
            detail=f"ticket {issue_id!r} is {issue.status.value!r}, not Awaiting Review - nothing to accept/reject",
        )

    resolution = get_latest_resolution(session, issue_id)
    if resolution is None:
        raise HTTPException(status_code=409, detail=f"ticket {issue_id!r} has no draft to react to")

    embedding_provider = get_provider_for(profile.embedding)
    updated_resolution = submit_feedback(
        issue, resolution.resolution_id, payload.accepted, payload.reason,
        profile, session, embedding_provider, ctx.vector_store,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
    )
    updated_issue = get_issue(session, issue_id)
    return FeedbackResponse(issue=updated_issue, resolution=updated_resolution)
