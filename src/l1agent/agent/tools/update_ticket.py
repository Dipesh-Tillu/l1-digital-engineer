"""Tool: update_ticket (SPEC.md §6.2).

Input: {issue_id, status, body?}. Output: {ok}.

Stubbed until M6 (SPEC.md §13): updates the local `issue_record` row so the
rest of the framework (API, eval harness) has a consistent view, but does
not call any external helpdesk API yet - that's the M6 swap.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from l1agent.schemas import IssueStatus, UpdateTicketOutput
from l1agent.store.relational import get_issue, upsert_issue


def update_ticket(issue_id: str, status: str, session: Session, body: str | None = None) -> UpdateTicketOutput:
    del body  # would become the outbound reply text once M6 wires a real helpdesk API
    issue = get_issue(session, issue_id)
    if issue is None:
        return UpdateTicketOutput(ok=False)
    issue.status = IssueStatus(status)
    issue.updated_at = datetime.now(timezone.utc)
    upsert_issue(session, issue)
    return UpdateTicketOutput(ok=True)
