"""Tool: escalate_to_l2 (SPEC.md §6.2).

Input: {issue_id, draft?, reason}. Output: {queue, ticket_ref, linked_issue_id?}.

Stubbed until M6 (real ticketing integration, SPEC.md §13): routes by the
product's EscalationPolicy and returns a queue + a synthetic ticket_ref, but
does not call any external helpdesk API yet.

Demo workflow addendum (l1-support-agent-demo-workflow-spec.md §1 step 10,
§4): when a `session` is passed, this also creates a *real*, persisted
`IssueRecord` for the hand-off - `escalated_from_issue_id` pointing back at
the original ticket - so "Escalated to L2 - reference SR-xxxx" is a ticket
that actually appears on the board, not just a string. Callers that don't
pass `session` (e.g. eval/replay.py, which deliberately never mutates the
DB it's evaluating against) get the old no-DB-effect behavior unchanged.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from l1agent.schemas import EscalateOutput, IssueRecord, IssueStatus, ProductProfile


def escalate_to_l2(
    issue_id: str,
    reason: str,
    profile: ProductProfile,
    draft: str | None = None,
    category: str | None = None,
    *,
    session=None,
    summary: str | None = None,
    body_text: str | None = None,
) -> EscalateOutput:
    queue = profile.escalation.category_routes.get(category, profile.escalation.default_queue) if category else profile.escalation.default_queue
    ticket_ref = f"ESC-{uuid.uuid4().hex[:8]}"

    linked_issue_id: str | None = None
    if session is not None:
        from l1agent.store.relational import upsert_issue

        now = datetime.now(timezone.utc)
        linked = IssueRecord(
            issue_id=ticket_ref,
            product_id=profile.product_id,
            summary=summary or f"[Escalated from {issue_id}] {reason}",
            body=body_text or draft or reason,
            category=category,
            status=IssueStatus.escalated,
            source_channel="escalation",
            created_at=now,
            updated_at=now,
            escalated_from_issue_id=issue_id,
            escalation_queue=queue,
        )
        upsert_issue(session, linked)
        linked_issue_id = ticket_ref

    return EscalateOutput(queue=queue, ticket_ref=ticket_ref, linked_issue_id=linked_issue_id)
