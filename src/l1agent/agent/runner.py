"""Orchestration loop (SPEC.md §6.3).

`draft_response` is the only step that actually runs the Claude Agent SDK
loop end to end; the surrounding steps are deterministic orchestration in
plain Python, not agent turns. Keep it this way - it's what makes the
gate's decision auditable outside the model's own reasoning.

Demo workflow addendum (l1-support-agent-demo-workflow-spec.md): the
original `handle_issue` ran retrieve -> compose -> gate -> act as one
inline call with no seam for a human to intervene between the gate and the
act. Per the addendum's assumption to confirm (§0), that's split here:

- `run_attempt` is retrieve -> compose -> gate only. It persists the
  attempt (outcome=pending) but never touches issue status, never
  escalates, never indexes a precedent - it's a pure "what did the agent
  find" step, safe to call once per attempt.
- `handle_issue` is unchanged from the outside (same signature, same
  return shape, same tests pass) - it's now just run_attempt() followed by
  an immediate, unconditional commit, exactly reproducing the old
  single-shot behavior for every existing non-demo caller (CLI, eval,
  the original POST /tickets contract).
- `start_ticket` / `submit_feedback` are the new demo-mode entry points
  (spec §3): start_ticket runs attempt 1 and routes by gate result;
  submit_feedback records an accept/reject decision and, on reject,
  re-runs run_attempt with the rejection reason threaded in as extra
  context, up to a bounded retry cap.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.agent.tools.draft_response import AgentCallFn, draft_response
from l1agent.agent.tools.escalate_to_l2 import escalate_to_l2
from l1agent.agent.tools.log_resolution import finalize_resolution_outcome, log_resolution
from l1agent.agent.tools.score_confidence import score_confidence
from l1agent.agent.tools.search_issues import search_issues
from l1agent.agent.tools.search_kb import search_kb
from l1agent.agent.tools.update_ticket import update_ticket
from l1agent.context import TurnContext
from l1agent.schemas import (
    IssueRecord,
    IssueStatus,
    ProductProfile,
    ResolutionAuthor,
    ResolutionOutcome,
    ResolutionRecord,
)
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.relational import get_resolution, upsert_issue
from l1agent.store.vector.base import VectorStore

# Default bounded-retry cap (spec §3): 2 rejections total = 3 attempts
# offered before a rejection forces escalation without a further attempt.
DEFAULT_MAX_ATTEMPTS = 3


def issue_query(issue: IssueRecord) -> str:
    return f"{issue.summary}\n{issue.body}"


def run_attempt(
    issue: IssueRecord,
    profile: ProductProfile,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    attempt_number: int = 1,
    extra_context: str | None = None,
    agent_call: AgentCallFn | None = None,
) -> ResolutionRecord:
    """One retrieve -> compose -> gate cycle. Persists the attempt
    (outcome=pending) but commits no outcome - the caller decides what to
    do with a passing/failing gate (see module docstring)."""
    if not issue.category:
        issue.category = classify_issue(issue.summary, issue.body, profile, embedding_provider).category
    issue.updated_at = datetime.now(timezone.utc)
    upsert_issue(session, issue)

    query = issue_query(issue)
    if extra_context:
        query = f"{query}\n{extra_context}"

    kb_hits = search_kb(query, profile.product_id, embedding_provider, vector_store, category=issue.category)
    issue_hits = search_issues(query, profile.product_id, embedding_provider, vector_store)

    turn = TurnContext(product_id=profile.product_id, category=issue.category)
    turn.record_kb_hits(kb_hits)
    turn.record_issue_hits(issue_hits)

    # The rejection reason also needs to reach draft_response's prompt, not
    # just the retrieval query - render_context_prompt reads issue.body, so
    # a local (unpersisted) copy carries it into that one call only.
    draft_issue = issue
    if extra_context:
        draft_issue = issue.model_copy(
            update={"body": f"{issue.body}\n\n[Requester rejected a previous attempt; reason: {extra_context}]"}
        )
    draft = draft_response(draft_issue, kb_hits, issue_hits, profile, turn, agent_call=agent_call)

    gate = score_confidence(draft.body, draft.cited_sources, issue.category, profile, turn)

    resolution = ResolutionRecord(
        resolution_id=str(uuid.uuid4()),
        issue_id=issue.issue_id,
        attempt_number=attempt_number,
        authored_by=ResolutionAuthor.agent,
        body=draft.body,
        cited_sources=draft.cited_sources,
        confidence_score=gate.score,
        outcome=ResolutionOutcome.pending,
        gate_passed=gate.passed,
        gate_reasons=gate.reasons,
        kb_hits=kb_hits,
        issue_hits=issue_hits,
        trace=turn.llm_trace,
    )
    log_resolution(resolution, session, embedding_provider, vector_store)
    return resolution


def handle_issue(
    issue: IssueRecord,
    profile: ProductProfile,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    agent_call: AgentCallFn | None = None,
) -> ResolutionRecord:
    """Original, unmodified-from-the-outside entry point: one attempt,
    committed immediately (gate pass -> resolved, gate fail -> escalated),
    no pause for confirmation. This is production/non-demo behavior -
    every existing caller (CLI, the original POST /tickets contract,
    eval/replay's own independent path) keeps working exactly as before.
    """
    resolution = run_attempt(issue, profile, session, embedding_provider, vector_store, agent_call=agent_call)
    if resolution.gate_passed:
        update_ticket(issue.issue_id, IssueStatus.resolved.value, session, body=resolution.body)
    else:
        escalate_to_l2(
            issue.issue_id, "; ".join(resolution.gate_reasons) or "confidence gate failed", profile,
            draft=resolution.body, category=issue.category,
        )
        update_ticket(issue.issue_id, IssueStatus.escalated.value, session, body=resolution.body)
    return resolution


def _route_after_attempt(
    issue: IssueRecord,
    resolution: ResolutionRecord,
    profile: ProductProfile,
    session: Session,
) -> ResolutionRecord:
    """Shared by start_ticket/submit_feedback's retry path (spec §3): a
    failing gate always escalates immediately, demo_mode or not - there is
    nothing to "pause for confirmation" on a draft that didn't clear the
    confidence bar. A passing gate either auto-sends (production) or pauses
    in Awaiting Review for an explicit accept/reject (demo_mode)."""
    if resolution.gate_passed:
        if issue.demo_mode:
            update_ticket(issue.issue_id, IssueStatus.awaiting_review.value, session)
        else:
            update_ticket(issue.issue_id, IssueStatus.resolved.value, session, body=resolution.body)
    else:
        escalate_to_l2(
            issue.issue_id, "; ".join(resolution.gate_reasons) or "confidence gate failed", profile,
            draft=resolution.body, category=issue.category, session=session,
            summary=issue.summary, body_text=issue.body,
        )
        update_ticket(issue.issue_id, IssueStatus.escalated.value, session)
    return resolution


def start_ticket(
    issue: IssueRecord,
    profile: ProductProfile,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    agent_call: AgentCallFn | None = None,
) -> ResolutionRecord:
    """Demo workflow §1 steps 3-6: New -> Working, run attempt 1, then
    route by gate result. `issue.attempt_count` is set to 1 here - it is
    the single source of truth the ticket board reads for "current attempt
    number" and that submit_feedback's retry cap checks against."""
    update_ticket(issue.issue_id, IssueStatus.working.value, session)
    resolution = run_attempt(issue, profile, session, embedding_provider, vector_store, attempt_number=1, agent_call=agent_call)
    issue.attempt_count = 1
    upsert_issue(session, issue)
    return _route_after_attempt(issue, resolution, profile, session)


def submit_feedback(
    issue: IssueRecord,
    resolution_id: str,
    accepted: bool,
    reason: str | None,
    profile: ProductProfile,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    agent_call: AgentCallFn | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ResolutionRecord:
    """Demo workflow §3: the accept/reject loop.

    On accept: closes exactly as log_resolution/update_ticket already do
    elsewhere, outcome=accepted - this is also the first place in the
    system an outcome actually becomes "accepted" and the feedback-loop
    precedent indexing actually fires (handle_issue's single-shot path
    never set an outcome beyond "pending").

    On reject: records the rejection + reason, increments the attempt
    counter, and either re-runs retrieve -> compose -> gate with the
    reason threaded in as extra context (if under the cap) or escalates
    immediately (cap exceeded) - regardless of gate result on that
    final attempt, per spec §3 step 5.
    """
    if accepted:
        finalize_resolution_outcome(
            resolution_id, ResolutionOutcome.accepted, session, embedding_provider, vector_store,
        )
        update_ticket(issue.issue_id, IssueStatus.resolved.value, session)
        return get_resolution(session, resolution_id)

    finalize_resolution_outcome(resolution_id, ResolutionOutcome.rejected, session, rejection_reason=reason)
    rejected = get_resolution(session, resolution_id)

    if issue.attempt_count >= max_attempts:
        # Cap exceeded on this rejection - escalate the last attempt as-is,
        # no further attempt is run (spec §3 step 5).
        escalate_to_l2(
            issue.issue_id, f"rejection cap ({max_attempts} attempts) exceeded", profile,
            draft=rejected.body if rejected else "", category=issue.category, session=session,
            summary=issue.summary, body_text=issue.body,
        )
        update_ticket(issue.issue_id, IssueStatus.escalated.value, session)
        return rejected

    update_ticket(issue.issue_id, IssueStatus.working.value, session)
    next_attempt_number = issue.attempt_count + 1
    resolution = run_attempt(
        issue, profile, session, embedding_provider, vector_store,
        attempt_number=next_attempt_number, extra_context=reason, agent_call=agent_call,
    )
    issue.attempt_count = next_attempt_number
    upsert_issue(session, issue)
    return _route_after_attempt(issue, resolution, profile, session)
