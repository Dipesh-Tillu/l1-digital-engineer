"""Pydantic models shared across ingestion, store, agent, gate, api, and cli.

See SPEC.md §4 for the design rationale behind each model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# §4.6 SourceRef - tagged union resolved by the connector registry
# --------------------------------------------------------------------------

class FileSourceRef(BaseModel):
    type: Literal["file"] = "file"
    path: str
    # CSV column mapping, only meaningful when `path` points at an issue export
    summary_col: str | None = None
    body_col: str | None = None
    category_col: str | None = None
    resolution_col: str | None = None  # historical resolution text, if present
    outcome_col: str | None = None  # "accepted" | "reopened", defaults to "accepted"


class WikiSourceRef(BaseModel):
    type: Literal["wiki"] = "wiki"
    connector: str  # "confluence" | "notion" | "sharepoint"
    space_key: str
    auth_ref: str  # name of a secret, resolved via env/secrets store - never a literal credential


class UrlSourceRef(BaseModel):
    """A doc page or a JSON ticket-API endpoint served over HTTP - the
    "live source" ingestion path alongside local files and (M5) wiki
    spaces. Fetches one resource per entry (not a crawler); add multiple
    kb_sources/issue_sources entries for multiple pages/endpoints."""

    type: Literal["url"] = "url"
    url: str
    format: Literal["auto", "html", "markdown", "pdf", "text", "csv", "json"] = "auto"
    auth_ref: str | None = None  # env var name holding a bearer token
    # issue/ticket field mapping - same convention as FileSourceRef's CSV mapping
    summary_col: str | None = None
    body_col: str | None = None
    category_col: str | None = None
    resolution_col: str | None = None
    outcome_col: str | None = None
    json_records_path: str | None = None  # dotted path to the record list, e.g. "data.tickets"
    next_page_path: str | None = None  # dotted path to a next-page URL, for cursor pagination
    max_pages: int = 20


SourceRef = Annotated[Union[FileSourceRef, WikiSourceRef, UrlSourceRef], Field(discriminator="type")]


# --------------------------------------------------------------------------
# §4.1 ProductProfile
# --------------------------------------------------------------------------

class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    default: str = "claude-sonnet-4-5"
    overrides: dict[str, str] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    provider: Literal["voyage", "openai", "local"] = "voyage"
    model: str | None = None  # None -> provider's own default
    api_key_env: str | None = None  # None -> provider's conventional env var name


class FineTuneConfig(BaseModel):
    """Weight-level fine-tuning for one support tier (SPEC-adjacent: see
    the fine-tuning design). `backend` is None until a fine-tune has been
    requested; `model_ref` is filled in by finetune/runner.py after a
    successful run and is what agent/tools/draft_response.py's dispatcher
    checks to route a tier's tickets to the fine-tuned checkpoint instead
    of the base provider model."""

    backend: Literal["openai", "local_lora"] | None = None
    base_model: str | None = None  # None -> backend's own default
    hyperparams: dict = Field(default_factory=dict)
    model_ref: str | None = None
    trained_at: datetime | None = None
    metrics: dict = Field(default_factory=dict)


class TierConfig(BaseModel):
    """A support tier (L1, L2, ...future L3) and which issue categories
    route to it. Fine-tuning a new checkpoint for a tier and setting
    `finetune.model_ref` is how a product moves a category from "answered
    by the base model" to "answered by a weight fine-tuned specifically
    for this tier" - the mechanism behind "later change the weights to be
    an L2/L3 engineer"."""

    role: str  # "l1_support" | "l2_support" | ... (free-form, extensible)
    categories: list[str] = Field(default_factory=lambda: ["*"])  # "*" = catch-all/default tier
    finetune: FineTuneConfig = Field(default_factory=FineTuneConfig)


class ConfidenceThreshold(BaseModel):
    category: str  # taxonomy leaf, or "*" for default
    min_citation_score: float = 0.72
    require_precedent: bool = False
    auto_resolve: bool = False


class EscalationPolicy(BaseModel):
    default_queue: str
    category_routes: dict[str, str] = Field(default_factory=dict)


class ProductProfile(BaseModel):
    product_id: str
    display_name: str
    issue_taxonomy: list[str]
    thresholds: list[ConfidenceThreshold]
    escalation: EscalationPolicy
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tiers: list[TierConfig] = Field(default_factory=lambda: [TierConfig(role="l1_support")])
    kb_sources: list[SourceRef] = Field(default_factory=list)
    issue_sources: list[SourceRef] = Field(default_factory=list)
    prompt_addendum: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def threshold_for(self, category: str | None) -> ConfidenceThreshold:
        if category:
            for t in self.thresholds:
                if t.category == category:
                    return t
        for t in self.thresholds:
            if t.category == "*":
                return t
        raise ValueError(
            f"no threshold for category={category!r} and no '*' default defined"
        )

    def tier_for(self, category: str | None) -> TierConfig:
        """Exact category matches win first (across all tiers, in list
        order); the "*" catch-all tier is only used when nothing claimed
        the category explicitly. This means adding a second tier and
        listing specific categories on it never requires editing the
        first tier's own `categories` list."""
        if category:
            for t in self.tiers:
                if category in t.categories:
                    return t
        for t in self.tiers:
            if "*" in t.categories:
                return t
        return self.tiers[0]


# --------------------------------------------------------------------------
# §4.2 KBDocument
# --------------------------------------------------------------------------

class KBDocument(BaseModel):
    doc_id: str
    product_id: str
    source_uri: str
    section_path: str
    chunk_text: str
    chunk_index: int
    token_count: int
    last_synced_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# KBHit / IssueHit - moved ahead of ResolutionRecord (§4.4) so its
# kb_hits/issue_hits transient fields (demo workflow addendum) can
# reference them without a forward-ref rebuild.
# --------------------------------------------------------------------------

class LLMTraceStep(BaseModel):
    """One step of a real draft_response agent turn - demo workflow
    addendum (per user request): what was actually sent to/received from
    the model, not a summary. Captured by the real provider backends
    (draft_response.py's _anthropic_agent_call / _openai_agent_call /
    _local_agent_call) via TurnContext.llm_trace. Transient - never
    persisted, never set by the fake `agent_call` tests inject, same as
    ResolutionRecord.kb_hits/issue_hits below."""

    kind: Literal["assistant_text", "tool_call", "tool_result"]
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result_summary: str | None = None


class LLMTrace(BaseModel):
    tier_role: str
    model_provider: str
    model: str
    system_prompt: str
    user_prompt: str
    steps: list[LLMTraceStep] = Field(default_factory=list)


class KBHit(BaseModel):
    doc_id: str
    section_path: str
    text: str
    score: float
    source_uri: str


class IssueHit(BaseModel):
    issue_id: str
    summary: str
    resolution_body: str
    score: float
    outcome: str


# --------------------------------------------------------------------------
# §4.3 IssueRecord
# --------------------------------------------------------------------------

class IssueStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    escalated = "escalated"
    reopened = "reopened"
    # --- demo workflow addendum (l1-support-agent-demo-workflow-spec.md §1) ---
    # Board-visible states for the chat+ticket demo. Non-demo callers
    # (handle_issue, the CLI, eval/replay) never set these - they still go
    # straight open -> resolved/escalated as before.
    new = "new"              # ticket created, not yet picked up
    working = "working"      # agent is retrieving/drafting/gating this attempt
    awaiting_review = "awaiting_review"  # demo_mode: gate passed, paused for accept/reject


class IssueRecord(BaseModel):
    issue_id: str
    product_id: str
    summary: str
    body: str
    category: str | None = None
    status: IssueStatus = IssueStatus.open
    resolution_id: str | None = None
    source_channel: str = "manual_import"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    # --- demo workflow addendum additions ---
    escalated_from_issue_id: str | None = None  # set only on the L2 ticket created via hand-off (spec §4)
    escalation_queue: str | None = None  # set only on the L2 ticket - which human queue it routed to
    demo_mode: bool = False  # per-ticket, not per-request: must persist across /run and /feedback calls
    attempt_count: int = 0  # number of retrieve->compose->gate attempts run so far (drives the reject cap)


# --------------------------------------------------------------------------
# §4.4 ResolutionRecord
# --------------------------------------------------------------------------

class ResolutionAuthor(str, Enum):
    agent = "agent"
    l2_human = "l2_human"


class ResolutionOutcome(str, Enum):
    pending = "pending"
    accepted = "accepted"
    reopened = "reopened"
    rejected = "rejected"  # requester rejected this specific attempt in demo mode (spec §4)


class ResolutionRecord(BaseModel):
    resolution_id: str
    issue_id: str
    attempt_number: int = 1  # one record per attempt, not one per issue (spec §4)
    authored_by: ResolutionAuthor
    body: str
    cited_sources: list[str] = Field(default_factory=list)
    confidence_score: float
    outcome: ResolutionOutcome = ResolutionOutcome.pending
    rejection_reason: str | None = None  # only set when outcome == rejected (spec §4)
    created_at: datetime = Field(default_factory=utcnow)
    # --- demo workflow addendum: gate detail, persisted alongside the core
    # fields above so the ticket board can show "every attempt's draft,
    # citations, and gate outcome" (spec §6), not just the latest.
    gate_passed: bool | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    # --- transient, NOT persisted in store/relational.py - populated only
    # on the object run_attempt() returns fresh, so the chat panel can show
    # "which search_kb/search_issues hits, scores" (spec §6) for the
    # current attempt without a second round-trip. Reloaded rows default
    # these to empty lists.
    kb_hits: list["KBHit"] = Field(default_factory=list)
    issue_hits: list["IssueHit"] = Field(default_factory=list)
    trace: LLMTrace | None = None  # the real LLM call(s) this attempt made - see LLMTrace above


# --------------------------------------------------------------------------
# Ingestion-internal shapes (§5.1) - not persisted, connector <-> pipeline only
# --------------------------------------------------------------------------

class RawDocument(BaseModel):
    uri: str
    title: str
    content_bytes: bytes
    content_type: str  # "text/markdown" | "application/pdf" | "text/html" | ...


class RawIssue(BaseModel):
    issue_id: str
    summary: str
    body: str
    category: str | None = None
    resolution_body: str | None = None
    outcome: str | None = None  # "accepted" | "reopened"; None if unresolved/unknown
    source_channel: str = "manual_import"
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Tool I/O shapes (§6.2) - implemented as Pydantic models for SDK schema derivation
# --------------------------------------------------------------------------

class SearchKBOutput(BaseModel):
    hits: list[KBHit]


class SearchIssuesOutput(BaseModel):
    hits: list[IssueHit]


class ClassifyIssueOutput(BaseModel):
    category: str
    confidence: float


class GateResult(BaseModel):
    passed: bool = Field(alias="pass")
    score: float
    reasons: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DraftResponseOutput(BaseModel):
    body: str
    cited_sources: list[str] = Field(default_factory=list)


class EscalateOutput(BaseModel):
    queue: str
    ticket_ref: str
    linked_issue_id: str | None = None  # set when a session was passed and a real L2 IssueRecord was created


class UpdateTicketOutput(BaseModel):
    ok: bool


class LogResolutionOutput(BaseModel):
    ok: bool
    indexed: bool


# --------------------------------------------------------------------------
# Fine-tuning shapes (finetune/) - not persisted directly; FineTuneResult's
# fields get copied onto the owning TierConfig.finetune and persisted as
# part of the ProductProfile blob (store/relational.py:ProductProfileRow).
# --------------------------------------------------------------------------

class TrainingExample(BaseModel):
    system: str
    user: str
    assistant: str
    role: str
    issue_id: str


class FineTuneResult(BaseModel):
    model_ref: str
    backend: Literal["openai", "local_lora"]
    base_model: str
    trained_at: datetime = Field(default_factory=utcnow)
    metrics: dict = Field(default_factory=dict)
