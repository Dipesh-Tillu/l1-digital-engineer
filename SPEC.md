# L1 Support Agent — Low-Level Design & Build Spec

Status: v0.2 (implementation-ready draft) · Companion to `l1-support-agent-blueprint.md` (conceptual architecture) and the rendered blueprint artifact.

## 0. How to use this document

This is written to be handed to Claude Code as a build spec. It assumes the reader has the companion blueprint for *why* each piece exists; this document specifies *what to build*: concrete schemas, interfaces, API contracts, and an ordered backlog of milestones with acceptance criteria. Where a decision was made for you (language, storage, embedding provider), the rationale and the swap-out interface are both given, so implementation can start without a design meeting, and a later swap doesn't touch calling code.

Build in the milestone order in §13. Each milestone is meant to be a self-contained unit of work with its own passing tests before the next one starts.

## 1. System overview

Recap of the four architectural layers from the blueprint, restated as the modules this spec defines:

| Blueprint layer | This spec's module(s) |
|---|---|
| Knowledge Layer | `ingestion/` (connectors, chunking, embedding) + `store/` (vector store, relational store) |
| Agent Core | `agent/` (system prompt, tools, orchestration loop) |
| Confidence gate | `gate/` (rule + score evaluation) |
| Feedback loop | `agent/tools/log_resolution.py` writing back through `store/` |

A thin `api/` service and `cli/` layer sit on top of all four so the framework can run either as a long-lived service (real ticketing integration) or as a local command a developer runs against a folder of sample data (the "plug and play, see how it behaves" use case).

## 2. Implementation decisions & rationale

Every choice below is made behind an interface named in parentheses, so it is a default, not a lock-in.

**Language & runtime — Python 3.11+.** The Claude Agent SDK and every embedding/vector-store client used below have first-class Python support; the ingestion side (PDF/HTML/CSV parsing) is also strongest in Python. If the eventual production ticketing stack is Node/TypeScript, only `api/` needs a thin rewrite — `agent/`, `ingestion/`, and `store/` are all reachable over the REST API in §8.

**Agent runtime — Claude Agent SDK (`claude-agent-sdk`, Python package).** Tools are registered as an in-process MCP server (`create_sdk_mcp_server` + `@tool`) and passed into `ClaudeAgentOptions`; the persona's system prompt and allowed-tools list live in `agent/prompts/` and `agent/options.py`. Pin the SDK version in `pyproject.toml` and re-check `create_sdk_mcp_server`/`@tool`/`ClaudeAgentOptions` signatures against current docs at build time — the shapes below are correct as of this writing but the SDK is young and moves.

**Model selection — a config field, not a code constant (`ModelConfig`).** `product_profile.model` sets the default; `product_profile.model_overrides` can pin a cheaper/faster model to `classify_issue` and a stronger one to `draft_response`, since those two calls have very different accuracy-per-dollar needs. This directly answers "choose the model we need" from the brief — it's a deploy-time choice, never hardcoded.

**Embeddings — Voyage AI (`voyage-3-large` default) behind an `EmbeddingProvider` interface.** Voyage is Anthropic's recommended embeddings partner and has a domain-tuned `voyage-3-large`/`voyage-code-3` family; swapping to OpenAI or a local model is a one-file change in `store/embeddings.py`.

**Vector store — Chroma (local, embedded, on-disk) behind a `VectorStore` interface.** Zero infra to stand up, which matters for the "plug and play against real product data to see how it behaves" prototype goal — a product team can onboard against a laptop or a single container. The interface (`upsert`, `query`, `delete_by_product`) is implemented a second time for Qdrant in `store/vector/qdrant_store.py` as the production swap-in once a deployment needs multi-node scale or managed hosting; nothing above `store/` changes.

**Relational metadata — SQLite by default, Postgres via the same SQLAlchemy models.** Ticket/resolution/product-profile records, not embeddings, live here. `DATABASE_URL` picks the backend; schema in §4.5 is written to be valid in both.

**API — FastAPI.** Async-native, and its Pydantic integration means the schemas in §4 are the request/response models directly — no duplicate DTO layer.

**CLI — Typer.** Thin wrapper over the same service functions the API calls, so `l1agent onboard ...` and `POST /products` never drift apart.

**Testing — pytest, with a synthetic "Acme Widgets" fixture product** (a handful of markdown docs + a CSV of 30 fake historical tickets) checked into `tests/fixtures/`. Every module's tests run against this fixture rather than mocks wherever practical, so the test suite doubles as a runnable demo of the whole pipeline.

## 3. Repository layout

```
l1-support-agent/
├── pyproject.toml
├── .env.example
├── README.md
├── SPEC.md                      # this document
├── config/
│   └── products/
│       └── acme-widgets.yaml    # example product_profile
├── src/l1agent/
│   ├── __init__.py
│   ├── schemas.py                # Pydantic models: §4.1–4.4
│   ├── ingestion/
│   │   ├── connectors/
│   │   │   ├── base.py           # KnowledgeConnector, IssueConnector ABCs
│   │   │   ├── file_connector.py # local files: md/pdf/html/csv/docx
│   │   │   └── wiki_connector.py # Confluence/Notion/SharePoint stub, §5.3
│   │   ├── chunking.py
│   │   └── pipeline.py           # orchestrates connector → chunk → embed → upsert
│   ├── store/
│   │   ├── embeddings.py         # EmbeddingProvider interface + Voyage impl
│   │   ├── vector/
│   │   │   ├── base.py           # VectorStore interface
│   │   │   ├── chroma_store.py
│   │   │   └── qdrant_store.py   # production swap-in, not built in M0–M4
│   │   └── relational.py         # SQLAlchemy models + session, §4.5
│   ├── agent/
│   │   ├── prompts/
│   │   │   └── l1_persona.md
│   │   ├── tools/                # one file per tool, §6.2
│   │   │   ├── get_product_profile.py
│   │   │   ├── search_kb.py
│   │   │   ├── search_issues.py
│   │   │   ├── classify_issue.py
│   │   │   ├── score_confidence.py
│   │   │   ├── draft_response.py
│   │   │   ├── escalate_to_l2.py
│   │   │   ├── update_ticket.py
│   │   │   └── log_resolution.py
│   │   ├── options.py            # ClaudeAgentOptions + model selection, §6.4
│   │   └── runner.py             # orchestration loop, §6.3
│   ├── gate/
│   │   └── confidence.py         # §7
│   ├── api/
│   │   ├── main.py
│   │   └── routes/                # tickets, products, eval — §8
│   ├── cli/
│   │   └── main.py                # §9
│   └── eval/
│       └── replay.py              # §10
└── tests/
    ├── fixtures/acme-widgets/
    │   ├── docs/*.md
    │   └── issues.csv
    ├── test_ingestion.py
    ├── test_agent_tools.py
    ├── test_gate.py
    ├── test_api.py
    └── test_second_product_cold_start.py   # Phase-2 rollout check, run as CI
```

## 4. Configuration & schemas

### 4.1 `ProductProfile`

```python
class ModelConfig(BaseModel):
    default: str = "claude-sonnet-4-5"
    overrides: dict[str, str] = {}   # e.g. {"classify_issue": "claude-haiku-4-5"}

class ConfidenceThreshold(BaseModel):
    category: str                    # taxonomy leaf, or "*" for default
    min_citation_score: float = 0.72
    require_precedent: bool = False  # must also match a past resolved issue
    auto_resolve: bool = False       # category-level kill switch, independent of score

class EscalationPolicy(BaseModel):
    default_queue: str
    category_routes: dict[str, str] = {}   # category -> queue/team id

class ProductProfile(BaseModel):
    product_id: str                  # slug, unique, immutable once created
    display_name: str
    issue_taxonomy: list[str]        # flat list or "parent/child" paths
    thresholds: list[ConfidenceThreshold]
    escalation: EscalationPolicy
    model: ModelConfig = ModelConfig()
    kb_sources: list[SourceRef]      # §4.6 connector config
    issue_sources: list[SourceRef]
    created_at: datetime
```

`SourceRef` is a small tagged union — `{"type": "file", "path": ...}` or `{"type": "wiki", "connector": "confluence", "space_key": ..., "auth_ref": ...}` — resolved by the connector registry in `ingestion/connectors/base.py`.

### 4.2 `KBDocument` (chunk-level, mirrors the vector store payload)

```python
class KBDocument(BaseModel):
    doc_id: str
    product_id: str
    source_uri: str
    section_path: str         # e.g. "Billing > Refunds > Partial refunds"
    chunk_text: str
    chunk_index: int
    token_count: int
    last_synced_at: datetime
```

### 4.3 `IssueRecord`

```python
class IssueStatus(str, Enum):
    open = "open"; resolved = "resolved"; escalated = "escalated"; reopened = "reopened"

class IssueRecord(BaseModel):
    issue_id: str
    product_id: str
    summary: str
    body: str
    category: str | None          # filled by classify_issue if absent on intake
    status: IssueStatus
    resolution_id: str | None
    source_channel: str           # "zendesk" | "email" | "manual_import" | ...
    created_at: datetime
    updated_at: datetime
```

### 4.4 `ResolutionRecord`

```python
class ResolutionAuthor(str, Enum):
    agent = "agent"; l2_human = "l2_human"

class ResolutionOutcome(str, Enum):
    pending = "pending"; accepted = "accepted"; reopened = "reopened"

class ResolutionRecord(BaseModel):
    resolution_id: str
    issue_id: str
    authored_by: ResolutionAuthor
    body: str
    cited_sources: list[str]      # doc_id or issue_id values
    confidence_score: float
    outcome: ResolutionOutcome
    created_at: datetime
```

### 4.5 Relational DDL (SQLite/Postgres-compatible)

```sql
CREATE TABLE product_profile (
    product_id        TEXT PRIMARY KEY,
    display_name       TEXT NOT NULL,
    profile_json        TEXT NOT NULL,   -- full ProductProfile, source of truth
    created_at         TIMESTAMP NOT NULL
);

CREATE TABLE issue_record (
    issue_id           TEXT PRIMARY KEY,
    product_id          TEXT NOT NULL REFERENCES product_profile(product_id),
    summary             TEXT NOT NULL,
    body                 TEXT NOT NULL,
    category             TEXT,
    status               TEXT NOT NULL,
    resolution_id        TEXT,
    source_channel        TEXT NOT NULL,
    created_at           TIMESTAMP NOT NULL,
    updated_at           TIMESTAMP NOT NULL
);
CREATE INDEX idx_issue_product ON issue_record(product_id);

CREATE TABLE resolution_record (
    resolution_id        TEXT PRIMARY KEY,
    issue_id              TEXT NOT NULL REFERENCES issue_record(issue_id),
    authored_by            TEXT NOT NULL,
    body                    TEXT NOT NULL,
    cited_sources           TEXT NOT NULL,  -- JSON array
    confidence_score        REAL NOT NULL,
    outcome                  TEXT NOT NULL,
    created_at              TIMESTAMP NOT NULL
);

CREATE TABLE kb_document_meta (
    doc_id           TEXT PRIMARY KEY,
    product_id        TEXT NOT NULL REFERENCES product_profile(product_id),
    source_uri         TEXT NOT NULL,
    section_path        TEXT,
    last_synced_at       TIMESTAMP NOT NULL
);
```

Embeddings themselves are never stored relationally — only in the vector store, keyed by `doc_id` / `issue_id`. `kb_document_meta` exists so a freshness sweep can run as a plain SQL query without touching the vector store.

### 4.6 Vector store collections

One Chroma collection per product per index: `kb__{product_id}` and `issues__{product_id}`. Every vector's metadata payload carries `{"product_id", "category", "doc_id"|"issue_id", "section_path"}` so a query can filter by `product_id` at the collection level (hard namespace boundary — see the cross-product-leakage risk in the blueprint) and by `category` when the gate needs a category-scoped lookup.

## 5. Ingestion pipeline

### 5.1 Connector interfaces (`ingestion/connectors/base.py`)

```python
class KnowledgeConnector(Protocol):
    def list_documents(self, source: SourceRef) -> Iterator[RawDocument]: ...

class IssueConnector(Protocol):
    def list_issues(self, source: SourceRef) -> Iterator[RawIssue]: ...
    def sync_cursor(self) -> str | None: ...   # for incremental sync, §5.3
```

`RawDocument = {uri, title, content_bytes, content_type}`; `RawIssue` mirrors `IssueRecord` minus the fields the pipeline fills in (category via `classify_issue`, embeddings).

### 5.2 File connector (build first — this is what M1 ships)

Handles local paths and the "uploaded files" source type from the brief: `.md`/`.txt` read directly, `.pdf` via `pypdf`, `.html` via `BeautifulSoup` (strip nav/boilerplate), `.csv` (issue history) parsed with the column-mapping block in the product's `SourceRef` (`{"summary_col": "Subject", "body_col": "Description", "category_col": "Type"}` — real ticket exports rarely use the framework's own field names).

### 5.3 Wiki connector (interface now, Confluence implementation in M5)

Same `KnowledgeConnector` interface; `list_documents` paginates the space, `sync_cursor` stores the space's last-modified watermark so re-syncs are incremental, not full re-index. Auth (`auth_ref`) resolves to a secret name, never a literal credential, in `ProductProfile` — the actual token lives in the secrets store from §12. Notion and SharePoint implementations follow the same shape; ship whichever the pilot product actually uses first (open question, carried over from the blueprint).

### 5.4 Chunking (`ingestion/chunking.py`)

Recursive character/markdown-header splitter, target 400–600 tokens per chunk with 15% overlap; `section_path` is built by tracking heading depth as the splitter walks the document, so a chunk always knows which `H1 > H2 > H3` it came from — this is what lets `search_kb` return something a person (or the confidence gate) can actually check against the source doc, not just a floating paragraph.

### 5.5 Embedding & upsert (`ingestion/pipeline.py`)

`run_ingestion(product_id)`: connector → chunk → `EmbeddingProvider.embed_batch()` (batched at 128, Voyage's practical batch ceiling) → `VectorStore.upsert(collection, ids, vectors, payloads)` → mirror `doc_id`/`source_uri`/`section_path` into `kb_document_meta`. Issue ingestion is the same shape against the `issues__{product_id}` collection, additionally calling `classify_issue` for any row missing a category.

### 5.6 CLI entry points

```
l1agent onboard --config config/products/acme-widgets.yaml   # create profile + full index
l1agent sync --product acme-widgets                           # incremental re-sync
```

## 6. Agent Core

### 6.1 System prompt (`agent/prompts/l1_persona.md`, excerpt)

```
You are a Level 1 support engineer for {{product_display_name}}. You resolve
issues only from what search_kb and search_issues return to you — never from
memory of how similar products usually work. Every claim in your response
must trace to a specific passage or precedent case you retrieved this turn.
If neither tool returns something that actually answers the report, say so
and call escalate_to_l2 rather than guessing. Match the tone of a competent,
plain-spoken colleague: no hedging, no apologizing on the product's behalf,
no promising timelines you cannot see.
```

Product-specific additions (tone notes, forbidden topics, glossary) are appended from an optional `prompt_addendum` field on `ProductProfile` rather than forking this file per product.

### 6.2 Tool definitions

Registered via `@tool` on an SDK MCP server; input/output shapes below (JSON Schema style — implement as Pydantic models and let the SDK derive the schema).

| Tool | Input | Output |
|---|---|---|
| `get_product_profile` | `{}` (bound to the running session's product) | `ProductProfile` (subset: taxonomy, thresholds, escalation) |
| `search_kb` | `{query: str, top_k: int = 6, category: str? }` | `{hits: [{doc_id, section_path, text, score, source_uri}]}` |
| `search_issues` | `{query: str, top_k: int = 5}` | `{hits: [{issue_id, summary, resolution_body, score, outcome}]}` |
| `classify_issue` | `{summary: str, body: str}` | `{category: str, confidence: float}` |
| `score_confidence` | `{draft: str, cited_sources: [str], category: str}` | `{pass: bool, score: float, reasons: [str]}` |
| `draft_response` | `{issue: IssueRecord, kb_hits: [...], issue_hits: [...]}` | `{body: str, cited_sources: [str]}` |
| `escalate_to_l2` | `{issue_id: str, draft: str?, reason: str}` | `{queue: str, ticket_ref: str}` |
| `update_ticket` | `{issue_id: str, status: str, body: str?}` | `{ok: bool}` |
| `log_resolution` | `{resolution: ResolutionRecord}` | `{ok: bool, indexed: bool}` |

`classify_issue` and `score_confidence` are implemented as **plain function calls, not sub-agent turns** — one embedding-similarity lookup against `issue_taxonomy` examples and one rules+score evaluation (§7), respectively. Routing them through another LLM call would add latency and cost without adding accuracy, since both are closed, well-specified decisions once the taxonomy and thresholds exist. Only `draft_response`'s prose generation needs the model.

### 6.3 Orchestration loop (`agent/runner.py`, pseudocode)

```python
def handle_issue(issue: IssueRecord) -> ResolutionRecord:
    profile = get_product_profile(issue.product_id)
    if not issue.category:
        issue.category = classify_issue(issue.summary, issue.body).category

    kb_hits = search_kb(issue_query(issue), category=issue.category)
    issue_hits = search_issues(issue_query(issue))

    draft = draft_response(issue, kb_hits, issue_hits)   # single Claude Agent SDK turn,
                                                           # model = profile.model.overrides
                                                           # .get("draft_response", profile.model.default)
    gate = score_confidence(draft.body, draft.cited_sources, issue.category)

    if gate.pass:
        update_ticket(issue.issue_id, status="resolved", body=draft.body)
        resolution = ResolutionRecord(authored_by="agent", outcome="pending", ...)
    else:
        escalate_to_l2(issue.issue_id, draft=draft.body, reason="; ".join(gate.reasons))
        resolution = ResolutionRecord(authored_by="agent", outcome="pending",
                                       confidence_score=gate.score, ...)  # still logged

    log_resolution(resolution)
    return resolution
```

The `draft_response` call is the only step that actually runs the Claude Agent SDK loop end to end (system prompt + `search_kb`/`search_issues` available as tools, so the model can re-query if the first retrieval was thin, up to a `max_turns` cap in `ClaudeAgentOptions`); the surrounding steps are deterministic orchestration in plain Python, not agent turns. Keep it this way — it's what makes the gate's decision auditable outside the model's own reasoning.

### 6.4 Model selection (`agent/options.py`)

```python
def build_options(profile: ProductProfile, tool_name: str) -> ClaudeAgentOptions:
    model = profile.model.overrides.get(tool_name, profile.model.default)
    return ClaudeAgentOptions(
        model=model,
        mcp_servers={"l1agent": l1agent_tool_server},
        allowed_tools=["search_kb", "search_issues"],
        system_prompt=render_persona(profile),
        max_turns=6,
    )
```

Swapping a product's model — or dropping in a newer Claude model as it becomes available — is a one-line change to `product_profile.model`, no redeploy of `agent/` itself required beyond a config reload.

## 7. Confidence gate (`gate/confidence.py`)

Evaluated in this order, short-circuiting on the first failure (cheapest checks first):

1. **Category gate** — `thresholds[category].auto_resolve` must be `true`. A category not marked auto-resolvable fails here regardless of score; this is the manual kill switch from the rollout plan's Phase 0/1.
2. **Citation gate** — `draft.cited_sources` must be non-empty and every cited `doc_id`/`issue_id` must appear in this turn's actual `search_kb`/`search_issues` results (checked against the tool call log, not trusted from the model's claim) — this is what stops a fabricated citation from passing.
3. **Score gate** — the top cited source's retrieval score must be `>= thresholds[category].min_citation_score`.
4. **Precedent gate** (optional, per category) — if `require_precedent` is set, at least one `search_issues` hit with `outcome == "accepted"` must be among the cited sources.

Output is `{pass, score, reasons}` where `reasons` lists every failed check, not just the first — useful both to the L2 human inheriting the case and to whoever is tuning thresholds later.

## 8. API surface (`api/routes/`)

| Method & path | Body | Response |
|---|---|---|
| `POST /products` | `ProductProfile` (sans `created_at`) | `201`, `{product_id}` — runs full ingestion synchronously for file sources, queues async job for wiki sources |
| `POST /products/{id}/sync` | `{}` | `202`, `{job_id}` |
| `POST /tickets` | `{product_id, summary, body, source_channel}` | `201`, `ResolutionRecord` (runs the full pipeline in §6.3 inline) |
| `GET /tickets/{issue_id}` | — | `IssueRecord` + latest `ResolutionRecord` |
| `POST /eval/replay` | `{product_id, dataset_path}` | `EvalReport`, §10.3 |

Auth for the prototype: a single `X-API-Key` header checked against `API_KEYS` env var (comma-separated); this is explicitly not production auth and is called out as such in `README.md` and in §14.

## 9. CLI surface (`cli/main.py`)

Mirrors §8 one-to-one for local, no-server use:

```
l1agent onboard --config config/products/acme-widgets.yaml
l1agent ticket submit --product acme-widgets --summary "..." --body "..."
l1agent eval replay --product acme-widgets --dataset tests/fixtures/acme-widgets/issues.csv
```

This is the "plug and play, see how it behaves" path from the brief: point `l1agent onboard` at a real product's docs folder and ticket export, then run `l1agent eval replay` against a held-out slice of that same export before anything talks to a live requester.

## 10. Evaluation harness (`eval/replay.py`)

### 10.1 Replay mode

Takes a dataset of `IssueRecord`s that already have a known `resolution_id`/outcome (held-out slice of real historical tickets, never used for indexing), runs each through `handle_issue()` with `update_ticket`/`escalate_to_l2` stubbed to no-ops, and compares the gate's decision plus (when the draft passed) a similarity score between the drafted and actual historical resolution.

### 10.2 Metrics computed

Matches the three numbers from the blueprint's evaluation section: **auto-resolution rate** (share the gate passed), **draft/actual similarity** on passed cases (a proxy for reopen risk, since true reopens don't exist in a replay), and **escalation precision proxy** (share of gate-failed cases where the original human resolution took materially more back-and-forth than a single reply — a rough stand-in until live data exists).

### 10.3 Report

`EvalReport` JSON with per-category breakdowns plus a flat summary; CLI prints the summary table, API returns the full JSON. This is the artifact a product team looks at before flipping any category's `auto_resolve` to `true` in production.

## 11. Testing plan

- **Unit**: chunking boundaries and `section_path` construction; each gate rule in isolation with constructed pass/fail fixtures; `ProductProfile` validation (bad threshold ranges, empty taxonomy).
- **Integration**: full `onboard → ticket submit` against the `acme-widgets` fixture, asserting a known-answerable ticket passes the gate and a known-unanswerable one escalates.
- **`test_second_product_cold_start.py`**: onboard a second synthetic product with a deliberately overlapping vocabulary to the first (e.g. both mention "refund"), submit a ticket to each, and assert neither retrieval crosses into the other's collection. This is the automated version of the blueprint's Phase 2 rollout check — it should be in CI from M2 onward, not deferred to a manual rollout phase.

## 12. Environment & secrets (`.env.example`)

```
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
DATABASE_URL=sqlite:///./l1agent.db
CHROMA_PERSIST_DIR=./chroma_data
API_KEYS=dev-local-key
# per-connector secrets referenced by name from ProductProfile.kb_sources[].auth_ref, e.g.:
CONFLUENCE_ACME_TOKEN=
```

## 13. Implementation backlog

| Milestone | Scope | Acceptance criteria |
|---|---|---|
| **M0 — Scaffold** | Repo layout, `pyproject.toml`, `schemas.py`, empty stubs for every module in §3 | `pytest` collects (even if most tests are `xfail`); `l1agent --help` runs |
| **M1 — Ingestion (files only)** | `file_connector.py`, `chunking.py`, `pipeline.py`, Chroma + SQLite stores | `l1agent onboard` against the `acme-widgets` fixture produces a populated `kb__acme-widgets` collection; chunk count and `section_path`s spot-checked in a test |
| **M2 — Agent Core + gate, single product** | All nine tools, `runner.py`, `gate/confidence.py`, cold-start test in place | `l1agent eval replay` on the fixture dataset produces a non-trivial auto-resolution rate (not 0%, not 100%) and every passed case has a verifiable citation |
| **M3 — API service** | FastAPI app wrapping M1+M2 | `POST /tickets` end to end against a running server, `POST /eval/replay` matches CLI output |
| **M4 — Second product proof** | Second fixture product, cold-start test passing in CI | Onboarding it required zero changes outside `config/products/` |
| **M5 — Wiki connector** | Confluence (or the pilot's actual tool) implementation of `KnowledgeConnector` | Incremental sync verified: a doc edit is reflected in `kb_document_meta.last_synced_at` and re-embedded without a full re-index |
| **M6 — Real ticketing integration** | `update_ticket`/`escalate_to_l2` wired to the actual helpdesk API, replacing the stub | Shadow-mode run (Phase 0 from the blueprint) against a small slice of live traffic, logged not sent |

M0–M4 are the "generic framework, provable with synthetic data" slice — everything needed to satisfy the brief's "plug and play... see how it behaves" ask before a single real credential is involved. M5–M6 are what turns it from a working prototype into the Phase 0 pilot from the blueprint's rollout plan.

## 14. Known gaps to resolve before M6

Carried over from the blueprint, now sharper: which ticketing system's API shape `update_ticket`/`escalate_to_l2` need to match; whether the pilot's wiki is Confluence, Notion, or SharePoint (picks the M5 connector); and a real `API_KEYS` / secrets-manager story before anything here sees production credentials — the `.env` approach in §12 is a prototype convenience, not a production auth design.
