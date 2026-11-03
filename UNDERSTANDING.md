# Understanding: L1 Support Agent

A working-notes companion to `README.md` (quickstart) and `SPEC.md` (full
design spec). This file summarizes *what the code actually does today*,
verified against the source — useful for getting oriented fast without
reading all ~30 source files or the 450-line spec.

## 1. What this project is

An L1 (tier-1) customer-support ticket resolver that only answers from
retrieved knowledge — a product's docs and its own past resolved tickets —
never from the model's general knowledge. Every drafted reply must cite a
specific passage or precedent case it actually retrieved in that turn. A
deterministic, non-LLM "confidence gate" then decides whether the draft is
trustworthy enough to auto-resolve the ticket, or whether it must escalate
to a human (L2).

It's built to be **multi-tenant by product**: each product ("Acme Widgets",
"Beta Gadgets", ...) gets its own config, its own vector-store namespace, and
its own confidence thresholds — onboarding a new product requires zero code
changes, only a new YAML file under `config/products/` (or a guided run of
`l1agent init`, §7).

Everything about the pipeline is configurable per product, not hardcoded:
which LLM drafts replies (Anthropic or OpenAI), which embedding provider
indexes the data (Voyage, OpenAI, or a dependency-free local fallback),
and where the docs/tickets come from (local files or a live URL/JSON API).
On top of that, a product can optionally **fine-tune actual model weights**
into a persona for one or more support tiers (L1, L2, ...) — see §4 and §8.

Status: implements milestones M0–M4 from `SPEC.md` §13 (the generic
framework, provable against synthetic fixture data, no live ticketing/wiki
credentials required) plus later additions beyond the original spec:
multi-provider LLM/embedding config, a URL/API ingestion connector, the
`l1agent init` installer wizard, and the weight-fine-tuning subsystem.
(M5 wiki connector and M6 real ticketing integration are still not built —
see §8 below.)

## 2. The pipeline, end to end

```
docs + ticket CSV/URL  →  ingestion  →  vector store + relational DB
                                            │
new ticket arrives                         │
      │                                     ▼
      ├─ classify_issue  (embedding similarity, no LLM call)
      ├─ search_kb + search_issues  (vector similarity lookups)
      ├─ draft_response  (the ONLY real LLM call — see below for which backend)
      └─ score_confidence  (deterministic rule+score gate, no LLM)
                │
        pass ───┴─── fail
         │             │
   auto-resolve   escalate_to_l2
         │             │
         └──> log_resolution (writes back to DB + vector store)
```

This whole sequence is `handle_issue()` in `src/l1agent/agent/runner.py:40`.
Only one step (`draft_response`) touches an LLM at all; everything else is
plain deterministic Python. That's a deliberate design choice (see §5) —
it's what makes the auto-resolve/escalate decision auditable independent of
the model's own reasoning.

`draft_response` itself dispatches to one of three backends depending on
the ticket's category and the product's config
(`agent/tools/draft_response.py:_default_agent_call`,
`ProductProfile.tier_for(category)`):

1. A **fine-tuned local LoRA checkpoint**, if the ticket's tier has one
   (`TierConfig.finetune.backend == "local_lora"` and `model_ref` set) —
   single-shot generation, no tool loop (see §4).
2. A **fine-tuned OpenAI checkpoint**, if the tier has one instead
   (`backend == "openai"`) — same multi-turn tool-calling loop as (3),
   just with `model=` pointed at the fine-tuned model id.
3. Otherwise, the product's **base provider** (`ProductProfile.model.provider`):
   Anthropic via the Claude Agent SDK, or OpenAI via a hand-rolled
   tool-calling loop — both let the model re-call `search_kb`/`search_issues`
   if the first retrieval was thin, up to `max_turns`.

A fine-tuned checkpoint always takes priority for its tier; nothing here
changes what the confidence gate checks afterward.

## 3. Module map (what's actually in `src/l1agent/`)

| Module | Role | Key file(s) |
|---|---|---|
| `schemas.py` | All Pydantic models: `ProductProfile`, `IssueRecord`, `ResolutionRecord`, tool I/O shapes, `TierConfig`/`FineTuneConfig` | single file |
| `context.py` | `AppContext` (wires vector store + DB session factory) and `TurnContext` (per-ticket state used by the gate to verify citations) | |
| `ingestion/` | Turns docs/CSVs/URLs into embedded, searchable chunks | `connectors/file_connector.py`, `connectors/url_connector.py`, `connectors/wiki_connector.py` (stub), `chunking.py`, `pipeline.py` |
| `store/` | Persistence: embeddings, vector search, relational metadata | `embeddings.py` (Voyage/OpenAI/local + `get_provider_for` factory), `vector/chroma_store.py` (+ `qdrant_store.py` unused swap-in), `relational.py` (SQLAlchemy) |
| `agent/` | The 9 tools + orchestration + LLM provider dispatch | `runner.py`, `options.py`, `prompts/{l1_support,l2_support}.md`, `tools/*.py` |
| `gate/confidence.py` | The 4-stage deterministic pass/fail gate | |
| `finetune/` | Weight-level fine-tuning: build a training set from a product's own history, run it through a backend, persist the checkpoint onto the tier | `dataset.py`, `backends/{openai_backend,local_lora_backend}.py`, `runner.py` |
| `api/` | FastAPI service exposing the same operations over HTTP | `main.py`, `routes/{products,tickets,eval}.py` |
| `cli/main.py` | Typer CLI — calls the *same* underlying functions as the API, so the two never drift | includes the `init` wizard and `finetune run` |
| `eval/replay.py` | Replays a held-out ticket dataset through the pipeline and reports auto-resolution rate etc. | |

## 4. The 9 agent tools

Defined in `agent/tools/`. For the Anthropic path they're wired into an
in-process MCP server (`agent/options.py`); the OpenAI path calls the same
underlying functions directly through a hand-rolled tool-calling loop
(`agent/tools/draft_response.py:_openai_agent_call`) instead of MCP. Only
`draft_response` ever touches an LLM:

| Tool | Uses LLM? | What it does |
|---|---|---|
| `get_product_profile` | No | Returns the current product's taxonomy/thresholds/escalation config |
| `search_kb` | No | Vector search over the product's KB chunks (Chroma) |
| `search_issues` | No | Vector search over the product's past resolved tickets |
| `classify_issue` | No | Embedding-similarity match against taxonomy examples — **not** an LLM call |
| `score_confidence` | No | The 4-stage gate in `gate/confidence.py` |
| `draft_response` | **Yes** | Dispatches to Anthropic, OpenAI, or a fine-tuned checkpoint (§2) — the only real LLM call |
| `escalate_to_l2` | No | Routes to a queue per `escalation.category_routes` |
| `update_ticket` | No | Writes status/body to the relational store |
| `log_resolution` | No | Persists the `ResolutionRecord` and indexes it for future `search_issues` hits |

This split (`SPEC.md` §6.2) is intentional: `classify_issue` and
`score_confidence` are "closed, well-specified decisions once the taxonomy
and thresholds exist" — routing them through an LLM would add latency/cost
with no accuracy gain.

A fine-tuned **local LoRA** checkpoint skips the tool-calling loop entirely
(`_local_agent_call`): `runner.py` already ran `search_kb`/`search_issues`
before calling `draft_response` at all, so the local model gets one
persona+context prompt and returns one completion, parsed for a trailing
`CITED_SOURCES: [...]` marker it was trained to produce
(`finetune/dataset.py` — see §8).

## 5. The confidence gate — the core trust mechanism

`gate/confidence.py`, evaluated in order, short-circuiting on first failure:

1. **Category gate** — the issue's category must have `auto_resolve: true`
   in the product's config. A manual kill-switch per category, independent
   of anything the model did.
2. **Citation gate** — the draft must cite at least one source, and every
   cited `doc_id`/`issue_id` must be checked **against `TurnContext`** (the
   actual tool-call log from this turn), not trusted from the model's own
   claim. This is what stops a hallucinated citation from passing.
3. **Score gate** — the top cited source's retrieval similarity score must
   clear `min_citation_score` for that category.
4. **Precedent gate** (optional per category) — if `require_precedent` is
   set, at least one cited source must be a past issue with
   `outcome == "accepted"`.

Output always lists *every* failed reason, not just the first one that
short-circuited scoring — useful for both the L2 human and later threshold
tuning. See `config/products/acme-widgets.yaml` for a real example: `*`
(default) is not auto-resolvable, but `billing/refunds` and
`shipping/delays` are, each with its own threshold.

## 6. Testability — no API keys required

Swap-out interfaces make the whole pipeline (RAG and fine-tuning both)
runnable and testable without any credentials or heavy dependencies:

- `EmbeddingProvider` (`store/embeddings.py`) — real impls call Voyage AI
  or OpenAI; tests inject a deterministic local fake. `get_provider_for`
  falls back to that same local fake automatically whenever a configured
  provider's API key isn't set, in production code too, not just tests.
- The `agent_call` hook inside `draft_response` (`agent/tools/draft_response.py`)
  — real impls run Anthropic/OpenAI/local-LoRA; tests inject a fake that
  returns a canned `DraftResponseOutput`. See `tests/conftest.py`.
- `FineTuneBackend` (`finetune/backends/base.py`) — real impls call the
  OpenAI fine-tuning API or train a local model; tests inject a fake
  backend to verify `finetune/runner.py`'s dataset-building and
  profile-persistence plumbing (`tests/test_finetune_runner.py`) without
  ever importing `torch` or `openai`.
- `httpx.get` is monkeypatched in `tests/test_url_connector.py` — the URL
  connector is tested with no real network calls either.

This is why `pytest` (60 tests, all passing as of this check) runs clean
with no `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY`/`OPENAI_API_KEY` set and
without the heavy `finetune` extra installed — the synthetic "Acme
Widgets" fixture in `tests/fixtures/` doubles as a runnable demo of the
entire pipeline.

`tests/test_second_product_cold_start.py` is worth calling out specifically:
it onboards a second product ("Beta Gadgets") with deliberately overlapping
vocabulary (both mention "refund") and asserts retrieval never crosses
product boundaries — the automated proof of the per-product namespace
isolation described in §4.6 of `SPEC.md`.

## 7. Two ways to drive it: CLI and API

Both are thin wrappers over the same functions (`run_ingestion`,
`handle_issue`, `replay`) — see `cli/main.py` and `api/routes/*.py` — so
they can't drift apart.

**CLI** (`l1agent`, entry point in `pyproject.toml`):
- `l1agent init` — interactive installer: configure a product (dataset,
  LLM/embedding provider), ingest it, optionally fine-tune a persona, and
  test it with a sample ticket — a guided front door to the commands below
- `l1agent onboard --config <yaml>` — create a product profile + full ingestion
- `l1agent sync --product <id>` — incremental re-sync
- `l1agent ticket submit --product <id> --summary ... --body ...` — runs the full pipeline, prints the `ResolutionRecord`
- `l1agent eval replay --product <id> --dataset <csv>` — replay a held-out dataset, print metrics
- `l1agent finetune run --product <id> --role <tier> [--backend ...]` — fine-tune a support tier's persona model on that product's own history

**API** (`uvicorn l1agent.api.main:app`):
- `POST /products`, `POST /products/{id}/sync`, `POST /tickets`,
  `GET /tickets/{issue_id}`, `POST /eval/replay`, `GET /healthz`
- Auth is a single `X-API-Key` header checked against the `API_KEYS` env var
  — explicitly called out (README + SPEC §14) as a local-dev convenience,
  **not** production auth.

## 8. What's not built yet (honest gaps)

- **M5 — real wiki connector**: `ingestion/connectors/wiki_connector.py`
  exists only as an interface stub (Confluence/Notion/SharePoint TBD —
  "ship whichever the pilot product actually uses first"). The URL/API
  connector (`url_connector.py`) covers the "live source" case for a
  single doc page or JSON ticket endpoint in the meantime, but it fetches
  one resource per config entry — it is not a crawler.
- **M6 — real ticketing integration**: `update_ticket`/`escalate_to_l2`
  currently write to the local relational store, not a live helpdesk API.
- **Vector store production swap**: `store/vector/qdrant_store.py` exists
  as the documented production alternative to Chroma but isn't exercised —
  Chroma (local, embedded, on-disk) is what actually runs, for every
  product regardless of its LLM/embedding provider choice.
- **Secrets management**: `.env` file is explicitly a prototype convenience,
  not a production secrets story (`SPEC.md` §14).
- **Fine-tuning is a proof of mechanism, not a quality claim**: the
  `local_lora` backend trains a small (1-3B parameter) open-weight model
  on however many historical resolved tickets a product has — for the
  synthetic fixtures, a few dozen. That's enough to prove a tier can be
  routed to a locally fine-tuned checkpoint end to end; it is not enough
  data or model capacity to expect production-grade output. Historical
  resolutions also don't carry machine-checkable citations, so each
  training example's `CITED_SOURCES` line is a heuristic (the
  top-retrieved passage at generation time), not verified ground truth —
  see `finetune/dataset.py`'s docstring.
- **No fine-tune run history**: `TierConfig.finetune` only remembers the
  *latest* run per tier (persisted directly on the `ProductProfile` blob,
  `store/relational.py:ProductProfileRow.profile_json`). A full audit-log
  table of every run would be a reasonable future addition.
- **Anthropic has no public fine-tuning API**: this is a hard external
  constraint, not a scope choice — a tier can only get a fine-tuned
  checkpoint via the `openai` or `local_lora` backend. A tier without one
  simply drafts through the product's configured base provider.

## 9. How to run it

Prereqs: Python 3.11+ (repo's `.venv` is on 3.12.10, already has the
package installed editable).

```bash
# one-time setup (already done in this checkout's .venv/)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional — only needed for real LLM/embedding/fine-tuning calls

# guided setup for your own product (dataset, LLM/embedding provider,
# optional fine-tuning, then a live test) - or onboard the fixture directly:
l1agent init
l1agent onboard --config config/products/acme-widgets.yaml

# file a ticket end-to-end (uses the product's configured LLM provider if
# keys are set in .env; otherwise this specific command path needs the
# keys - tests use fakes instead)
l1agent ticket submit --product acme-widgets \
  --summary "Refund not received" \
  --body "I returned my widget 2 weeks ago and haven't been refunded."

# replay a held-out dataset and see aggregate metrics
l1agent eval replay --product acme-widgets --dataset tests/fixtures/acme-widgets/issues.csv

# optional: fine-tune a persona for one tier (needs OPENAI_API_KEY for
# --backend openai, or `pip install -e ".[finetune]"` for --backend local_lora)
l1agent finetune run --product acme-widgets --role l1_support --backend local_lora

# or run it as a service instead of the CLI
uvicorn l1agent.api.main:app --reload
```

Run the test suite (no API keys or heavy dependencies needed — everything
external is faked deterministically):

```bash
source .venv/bin/activate
pytest        # 60 tests, all passing as of this check
```

To see the pipeline behave on a *second* product with overlapping
vocabulary (proving namespace isolation), look at
`config/products/beta-gadgets.yaml` and
`tests/test_second_product_cold_start.py`.

## 10. Where to look next, by question

- "How does a chunk know its position in a doc?" → `ingestion/chunking.py`
  (recursive markdown-header splitter, `section_path` tracks heading depth)
- "How do I add a new product?" → `l1agent init` (guided), or write a new
  `config/products/<id>.yaml` matching `ProductProfile` in `schemas.py`
  and run `l1agent onboard --config ...`
- "How do I point a product at a live URL/API instead of local files?" →
  a `type: url` entry in `kb_sources`/`issue_sources`
  (`UrlSourceRef` in `schemas.py`, one resource per entry)
- "How do I change which LLM or embedding provider a product uses?" →
  `model.provider`/`model.default`/`model.overrides` and
  `embedding.provider`/`embedding.model` in the product's YAML — see
  README's "Configuring the LLM, embeddings, and data sources"
- "How do I make a tier use a fine-tuned model instead of the base
  provider?" → `l1agent finetune run --product <id> --role <tier>`; it
  writes `TierConfig.finetune.model_ref` onto the profile, which
  `draft_response`'s dispatcher checks before falling back to
  `model.provider` (`agent/tools/draft_response.py:_default_agent_call`)
- "How is a fabricated citation caught?" → `TurnContext` in `context.py`
  records real tool-call results; `gate/confidence.py`'s citation gate
  checks against that log, never against the model's own claim — this
  holds regardless of which LLM backend or fine-tuned checkpoint drafted
  the reply
- "Full design rationale for any of the above" → `SPEC.md` for the
  original M0–M4 design (organized by the section numbers referenced
  throughout this file and in the code's own docstrings/comments); the
  multi-provider/URL/fine-tuning additions are documented in code
  docstrings (`schemas.py`, `finetune/`) rather than a SPEC.md section,
  since they were added after that document was written
