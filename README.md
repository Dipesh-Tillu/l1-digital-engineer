# L1 Support Agent

Knowledge-grounded L1 ticket resolution: ingest a product's docs + historical
tickets, draft a resolution grounded only in retrieved content, gate it
through a deterministic confidence check, and auto-resolve or escalate.

See `SPEC.md` for the full design. This implements milestones M0-M4: the
generic framework, provable against synthetic fixture data, with no live
ticketing or wiki credentials required.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in provider keys to run drafts/embeddings/fine-tuning for real
```

`l1agent` is installed *inside* the `.venv` you just created, not on your
system PATH — you must `source .venv/bin/activate` again in **every new
terminal tab/window** before `l1agent ...` will work. If you get
`command not found: l1agent`, that's almost always why: see
[Troubleshooting](#troubleshooting-command-not-found-l1agent) below.

The fastest way to stand up your own product is the interactive installer,
which walks through picking a dataset, an LLM/embedding provider, and
optionally fine-tuning a persona model, then lets you test it immediately:

```bash
l1agent init
```

Or drive each step directly (what `init` calls under the hood), against
the bundled synthetic fixture:

```bash
l1agent onboard --config config/products/acme-widgets.yaml
l1agent ticket submit --product acme-widgets --summary "Refund not received" \
    --body "I returned my widget 2 weeks ago and haven't been refunded."
l1agent eval replay --product acme-widgets --dataset tests/fixtures/acme-widgets/issues.csv
```

Or run the API:

```bash
uvicorn l1agent.api.main:app --reload
```

## Live demo (chat intake + ticket board)

A thin demo layer for showing a real ticket's lifecycle end to end to a
live audience - see `l1-support-agent-demo-workflow-spec.md` for the full
spec. It's a static page (`src/l1agent/api/static/index.html`) served by
the same FastAPI app, calling the exact same agent core as everything
else - no separate reasoning path.

```bash
uvicorn l1agent.api.main:app --reload
# open http://localhost:8000/  (redirects to /demo/)
```

**Before showing this live, both of these must be true, or the demo will
error out or silently produce meaningless results:**

1. **`ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` must be set in `.env`** (real
   values, not the blank placeholders `.env.example` ships with).
   `search_kb`/`search_issues` query the *already-ingested* Chroma
   collections, which were embedded with real Voyage vectors
   (dimension 1024) - without `VOYAGE_API_KEY`, the embedding provider
   silently falls back to a 256-dim local hash provider and every query
   **fails outright** with a Chroma dimension-mismatch error, not just a
   degraded score. Without `ANTHROPIC_API_KEY`, `draft_response` raises
   before it ever gets a citation to check.
2. **At least one category needs `auto_resolve: true`.** As shipped,
   `config/products/powerScale.yaml` had every category's confidence gate
   at `auto_resolve: false` (§7's manual kill switch) - fine for the
   original autonomous-pipeline build, but it means the gate could never
   pass and the demo's Accept path would be unreachable. This has been
   flipped to `true` for all 8 named PowerScale categories (the `'*'`
   catch-all default is left off, same convention as the other two
   product configs) and pushed into the already-onboarded product profile
   row in `l1agent.db` directly - re-review the thresholds in that file if
   `auto_resolve: true` isn't actually appropriate for a real support
   product yet.

The chat panel always creates tickets in **demo mode**
(`demo_mode: true`): a passing confidence gate pauses in "Awaiting Review"
for an explicit Accept/Reject instead of auto-sending. Production/non-demo
callers (the CLI, `POST /tickets` without `demo_mode`, `eval replay`) are
completely unaffected - same auto-resolve-or-escalate behavior as before.

## Configuring the LLM, embeddings, and data sources

Everything is per-product config (`config/products/<id>.yaml`, a
`ProductProfile` - see `src/l1agent/schemas.py`), not a code change:

- **Dataset**: `kb_sources`/`issue_sources` accept a local folder/CSV
  (`type: file`) or a live URL/JSON API (`type: url`, one resource per
  entry - not a crawler; add multiple entries for multiple pages).
- **LLM**: `model.provider` is `anthropic` (default, via the Claude Agent
  SDK) or `openai`; `model.default`/`model.overrides` pick the model id.
- **Embeddings**: `embedding.provider` is `voyage` (default), `openai`, or
  `local` (a dependency-free deterministic fallback used automatically
  when no provider API key is set - fine for a first look, not for real
  semantic retrieval quality).

Install the real provider SDKs with `pip install -e ".[providers]"`
(`voyageai`, `openai`); they're optional and lazily imported.

## Fine-tuning a support-engineer persona (optional)

Beyond RAG-indexing (always on - this framework never answers from
memorized facts, only from what's retrieved), a product can also
fine-tune actual model weights into a persona for one or more support
**tiers** (`ProductProfile.tiers`, e.g. `l1_support`, `l2_support`).
Every training example is itself
retrieval-augmented (persona + this turn's retrieved passages in, the
historical human resolution out), so fine-tuning shapes tone/response
structure, not facts - the confidence gate still verifies every citation
against what was actually retrieved at answer time, fine-tuned or not.

```bash
l1agent finetune run --product acme-widgets --role l1_support --backend local_lora
# or: --backend openai (requires OPENAI_API_KEY; real cost, real time)
```

`local_lora` trains a small open-weight model on your machine via LoRA -
install the heavy extra first: `pip install -e ".[finetune]"` (torch,
transformers, peft, trl, datasets). On a handful of synthetic examples this
proves the mechanism end to end; it is not a production quality claim for
a model this small trained on this little data. Once a tier has a
fine-tuned checkpoint, tickets routed to that tier automatically use it -
no code change, just the product profile getting updated in place.
Add a second `TierConfig` (e.g. `l2_support`) and fine-tune it separately
to move a set of categories to a more capable/differently-trained
checkpoint later.

## Troubleshooting: `command not found: l1agent`

This means your shell can't find the `l1agent` executable — it's not that
anything is broken, it's almost always one of these:

1. **The virtualenv isn't activated in this terminal.** Activation only
   lasts for the current shell session; opening a new tab/window, or
   restarting your terminal, drops it. Fix:
   ```bash
   cd /path/to/l1-support-agent
   source .venv/bin/activate
   ```
   You'll know it worked because your prompt gets a `(.venv)` prefix.
   Confirm with `which l1agent` — it should print a path ending in
   `.venv/bin/l1agent`; if it prints nothing, the venv isn't active.

2. **The package was never installed.** If step 1 shows `(.venv)` active
   but `which l1agent` still finds nothing, run the install:
   ```bash
   pip install -e ".[dev]"
   ```
   from the project root (the directory containing `pyproject.toml`).

3. **You're using a different venv/Python than the one you installed into**
   (e.g. a fresh `python -m venv .venv` overwrote the old one, or you're in
   a different clone of the repo). Re-run steps 1-2 for the venv you're
   actually in.

If you'd rather not depend on the console-script entry point at all, you
can always invoke the CLI as a module instead — this works the same way
as long as the venv's `pip install -e` has run, with no dependency on
PATH:

```bash
python -m l1agent.cli.main --help
python -m l1agent.cli.main init
```

## Prototype auth

The API's `X-API-Key` check against `API_KEYS` is a local-dev convenience,
not a production auth design — see SPEC.md §14.

## Testing

```bash
pytest
```

The test suite runs against the synthetic "Acme Widgets" fixture in
`tests/fixtures/` rather than mocks wherever practical. Embeddings and the
`draft_response` LLM call are both behind swap-out interfaces
(`EmbeddingProvider`, the agent-call hook in `draft_response`); tests inject
deterministic local fakes for these so the suite runs without API keys.
Point `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY`/`OPENAI_API_KEY` at real
credentials to exercise the real providers end to end. The URL connector is
tested against a monkeypatched `httpx`, and fine-tuning's dataset-building
and profile-persistence logic is tested against an injected fake backend
(`tests/test_finetune_runner.py`) - neither test file ever imports `torch`
or `openai`.
