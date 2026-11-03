"""CLI surface (SPEC.md §9). Mirrors api/routes/*.py one-to-one for local,
no-server use - it calls the exact same service functions
(`run_ingestion`, `handle_issue`, `replay`) so the two surfaces can't drift
apart.
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from l1agent.agent.runner import handle_issue
from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.context import AppContext
from l1agent.eval.replay import replay
from l1agent.finetune.runner import run_finetune
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import (
    ConfidenceThreshold,
    EmbeddingConfig,
    EscalationPolicy,
    FileSourceRef,
    IssueRecord,
    IssueStatus,
    ModelConfig,
    ProductProfile,
    TierConfig,
    UrlSourceRef,
)
from l1agent.store.embeddings import get_provider_for
from l1agent.store.relational import get_product_profile, session_scope, upsert_product_profile

app = typer.Typer(help="L1 Support Agent - local CLI mirroring the API surface.")
ticket_app = typer.Typer(help="Manage tickets.")
eval_app = typer.Typer(help="Run the evaluation harness.")
finetune_app = typer.Typer(help="Fine-tune a support-tier persona model.")
app.add_typer(ticket_app, name="ticket")
app.add_typer(eval_app, name="eval")
app.add_typer(finetune_app, name="finetune")


def _classify_fn(embedding_provider):
    def _fn(summary: str, body: str, profile: ProductProfile):
        return classify_issue(summary, body, profile, embedding_provider)

    return _fn


@app.command()
def onboard(config: Path = typer.Option(..., exists=True, help="Path to a product_profile YAML file.")):
    """Create a product profile and run a full ingestion (SPEC.md §5.6)."""
    profile = ProductProfile.model_validate(yaml.safe_load(config.read_text()))
    ctx = AppContext()
    embedding_provider = get_provider_for(profile.embedding)
    with session_scope() as session:
        upsert_product_profile(session, profile)
        session.flush()
        report = run_ingestion(profile, embedding_provider, ctx.vector_store, session, _classify_fn(embedding_provider))
    typer.echo(
        f"Onboarded {profile.product_id!r}: {report.kb_chunks_indexed} KB chunks, "
        f"{report.issues_indexed} issues indexed."
    )


@app.command()
def sync(product: str = typer.Option(..., help="product_id to re-sync.")):
    """Incremental re-sync (SPEC.md §5.6)."""
    ctx = AppContext()
    with session_scope() as session:
        profile = get_product_profile(session, product)
        if profile is None:
            typer.echo(f"error: product {product!r} not found", err=True)
            raise typer.Exit(code=1)
        embedding_provider = get_provider_for(profile.embedding)
        report = run_ingestion(profile, embedding_provider, ctx.vector_store, session, _classify_fn(embedding_provider))
    typer.echo(f"Synced {product!r}: {report.kb_chunks_indexed} KB chunks, {report.issues_indexed} issues indexed.")


@ticket_app.command("submit")
def ticket_submit(
    product: str = typer.Option(..., help="product_id to file the ticket against."),
    summary: str = typer.Option(...),
    body: str = typer.Option(...),
    source_channel: str = typer.Option("manual_import"),
):
    """Submit a ticket and run the full pipeline (SPEC.md §6.3)."""
    ctx = AppContext()
    with session_scope() as session:
        profile = get_product_profile(session, product)
        if profile is None:
            typer.echo(f"error: product {product!r} not found", err=True)
            raise typer.Exit(code=1)
        issue = IssueRecord(
            issue_id=str(uuid.uuid4()),
            product_id=product,
            summary=summary,
            body=body,
            status=IssueStatus.open,
            source_channel=source_channel,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        embedding_provider = get_provider_for(profile.embedding)
        resolution = handle_issue(issue, profile, session, embedding_provider, ctx.vector_store)
    typer.echo(json.dumps(resolution.model_dump(mode="json"), indent=2))


@eval_app.command("replay")
def eval_replay(
    product: str = typer.Option(..., help="product_id to evaluate."),
    dataset: Path = typer.Option(..., exists=True, help="Held-out ticket CSV, same columns as issue_sources."),
):
    """Replay a held-out dataset through the pipeline and print an EvalReport (SPEC.md §10)."""
    ctx = AppContext()
    with session_scope() as session:
        profile = get_product_profile(session, product)
        if profile is None:
            typer.echo(f"error: product {product!r} not found", err=True)
            raise typer.Exit(code=1)
        embedding_provider = get_provider_for(profile.embedding)
        report = replay(profile, str(dataset), embedding_provider, ctx.vector_store)

    typer.echo(f"product: {report.product_id}")
    typer.echo(f"total_cases: {report.total_cases}")
    typer.echo(f"auto_resolution_rate: {report.auto_resolution_rate:.2%}")
    sim = report.avg_draft_actual_similarity
    typer.echo(f"avg_draft_actual_similarity: {sim:.3f}" if sim is not None else "avg_draft_actual_similarity: n/a")
    prec = report.escalation_precision_proxy
    typer.echo(f"escalation_precision_proxy: {prec:.2%}" if prec is not None else "escalation_precision_proxy: n/a")
    typer.echo("by_category:")
    for cat in report.by_category:
        typer.echo(f"  {cat['category']}: {json.dumps(cat)}")


@finetune_app.command("run")
def finetune_run(
    product: str = typer.Option(..., help="product_id to fine-tune."),
    role: str = typer.Option(..., help="Tier role to fine-tune, e.g. l1_support or l2_support."),
    backend: str = typer.Option(
        None, help="Override the tier's configured backend for this run: 'local_lora' or 'openai'."
    ),
):
    """Fine-tune a support-tier persona model on this product's own
    historical resolved tickets (finetune/runner.py). Requires either
    OPENAI_API_KEY (backend=openai, real cost/time) or the 'finetune'
    extra installed (backend=local_lora, heavy local compute)."""
    ctx = AppContext()
    with session_scope() as session:
        profile = get_product_profile(session, product)
        if profile is None:
            typer.echo(f"error: product {product!r} not found", err=True)
            raise typer.Exit(code=1)
        tier = next((t for t in profile.tiers if t.role == role), None)
        if tier is None:
            typer.echo(f"error: product {product!r} has no tier with role {role!r}", err=True)
            raise typer.Exit(code=1)
        if backend:
            tier.finetune.backend = backend

        embedding_provider = get_provider_for(profile.embedding)
        typer.echo(f"Building training examples for role {role!r}...")
        try:
            result = run_finetune(profile, role, session, embedding_provider, ctx.vector_store)
        except (ValueError, RuntimeError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Fine-tuned {product!r}/{role!r} via {result.backend}: model_ref={result.model_ref!r}")
    typer.echo(f"metrics: {json.dumps(result.metrics)}")


def _peek_csv_headers(path: str) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), [])
    except OSError:
        return []


def _best_match(headers: list[str], candidates: list[str]) -> str:
    lowered = {h.lower(): h for h in headers}
    for cand in candidates:
        for h_lower, h in lowered.items():
            if cand in h_lower:
                return h
    return candidates[0]


def _prompt_kb_source() -> FileSourceRef | UrlSourceRef:
    kind = typer.prompt("Knowledge base source - local folder or URL? [local/url]", default="local")
    if kind == "url":
        url = typer.prompt("KB URL")
        auth_ref = typer.prompt("Env var holding a bearer token (blank if none)", default="")
        return UrlSourceRef(url=url, auth_ref=auth_ref or None)
    path = typer.prompt("Path to local docs folder")
    return FileSourceRef(path=path)


def _prompt_issue_source() -> FileSourceRef | UrlSourceRef:
    kind = typer.prompt("Ticket history source - local CSV or URL/API? [local/url]", default="local")
    if kind == "url":
        url = typer.prompt("Ticket API URL")
        auth_ref = typer.prompt("Env var holding a bearer token (blank if none)", default="")
        json_records_path = typer.prompt(
            "Dotted path to the record list in the JSON response (blank if the response is a top-level array)",
            default="",
        )
        summary_col = typer.prompt("JSON field for the ticket summary", default="summary")
        body_col = typer.prompt("JSON field for the ticket body", default="body")
        category_col = typer.prompt("JSON field for category (blank if none)", default="")
        resolution_col = typer.prompt("JSON field for the historical resolution text (blank if none)", default="resolution")
        outcome_col = typer.prompt("JSON field for outcome (blank if none)", default="")
        return UrlSourceRef(
            url=url, auth_ref=auth_ref or None, json_records_path=json_records_path or None,
            summary_col=summary_col, body_col=body_col,
            category_col=category_col or None, resolution_col=resolution_col or None,
            outcome_col=outcome_col or None,
        )
    path = typer.prompt("Path to local ticket CSV")
    headers = _peek_csv_headers(path)
    summary_col = typer.prompt("CSV column for the ticket summary", default=_best_match(headers, ["summary", "subject"]))
    body_col = typer.prompt("CSV column for the ticket body", default=_best_match(headers, ["body", "description"]))
    category_col = typer.prompt("CSV column for category (blank if none)", default=_best_match(headers, ["category", "type"]))
    resolution_col = typer.prompt(
        "CSV column for the historical resolution text (blank if none)", default=_best_match(headers, ["resolution"])
    )
    outcome_col = typer.prompt("CSV column for outcome (blank if none)", default=_best_match(headers, ["outcome"]))
    return FileSourceRef(
        path=path, summary_col=summary_col, body_col=body_col,
        category_col=category_col or None, resolution_col=resolution_col or None,
        outcome_col=outcome_col or None,
    )


def _run_finetune_step(profile: ProductProfile, role: str, embedding_provider, vector_store) -> None:
    backend = typer.prompt(f"Fine-tune backend for {role!r} [local_lora/openai]", default="local_lora")
    tier = next(t for t in profile.tiers if t.role == role)
    tier.finetune.backend = backend
    with session_scope() as session:
        upsert_product_profile(session, profile)
        session.flush()
        try:
            result = run_finetune(profile, role, session, embedding_provider, vector_store)
            typer.echo(f"Fine-tuned {role!r} via {result.backend}: model_ref={result.model_ref!r}")
        except (ValueError, RuntimeError) as e:
            typer.echo(f"warning: fine-tuning {role!r} failed: {e}", err=True)


@app.command()
def init():
    """Interactive installer: configure a product, ingest its docs/tickets,
    optionally fine-tune a support-tier persona, and test it end to end.
    A guided front door to the same onboard/ticket-submit/finetune-run
    functions used elsewhere in this CLI - nothing here is a separate
    code path."""
    typer.echo("L1 Support Agent - product setup wizard\n")

    product_id = typer.prompt("Product ID (slug)")
    display_name = typer.prompt("Display name", default=product_id.replace("-", " ").title())

    typer.echo("")
    kb_source = _prompt_kb_source()
    typer.echo("")
    issue_source = _prompt_issue_source()

    typer.echo("")
    taxonomy_raw = typer.prompt("Issue taxonomy (comma-separated categories)")
    categories = [c.strip() for c in taxonomy_raw.split(",") if c.strip()]
    if not categories:
        typer.echo("error: at least one category is required", err=True)
        raise typer.Exit(code=1)
    thresholds = [ConfidenceThreshold(category="*")] + [ConfidenceThreshold(category=c) for c in categories]
    typer.echo(
        "Every category defaults to auto_resolve=false (manual kill switch) - "
        f"edit config/products/{product_id}.yaml and flip a category on once `l1agent eval replay` looks good."
    )

    default_queue = typer.prompt("Default escalation queue", default="l2-general")
    escalation = EscalationPolicy(default_queue=default_queue)

    typer.echo("")
    llm_provider = typer.prompt("LLM provider [anthropic/openai]", default="anthropic")
    default_model = "claude-sonnet-4-5" if llm_provider == "anthropic" else "gpt-4o-mini"
    llm_model = typer.prompt("Model", default=default_model)
    llm_key_var = "ANTHROPIC_API_KEY" if llm_provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(llm_key_var):
        typer.echo(f"warning: {llm_key_var} is not set - drafting will fail until it is.")
    model_config = ModelConfig(provider=llm_provider, default=llm_model)

    embed_provider = typer.prompt("Embedding provider [voyage/openai/local]", default="voyage")
    embed_key_var = {"voyage": "VOYAGE_API_KEY", "openai": "OPENAI_API_KEY", "local": None}[embed_provider]
    if embed_key_var and not os.environ.get(embed_key_var):
        typer.echo(f"warning: {embed_key_var} is not set - falling back to local hash embeddings (lower retrieval quality) until it is.")
    embedding_config = EmbeddingConfig(provider=embed_provider)

    profile = ProductProfile(
        product_id=product_id,
        display_name=display_name,
        issue_taxonomy=categories,
        thresholds=thresholds,
        escalation=escalation,
        model=model_config,
        embedding=embedding_config,
        kb_sources=[kb_source],
        issue_sources=[issue_source],
    )

    config_path = Path("config/products") / f"{product_id}.yaml"
    if config_path.exists() and not typer.confirm(f"\n{config_path} already exists - overwrite?", default=False):
        raise typer.Exit(code=1)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(json.loads(profile.model_dump_json()), sort_keys=False))
    typer.echo(f"Wrote {config_path}")

    ctx = AppContext()
    embedding_provider = get_provider_for(profile.embedding)
    typer.echo("\nIngesting (RAG-indexing) the knowledge base and ticket history...")
    with session_scope() as session:
        upsert_product_profile(session, profile)
        session.flush()
        report = run_ingestion(profile, embedding_provider, ctx.vector_store, session, _classify_fn(embedding_provider))
    typer.echo(f"Indexed {report.kb_chunks_indexed} KB chunks, {report.issues_indexed} issues.")

    if typer.confirm("\nFine-tune a support-engineer persona now (l1_support)?", default=False):
        _run_finetune_step(profile, "l1_support", embedding_provider, ctx.vector_store)

        if typer.confirm("Also create a second tier (l2_support) now?", default=False):
            l2_raw = typer.prompt(f"Which categories should route to l2_support? (comma-separated, from: {', '.join(categories)})")
            l2_categories = [c.strip() for c in l2_raw.split(",") if c.strip() in categories]
            if not l2_categories:
                typer.echo("No valid categories entered - skipping l2_support tier.")
            else:
                profile.tiers.append(TierConfig(role="l2_support", categories=l2_categories))
                _run_finetune_step(profile, "l2_support", embedding_provider, ctx.vector_store)

    if typer.confirm("\nTest it now with a sample ticket?", default=True):
        summary = typer.prompt("Ticket summary")
        body = typer.prompt("Ticket body")
        issue = IssueRecord(
            issue_id=str(uuid.uuid4()),
            product_id=product_id,
            summary=summary,
            body=body,
            status=IssueStatus.open,
            source_channel="manual_import",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        try:
            with session_scope() as session:
                db_profile = get_product_profile(session, product_id)
                resolution = handle_issue(issue, db_profile, session, embedding_provider, ctx.vector_store)
            typer.echo(json.dumps(resolution.model_dump(mode="json"), indent=2))
        except Exception as e:  # drafting can fail on missing API keys, network, etc. - surface it plainly
            typer.echo(f"error running the sample ticket: {e}", err=True)


if __name__ == "__main__":
    app()
