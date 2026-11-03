"""ClaudeAgentOptions + model selection (SPEC.md §6.4).

Swapping a product's model - or dropping in a newer Claude model - is a
one-line change to `product_profile.model`; no redeploy of `agent/` beyond a
config reload.

NOTE: `claude-agent-sdk` is young and its exact signatures move; SPEC.md §2
explicitly calls out re-checking `create_sdk_mcp_server`/`@tool`/
`ClaudeAgentOptions` against current docs at build time. This module is
correct as of this writing but is the one place in the codebase that
depends on that external contract - if the SDK's API has shifted, this is
the file to update.
"""
from __future__ import annotations

from pathlib import Path

from l1agent.schemas import ProductProfile

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_ROLE = "l1_support"


def render_persona(profile: ProductProfile, role: str = _DEFAULT_ROLE) -> str:
    """`role` picks `agent/prompts/{role}.md` (schemas.TierConfig.role) so
    each support tier can have its own persona depth/tone while every
    persona still shares the same "only from retrieved content" rule.
    Falls back to the L1 persona for an unrecognized role rather than
    erroring, since a role string is free-form product config."""
    persona_path = _PROMPTS_DIR / f"{role}.md"
    if not persona_path.exists():
        persona_path = _PROMPTS_DIR / f"{_DEFAULT_ROLE}.md"
    template = persona_path.read_text(encoding="utf-8")
    return (
        template.replace("{{product_display_name}}", profile.display_name)
        .replace("{{prompt_addendum}}", profile.prompt_addendum or "")
        .strip()
    )


def model_for(profile: ProductProfile, tool_name: str) -> str:
    return profile.model.overrides.get(tool_name, profile.model.default)


def build_options(profile: ProductProfile, tool_name: str, mcp_server, role: str = _DEFAULT_ROLE):
    """Returns a `claude_agent_sdk.ClaudeAgentOptions` configured for one
    tool's model + persona + retrieval tools. Only `draft_response` (§6.3)
    actually constructs a client against this; everything else in the
    pipeline is deterministic Python. `mcp_server` is product-bound - build
    it with `build_mcp_server_for()`."""
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        model=model_for(profile, tool_name),
        mcp_servers={"l1agent": mcp_server},
        allowed_tools=[
            "mcp__l1agent__search_kb",
            "mcp__l1agent__search_issues",
            "mcp__l1agent__submit_draft",
        ],
        system_prompt=render_persona(profile, role),
        max_turns=6,
    )


def build_mcp_server_for(profile: ProductProfile, embedding_provider, vector_store, turn):
    """Product-bound MCP server: search_kb/search_issues call the real
    tools against this product's collections and record hits onto `turn`
    (context.TurnContext) so the gate can verify citations afterward."""
    import json

    from claude_agent_sdk import create_sdk_mcp_server, tool

    from l1agent.agent.tools.search_issues import search_issues as _search_issues_fn
    from l1agent.agent.tools.search_kb import search_kb as _search_kb_fn

    @tool("search_kb", "Search the product knowledge base for passages relevant to a query",
          {"query": str, "top_k": int, "category": str})
    async def _search_kb_tool(args: dict) -> dict:
        hits = _search_kb_fn(
            args["query"], profile.product_id, embedding_provider, vector_store,
            top_k=args.get("top_k", 6) or 6, category=args.get("category") or None,
        )
        turn.record_kb_hits(hits)
        return {"content": [{"type": "text", "text": json.dumps([h.model_dump() for h in hits])}]}

    @tool("search_issues", "Search past resolved issues for precedent",
          {"query": str, "top_k": int})
    async def _search_issues_tool(args: dict) -> dict:
        hits = _search_issues_fn(
            args["query"], profile.product_id, embedding_provider, vector_store,
            top_k=args.get("top_k", 5) or 5,
        )
        turn.record_issue_hits(hits)
        return {"content": [{"type": "text", "text": json.dumps([h.model_dump() for h in hits])}]}

    @tool("submit_draft", "Submit the final drafted reply with its citations",
          {"body": str, "cited_sources": list})
    async def _submit_draft_tool(args: dict) -> dict:
        return {"content": [{"type": "text", "text": json.dumps(args)}]}

    return create_sdk_mcp_server(
        name="l1agent", version="0.1.0",
        tools=[_search_kb_tool, _search_issues_tool, _submit_draft_tool],
    )
