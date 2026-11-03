"""Tool: draft_response (SPEC.md §6.2, §6.3).

Input: {issue, kb_hits, issue_hits}. Output: {body, cited_sources}.

The *only* step that actually runs an LLM turn. Everything else in
`runner.py` is deterministic orchestration in plain Python - that's what
makes the gate's decision auditable outside the model's own reasoning.

Provider dispatch: a product's TierConfig (schemas.ProductProfile.tier_for)
decides which support tier a ticket's category belongs to. If that tier
has a fine-tuned checkpoint (`TierConfig.finetune.model_ref`, set by
`finetune/runner.py` after a successful run), drafting routes to it;
otherwise it falls back to the product's configured base LLM provider
(`ProductProfile.model.provider`: "anthropic" | "openai"). This is the
mechanism behind "fine-tune a new checkpoint to make this tier an L2/L3
engineer" - flipping which model a tier uses is a config/profile change,
not a code change.

Only the Anthropic and OpenAI paths run a real multi-turn tool-calling
loop (so the model can re-query search_kb/search_issues if the first
retrieval was thin, up to `max_turns`); the local LoRA path is single-shot
since `runner.py` already retrieves kb_hits/issue_hits before calling
draft_response at all - see finetune/backends/local_lora_backend.py.

`agent_call` is an injectable seam so tests (and any offline demo) can run
the whole pipeline without any provider's API key - see tests/conftest.py
for the fake used there. It is unchanged by all of the above: tests never
exercise any real backend's network or model-loading path.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from l1agent.context import TurnContext
from l1agent.schemas import DraftResponseOutput, IssueRecord, IssueHit, KBHit, LLMTrace, LLMTraceStep, ProductProfile, TierConfig

AgentCallFn = Callable[[IssueRecord, list[KBHit], list[IssueHit], ProductProfile, TurnContext], DraftResponseOutput]

_CITED_SOURCES_RE = re.compile(r"CITED_SOURCES:\s*(\[[^\]]*\])", re.IGNORECASE | re.DOTALL)
_TOOL_RESULT_PREVIEW_CHARS = 400


def _preview(value, limit: int = _TOOL_RESULT_PREVIEW_CHARS) -> str:
    """Trims a raw tool-result payload to something worth showing in the
    demo's LLM trace panel - real content, just not the full (sometimes
    multi-KB) JSON blob of retrieved passages."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _normalize_citation(source: str) -> str:
    """Models tend to cite passages exactly as render_context_prompt labels
    them (`[doc_id=abc]`, `issue_id=xyz`) rather than the bare id the
    confidence gate compares against - strip that labelling back off."""
    s = str(source).strip().strip("[]").strip()
    for prefix in ("doc_id=", "issue_id="):
        if s.startswith(prefix):
            return s[len(prefix):].strip()
    return s


def render_context_prompt(issue: IssueRecord, kb_hits: list[KBHit], issue_hits: list[IssueHit]) -> str:
    lines = [
        f"A customer reported the following issue for category "
        f"{issue.category or 'unclassified'!r}:",
        f"Summary: {issue.summary}",
        f"Body: {issue.body}",
        "",
        "Knowledge base passages already retrieved this turn (you may call "
        "search_kb again if these don't answer it):",
    ]
    for hit in kb_hits:
        lines.append(f"- [doc_id={hit.doc_id}] ({hit.section_path}, score={hit.score:.2f}): {hit.text}")
    lines.append("")
    lines.append("Past resolved issues already retrieved this turn:")
    for hit in issue_hits:
        lines.append(
            f"- [issue_id={hit.issue_id}] ({hit.outcome}, score={hit.score:.2f}) "
            f"{hit.summary}: {hit.resolution_body}"
        )
    lines.append("")
    lines.append("Draft the reply now and call submit_draft.")
    return "\n".join(lines)


def _openai_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": "Search the product knowledge base for passages relevant to a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "category": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_issues",
                "description": "Search past resolved issues for precedent",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_draft",
                "description": "Submit the final drafted reply with its citations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "body": {"type": "string"},
                        "cited_sources": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["body", "cited_sources"],
                },
            },
        },
    ]


def _anthropic_agent_call(
    issue: IssueRecord,
    kb_hits: list[KBHit],
    issue_hits: list[IssueHit],
    profile: ProductProfile,
    turn: TurnContext,
    tier: TierConfig,
) -> DraftResponseOutput:
    """Runs one real Claude Agent SDK turn. Requires ANTHROPIC_API_KEY and
    `claude-agent-sdk` installed. See agent/options.py's module docstring
    for the SDK-version caveat."""
    import anyio

    from l1agent.agent.options import build_mcp_server_for, build_options, model_for, render_persona
    from l1agent.context import AppContext
    from l1agent.store.embeddings import get_provider_for

    embedding_provider = get_provider_for(profile.embedding)
    vector_store = AppContext().vector_store

    prompt = render_context_prompt(issue, kb_hits, issue_hits)
    trace = LLMTrace(
        tier_role=tier.role, model_provider="anthropic", model=model_for(profile, "draft_response"),
        system_prompt=render_persona(profile, tier.role), user_prompt=prompt,
    )

    async def _run() -> DraftResponseOutput:
        from claude_agent_sdk import ClaudeSDKClient

        server = build_mcp_server_for(profile, embedding_provider, vector_store, turn)
        options = build_options(profile, "draft_response", server, role=tier.role)

        from claude_agent_sdk import TextBlock, ToolResultBlock, ToolUseBlock

        body_text = ""
        cited: list[str] = []
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                for block in getattr(message, "content", None) or []:
                    if isinstance(block, TextBlock) and block.text.strip():
                        trace.steps.append(LLMTraceStep(kind="assistant_text", text=block.text))
                    elif isinstance(block, ToolUseBlock):
                        name = block.name.replace("mcp__l1agent__", "")
                        trace.steps.append(LLMTraceStep(kind="tool_call", tool_name=name, tool_input=dict(block.input or {})))
                        if name == "submit_draft":
                            body_text = block.input.get("body", "")
                            cited = list(block.input.get("cited_sources", []))
                    elif isinstance(block, ToolResultBlock):
                        trace.steps.append(LLMTraceStep(kind="tool_result", tool_result_summary=_preview(block.content)))
        return DraftResponseOutput(body=body_text, cited_sources=cited)

    result = anyio.run(_run)
    turn.llm_trace = trace
    return result


def _openai_agent_call(
    issue: IssueRecord,
    kb_hits: list[KBHit],
    issue_hits: list[IssueHit],
    profile: ProductProfile,
    turn: TurnContext,
    tier: TierConfig,
    model_override: str | None = None,
) -> DraftResponseOutput:
    """Runs a manual tool-calling loop against OpenAI's chat completions
    API - the OpenAI-provider equivalent of `_anthropic_agent_call`'s
    Claude Agent SDK loop. `model_override` (a fine-tuned model id) takes
    precedence over `profile.model` when a tier has an OpenAI fine-tune."""
    import os

    from l1agent.agent.options import model_for, render_persona
    from l1agent.agent.tools.search_issues import search_issues as _search_issues_fn
    from l1agent.agent.tools.search_kb import search_kb as _search_kb_fn
    from l1agent.context import AppContext
    from l1agent.store.embeddings import get_provider_for

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; required when model.provider == 'openai' "
            "(or a tier's fine-tuned OpenAI checkpoint is configured)."
        )
    import openai

    client = openai.OpenAI(api_key=api_key)
    embedding_provider = get_provider_for(profile.embedding)
    vector_store = AppContext().vector_store
    model = model_override or model_for(profile, "draft_response")

    system_prompt = render_persona(profile, tier.role)
    user_prompt = render_context_prompt(issue, kb_hits, issue_hits)
    trace = LLMTrace(
        tier_role=tier.role, model_provider="openai", model=model,
        system_prompt=system_prompt, user_prompt=user_prompt,
    )
    turn.llm_trace = trace  # set upfront so it's populated even on an early raise below

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tools = _openai_tools_schema()

    for _ in range(6):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
        msg = response.choices[0].message
        if not msg.tool_calls:
            if msg.content and msg.content.strip():
                trace.steps.append(LLMTraceStep(kind="assistant_text", text=msg.content))
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Call submit_draft with your final answer now."})
            continue

        if msg.content and msg.content.strip():
            trace.steps.append(LLMTraceStep(kind="assistant_text", text=msg.content))
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            trace.steps.append(LLMTraceStep(kind="tool_call", tool_name=name, tool_input=args))

            if name == "search_kb":
                hits = _search_kb_fn(
                    args.get("query", ""), profile.product_id, embedding_provider, vector_store,
                    top_k=args.get("top_k") or 6, category=args.get("category") or None,
                )
                turn.record_kb_hits(hits)
                tool_result = [h.model_dump() for h in hits]
            elif name == "search_issues":
                hits = _search_issues_fn(
                    args.get("query", ""), profile.product_id, embedding_provider, vector_store,
                    top_k=args.get("top_k") or 5,
                )
                turn.record_issue_hits(hits)
                tool_result = [h.model_dump() for h in hits]
            elif name == "submit_draft":
                return DraftResponseOutput(body=args.get("body", ""), cited_sources=list(args.get("cited_sources", [])))
            else:
                tool_result = {"error": f"unknown tool {name!r}"}

            trace.steps.append(LLMTraceStep(kind="tool_result", tool_result_summary=_preview(tool_result)))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})

    return DraftResponseOutput(body="", cited_sources=[])


def _local_agent_call(
    issue: IssueRecord,
    kb_hits: list[KBHit],
    issue_hits: list[IssueHit],
    profile: ProductProfile,
    turn: TurnContext,
    tier: TierConfig,
) -> DraftResponseOutput:
    """Single-shot generation against a locally LoRA-fine-tuned checkpoint
    (finetune/backends/local_lora_backend.py). No tool-calling loop: the
    retrieval `runner.py` already did before calling draft_response is the
    only context this pass gets, matching how the training examples in
    finetune/dataset.py were built. The model is expected to end its reply
    with a `CITED_SOURCES: [...]` line (the training data teaches this
    format); if it doesn't, cited_sources comes back empty and the
    confidence gate escalates on the empty-citation rule, same as any
    other provider's ungrounded answer would."""
    from l1agent.agent.options import render_persona
    from l1agent.finetune.backends.local_lora_backend import DEFAULT_BASE_MODEL, LocalLoRABackend

    ft = tier.finetune
    if not ft.model_ref:
        raise RuntimeError(
            f"tier {tier.role!r} has finetune.backend='local_lora' but no model_ref - "
            "run `l1agent finetune run` for this tier first."
        )

    system = render_persona(profile, tier.role)
    user = render_context_prompt(issue, kb_hits, issue_hits)
    text = LocalLoRABackend().generate(ft.model_ref, ft.base_model or DEFAULT_BASE_MODEL, system, user)

    cited: list[str] = []
    match = _CITED_SOURCES_RE.search(text)
    if match:
        try:
            cited = [str(c) for c in json.loads(match.group(1))]
        except json.JSONDecodeError:
            cited = []
        text = text[: match.start()].rstrip()

    turn.llm_trace = LLMTrace(
        tier_role=tier.role, model_provider="local_lora", model=ft.model_ref,
        system_prompt=system, user_prompt=user,
        steps=[LLMTraceStep(kind="assistant_text", text=text)],
    )
    return DraftResponseOutput(body=text, cited_sources=cited)


def _default_agent_call(
    issue: IssueRecord,
    kb_hits: list[KBHit],
    issue_hits: list[IssueHit],
    profile: ProductProfile,
    turn: TurnContext,
) -> DraftResponseOutput:
    tier = profile.tier_for(issue.category)
    ft = tier.finetune

    if ft.model_ref and ft.backend == "openai":
        return _openai_agent_call(issue, kb_hits, issue_hits, profile, turn, tier, model_override=ft.model_ref)
    if ft.model_ref and ft.backend == "local_lora":
        return _local_agent_call(issue, kb_hits, issue_hits, profile, turn, tier)

    if profile.model.provider == "openai":
        return _openai_agent_call(issue, kb_hits, issue_hits, profile, turn, tier)
    return _anthropic_agent_call(issue, kb_hits, issue_hits, profile, turn, tier)


def draft_response(
    issue: IssueRecord,
    kb_hits: list[KBHit],
    issue_hits: list[IssueHit],
    profile: ProductProfile,
    turn: TurnContext,
    *,
    agent_call: AgentCallFn | None = None,
) -> DraftResponseOutput:
    call = agent_call or _default_agent_call
    result = call(issue, kb_hits, issue_hits, profile, turn)
    result.cited_sources = [_normalize_citation(c) for c in result.cited_sources]
    return result
