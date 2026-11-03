"""Tool: score_confidence (SPEC.md §6.2, §6.3).

Input: {draft, cited_sources, category}. Output: {pass, score, reasons}.

Implemented as a **plain function call, not a sub-agent turn** - one
rules+score evaluation against `gate/confidence.py` (§7). Requires the
current turn's actual search results (via TurnContext) so citations are
checked against the tool-call log, not trusted from the model's claim.
"""
from __future__ import annotations

from l1agent.context import TurnContext
from l1agent.gate.confidence import evaluate_confidence
from l1agent.schemas import GateResult, ProductProfile


def score_confidence(
    draft: str,
    cited_sources: list[str],
    category: str | None,
    profile: ProductProfile,
    turn: TurnContext,
) -> GateResult:
    del draft  # not needed by the current rule set; kept in the signature per SPEC.md §6.2
    return evaluate_confidence(profile, category, cited_sources, turn)
