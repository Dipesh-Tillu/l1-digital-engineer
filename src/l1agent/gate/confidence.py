"""Confidence gate (SPEC.md §7).

Evaluated in this order, short-circuiting on the first failure (cheapest
checks first):

1. Category gate - thresholds[category].auto_resolve must be True.
2. Citation gate - cited_sources must be non-empty and every cited id must
   appear in this turn's *actual* search_kb/search_issues results (checked
   against the tool call log via TurnContext, not trusted from the model's
   claim) - this is what stops a fabricated citation from passing.
3. Score gate - the top cited source's retrieval score must be
   >= thresholds[category].min_citation_score.
4. Precedent gate (optional, per category) - if require_precedent is set,
   at least one accepted-outcome search_issues hit must be among the cited
   sources.

Output lists every failed check, not just the first - useful both to the L2
human inheriting the case and to whoever is tuning thresholds later.
"""
from __future__ import annotations

from l1agent.context import TurnContext
from l1agent.schemas import GateResult, ProductProfile


def evaluate_confidence(
    profile: ProductProfile,
    category: str | None,
    cited_sources: list[str],
    turn: TurnContext,
) -> GateResult:
    threshold = profile.threshold_for(category)
    reasons: list[str] = []

    # 1. Category gate - short-circuits everything else.
    if not threshold.auto_resolve:
        return GateResult(**{
            "pass": False,
            "score": 0.0,
            "reasons": [
                f"category {threshold.category!r} is not marked auto_resolve "
                "(manual kill switch; see rollout Phase 0/1)"
            ],
        })

    # 2. Citation gate.
    if not cited_sources:
        reasons.append("draft cited no sources")
    else:
        known_ids = turn.kb_hit_ids | turn.issue_hit_ids
        unverifiable = [s for s in cited_sources if s not in known_ids]
        if unverifiable:
            reasons.append(
                f"cited source(s) not present in this turn's retrieval results: {unverifiable}"
            )

    if reasons:
        # Citations failed outright - no meaningful score to compute.
        return GateResult(**{"pass": False, "score": 0.0, "reasons": reasons})

    # 3. Score gate - top cited source's retrieval score.
    cited_scores = [turn.scores_by_id.get(s, 0.0) for s in cited_sources]
    top_score = max(cited_scores) if cited_scores else 0.0
    if top_score < threshold.min_citation_score:
        reasons.append(
            f"top cited source score {top_score:.3f} < required "
            f"{threshold.min_citation_score:.3f}"
        )

    # 4. Precedent gate (optional).
    if threshold.require_precedent:
        cited_accepted = [s for s in cited_sources if s in turn.accepted_issue_hit_ids]
        if not cited_accepted:
            reasons.append(
                "category requires a precedent citation (an accepted-outcome "
                "past issue) but none was cited"
            )

    return GateResult(**{"pass": len(reasons) == 0, "score": top_score, "reasons": reasons})
