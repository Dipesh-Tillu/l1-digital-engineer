from __future__ import annotations

from l1agent.context import TurnContext
from l1agent.gate.confidence import evaluate_confidence
from l1agent.schemas import ConfidenceThreshold, EscalationPolicy, ProductProfile


def make_profile(**threshold_kwargs) -> ProductProfile:
    threshold = ConfidenceThreshold(category="billing/refunds", **threshold_kwargs)
    default = ConfidenceThreshold(category="*", min_citation_score=0.72, auto_resolve=False)
    return ProductProfile(
        product_id="test-product",
        display_name="Test Product",
        issue_taxonomy=["billing/refunds"],
        thresholds=[default, threshold],
        escalation=EscalationPolicy(default_queue="l2-general"),
    )


def make_turn(kb_scores: dict[str, float] | None = None, accepted_issue_ids: set[str] | None = None) -> TurnContext:
    turn = TurnContext(product_id="test-product", category="billing/refunds")
    for doc_id, score in (kb_scores or {}).items():
        turn.kb_hit_ids.add(doc_id)
        turn.scores_by_id[doc_id] = score
    for issue_id in accepted_issue_ids or set():
        turn.issue_hit_ids.add(issue_id)
        turn.accepted_issue_hit_ids.add(issue_id)
        turn.scores_by_id.setdefault(issue_id, 0.9)
    return turn


def test_category_gate_fails_when_not_auto_resolve():
    profile = make_profile(min_citation_score=0.5, auto_resolve=False)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1"], turn)
    assert result.passed is False
    assert any("auto_resolve" in r for r in result.reasons)


def test_citation_gate_fails_on_empty_citations():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "billing/refunds", [], turn)
    assert result.passed is False
    assert any("cited no sources" in r for r in result.reasons)


def test_citation_gate_fails_on_fabricated_citation():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-does-not-exist"], turn)
    assert result.passed is False
    assert any("not present in this turn" in r for r in result.reasons)


def test_score_gate_fails_below_threshold():
    profile = make_profile(min_citation_score=0.8, auto_resolve=True)
    turn = make_turn({"doc-1": 0.5})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1"], turn)
    assert result.passed is False
    assert any("score" in r for r in result.reasons)


def test_precedent_gate_fails_without_accepted_precedent():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True, require_precedent=True)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1"], turn)
    assert result.passed is False
    assert any("precedent" in r for r in result.reasons)


def test_precedent_gate_passes_with_accepted_precedent_cited():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True, require_precedent=True)
    turn = make_turn({"doc-1": 0.9}, accepted_issue_ids={"issue-1"})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1", "issue-1"], turn)
    assert result.passed is True
    assert result.reasons == []


def test_all_gates_pass():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1"], turn)
    assert result.passed is True
    assert result.score == 0.9
    assert result.reasons == []


def test_reasons_lists_every_failure_not_just_first():
    profile = make_profile(min_citation_score=0.99, auto_resolve=True, require_precedent=True)
    turn = make_turn({"doc-1": 0.5})
    result = evaluate_confidence(profile, "billing/refunds", ["doc-1"], turn)
    assert result.passed is False
    # score gate and precedent gate should both be reported
    assert len(result.reasons) >= 2


def test_unknown_category_falls_back_to_default_threshold():
    profile = make_profile(min_citation_score=0.5, auto_resolve=True)
    turn = make_turn({"doc-1": 0.9})
    result = evaluate_confidence(profile, "some/unlisted-category", ["doc-1"], turn)
    # falls back to "*" threshold, which has auto_resolve=False
    assert result.passed is False
