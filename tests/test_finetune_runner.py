from __future__ import annotations

from datetime import datetime, timezone

import pytest

from l1agent.agent.tools.classify_issue import classify_issue
from l1agent.finetune.runner import run_finetune
from l1agent.ingestion.pipeline import run_ingestion
from l1agent.schemas import FineTuneResult, TrainingExample
from l1agent.store.relational import get_product_profile, session_scope
from tests.conftest import load_profile


def _onboard(ctx):
    profile = load_profile("acme-widgets")
    with session_scope(ctx.database_url) as session:
        run_ingestion(
            profile, ctx.embedding_provider, ctx.vector_store, session,
            lambda summary, body, p: classify_issue(summary, body, p, ctx.embedding_provider),
        )
    return profile


class _FakeBackend:
    """Never imports torch/openai - verifies run_finetune's plumbing only."""

    def __init__(self):
        self.received_examples: list[TrainingExample] | None = None

    def run(self, examples, config, *, product_id, role):
        self.received_examples = examples
        return FineTuneResult(
            model_ref=f"fake-model-for-{product_id}-{role}",
            backend="local_lora",
            base_model="fake-base-model",
            trained_at=datetime.now(timezone.utc),
            metrics={"n_examples": len(examples)},
        )


def test_run_finetune_persists_result_onto_the_matching_tier(ctx):
    profile = _onboard(ctx)
    fake = _FakeBackend()

    with session_scope(ctx.database_url) as session:
        result = run_finetune(profile, "l1_support", session, ctx.embedding_provider, ctx.vector_store, backend=fake)

    assert result.model_ref == "fake-model-for-acme-widgets-l1_support"
    assert fake.received_examples  # dataset actually got built and passed through

    tier = next(t for t in profile.tiers if t.role == "l1_support")
    assert tier.finetune.model_ref == result.model_ref
    assert tier.finetune.backend == "local_lora"
    assert tier.finetune.base_model == "fake-base-model"
    assert tier.finetune.trained_at is not None

    with session_scope(ctx.database_url) as session:
        reloaded = get_product_profile(session, "acme-widgets")
    reloaded_tier = next(t for t in reloaded.tiers if t.role == "l1_support")
    assert reloaded_tier.finetune.model_ref == result.model_ref


def test_run_finetune_raises_on_unknown_role(ctx):
    profile = _onboard(ctx)
    with session_scope(ctx.database_url) as session:
        with pytest.raises(ValueError, match="no tier configured"):
            run_finetune(profile, "l3_support", session, ctx.embedding_provider, ctx.vector_store, backend=_FakeBackend())


def test_run_finetune_raises_when_no_training_data_for_role(ctx):
    from l1agent.schemas import TierConfig

    profile = _onboard(ctx)
    # A role no historical issue is routed to - dataset will be empty.
    profile.tiers.append(TierConfig(role="l3_support", categories=["nonexistent-category"]))

    with session_scope(ctx.database_url) as session:
        with pytest.raises(ValueError, match="nothing to fine-tune on"):
            run_finetune(profile, "l3_support", session, ctx.embedding_provider, ctx.vector_store, backend=_FakeBackend())
