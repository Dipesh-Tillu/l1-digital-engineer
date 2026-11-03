from __future__ import annotations

from l1agent.schemas import (
    EmbeddingConfig,
    FineTuneConfig,
    ModelConfig,
    ProductProfile,
    TierConfig,
    UrlSourceRef,
)
from tests.conftest import load_profile


def test_existing_fixture_profiles_still_validate_unchanged():
    acme = load_profile("acme-widgets")
    beta = load_profile("beta-gadgets")
    assert acme.model.provider == "anthropic"
    assert acme.embedding.provider == "voyage"
    assert [t.role for t in acme.tiers] == ["l1_support"]
    assert beta.model.provider == "anthropic"


def test_url_source_ref_validates_and_is_part_of_the_source_union():
    profile = ProductProfile(
        product_id="p", display_name="P", issue_taxonomy=["a"], thresholds=[],
        escalation={"default_queue": "q"},
        kb_sources=[{"type": "url", "url": "https://example.com/docs"}],
        issue_sources=[
            {
                "type": "url",
                "url": "https://example.com/api/tickets",
                "json_records_path": "data.tickets",
                "resolution_col": "resolution",
            }
        ],
    )
    assert isinstance(profile.kb_sources[0], UrlSourceRef)
    assert isinstance(profile.issue_sources[0], UrlSourceRef)
    assert profile.issue_sources[0].json_records_path == "data.tickets"


def test_model_config_and_embedding_config_defaults():
    assert ModelConfig().provider == "anthropic"
    assert ModelConfig(provider="openai").provider == "openai"
    assert EmbeddingConfig().provider == "voyage"


def test_tier_for_exact_category_beats_catch_all_regardless_of_order():
    profile = ProductProfile(
        product_id="p", display_name="P", issue_taxonomy=["a", "b"], thresholds=[],
        escalation={"default_queue": "q"},
        tiers=[
            TierConfig(role="l1_support", categories=["*"]),
            TierConfig(role="l2_support", categories=["b"]),
        ],
    )
    assert profile.tier_for("b").role == "l2_support"
    assert profile.tier_for("a").role == "l1_support"
    assert profile.tier_for(None).role == "l1_support"
    assert profile.tier_for("unknown-category").role == "l1_support"


def test_finetune_config_starts_unset_until_a_run_populates_it():
    ft = FineTuneConfig()
    assert ft.backend is None
    assert ft.model_ref is None
