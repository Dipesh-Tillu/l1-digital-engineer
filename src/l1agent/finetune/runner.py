"""Orchestrates a fine-tuning run for one product/tier: build a training
set from that product's own ingested history (`finetune/dataset.py`),
dispatch to the configured backend (OpenAI hosted, or a local LoRA run),
and persist the resulting checkpoint reference back onto the product's
`TierConfig` so `draft_response`'s dispatcher picks it up on the next
ticket routed to that tier (`schemas.ProductProfile.tier_for`,
`agent/tools/draft_response.py:_default_agent_call`).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from l1agent.finetune.backends.base import FineTuneBackend
from l1agent.finetune.dataset import build_training_examples
from l1agent.schemas import FineTuneResult, ProductProfile
from l1agent.store.embeddings import EmbeddingProvider
from l1agent.store.relational import upsert_product_profile
from l1agent.store.vector.base import VectorStore


def _resolve_backend(name: str | None) -> FineTuneBackend:
    if name == "openai":
        from l1agent.finetune.backends.openai_backend import OpenAIFineTuneBackend

        return OpenAIFineTuneBackend()
    if name == "local_lora":
        from l1agent.finetune.backends.local_lora_backend import LocalLoRABackend

        return LocalLoRABackend()
    raise ValueError(f"tier.finetune.backend must be 'openai' or 'local_lora' to run fine-tuning (got {name!r})")


def run_finetune(
    profile: ProductProfile,
    role: str,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    backend: FineTuneBackend | None = None,
) -> FineTuneResult:
    tier = next((t for t in profile.tiers if t.role == role), None)
    if tier is None:
        raise ValueError(f"product {profile.product_id!r} has no tier configured with role {role!r}")

    examples = build_training_examples(profile, role, embedding_provider, vector_store)
    if not examples:
        raise ValueError(
            f"no historical resolved issues route to role {role!r} on product "
            f"{profile.product_id!r} - nothing to fine-tune on"
        )

    chosen = backend or _resolve_backend(tier.finetune.backend)
    result = chosen.run(examples, tier.finetune, product_id=profile.product_id, role=role)

    tier.finetune.model_ref = result.model_ref
    tier.finetune.backend = result.backend
    tier.finetune.base_model = result.base_model
    tier.finetune.trained_at = result.trained_at
    tier.finetune.metrics = result.metrics
    upsert_product_profile(session, profile)

    return result
