"""Tool: get_product_profile (SPEC.md §6.2).

Input: {} (bound to the running session's product).
Output: ProductProfile subset - taxonomy, thresholds, escalation.
"""
from __future__ import annotations

from l1agent.schemas import ProductProfile


def get_product_profile(profile: ProductProfile) -> dict:
    return {
        "product_id": profile.product_id,
        "display_name": profile.display_name,
        "issue_taxonomy": profile.issue_taxonomy,
        "thresholds": [t.model_dump() for t in profile.thresholds],
        "escalation": profile.escalation.model_dump(),
    }
