"""FineTuneBackend interface - implemented by `openai_backend.py` (hosted,
billed) and `local_lora_backend.py` (local, free, heavier setup). Tests
inject a fake implementing this same Protocol so `finetune/runner.py`'s
dataset-building/profile-persistence plumbing is verified without ever
touching real network or model-loading code - mirrors the `agent_call`
seam in `agent/tools/draft_response.py`.
"""
from __future__ import annotations

from typing import Protocol

from l1agent.schemas import FineTuneConfig, FineTuneResult, TrainingExample


class FineTuneBackend(Protocol):
    def run(
        self,
        examples: list[TrainingExample],
        config: FineTuneConfig,
        *,
        product_id: str,
        role: str,
    ) -> FineTuneResult: ...
