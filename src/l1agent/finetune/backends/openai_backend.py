"""OpenAI hosted fine-tuning backend.

This is real, billed external infrastructure - uploading a training file
and creating a fine-tuning job costs money and can take minutes to
complete. Only invoked when a product's `TierConfig.finetune.backend ==
"openai"` (via `l1agent finetune run --backend openai` or the `l1agent
init` wizard). `openai` is a lazy import - see pyproject.toml's
`providers` optional-dependency group.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone

from l1agent.schemas import FineTuneConfig, FineTuneResult, TrainingExample

DEFAULT_BASE_MODEL = "gpt-4o-mini-2024-07-18"
_POLL_INTERVAL_SECONDS = 10
_MAX_POLLS = 180  # ~30 minutes at the interval above


class OpenAIFineTuneBackend:
    def run(
        self,
        examples: list[TrainingExample],
        config: FineTuneConfig,
        *,
        product_id: str,
        role: str,
    ) -> FineTuneResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Required for the 'openai' fine-tune backend.")
        import openai  # local import: optional dependency, only needed for real calls

        client = openai.OpenAI(api_key=api_key)
        base_model = config.base_model or DEFAULT_BASE_MODEL

        training_path = self._write_jsonl(examples)
        try:
            with open(training_path, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="fine-tune")

            create_kwargs: dict = {"training_file": uploaded.id, "model": base_model}
            if config.hyperparams:
                create_kwargs["hyperparameters"] = config.hyperparams
            job = client.fine_tuning.jobs.create(**create_kwargs)

            for _ in range(_MAX_POLLS):
                job = client.fine_tuning.jobs.retrieve(job.id)
                if job.status in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(_POLL_INTERVAL_SECONDS)

            if job.status != "succeeded":
                raise RuntimeError(f"OpenAI fine-tuning job {job.id} ended with status {job.status!r}")

            return FineTuneResult(
                model_ref=job.fine_tuned_model,
                backend="openai",
                base_model=base_model,
                trained_at=datetime.now(timezone.utc),
                metrics={"n_examples": len(examples), "job_id": job.id, "role": role, "product_id": product_id},
            )
        finally:
            os.unlink(training_path)

    @staticmethod
    def _write_jsonl(examples: list[TrainingExample]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": ex.system},
                                {"role": "user", "content": ex.user},
                                {"role": "assistant", "content": ex.assistant},
                            ]
                        }
                    )
                    + "\n"
                )
            return f.name
