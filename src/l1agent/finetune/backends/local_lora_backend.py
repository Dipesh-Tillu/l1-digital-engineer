"""Local LoRA fine-tuning backend - trains a small open-weight model on
this product's persona-shaped training examples, entirely on the user's
own machine. No per-call API cost, no external billing, but a heavy
optional dependency (torch/transformers/peft/trl/datasets - install with
`pip install -e ".[finetune]"`) and CPU-only training is slow.

Version caveat (same posture SPEC.md already takes for
`claude-agent-sdk`): `peft`/`trl`/`transformers` APIs move fast - verify
exact signatures against the installed versions if this errors.

Honest expectation-setting: a 1-3B model LoRA-tuned on a few dozen
synthetic examples will have mediocre output quality. This backend proves
the weight-swap mechanism (a product config can point a tier at a locally
fine-tuned checkpoint and `draft_response` will use it) end to end - it is
not a claim of production-grade local support-engineer quality.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from l1agent.schemas import FineTuneConfig, FineTuneResult, TrainingExample

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_MODELS_DIR = Path("models")


def _require_deps() -> None:
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Local LoRA fine-tuning requires the 'finetune' extra: "
            'pip install -e ".[finetune]" (torch, transformers, peft, trl, datasets).'
        ) from e


class LocalLoRABackend:
    def run(
        self,
        examples: list[TrainingExample],
        config: FineTuneConfig,
        *,
        product_id: str,
        role: str,
    ) -> FineTuneResult:
        _require_deps()
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        base_model = config.base_model or DEFAULT_BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model)

        def _to_text(ex: TrainingExample) -> str:
            messages = [
                {"role": "system", "content": ex.system},
                {"role": "user", "content": ex.user},
                {"role": "assistant", "content": ex.assistant},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        dataset = Dataset.from_dict({"text": [_to_text(ex) for ex in examples]})

        lora_config = LoraConfig(
            r=config.hyperparams.get("lora_r", 8),
            lora_alpha=config.hyperparams.get("lora_alpha", 16),
            lora_dropout=config.hyperparams.get("lora_dropout", 0.05),
            task_type="CAUSAL_LM",
        )

        adapter_dir = _MODELS_DIR / product_id / role / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        epochs = config.hyperparams.get("epochs", 3)

        sft_config = SFTConfig(
            output_dir=str(adapter_dir.parent / "_train_output"),
            num_train_epochs=epochs,
            per_device_train_batch_size=config.hyperparams.get("batch_size", 2),
            dataset_text_field="text",
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        return FineTuneResult(
            model_ref=str(adapter_dir),
            backend="local_lora",
            base_model=base_model,
            trained_at=datetime.now(timezone.utc),
            metrics={"n_examples": len(examples), "epochs": epochs, "role": role, "product_id": product_id},
        )

    def generate(self, adapter_dir: str, base_model: str, system: str, user: str, *, max_new_tokens: int = 512) -> str:
        """Single-shot generation against a saved LoRA adapter - used by
        `agent/tools/draft_response.py:_local_agent_call`. No tool-calling
        loop: retrieval already happened in `runner.py` before drafting,
        so this call gets one persona + context prompt and returns one
        completion, expected to end with a `CITED_SOURCES: [...]` line per
        the training data's format (finetune/dataset.py)."""
        _require_deps()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
