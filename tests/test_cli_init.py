from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from l1agent.cli.main import app
from l1agent.schemas import ProductProfile
from l1agent.store.relational import reset_engine_cache

REPO_ROOT = Path(__file__).parent.parent


def _wizard_input(docs_path: str, issues_path: str) -> str:
    # Order must match every typer.prompt/typer.confirm call in
    # cli/main.py:init(), in sequence. Blank lines accept the printed
    # default. Embedding provider is "local" and both the fine-tune and
    # sample-ticket steps are skipped so this test needs no API keys and no
    # heavy optional dependencies.
    return "\n".join(
        [
            "wizard-test-product",  # product_id
            "",  # display_name (default)
            "",  # kb source kind (default: local)
            docs_path,  # kb docs path
            "",  # issue source kind (default: local)
            issues_path,  # issue CSV path
            "",  # summary_col (default guessed)
            "",  # body_col (default guessed)
            "",  # category_col (default guessed)
            "",  # resolution_col (default guessed)
            "",  # outcome_col (default guessed)
            "billing/refunds,shipping/delays",  # taxonomy
            "",  # escalation queue (default)
            "",  # llm provider (default: anthropic)
            "",  # llm model (default)
            "local",  # embedding provider
            "n",  # fine-tune now? -> no
            "n",  # test with a sample ticket now? -> no
            "",
        ]
    )


def test_init_wizard_writes_config_and_ingests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_engine_cache()

    docs_path = str(REPO_ROOT / "tests/fixtures/acme-widgets/docs")
    issues_path = str(REPO_ROOT / "tests/fixtures/acme-widgets/issues.csv")

    runner = CliRunner()
    result = runner.invoke(app, ["init"], input=_wizard_input(docs_path, issues_path))

    assert result.exit_code == 0, result.output

    config_path = tmp_path / "config" / "products" / "wizard-test-product.yaml"
    assert config_path.exists()

    profile = ProductProfile.model_validate(yaml.safe_load(config_path.read_text()))
    assert profile.product_id == "wizard-test-product"
    assert profile.embedding.provider == "local"
    assert profile.issue_taxonomy == ["billing/refunds", "shipping/delays"]
    assert all(t.auto_resolve is False for t in profile.thresholds), "wizard defaults must be conservative"

    assert "Indexed" in result.output
    assert "26 KB chunks" in result.output or "KB chunks" in result.output

    reset_engine_cache()


def test_init_wizard_refuses_to_silently_overwrite_existing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    reset_engine_cache()

    config_dir = tmp_path / "config" / "products"
    config_dir.mkdir(parents=True)
    (config_dir / "wizard-test-product.yaml").write_text("product_id: wizard-test-product\n")

    docs_path = str(REPO_ROOT / "tests/fixtures/acme-widgets/docs")
    issues_path = str(REPO_ROOT / "tests/fixtures/acme-widgets/issues.csv")

    runner = CliRunner()
    # The wizard inserts one extra confirm prompt ("overwrite?") right after
    # the embedding-provider prompt when the target file already exists;
    # the next queued line ("n") answers it, which is a "no" - the command
    # should exit non-zero right there, before ingestion ever runs.
    result = runner.invoke(app, ["init"], input=_wizard_input(docs_path, issues_path))

    assert result.exit_code != 0
    assert "Indexed" not in result.output
    reset_engine_cache()
