from __future__ import annotations

from l1agent.schemas import EmbeddingConfig
from l1agent.store.embeddings import (
    LocalHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    get_provider_for,
)


def test_local_provider_always_returns_local_hash():
    provider = get_provider_for(EmbeddingConfig(provider="local"))
    assert isinstance(provider, LocalHashEmbeddingProvider)


def test_voyage_falls_back_to_local_hash_when_key_missing(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    provider = get_provider_for(EmbeddingConfig(provider="voyage"))
    assert isinstance(provider, LocalHashEmbeddingProvider)


def test_openai_falls_back_to_local_hash_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider_for(EmbeddingConfig(provider="openai"))
    assert isinstance(provider, LocalHashEmbeddingProvider)


def test_voyage_provider_instantiated_when_key_present(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    provider = get_provider_for(EmbeddingConfig(provider="voyage", model="voyage-3-large"))
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.model == "voyage-3-large"
    assert provider.api_key == "fake-key"
    # never actually calls the network - the real client is lazily built
    # inside _get_client(), which this test never triggers.


def test_openai_provider_instantiated_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    provider = get_provider_for(EmbeddingConfig(provider="openai"))
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.api_key == "fake-key"


def test_custom_api_key_env_name_is_respected(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_VOYAGE_KEY", "fake-key")
    provider = get_provider_for(EmbeddingConfig(provider="voyage", api_key_env="MY_CUSTOM_VOYAGE_KEY"))
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.api_key == "fake-key"
