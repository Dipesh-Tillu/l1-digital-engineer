"""EmbeddingProvider interface + implementations.

Default is Voyage AI (`voyage-3-large`), per SPEC.md §2. Swapping providers
is a one-file change: implement `EmbeddingProvider` and point
`get_default_provider()` (or the caller) at it.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from l1agent.schemas import EmbeddingConfig


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Callers batch at <=128 items (Voyage's practical ceiling)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class VoyageEmbeddingProvider:
    """Production default. Requires VOYAGE_API_KEY."""

    def __init__(self, model: str = "voyage-3-large", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.dimension = 1024
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "VOYAGE_API_KEY is not set. Set it in your environment, or "
                    "inject a different EmbeddingProvider (see store/embeddings.py)."
                )
            import voyageai  # local import: optional dependency, only needed for real calls

            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            result = client.embed(batch, model=self.model, input_type="document")
            out.extend(result.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        result = client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]


class LocalHashEmbeddingProvider:
    """Deterministic, dependency-free embedding used by tests and offline demos.

    Not semantically meaningful in the way a trained model is, but it is
    stable, cheap, and gives near-duplicate/overlapping text high cosine
    similarity via shared token hashing - enough to exercise chunking,
    retrieval, ranking, and the confidence gate without network calls or
    API keys. Swap in VoyageEmbeddingProvider (or another EmbeddingProvider)
    for real semantic retrieval quality.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbeddingProvider:
    """Alternate real embedding backend. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.dimension = 3072
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Set it in your environment, or "
                    "inject a different EmbeddingProvider (see store/embeddings.py)."
                )
            import openai  # local import: optional dependency, only needed for real calls

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            result = client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in result.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        result = client.embeddings.create(model=self.model, input=[text])
        return result.data[0].embedding


def get_provider_for(config: EmbeddingConfig) -> EmbeddingProvider:
    """Config-driven embedding provider resolution - what actually lets a
    product *choose* its embedding provider (SPEC.md's env-var-only
    `get_default_provider()` below can't express that; this is what
    cli/main.py and api/routes/*.py call once a ProductProfile is loaded).

    Falls back to the local hash provider whenever the chosen provider's
    API key isn't set, same dev-friendly behavior as `get_default_provider`.
    """
    if config.provider == "local":
        return LocalHashEmbeddingProvider()

    default_env_var = {"voyage": "VOYAGE_API_KEY", "openai": "OPENAI_API_KEY"}[config.provider]
    key = os.environ.get(config.api_key_env or default_env_var)
    if not key:
        return LocalHashEmbeddingProvider()

    if config.provider == "voyage":
        return VoyageEmbeddingProvider(model=config.model or "voyage-3-large", api_key=key)
    return OpenAIEmbeddingProvider(model=config.model or "text-embedding-3-large", api_key=key)


def get_default_provider() -> EmbeddingProvider:
    """Resolves the embedding provider for production use.

    Falls back to the local hash provider when no VOYAGE_API_KEY is set, so
    `l1agent onboard` still works out of the box for the "plug and play, see
    how it behaves" prototype path - swap to VoyageEmbeddingProvider by
    setting VOYAGE_API_KEY for real semantic retrieval.
    """
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbeddingProvider()
    return LocalHashEmbeddingProvider()
