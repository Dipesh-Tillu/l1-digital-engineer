"""Shared FastAPI dependencies: auth, DB session, and the process-wide
AppContext (embedding provider + vector store). SPEC.md §8.

Auth for the prototype: a single `X-API-Key` header checked against
`API_KEYS` (comma-separated). This is explicitly not production auth - see
SPEC.md §14 and README.md.
"""
from __future__ import annotations

import os
from typing import Iterator

from fastapi import Header, HTTPException
from sqlalchemy.orm import Session

from l1agent.context import AppContext

_app_context: AppContext | None = None


def get_app_context() -> AppContext:
    # Lazy: constructing AppContext() touches disk (Chroma's persist dir) -
    # deferring it until first request keeps `import l1agent.api.main` (and
    # `pytest` collection) side-effect-free, and lets tests swap in a
    # tmp-path-scoped context before anything real gets created.
    global _app_context
    if _app_context is None:
        _app_context = AppContext()
    return _app_context


def get_session() -> Iterator[Session]:
    factory = get_app_context().session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    allowed = {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}
    if not allowed:
        # No API_KEYS configured: prototype convenience, auth is a no-op.
        return
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
