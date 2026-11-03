"""FastAPI app (SPEC.md §8). Run with `uvicorn l1agent.api.main:app`."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from l1agent.api.routes import eval as eval_routes
from l1agent.api.routes import products, tickets

app = FastAPI(
    title="L1 Support Agent API",
    description="Prototype API - X-API-Key auth only, see SPEC.md §14.",
)

app.include_router(products.router)
app.include_router(tickets.router)
app.include_router(eval_routes.router)

# Demo workflow addendum (l1-support-agent-demo-workflow-spec.md §6): the
# chat + ticket board UI is a static page calling the API above - no
# separate frontend server/build step, and same-origin so there's no CORS
# to configure for the live demo.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/demo", StaticFiles(directory=_STATIC_DIR, html=True), name="demo")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/demo/")


@app.get("/healthz")
def healthz():
    return {"ok": True}
