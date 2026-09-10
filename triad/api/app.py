"""The TRIAD-RAG demo API: FastAPI app factory. Serves JSON under ``/api`` and,
when built, the static UI (``ui/dist``) at ``/``.

``create_app(service)`` takes any ``DemoService`` implementation — the fake one
for local dev/tests, or a real pipeline adapter — so routes never know which
they're talking to.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from triad.api import serialize
from triad.api.service import DemoService

__all__ = ["create_app", "UI_DIST_DIR"]

# Resolved relative to this file, never CWD: triad/api/app.py -> triad/api -> triad -> repo root -> ui/dist
UI_DIST_DIR = Path(__file__).resolve().parents[2] / "ui" / "dist"


class AskRequest(BaseModel):
    tenant: str
    question: str
    defense: bool


class ProbeRequest(BaseModel):
    as_tenant: str
    target_tenant: str


def create_app(service: DemoService) -> FastAPI:
    app = FastAPI(title="TRIAD-RAG demo API")

    @app.get("/api/meta")
    def get_meta():
        return serialize.meta_dict(service.meta())

    @app.get("/api/tenants")
    def get_tenants():
        return [serialize.tenant_dict(t) for t in service.list_tenants()]

    @app.post("/api/ask")
    def post_ask(body: AskRequest):
        result = service.ask(tenant=body.tenant, question=body.question, defense=body.defense)
        return serialize.ask_dict(result)

    @app.get("/api/quarantine")
    def get_quarantine():
        return [serialize.quarantine_dict(q) for q in service.list_quarantine()]

    @app.post("/api/quarantine/{item_id:path}/release")
    def post_release(item_id: str):
        released = service.release_quarantine(item_id)
        if not released:
            raise HTTPException(status_code=404, detail=f"quarantine item {item_id!r} not found")
        return {"released": True}

    @app.post("/api/probe")
    def post_probe(body: ProbeRequest):
        result = service.probe(as_tenant=body.as_tenant, target_tenant=body.target_tenant)
        return serialize.probe_dict(result)

    @app.get("/api/trace/{chunk_id:path}")
    def get_trace(chunk_id: str):
        steps = service.trace_for_chunk(chunk_id)
        if steps is None:
            raise HTTPException(status_code=404, detail=f"chunk {chunk_id!r} not found")
        return [serialize.trace_step_dict(s) for s in steps]

    @app.get("/api/results")
    def get_results():
        return [serialize.results_row_dict(r) for r in service.results_table()]

    # Static UI mount MUST be registered last: StaticFiles(html=True) mounted at
    # "/" matches everything, so any /api/* route added after this would be
    # shadowed. Guarded by exists() so an unbuilt ui/dist never breaks app
    # construction (and therefore never breaks the /api/* tests that run
    # without a UI build).
    if UI_DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIST_DIR), html=True), name="ui")

    return app
