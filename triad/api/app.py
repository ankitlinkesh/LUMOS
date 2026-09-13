"""The TRIAD-RAG HTTP service: FastAPI app factory. Serves JSON under ``/api``.

``create_app(service)`` takes any ``DemoService`` implementation -- the fake
one for local dev/tests, or the real pipeline adapter (``real_adapter.py``) --
so routes never know which they're talking to.

INTEGRATION CONTRACT -- read this before putting this service behind any
untrusted caller.

This service has no login layer and takes no position on who is allowed to
ask what. ``POST /api/ask`` reads its ``tenant`` straight out of the request
body. The project's whole point is that a retrieved chunk must never widen
its own retrieval scope -- but that only holds if something upstream of this
service has already bound the caller to a tenant via its own authenticated
session and refuses to forward a request whose tenant disagrees with it. If
you expose these endpoints directly to end users, with the tenant (or the
``defense`` flag) taken from an unauthenticated or user-controlled source,
you personally reintroduce the exact cross-tenant leak Stage 2 exists to
prevent -- this service will faithfully run the leaky retriever if asked to
(``POST /api/ask`` with ``"defense": false``), because *proving that leak is
real* is one of this project's own measurements (see the README's Stage 2
section and ``triad.eval.tenant_leak``). Treat tenant binding and the
``defense`` flag as your integration's responsibility, not this service's.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from triad.api import serialize
from triad.api.service import DemoService, NoUsableTargetDocument

__all__ = ["create_app"]


class AskRequest(BaseModel):
    tenant: str
    question: str
    defense: bool


class ProbeRequest(BaseModel):
    as_tenant: str
    target_tenant: str


def create_app(service: DemoService) -> FastAPI:
    app = FastAPI(title="TRIAD-RAG API")

    @app.get("/api/meta")
    def get_meta():
        return serialize.meta_dict(service.meta())

    @app.get("/api/tenants")
    def get_tenants():
        return [serialize.tenant_dict(t) for t in service.list_tenants()]

    @app.post("/api/ask")
    def post_ask(body: AskRequest):
        # See this module's docstring: tenant and defense come straight from
        # the request body. The caller is trusted to have already bound
        # `tenant` to its own authenticated session and to control who, if
        # anyone, may set `defense: false`.
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
        try:
            result = service.probe(as_tenant=body.as_tenant, target_tenant=body.target_tenant)
        except NoUsableTargetDocument as exc:
            # A precondition failure the caller can act on (pick a different
            # target_tenant) -- never silently degrade to a generic query,
            # which is exactly the bug this replaces.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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

    return app
