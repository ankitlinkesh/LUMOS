"""The TRIAD-RAG demo API: FastAPI app factory. Serves JSON under ``/api`` and,
when built, the static UI (``ui/dist``) at ``/``.

``create_app(service)`` takes any ``DemoService`` implementation — the fake one
for local dev/tests, or a real pipeline adapter — so routes never know which
they're talking to. It also takes an optional ``auth_service`` (an
``AuthService``, see ``auth.py``) so tests can inject a fabricated user set
and a fake clock instead of depending on the committed ``demo_users.json``
and real wall-clock time.

Every ``/api/*`` route is enforced by ``_PolicyRoute`` below, which looks up
``(method, path)`` in ``auth.POLICY`` -- the single permission table -- BEFORE
FastAPI parses the request body or resolves any other dependency, so an
unauthenticated request to a protected route gets 401 even if its JSON body
is garbage. A route with no entry in ``POLICY`` default-denies (401/403)
rather than default-allowing; ``tests/test_auth.py`` additionally enumerates
every registered route and fails the build if one has no policy entry, so
"add a route, forget the policy" cannot silently ship.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from triad.api import serialize
from triad.api.auth import (
    POLICY,
    PUBLIC,
    SESSION_COOKIE_NAME,
    AuthService,
    LoginFailed,
    Principal,
    permissions_for,
)
from triad.api.service import DemoService, NoUsableTargetDocument

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


class LoginRequest(BaseModel):
    username: str
    password: str


def current_principal(request: Request) -> Principal | None:
    """The validated session's Principal, or None for a public route with no
    (or an invalid) session cookie. Set on ``request.state`` by
    ``_PolicyRoute`` before any handler or other dependency runs, so this is
    a pure read, never a second lookup against the session store."""
    return getattr(request.state, "principal", None)


def _require_principal(request: Request) -> Principal:
    """For handlers on routes _PolicyRoute has already restricted to
    authenticated roles: the policy check guarantees a Principal is present
    by the time the handler body runs, so this narrows the type without a
    redundant None-check scattered through every handler."""
    principal = current_principal(request)
    assert principal is not None, "_PolicyRoute must reject unauthenticated requests before the handler runs"
    return principal


class _PolicyRoute(APIRoute):
    """An APIRoute that consults ``auth.POLICY`` for its own (method, path)
    on every request, before FastAPI's normal dependency/body resolution
    runs. This is the FastAPI-documented way to wrap request handling
    (see "custom request and route classes"); it is structurally impossible
    to register a route on a router using this class without going through
    this check -- there is no code path that reaches a handler body without
    it.
    """

    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()
        path = self.path

        async def custom_handler(request: Request) -> Response:
            auth: AuthService = request.app.state.auth_service
            token = request.cookies.get(SESSION_COOKIE_NAME)
            principal = auth.session_for(token)

            entry = POLICY.get((request.method, path))
            if entry is None:
                # Default deny: a route with no policy entry is treated as
                # protected-and-forbidden, never as open. The route-coverage
                # test in tests/test_auth.py is what actually prevents this
                # from shipping (it fails the build), but this is the
                # runtime fail-safe if that test is ever skipped.
                raise HTTPException(status_code=401 if principal is None else 403)
            if entry != PUBLIC:
                if principal is None:
                    raise HTTPException(status_code=401, detail="authentication required")
                if principal.role not in entry:
                    raise HTTPException(status_code=403, detail="forbidden for this role")

            request.state.principal = principal
            return await original_handler(request)

        return custom_handler


def create_app(service: DemoService, auth_service: AuthService | None = None) -> FastAPI:
    app = FastAPI(title="TRIAD-RAG demo API")
    app.state.auth_service = auth_service if auth_service is not None else AuthService()
    # Every route registered from here on (via @app.get/@app.post) goes
    # through _PolicyRoute -- see its docstring.
    app.router.route_class = _PolicyRoute

    # -- auth ---------------------------------------------------------------

    @app.post("/api/login")
    def post_login(body: LoginRequest, response: Response, request: Request):
        auth: AuthService = request.app.state.auth_service
        try:
            token = auth.login(body.username, body.password)
        except LoginFailed as exc:
            # Same generic message for "no such user", "wrong password", and
            # "locked out" -- a client cannot distinguish them from the
            # response body. Retry-After is a plain number, never a hint
            # about which case applied.
            headers = {}
            if exc.locked and exc.retry_after_seconds is not None:
                headers["Retry-After"] = str(max(1, int(exc.retry_after_seconds)))
            raise HTTPException(status_code=401, detail="invalid username or password", headers=headers) from exc

        principal = auth.session_for(token)
        assert principal is not None
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="strict",
            # No `secure=True`: this demo is served over plain HTTP on
            # localhost (see README's Demo UI section). A TLS deployment
            # should flip this on; doing so here would silently make login
            # look broken (the browser drops a Secure cookie on http://).
            max_age=int(auth.session_ttl_seconds),
            path="/",
        )
        return {"username": principal.username, "role": principal.role, "tenant": principal.tenant}

    @app.post("/api/logout")
    def post_logout(request: Request, response: Response):
        auth: AuthService = request.app.state.auth_service
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            auth.logout(token)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return {"logged_out": True}

    @app.get("/api/me")
    def get_me(request: Request):
        principal = _require_principal(request)
        return {
            "username": principal.username,
            "role": principal.role,
            "tenant": principal.tenant,
            "permissions": permissions_for(principal.role),
        }

    # -- demo API -------------------------------------------------------------

    @app.get("/api/meta")
    def get_meta():
        return serialize.meta_dict(service.meta())

    @app.get("/api/tenants")
    def get_tenants():
        return [serialize.tenant_dict(t) for t in service.list_tenants()]

    @app.post("/api/ask")
    def post_ask(body: AskRequest, request: Request):
        principal = _require_principal(request)
        # Tenant scope comes from the SESSION, never the request body. A
        # mismatch is treated as a tampering attempt and refused outright
        # (403) rather than silently substituted with the session's tenant,
        # so the attempt is visible instead of hidden.
        if body.tenant != principal.tenant:
            raise HTTPException(status_code=403, detail="tenant scope mismatch")
        # Defense is forced ON for employees regardless of what the request
        # asked for: the OFF path is the deliberately vulnerable leaky
        # retriever, and it really does return other tenants' mail.
        result = service.ask(tenant=principal.tenant, question=body.question, defense=True)
        return serialize.ask_dict(result)

    @app.get("/api/quarantine")
    def get_quarantine(request: Request):
        principal = _require_principal(request)
        items = service.list_quarantine()
        if principal.role == "ceo":
            # Aggregate counts only -- no id, preview, or reason string.
            return serialize.quarantine_aggregate_dict(items)
        return [serialize.quarantine_dict(q) for q in items]

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
    def get_trace(chunk_id: str, request: Request):
        principal = _require_principal(request)
        if principal.role == "employee":
            # Resolve tenant BEFORE fetching/returning any trace content --
            # trace steps can quote the chunk's own reasons, which quote
            # email text. A foreign or unknown chunk gets the SAME 403 with
            # no reason string, so an employee can't distinguish "not mine"
            # from "doesn't exist" by status code or body.
            tenant = service.chunk_tenant(chunk_id)
            if tenant is None or tenant != principal.tenant:
                raise HTTPException(status_code=403, detail="forbidden")
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
    # without a UI build). A Mount is not an APIRoute, so it is never subject
    # to _PolicyRoute -- static files (and /api/login, /api/meta) stay public
    # by construction, not by policy-table entry.
    if UI_DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIST_DIR), html=True), name="ui")

    return app
