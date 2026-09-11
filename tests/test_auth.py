"""Authentication and role-based access-control tests for the TRIAD-RAG demo
API (triad/api/auth.py + the enforcement wired into triad/api/app.py).

Uses a small fabricated AuthService (never the committed demo_users.json, and
never real wall-clock time for the lockout tests) so this file is pure and
deterministic. Driven against FakeDemoService via FastAPI's TestClient, same
as tests/test_api_endpoints.py.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from triad.api.app import create_app
from triad.api.auth import POLICY, PUBLIC, AuthService, UserRecord, hash_password
from triad.api.service import FakeDemoService

_PASSWORD = "correct horse battery staple"
_ITERATIONS = 1000  # low on purpose: tests hash a password on every login call


def _user(username: str, role: str, tenant: str | None) -> UserRecord:
    salt = secrets.token_bytes(16)
    digest = hash_password(_PASSWORD, salt=salt, iterations=_ITERATIONS)
    return UserRecord(username=username, role=role, salt_hex=salt.hex(),
                        hash_hex=digest.hex(), iterations=_ITERATIONS, tenant=tenant)


ROLES = ("employee", "dbmanager", "securityhead", "ceo")


def _make_users() -> dict[str, UserRecord]:
    return {
        "employee": _user("employee", "employee", "tenant-a"),
        "employee-other": _user("employee-other", "employee", "tenant-b"),
        "dbmanager": _user("dbmanager", "dbmanager", None),
        "securityhead": _user("securityhead", "securityhead", None),
        "ceo": _user("ceo", "ceo", None),
    }


def make_app(auth_service: AuthService | None = None):
    return create_app(FakeDemoService(), auth_service=auth_service or AuthService(users=_make_users()))


def login(client: TestClient, username: str) -> None:
    r = client.post("/api/login", json={"username": username, "password": _PASSWORD})
    assert r.status_code == 200, r.text


def client_as(app, role: str) -> TestClient:
    c = TestClient(app)
    login(c, role)
    return c


# ---------------------------------------------------------------------------
# Route-coverage: every registered /api/* route must have a POLICY entry.
# ---------------------------------------------------------------------------


def _api_routes(app):
    """(method, path) pairs for every /api/* route on the live app, skipping
    the HEAD/OPTIONS methods Starlette auto-adds (those default-deny too,
    since POLICY only ever has GET/POST entries, but they're not worth
    separately enumerating -- default-deny already covers them)."""
    routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is None or methods is None or not path.startswith("/api/"):
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.append((method, path))
    return routes


def test_every_api_route_has_a_policy_entry():
    app = make_app()
    missing = [rp for rp in _api_routes(app) if rp not in POLICY]
    assert missing == [], f"routes with no auth.POLICY entry (would default-deny, but must be explicit): {missing}"


def test_policy_table_has_no_stale_entries():
    # The mirror check: every POLICY entry corresponds to a real route, so a
    # removed/renamed endpoint doesn't leave a dead, misleading policy row.
    app = make_app()
    live = set(_api_routes(app))
    stale = [rp for rp in POLICY if rp not in live]
    assert stale == [], f"POLICY entries with no matching live route: {stale}"


def test_mutation_check_unpoliced_route_fails_coverage():
    # Prove the coverage test actually catches a missing policy entry: add a
    # route with NO POLICY entry directly to a live app's router, then rerun
    # the same check the real test above runs, and confirm it fails.
    app = make_app()

    @app.get("/api/unpoliced-mutation-check")
    def _unpoliced():
        return {"should": "never ship"}

    missing = [rp for rp in _api_routes(app) if rp not in POLICY]
    assert ("GET", "/api/unpoliced-mutation-check") in missing


def test_unpoliced_route_default_denies_at_runtime_too():
    # Not just a test-suite gap: an unpoliced route actually refuses traffic,
    # as a fail-safe under the coverage test above. Built as a standalone
    # FastAPI app (not through create_app) so a locally-built ui/dist's
    # StaticFiles mount at "/" -- which, in a real app, is always registered
    # LAST and would shadow a route added after the fact -- can't interfere;
    # this isolates _PolicyRoute's own behavior.
    from fastapi import FastAPI

    from triad.api.app import _PolicyRoute
    from triad.api.auth import SESSION_COOKIE_NAME

    auth = AuthService(users=_make_users())
    app = FastAPI()
    app.state.auth_service = auth
    app.router.route_class = _PolicyRoute

    @app.get("/api/unpoliced-mutation-check")
    def _unpoliced():
        return {"should": "never ship"}

    anon = TestClient(app)
    assert anon.get("/api/unpoliced-mutation-check").status_code == 401

    # This minimal app has no /api/login route of its own -- create the
    # session directly against the AuthService (the same thing the real
    # /api/login handler does under the hood) and hand the client its cookie.
    token = auth.login("ceo", _PASSWORD)
    authed = TestClient(app)
    authed.cookies.set(SESSION_COOKIE_NAME, token)
    assert authed.get("/api/unpoliced-mutation-check").status_code == 403


# ---------------------------------------------------------------------------
# Default-deny: unauthenticated = 401, authenticated-but-forbidden = 403.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/tenants"),
    ("GET", "/api/quarantine"),
    ("POST", "/api/ask"),
    ("POST", "/api/probe"),
    ("GET", "/api/results"),
    ("GET", "/api/me"),
])
def test_unauthenticated_request_is_401(method, path):
    app = make_app()
    client = TestClient(app)
    r = client.request(method, path, json={} if method == "POST" else None)
    assert r.status_code == 401


def test_meta_is_public():
    app = make_app()
    client = TestClient(app)
    assert client.get("/api/meta").status_code == 200


# ---------------------------------------------------------------------------
# The full role x route matrix.
# ---------------------------------------------------------------------------

# (method, path, body-or-None) -> {role: expected_status}. Only the roles
# explicitly listed are checked for that row -- rows list every role so
# nothing is silently skipped.
_MATRIX = [
    ("POST", "/api/ask", {"tenant": "tenant-a", "question": "q", "defense": True},
     {"employee": 200, "dbmanager": 403, "securityhead": 403, "ceo": 403}),
    ("GET", "/api/tenants", None,
     {"employee": 403, "dbmanager": 200, "securityhead": 200, "ceo": 200}),
    ("GET", "/api/quarantine", None,
     {"employee": 403, "dbmanager": 200, "securityhead": 200, "ceo": 200}),
    ("POST", "/api/probe", {"as_tenant": "tenant-a", "target_tenant": "tenant-b"},
     {"employee": 403, "dbmanager": 403, "securityhead": 200, "ceo": 403}),
    ("GET", "/api/results", None,
     {"employee": 403, "dbmanager": 403, "securityhead": 200, "ceo": 200}),
]


@pytest.mark.parametrize("method,path,body,expected", _MATRIX, ids=[m[1] for m in _MATRIX])
def test_role_matrix(method, path, body, expected):
    for role, want in expected.items():
        app = make_app()
        client = client_as(app, role)
        r = client.request(method, path, json=body)
        assert r.status_code == want, f"{role} {method} {path}: got {r.status_code}, want {want}"


def test_release_requires_securityhead_not_dbmanager():
    # release is on its own because the target item id must exist to
    # distinguish 403 (wrong role) from 404 (unknown id) meaningfully.
    for role, want in (("employee", 403), ("dbmanager", 403), ("securityhead", 200), ("ceo", 403)):
        app = make_app()
        item_id = client_as(app, "securityhead").get("/api/quarantine").json()[0]["id"]
        client = client_as(app, role)
        r = client.post(f"/api/quarantine/{item_id}/release")
        assert r.status_code == want, f"{role}: got {r.status_code}, want {want}"


def test_trace_requires_employee_dbmanager_or_securityhead_not_ceo():
    app = make_app()
    chunk_id = client_as(app, "securityhead").get("/api/quarantine").json()[0]["id"]
    for role, want in (("dbmanager", 200), ("securityhead", 200), ("ceo", 403)):
        client = client_as(app, role)
        r = client.get(f"/api/trace/{chunk_id}")
        assert r.status_code == want, f"{role}: got {r.status_code}, want {want}"


# ---------------------------------------------------------------------------
# The employee tamper case.
# ---------------------------------------------------------------------------


def test_employee_tampered_tenant_is_403_and_correct_request_scoped_to_own_tenant():
    app = make_app()
    client = client_as(app, "employee")  # bound to tenant-a

    tampered = client.post("/api/ask", json={"tenant": "tenant-b", "question": "q", "defense": True})
    assert tampered.status_code == 403

    correct = client.post("/api/ask", json={"tenant": "tenant-a", "question": "q", "defense": True})
    assert correct.status_code == 200


def test_employee_cannot_turn_defense_off():
    app = make_app()
    client = client_as(app, "employee")
    r = client.post("/api/ask", json={"tenant": "tenant-a", "question": "q", "defense": False})
    assert r.status_code == 200
    assert r.json()["leak_mode"] is False


# ---------------------------------------------------------------------------
# Trace scoping and the "no reason string leak" rule.
# ---------------------------------------------------------------------------


def test_employee_trace_of_foreign_chunk_is_refused_with_no_reason_text():
    app = make_app()
    securityhead = client_as(app, "securityhead")
    foreign_items = securityhead.get("/api/quarantine").json()
    # doc-x-poison belongs to tenant-x, doc-y-poison to tenant-b -- neither is
    # tenant-a, the employee fixture's own tenant.
    foreign_id = next(i["id"] for i in foreign_items if i["tenant"] != "tenant-a")
    # Confirm the real reason text this trace would otherwise reveal.
    full_trace_text = securityhead.get(f"/api/trace/{foreign_id}").text

    employee = client_as(app, "employee")
    r = employee.get(f"/api/trace/{foreign_id}")
    assert r.status_code in (403, 404)
    body_text = r.text
    # No fragment of the real trace/reason content leaked into the refusal.
    assert body_text != full_trace_text
    for reason_fragment in ("hidden instruction", "exfil", "zero-width", "EXECUTE_USERQUERY"):
        assert reason_fragment not in body_text


def test_employee_trace_status_is_consistent_for_unknown_and_foreign():
    app = make_app()
    employee = client_as(app, "employee")
    r_unknown = employee.get("/api/trace/does-not-exist-at-all")
    r_foreign = employee.get("/api/trace/doc-x-poison")
    assert r_unknown.status_code == r_foreign.status_code == 403


def test_employee_can_trace_own_tenant_chunk():
    app = make_app()
    employee = client_as(app, "employee")  # tenant-a
    r = employee.get("/api/trace/doc-a-budget")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# CEO quarantine: aggregate only.
# ---------------------------------------------------------------------------


def test_ceo_quarantine_is_aggregate_only():
    app = make_app()
    securityhead = client_as(app, "securityhead")
    full = securityhead.get("/api/quarantine").json()
    full_text = securityhead.get("/api/quarantine").text

    ceo = client_as(app, "ceo")
    r = ceo.get("/api/quarantine")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"total", "by_tenant", "by_flag"}
    assert body["total"] == len(full)

    body_text = r.text
    for item in full:
        assert item["id"] not in body_text
        assert item["preview"] not in body_text
        for reason in item["reasons"]:
            assert reason not in body_text
    assert body_text != full_text


# ---------------------------------------------------------------------------
# /api/me
# ---------------------------------------------------------------------------


def test_me_reports_identity_and_permissions():
    app = make_app()
    client = client_as(app, "securityhead")
    r = client.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "securityhead"
    assert body["role"] == "securityhead"
    assert body["tenant"] is None
    assert "POST /api/probe" in body["permissions"]
    assert "POST /api/ask" not in body["permissions"]


def test_employee_me_reports_own_tenant():
    app = make_app()
    client = client_as(app, "employee")
    body = client.get("/api/me").json()
    assert body["tenant"] == "tenant-a"


# ---------------------------------------------------------------------------
# Login mechanics: generic error, cookie flags, lockout, logout.
# ---------------------------------------------------------------------------


def test_wrong_password_and_unknown_username_get_the_same_generic_error():
    app = make_app()
    client = TestClient(app)
    r_wrong_pw = client.post("/api/login", json={"username": "employee", "password": "not-it"})
    r_no_user = client.post("/api/login", json={"username": "nobody-like-this", "password": "not-it"})
    assert r_wrong_pw.status_code == r_no_user.status_code == 401
    assert r_wrong_pw.json() == r_no_user.json()


def test_login_sets_httponly_samesite_strict_cookie():
    app = make_app()
    client = TestClient(app)
    r = client.post("/api/login", json={"username": "employee", "password": _PASSWORD})
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert "httponly" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()
    assert "secure" not in set_cookie.lower()  # demo runs over plain HTTP


def test_logout_invalidates_the_session():
    app = make_app()
    client = TestClient(app)
    login(client, "employee")
    assert client.get("/api/me").status_code == 200

    r = client.post("/api/logout")
    assert r.status_code == 200

    assert client.get("/api/me").status_code == 401


def test_lockout_after_five_failures_blocks_even_the_correct_password():
    now = [1_000_000.0]
    auth = AuthService(users=_make_users(), clock=lambda: now[0], max_failures=5, lockout_seconds=60)
    app = make_app(auth)
    client = TestClient(app)

    for _ in range(5):
        r = client.post("/api/login", json={"username": "employee", "password": "wrong"})
        assert r.status_code == 401

    r = client.post("/api/login", json={"username": "employee", "password": _PASSWORD})
    assert r.status_code == 401, "locked out even with the correct password"

    now[0] += 61
    r = client.post("/api/login", json={"username": "employee", "password": _PASSWORD})
    assert r.status_code == 200, "lockout window elapsed"


def test_successful_login_resets_the_failure_counter():
    now = [2_000_000.0]
    auth = AuthService(users=_make_users(), clock=lambda: now[0], max_failures=5, lockout_seconds=60)
    app = make_app(auth)
    client = TestClient(app)

    for _ in range(3):
        assert client.post("/api/login", json={"username": "employee", "password": "wrong"}).status_code == 401
    assert client.post("/api/login", json={"username": "employee", "password": _PASSWORD}).status_code == 200

    # Counter reset by the success; three more failures shouldn't lock yet.
    for _ in range(3):
        assert client.post("/api/login", json={"username": "employee", "password": "wrong"}).status_code == 401
    r = client.post("/api/login", json={"username": "employee", "password": _PASSWORD})
    assert r.status_code == 200, "should not be locked out -- the earlier success reset the counter"


def test_expired_session_is_401():
    now = [5_000_000.0]
    auth = AuthService(users=_make_users(), clock=lambda: now[0], session_ttl_seconds=100)
    app = make_app(auth)
    client = TestClient(app)
    login(client, "employee")
    assert client.get("/api/me").status_code == 200

    now[0] += 101
    r = client.get("/api/me")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# The committed demo_users.json actually loads and matches the README.
# ---------------------------------------------------------------------------


def test_demo_users_file_loads_and_documented_passwords_work():
    from triad.api.auth import AuthService as _AuthService  # default users=None -> loads demo_users.json

    auth = _AuthService()
    demo_creds = {
        "allen-p": "AllenDemo!2026",
        "arnold-j": "ArnoldDemo!2026",
        "arora-h": "AroraDemo!2026",
        "badeer-r": "BadeerDemo!2026",
        "bailey-s": "BaileyDemo!2026",
        "bass-e": "BassDemo!2026",
        "dbmanager": "DbManagerDemo!2026",
        "securityhead": "SecurityHeadDemo!2026",
        "ceo": "CeoDemo!2026",
    }
    for username, password in demo_creds.items():
        token = auth.login(username, password)
        principal = auth.session_for(token)
        assert principal is not None
        assert principal.username == username


def test_demo_users_employee_tenant_equals_username():
    from triad.api.auth import load_users_from_file

    users = load_users_from_file()
    for username in ("allen-p", "arnold-j", "arora-h", "badeer-r", "bailey-s", "bass-e"):
        user = users[username]
        assert user.role == "employee"
        assert user.tenant == username
    for username in ("dbmanager", "securityhead", "ceo"):
        assert users[username].tenant is None
