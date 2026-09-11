"""Shape tests for every /api/* endpoint, driven against FakeDemoService via
FastAPI's TestClient. A fresh service/app is built per test (fixture-style
helper functions, not a module-level singleton) so test order never matters —
release_quarantine mutates state.

Every protected route now requires a session (see triad/api/auth.py), so
every test logs in first. The test AuthService below is a small fabricated
user set bound to FakeDemoService's own tenant ids (tenant-a/b/c) -- it is
deliberately NOT the committed triad/api/demo_users.json, so these tests
never depend on (or need to know) the real demo passwords documented in the
README. Role-matrix and login-mechanics tests live in tests/test_auth.py;
this file only re-covers each endpoint's *shape*, now reached as an
appropriately-roled user.
"""

from __future__ import annotations

import secrets

from fastapi.testclient import TestClient

from triad.api.app import create_app
from triad.api.auth import AuthService, UserRecord, hash_password
from triad.api.service import FakeDemoService

_PASSWORD = "test-password-not-real"
_ITERATIONS = 1000  # low on purpose: these tests hash a password on every login


def _user(username: str, role: str, tenant: str | None) -> UserRecord:
    salt = secrets.token_bytes(16)
    digest = hash_password(_PASSWORD, salt=salt, iterations=_ITERATIONS)
    return UserRecord(username=username, role=role, salt_hex=salt.hex(),
                        hash_hex=digest.hex(), iterations=_ITERATIONS, tenant=tenant)


def _make_auth_service() -> AuthService:
    users = {
        "emp-a": _user("emp-a", "employee", "tenant-a"),
        "dbmanager": _user("dbmanager", "dbmanager", None),
        "securityhead": _user("securityhead", "securityhead", None),
        "ceo": _user("ceo", "ceo", None),
    }
    return AuthService(users=users)


def _logged_in(app, username: str) -> TestClient:
    client = TestClient(app)
    r = client.post("/api/login", json={"username": username, "password": _PASSWORD})
    assert r.status_code == 200, r.text
    return client


def make_client() -> TestClient:
    """A securityhead session -- covers tenants/quarantine/release/probe/
    trace/results, i.e. everything except /api/ask."""
    app = create_app(FakeDemoService(), auth_service=_make_auth_service())
    return _logged_in(app, "securityhead")


def make_employee_client() -> TestClient:
    """An employee session bound to tenant-a, for /api/ask tests."""
    app = create_app(FakeDemoService(), auth_service=_make_auth_service())
    return _logged_in(app, "emp-a")


def test_meta_shape_and_fake_service_reports_fake():
    # /api/meta is public -- no login needed, and the DEMO MODE banner
    # depends on that (it must be reachable before the login screen renders).
    client = TestClient(create_app(FakeDemoService(), auth_service=_make_auth_service()))
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"service", "data_source", "note"}
    assert body["service"] in {"fake", "real"}
    assert isinstance(body["note"], str) and body["note"]
    # FakeDemoService specifically must self-report as fake, not just "some
    # valid value" -- this is the flag the UI's DEMO MODE banner keys off.
    assert body["service"] == "fake"


def test_tenants_shape():
    client = make_client()
    r = client.get("/api/tenants")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) >= 1
    for t in body:
        assert set(t.keys()) >= {"id", "label", "n_docs"}
        assert isinstance(t["id"], str)
        assert isinstance(t["n_docs"], int)


def _assert_chunk_shape(c: dict):
    assert set(c.keys()) >= {"id", "tenant", "score", "preview", "source_type", "data_source", "taint"}
    assert set(c["taint"].keys()) == {"quarantined", "flags", "score", "reasons"}
    assert isinstance(c["taint"]["flags"], list)
    assert isinstance(c["taint"]["reasons"], list)


def test_ask_defense_off_shape_and_content():
    # An employee's defense is forced ON server-side regardless of what the
    # request asks for (see tests/test_auth.py for the dedicated proof of
    # that); FakeDemoService's own "leak_mode" scenario is exercised here via
    # its tenant fallback (an unknown tenant like "tenant-a" maps to the
    # clean doc), so this test now asserts the defended shape.
    client = make_employee_client()
    r = client.post("/api/ask", json={"tenant": "tenant-a", "question": "What is the Q3 budget?", "defense": True})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "answer", "declined", "decline_reason", "leak_mode", "latency_ms",
        "chunks", "trace", "cached", "data_source", "n_chunks_real", "n_chunks_synthetic",
    }
    for c in body["chunks"]:
        _assert_chunk_shape(c)
    for e in body["trace"]:
        assert set(e.keys()) == {"chunk_id", "stage", "event", "detail"}
        assert e["stage"] in {"ingest", "retrieve", "prompt", "egress"}


def test_ask_defense_on_declines_or_stays_safe():
    client = make_employee_client()
    r = client.post("/api/ask", json={"tenant": "tenant-a", "question": "What is the Q3 budget?", "defense": True})
    assert r.status_code == 200
    body = r.json()
    assert body["leak_mode"] is False
    # No quarantined/poisoned chunk should ever reach the defended answer.
    for c in body["chunks"]:
        assert c["taint"]["quarantined"] is False


def test_ask_ignores_defense_false_from_an_employee():
    # The critical server-side rule: an employee cannot turn defense off by
    # asking for it. defense=False in the body must still come back defended.
    client = make_employee_client()
    r = client.post("/api/ask", json={"tenant": "tenant-a", "question": "What is the Q3 budget?", "defense": False})
    assert r.status_code == 200
    assert r.json()["leak_mode"] is False


def test_ask_tampered_tenant_is_403():
    client = make_employee_client()
    r = client.post("/api/ask", json={"tenant": "tenant-b", "question": "anything", "defense": True})
    assert r.status_code == 403


def test_quarantine_list_and_release():
    client = make_client()
    r = client.get("/api/quarantine")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1
    for q in items:
        assert set(q.keys()) == {"id", "tenant", "preview", "score", "flags", "reasons"}
    target_id = items[0]["id"]

    r2 = client.post(f"/api/quarantine/{target_id}/release")
    assert r2.status_code == 200
    assert r2.json() == {"released": True}

    r3 = client.get("/api/quarantine")
    remaining_ids = {q["id"] for q in r3.json()}
    assert target_id not in remaining_ids


def test_release_unknown_id_is_404():
    client = make_client()
    r = client.post("/api/quarantine/does-not-exist/release")
    assert r.status_code == 404


def test_probe_shape_secure_vs_leaky():
    client = make_client()
    r = client.post("/api/probe", json={"as_tenant": "tenant-a", "target_tenant": "tenant-b"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "secure", "leaky", "property_test", "query", "target_gold_chunk_id", "gold_leaked",
    }
    for side_name in ("secure", "leaky"):
        side = body[side_name]
        assert set(side.keys()) == {"leaked", "n_foreign", "declined", "chunks"}
        for c in side["chunks"]:
            _assert_chunk_shape(c)

    assert body["secure"]["leaked"] is False
    assert body["secure"]["n_foreign"] == 0
    assert body["leaky"]["leaked"] is True
    assert body["leaky"]["n_foreign"] >= 1
    assert isinstance(body["query"], str) and body["query"]
    assert body["gold_leaked"] is True

    assert body["property_test"] is not None
    assert set(body["property_test"].keys()) == {"passed", "total", "fake"}
    assert body["property_test"]["passed"] == body["property_test"]["total"]
    # The fake service must mark its invented pass count as fake -- this is
    # what lets the UI grey out / relabel the badge instead of quoting it as
    # a real measurement (the real adapter reports fake=False once wired).
    assert body["property_test"]["fake"] is True


def test_probe_same_tenant_has_no_property_test():
    client = make_client()
    r = client.post("/api/probe", json={"as_tenant": "tenant-a", "target_tenant": "tenant-a"})
    body = r.json()
    assert body["property_test"] is None


def test_trace_for_known_chunk():
    client = make_client()
    q = client.get("/api/quarantine").json()
    chunk_id = q[0]["id"]
    r = client.get(f"/api/trace/{chunk_id}")
    assert r.status_code == 200
    steps = r.json()
    assert len(steps) >= 1
    stages_seen = []
    for s in steps:
        assert set(s.keys()) == {"stage", "status", "detail", "at"}
        assert s["stage"] in {"ingest", "retrieve", "prompt", "egress"}
        stages_seen.append(s["stage"])
    egress_step = next(s for s in steps if s["stage"] == "egress")
    assert egress_step["status"] == "disabled"


def test_trace_for_unknown_chunk_is_404():
    client = make_client()
    r = client.get("/api/trace/no-such-chunk")
    assert r.status_code == 404


def test_results_rows_are_marked_fake():
    client = make_client()
    r = client.get("/api/results")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    for row in rows:
        assert set(row.keys()) == {
            "attack", "asr_before", "asr_after", "clean_accuracy",
            "added_latency_ms", "fpr", "n", "data_source", "fake",
        }
        assert row["fake"] is True, "FakeDemoService must mark every row it invents as fake"
