"""Static-route test: when ui/dist is built, "/" serves ui/dist/index.html.
Skipped (not failed) when the UI hasn't been built, since building the UI is a
separate step (npm install/build) that shouldn't gate the Python-only test run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from triad.api.app import UI_DIST_DIR, create_app
from triad.api.service import FakeDemoService


@pytest.mark.skipif(not (UI_DIST_DIR / "index.html").exists(), reason="ui/dist not built yet (run `npm run build` in ui/)")
def test_root_serves_built_index_html():
    app = create_app(FakeDemoService())
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    on_disk = (UI_DIST_DIR / "index.html").read_text(encoding="utf-8")
    assert r.text == on_disk


def test_api_routes_work_even_when_ui_unbuilt_or_built():
    # Regardless of build state, /api/* must never be shadowed by the static mount.
    app = create_app(FakeDemoService())
    client = TestClient(app)
    r = client.get("/api/tenants")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
