"""Server-side authentication and role-based access control for the TRIAD-RAG
demo API.

This is a hackathon-demo auth layer, not a production one: sessions live in an
in-process dict (a server restart logs everyone out) and there is no
password-reset flow. Two things ARE production-grade on purpose, because this
is a security product and judges will probe it with ``curl``:

1. Every permission decision happens on the SERVER. ``POLICY`` below is the
   single source of truth for which role may call which ``/api/*`` route; the
   frontend only hides UI for convenience, it enforces nothing. See
   ``tests/test_auth.py``'s route-coverage test, which enumerates every
   registered route on the live FastAPI app and fails the build if any route
   has no entry here.
2. Passwords are never stored or compared in plaintext. ``AuthService.login``
   hashes with PBKDF2-HMAC-SHA256 (matching ``scripts/gen_demo_users.py``,
   which is what wrote ``demo_users.json``) and compares with
   ``hmac.compare_digest`` (constant-time, no early-exit timing leak).

``AuthService`` is constructor-injected (a ``users`` dict and a ``clock``
callable), matching the DI pattern the rest of this codebase uses for
``Pipeline``'s ``store``/``embedder``/``clock`` -- tests never depend on the
real committed ``demo_users.json`` or real wall-clock time to exercise login
throttling or session expiry deterministically.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

__all__ = [
    "Role",
    "ALL_ROLES",
    "Principal",
    "UserRecord",
    "LoginFailed",
    "AuthService",
    "POLICY",
    "PUBLIC",
    "permissions_for",
    "load_users_from_file",
    "DEFAULT_USERS_PATH",
    "SESSION_COOKIE_NAME",
    "PBKDF2_ITERATIONS_DEFAULT",
]

Role = Literal["employee", "dbmanager", "securityhead", "ceo"]
ALL_ROLES: frozenset[str] = frozenset({"employee", "dbmanager", "securityhead", "ceo"})

DEFAULT_USERS_PATH = Path(__file__).resolve().parent / "demo_users.json"
SESSION_COOKIE_NAME = "triad_session"

PBKDF2_ITERATIONS_DEFAULT = 200_000
_SESSION_TTL_SECONDS_DEFAULT = 8 * 60 * 60
_LOGIN_MAX_FAILURES_DEFAULT = 5
_LOGIN_LOCKOUT_SECONDS_DEFAULT = 60


@dataclass(frozen=True)
class UserRecord:
    """One row of ``demo_users.json``. ``tenant`` is set only for
    ``role == "employee"`` -- the account's bound tenant, never overridable
    by anything in a request (see ``app.py``'s ``post_ask``)."""

    username: str
    role: Role
    salt_hex: str
    hash_hex: str
    iterations: int
    tenant: str | None = None


@dataclass(frozen=True)
class Principal:
    """Who a validated session belongs to. Handed to route handlers via
    ``Depends(current_principal)``; never constructed from request-supplied
    data -- always looked up server-side from the session store."""

    username: str
    role: Role
    tenant: str | None


class LoginFailed(Exception):
    """Raised by ``AuthService.login`` for a wrong username, wrong password,
    OR an active lockout -- callers (``app.py``) turn every case into the
    same generic 401 so a client can't distinguish "no such user" from
    "wrong password" from "locked out" by response content alone."""

    def __init__(self, *, locked: bool, retry_after_seconds: float | None = None) -> None:
        super().__init__("login failed")
        self.locked = locked
        self.retry_after_seconds = retry_after_seconds


@dataclass
class _Session:
    username: str
    role: Role
    tenant: str | None
    expires_at: float


@dataclass
class _FailureState:
    count: int = 0
    locked_until: float | None = None


def load_users_from_file(path: Path = DEFAULT_USERS_PATH) -> dict[str, UserRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    default_iterations = data.get("pbkdf2_iterations", PBKDF2_ITERATIONS_DEFAULT)
    users: dict[str, UserRecord] = {}
    for row in data["users"]:
        users[row["username"]] = UserRecord(
            username=row["username"],
            role=row["role"],
            salt_hex=row["salt"],
            hash_hex=row["hash"],
            iterations=row.get("iterations", default_iterations),
            tenant=row.get("tenant"),
        )
    return users


def hash_password(password: str, *, salt: bytes, iterations: int = PBKDF2_ITERATIONS_DEFAULT) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


class AuthService:
    """In-memory sessions + PBKDF2 password checks + a per-username failed-
    login throttle. One instance is shared for the life of the server
    process (stored on ``app.state.auth_service``, see ``create_app``)."""

    def __init__(
        self,
        users: dict[str, UserRecord] | None = None,
        *,
        session_ttl_seconds: float = _SESSION_TTL_SECONDS_DEFAULT,
        max_failures: int = _LOGIN_MAX_FAILURES_DEFAULT,
        lockout_seconds: float = _LOGIN_LOCKOUT_SECONDS_DEFAULT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._users = users if users is not None else load_users_from_file()
        self._sessions: dict[str, _Session] = {}
        self._failures: dict[str, _FailureState] = {}
        self.session_ttl_seconds = session_ttl_seconds
        self._max_failures = max_failures
        self._lockout_seconds = lockout_seconds
        self._clock = clock

    # -- login/logout --------------------------------------------------

    def login(self, username: str, password: str) -> str:
        """Returns a fresh session token on success. Raises ``LoginFailed``
        on a wrong username, a wrong password, or an active lockout -- the
        throttle keys on the username STRING regardless of whether that
        username exists, so a nonexistent username can't be distinguished
        from a real one that's merely mid-lockout."""
        now = self._clock()
        failure = self._failures.get(username)
        if failure is not None and failure.locked_until is not None:
            if now < failure.locked_until:
                raise LoginFailed(locked=True, retry_after_seconds=failure.locked_until - now)
            # Lockout window elapsed: clear it so this attempt is judged on
            # its own merits.
            self._failures.pop(username, None)

        user = self._users.get(username)
        ok = False
        if user is not None:
            candidate = hash_password(password, salt=bytes.fromhex(user.salt_hex), iterations=user.iterations)
            ok = hmac.compare_digest(candidate, bytes.fromhex(user.hash_hex))
        if not ok:
            self._record_failure(username, now)
            raise LoginFailed(locked=False)

        self._failures.pop(username, None)
        token = secrets.token_urlsafe(32)
        assert user is not None  # ok is True only when user is not None
        self._sessions[token] = _Session(
            username=user.username, role=user.role, tenant=user.tenant,
            expires_at=now + self.session_ttl_seconds,
        )
        return token

    def _record_failure(self, username: str, now: float) -> None:
        failure = self._failures.get(username) or _FailureState()
        failure.count += 1
        if failure.count >= self._max_failures:
            failure.locked_until = now + self._lockout_seconds
            failure.count = 0
        self._failures[username] = failure

    def logout(self, token: str) -> None:
        self._sessions.pop(token, None)

    # -- session lookup --------------------------------------------------

    def session_for(self, token: str | None) -> Principal | None:
        if not token:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        if self._clock() >= session.expires_at:
            self._sessions.pop(token, None)
            return None
        return Principal(username=session.username, role=session.role, tenant=session.tenant)


# ---------------------------------------------------------------------------
# The permission matrix. ONE table, read by app.py's PolicyRoute wrapper on
# every request and by tests/test_auth.py's route-coverage test. A value of
# PUBLIC means no session is required at all; any other value is the
# frozenset of roles allowed to call that (method, path) pair. A route with
# no entry here default-denies (401 unauthenticated / 403 authenticated) --
# see app.py's _PolicyRoute -- so a route added without updating this table
# fails closed at runtime AND fails the coverage test.
#
# Path strings are FastAPI's own route-template spelling (including the
# ":path" converter on segments that hold real ids with "/" in them, e.g.
# EnronQA chunk ids like "allen-p/all_documents/423.") -- copy them out of
# app.py's @app.get/@app.post decorators exactly, not out of the OpenAPI doc.
# ---------------------------------------------------------------------------

PUBLIC = "public"

POLICY: dict[tuple[str, str], "str | frozenset[str]"] = {
    ("GET", "/api/meta"): PUBLIC,  # the DEMO MODE banner needs this before login
    ("POST", "/api/login"): PUBLIC,
    ("POST", "/api/logout"): ALL_ROLES,
    ("GET", "/api/me"): ALL_ROLES,
    ("POST", "/api/ask"): frozenset({"employee"}),
    ("GET", "/api/tenants"): frozenset({"dbmanager", "securityhead", "ceo"}),
    ("GET", "/api/quarantine"): frozenset({"dbmanager", "securityhead", "ceo"}),
    ("POST", "/api/quarantine/{item_id:path}/release"): frozenset({"securityhead"}),
    ("POST", "/api/probe"): frozenset({"securityhead"}),
    ("GET", "/api/trace/{chunk_id:path}"): frozenset({"employee", "dbmanager", "securityhead"}),
    ("GET", "/api/results"): frozenset({"securityhead", "ceo"}),
}


def permissions_for(role: Role) -> list[str]:
    """The list of ``"METHOD /api/path"`` strings a role may call, derived
    straight from ``POLICY`` -- the exact same table that enforces access, so
    ``GET /api/me``'s ``permissions`` field can never drift from what the
    server actually allows."""
    perms = [
        f"{method} {path}"
        for (method, path), entry in POLICY.items()
        if entry != PUBLIC and role in entry
    ]
    return sorted(perms)
