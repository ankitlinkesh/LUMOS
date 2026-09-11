"""Regenerate ``triad/api/demo_users.json``.

Run it whenever the demo password list below changes, or if you just want a
fresh set of random salts (the passwords are unaffected by re-running this --
only the stored salt/hash bytes change):

    .venv/Scripts/python.exe scripts/gen_demo_users.py

The passwords here are the ONLY place they exist in this repo outside the
README's "Demo credentials" section -- keep the two in sync. These are demo
credentials for a hackathon judge to log in with; they are not secrets in the
``secrets/`` sense (no production system depends on them), but this script
still never prints a password to stdout other than in the summary table,
and the committed JSON only ever holds PBKDF2 hashes, never plaintext.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from triad.api.auth import PBKDF2_ITERATIONS_DEFAULT, hash_password

OUT_PATH = Path(__file__).resolve().parents[1] / "triad" / "api" / "demo_users.json"

# username -> (role, tenant, password). tenant is None for the three
# non-employee roles. Employee usernames double as their bound tenant id
# (see the README's "Roles and access control" section) -- these six are
# EnronQA's own real tenant ids, the same ones Pipeline.demo() ingests, so
# the accounts work unmodified against both the fake and --real services.
_ACCOUNTS: dict[str, tuple[str, str | None, str]] = {
    "allen-p": ("employee", "allen-p", "AllenDemo!2026"),
    "arnold-j": ("employee", "arnold-j", "ArnoldDemo!2026"),
    "arora-h": ("employee", "arora-h", "AroraDemo!2026"),
    "badeer-r": ("employee", "badeer-r", "BadeerDemo!2026"),
    "bailey-s": ("employee", "bailey-s", "BaileyDemo!2026"),
    "bass-e": ("employee", "bass-e", "BassDemo!2026"),
    "dbmanager": ("dbmanager", None, "DbManagerDemo!2026"),
    "securityhead": ("securityhead", None, "SecurityHeadDemo!2026"),
    "ceo": ("ceo", None, "CeoDemo!2026"),
}


def main() -> None:
    users = []
    for username, (role, tenant, password) in _ACCOUNTS.items():
        salt = secrets.token_bytes(16)
        digest = hash_password(password, salt=salt, iterations=PBKDF2_ITERATIONS_DEFAULT)
        row = {
            "username": username,
            "role": role,
            "salt": salt.hex(),
            "hash": digest.hex(),
            "iterations": PBKDF2_ITERATIONS_DEFAULT,
        }
        if tenant is not None:
            row["tenant"] = tenant
        users.append(row)

    OUT_PATH.write_text(
        json.dumps({"pbkdf2_iterations": PBKDF2_ITERATIONS_DEFAULT, "users": users}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(users)} users to {OUT_PATH}")
    print("passwords are documented in README.md's 'Demo credentials' section -- not printed here.")


if __name__ == "__main__":
    main()
