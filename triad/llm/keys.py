"""Load Groq API keys from a secrets file or environment, never echoing them.

Keys sharing an ``org`` label share one rate-limit bucket (Groq limits are
per-organization: https://console.groq.com/docs/rate-limits). A key with no
``org`` is its own bucket, so unrelated free accounts don't get throttled
together by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from triad import config

_PLACEHOLDER = "PASTE_KEY_HERE"


class KeysMissing(RuntimeError):
    """Raised when no usable API key was found anywhere."""


def _mask(value: str) -> str:
    """Show only the last 4 characters; never enough to reconstruct the key.
    Uses plain ASCII (not an ellipsis glyph) so it renders correctly on a
    cp1252 Windows console, not just UTF-8 terminals."""
    if len(value) <= 4:
        return "gsk_..." + "*" * len(value)
    return "gsk_..." + value[-4:]


@dataclass(frozen=True, repr=False)
class ApiKey:
    """A Groq API key. ``repr``/``str`` MUST mask the value: these objects end up
    in logs, error messages and test failure output, and a leaked key here would
    be a real credential leak, not just a bug."""

    value: str
    label: str | None = None
    org: str | None = None

    def __post_init__(self) -> None:
        if not self.value.startswith("gsk_"):
            raise ValueError("malformed key: must start with 'gsk_' (value withheld)")

    @property
    def bucket(self) -> str:
        """Rate-limit bucket id: keys sharing an org share one bucket, otherwise
        each key is its own bucket (identified by its masked tail, not the raw value
        -- the bucket id itself must be safe to log)."""
        return f"org:{self.org}" if self.org else f"key:{_mask(self.value)}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ApiKey({_mask(self.value)}, label={self.label!r}, org={self.org!r})"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return _mask(self.value)


def _parse_line(line: str, *, lineno: int, source: str) -> ApiKey | None:
    """Parse one key line. Returns None for a comment/placeholder/blank line.
    Raises ValueError (line number only, never content) for a malformed key."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if _PLACEHOLDER in stripped:
        return None

    parts = stripped.split()
    raw = parts[0]
    label: str | None = None
    org: str | None = None
    for tok in parts[1:]:
        if tok.startswith("org="):
            org = tok[len("org="):] or None
        elif label is None:
            label = tok

    if not raw.startswith("gsk_"):
        raise ValueError(f"malformed key at {source} line {lineno}: does not start with 'gsk_'")
    return ApiKey(value=raw, label=label, org=org)


def _parse_env_entry(entry: str, *, index: int) -> ApiKey | None:
    """Same grammar as a file line, so GROQ_API_KEYS entries can carry
    'gsk_xxx label org=foo' too, just comma-separated instead of newline-separated."""
    return _parse_line(entry, lineno=index, source="GROQ_API_KEYS")


def load_keys(path: Path = config.KEYS_FILE, env: "os._Environ[str] | dict[str, str]" = os.environ) -> tuple[ApiKey, ...]:
    """Load keys from ``env['GROQ_API_KEYS']`` (comma-separated) if non-empty,
    otherwise from the file at ``path``. Never logs or returns file contents on
    failure -- only the path, so a missing/empty file can be diagnosed without
    ever printing what might be inside it.
    """
    env_value = (env.get("GROQ_API_KEYS") or "").strip()
    keys: list[ApiKey] = []

    if env_value:
        for i, entry in enumerate(env_value.split(","), start=1):
            parsed = _parse_env_entry(entry, index=i)
            if parsed is not None:
                keys.append(parsed)
    else:
        p = Path(path)
        if not p.exists():
            raise KeysMissing(
                f"no Groq API keys found: {p} does not exist. "
                f"Paste one or more keys (one per line, starting with gsk_) into that file."
            )
        with p.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                parsed = _parse_line(line, lineno=lineno, source=str(p))
                if parsed is not None:
                    keys.append(parsed)

    if not keys:
        raise KeysMissing(
            f"no usable Groq API keys found in {path} (or GROQ_API_KEYS). "
            f"Paste one or more keys (one per line, starting with gsk_) into that file."
        )
    return tuple(keys)
