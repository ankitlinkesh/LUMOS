"""Content-addressed disk cache for Groq chat completions.

The cache key MUST include the ``principal`` (tenant): two tenants asking the
exact same question must never share an answer, because serving tenant A's
cached response to tenant B would itself be a cross-tenant leak -- the same
invariant Stage 2 (retrieval) enforces, just at the LLM layer. The API key
must NEVER appear in the key or the stored file: the key is a credential
naming *who is paying*, not part of *what was asked*, and it must never leak
into a cache file that could be inspected or shipped around.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from triad import config

_RAW_KEY_RE = re.compile(r"gsk_[A-Za-z0-9]{16,}")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    *,
    principal: str,
    scope: Sequence[str],
    model: str,
    messages: Sequence[Mapping[str, str]],
    params: Mapping[str, Any],
) -> str:
    """sha256 of the canonical JSON of the full request shape. ``params`` must be
    the FULLY RESOLVED param dict (including any defaults the client injects,
    e.g. ``reasoning_effort``) -- hashing before resolving defaults would let a
    request with an explicit override collide with one that got the default."""
    payload = {
        "principal": principal,
        "scope": sorted(scope),
        "model": model,
        "messages": [dict(m) for m in messages],
        "params": dict(params),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    text: str
    model: str
    usage: Mapping[str, Any]
    created_at: float
    key_label: str | None  # masked label only, e.g. "gsk_...abcd" -- never the raw key


class DiskCache:
    """A dumb, atomic, content-addressed KV store on disk. One file per key."""

    def __init__(self, directory: Path = config.LLM_CACHE_DIR / "responses"):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None  # a corrupt cache entry is a miss, not a crash
        try:
            return CacheEntry(
                text=raw["text"],
                model=raw["model"],
                usage=raw.get("usage", {}),
                created_at=raw.get("created_at", 0.0),
                key_label=raw.get("key_label"),
            )
        except KeyError:
            return None

    def put(self, key: str, entry: CacheEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": entry.text,
            "model": entry.model,
            "usage": dict(entry.usage),
            "created_at": entry.created_at,
            "key_label": entry.key_label,
        }
        assert not _RAW_KEY_RE.search(_canonical(payload)), \
            "refusing to write a raw-looking key into the cache (only a masked label is allowed)"
        path = self._path(key)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)  # atomic, same directory as target (Windows-safe)
