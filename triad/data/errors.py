"""The one exception every real loader raises on failure.

A bare ``except Exception`` in the registry would let a real bug (a typo'd column
name, a broken parser) get silently reported as "dataset unavailable, fell back to
synthetic" -- the exact class of bug N.O.V.A's `llm doctor` and openWakeWord
postmortems both trace to a swallowed exception. Loaders raise this ONE type for
"I could not produce real records"; the registry catches only this (plus `OSError`
for the filesystem edge it doesn't already cover) and lets everything else propagate
as a loud crash.
"""

from __future__ import annotations


class DataUnavailable(Exception):
    """A real dataset loader could not produce records: missing file, empty result,
    or a row that violates the dataset's own documented shape."""
