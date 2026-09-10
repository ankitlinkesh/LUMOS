"""Loaders for TRIAD-RAG's evaluation datasets.

Every loader returns the frozen contract types (`triad.contract.Chunk`) or the
eval-only record types in `triad.data.types`. Real loaders raise
`triad.data.errors.DataUnavailable` on any missing/malformed input; nothing here
catches a bare `Exception` — see `triad.data.registry` for why.
"""
