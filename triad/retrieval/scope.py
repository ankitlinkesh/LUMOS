"""The tenant set a search is allowed to touch.

``narrow`` only ever intersects. The author's previous project (N.O.V.A) had a
real containment escape from nesting scopes by *replacement*: a sub-task
inherited a fresh scope instead of a restriction of its parent's, and regained
permissions the parent never had. The fix there -- and the rule encoded here --
is a most-restrictive-wins stack: narrowing a scope can only ever shrink it,
never hand back tenants the caller didn't already have.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    """An immutable set of tenants a search may see. An empty scope is a valid,
    deliberate state: it searches nothing (fail closed), not "everything"."""

    tenants: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.tenants, frozenset):
            object.__setattr__(self, "tenants", frozenset(self.tenants))

    @classmethod
    def of(cls, *tenants: str) -> "Scope":
        return cls(frozenset(tenants))

    def narrow(self, other: "Scope") -> "Scope":
        """Intersect with ``other``. Never widens: the result's tenants are always
        a subset of both inputs, so a child scope can never regain a tenant its
        parent excluded."""
        return Scope(self.tenants & other.tenants)

    def __contains__(self, tenant: str) -> bool:
        return tenant in self.tenants

    def __bool__(self) -> bool:
        return bool(self.tenants)

    def __len__(self) -> int:
        return len(self.tenants)
