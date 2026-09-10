"""Scope: most-restrictive-wins. ``narrow`` intersects, never widens -- the
containment property a previous project's real bug (nesting by replacement)
violated."""

from __future__ import annotations

from triad.retrieval.scope import Scope


def test_narrow_intersects():
    a = Scope.of("alice", "bob", "carol")
    b = Scope.of("bob", "carol", "dave")
    assert a.narrow(b) == Scope.of("bob", "carol")


def test_narrow_never_widens_even_with_disjoint_sets():
    a = Scope.of("alice")
    b = Scope.of("bob")
    result = a.narrow(b)
    assert result.tenants == frozenset()
    assert len(result) == 0


def test_narrow_of_empty_scope_stays_empty():
    empty = Scope(frozenset())
    full = Scope.of("alice", "bob")
    assert empty.narrow(full) == empty
    assert full.narrow(empty) == empty


def test_narrowing_never_regains_a_tenant_the_parent_excluded():
    """The exact containment escape this type is designed to prevent: a child
    scope built by narrowing must never end up with a tenant absent from the
    parent, no matter what the "other" side of narrow contains."""
    parent = Scope.of("alice")
    attacker_supplied = Scope.of("alice", "bob", "eve")  # tries to add bob/eve
    child = parent.narrow(attacker_supplied)
    assert child.tenants <= parent.tenants
    assert "bob" not in child and "eve" not in child


def test_membership_and_truthiness():
    s = Scope.of("alice")
    assert "alice" in s
    assert "bob" not in s
    assert bool(s) is True
    assert bool(Scope(frozenset())) is False


def test_accepts_non_frozenset_iterable_in_constructor():
    s = Scope(["alice", "alice", "bob"])
    assert s.tenants == frozenset({"alice", "bob"})
