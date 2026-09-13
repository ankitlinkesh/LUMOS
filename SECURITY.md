# Security Policy

## This is research-grade software, not a hardened product

TRIAD-RAG is an early-stage (`0.1.0`) research project measuring RAG-specific
attacks and defenses. It has not had a security audit, its detection rates are
well short of 100% on several measured attack classes (see the README's
"Measured results" and "Scope and limits" sections — the numbers are reported
honestly, including the ones that are bad), and `triad/api/` is an
integration-contract HTTP surface, not an authorization system: it trusts its
caller to bind a tenant to its own authenticated session (see
`triad/api/app.py`'s module docstring). Do not deploy it as-is in front of
untrusted users, and do not treat a clean pytest run as evidence of security —
several of the bugs described in this project's own history were found by
driving the system live, not by tests.

## Reporting a vulnerability

If you find a security issue in this codebase — a way to defeat a claimed
invariant (e.g. widen retrieval scope across tenants, get untrusted content to
authorize a tool call or cause an egress without escalation), a flaw in the
detection logic, or a bug in the HTTP service beyond what's already documented
in the README's "Scope and limits" section — please report it privately rather
than opening a public issue:

- Preferred: open a [GitHub Security Advisory](https://github.com/ankitlinkesh/LUMOS/security/advisories/new)
  on this repository. This notifies maintainers privately and gives you a
  private channel to share details, including a proof-of-concept.
- If that's not available to you, open a regular issue asking for a private
  contact channel, without including exploit details in the issue itself.

Please include: what invariant or behavior is violated, a minimal reproduction
(a script or a `pytest` case is ideal, since this project already has the
fixtures for that), and which commit you tested against.

There is no bug bounty. Response time is best-effort — this is a personal
research project, not a maintained product with an SLA.

## Known, disclosed limitations

The README's "Scope and limits" section documents every attack class and
condition this project is currently known to miss or handle poorly (e.g.
Stage 1A scoring 0% on BIPIA-style bare-imperative injections, `batch_cluster`
needing poison to arrive in one ingestion batch, no per-component ablation for
the combined defense). Reporting one of those specific, already-documented
gaps is still welcome if you have a concrete new finding, a fix, or a
reproduction that sharpens the existing caveat — but check that section first
so reports focus on genuinely new findings.
