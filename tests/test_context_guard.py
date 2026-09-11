from triad.context_guard import ContextGuard, UserContext
from triad.contract import Chunk, Provenance, ScoredChunk, TaintVerdict


def scored(cid, tenant, text, *, metadata=None, quarantined=False):
    return ScoredChunk(
        Chunk(
            id=cid,
            text=text,
            tenant=tenant,
            source_type="document",
            provenance=Provenance("test", cid, "synthetic"),
            metadata=metadata or {},
            taint=TaintVerdict(quarantined=quarantined),
        ),
        1.0,
    )


def test_context_guard_blocks_foreign_and_quarantined_chunks():
    result = ContextGuard().guard(
        UserContext("acme"),
        [scored("own", "acme", "safe"), scored("foreign", "other", "secret"),
         scored("held", "acme", "held", quarantined=True)],
    )
    assert [x.chunk.id for x in result.prompt_chunks] == ["own"]
    assert result.blocked == ("foreign", "held")
    assert [d.reasons for d in result.decisions[1:]] == [("tenant-isolation",), ("quarantined-content",)]


def test_context_guard_redacts_secrets_without_redacting_normal_email_by_default():
    result = ContextGuard().guard(
        UserContext("acme"),
        [scored("c1", "acme", "Contact alice@example.com with sk-live-ABCDEF1234567890")],
    )
    assert result.blocked == ()
    assert len(result.redacted) == 1
    assert "sk-live-ABCDEF1234567890" not in result.redacted[0].chunk.text
    assert "alice@example.com" in result.redacted[0].chunk.text
    assert result.decisions[0].action == "redact"


def test_context_guard_can_redact_email_when_policy_requests_it():
    result = ContextGuard(redact_email=True).guard(
        UserContext("acme"), [scored("c1", "acme", "Contact alice@example.com")]
    )
    assert "alice@example.com" not in result.redacted[0].chunk.text
