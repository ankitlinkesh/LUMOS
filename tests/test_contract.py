import pytest

from triad.contract import Chunk, GuardDecision, Provenance, RetrievalResult, ScoredChunk, TaintVerdict


def chunk(cid="c1", tenant="alice", source="real"):
    return Chunk(id=cid, text="hello", tenant=tenant, source_type="email",
                 provenance=Provenance("enronqa", cid, source))


def test_provenance_rejects_unknown_data_source():
    with pytest.raises(ValueError):
        Provenance("enronqa", "x", "made_up")


def test_chunk_must_have_a_tenant():
    with pytest.raises(ValueError):
        chunk(tenant="  ")


def test_retrieved_content_is_untrusted_by_default():
    assert chunk().taint.untrusted is True and chunk().taint.quarantined is False


def test_declined_retrieval_must_explain_itself():
    with pytest.raises(ValueError):
        RetrievalResult(query="q", tenant="alice", chunks=(), declined=True)


def test_foreign_chunks_detects_cross_tenant_results():
    mine, theirs = ScoredChunk(chunk("a", "alice"), 0.9), ScoredChunk(chunk("b", "bob"), 0.8)
    r = RetrievalResult(query="q", tenant="alice", chunks=(mine, theirs), scope_applied=("alice",))
    assert r.foreign_chunks() == (theirs,)


def test_foreign_chunks_defaults_scope_to_the_caller():
    r = RetrievalResult(query="q", tenant="alice", chunks=(ScoredChunk(chunk("b", "bob"), 0.8),))
    assert len(r.foreign_chunks()) == 1


def test_guard_decision_cannot_allow_and_escalate():
    with pytest.raises(ValueError):
        GuardDecision(stage="egress", allow=True, escalate=True)


def test_blocking_needs_a_reason():
    with pytest.raises(ValueError):
        GuardDecision(stage="ingest", allow=False)
    d = GuardDecision.block("ingest", "hidden instruction found")
    assert not d.allow and d.reasons == ("hidden instruction found",)


def test_needs_human_escalates_without_allowing():
    d = GuardDecision.needs_human("egress", "tool call motivated only by retrieved text")
    assert d.escalate and not d.allow


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        GuardDecision.ok("somewhere")


def test_taint_defaults_are_immutable():
    t = TaintVerdict()
    with pytest.raises(Exception):
        t.quarantined = True  # frozen
