"""Fast tests against tiny hand-built fixtures -- never the real (5GB) data files.
Real-data assertions live in test_data_real.py, marked `slow`.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from triad.data import bipia, enronqa, llmail, poisonedrag
from triad.data.errors import DataUnavailable

# --------------------------------------------------------------------------- enronqa

_ENRON_SCHEMA = pa.schema([
    ("email", pa.string()),
    ("questions", pa.list_(pa.string())),
    ("gold_answers", pa.list_(pa.string())),
    ("incorrect_answers", pa.list_(pa.list_(pa.string()))),
    ("path", pa.string()),
    ("user", pa.string()),
])


def _write_split(path, rows):
    table = pa.Table.from_pylist(rows, schema=_ENRON_SCHEMA)
    pq.write_table(table, path)


def _row(path, user, email="body text", questions=("q1",), gold=("g1",), incorrect=(("bad1", "bad2"),)):
    return {"email": email, "questions": list(questions), "gold_answers": list(gold),
            "incorrect_answers": [list(x) for x in incorrect], "path": path, "user": user}


def test_load_emails_dedupes_by_path_across_splits(tmp_path):
    # The measured trap: the SAME email (same path) appears in all three splits
    # because splits are over questions, not emails.
    shared = _row("alice/inbox/1.", "alice", questions=("q-train",))
    _write_split(tmp_path / "train-00000-of-00001.parquet", [shared])
    _write_split(tmp_path / "dev-00000-of-00001.parquet",
                 [_row("alice/inbox/1.", "alice", questions=("q-dev",))])
    _write_split(tmp_path / "test-00000-of-00001.parquet",
                 [_row("alice/inbox/1.", "alice", questions=("q-test",))])

    emails = enronqa.load_emails(root=tmp_path)
    assert len(emails) == 1
    assert emails[0].id == "alice/inbox/1."
    assert emails[0].tenant == "alice"
    assert emails[0].provenance.data_source == "real"


def test_load_emails_filters_by_user_and_respects_limit(tmp_path):
    _write_split(tmp_path / "train-00000-of-00001.parquet", [
        _row("a/1.", "alice"), _row("b/1.", "bob"), _row("a/2.", "alice"),
    ])
    only_alice = enronqa.load_emails(root=tmp_path, users=["alice"])
    assert {c.tenant for c in only_alice} == {"alice"}
    limited = enronqa.load_emails(root=tmp_path, limit=1)
    assert len(limited) == 1


def test_load_emails_no_files_raises_data_unavailable(tmp_path):
    with pytest.raises(DataUnavailable):
        enronqa.load_emails(root=tmp_path / "does-not-exist")


def test_load_qa_train_split_reads_both_shards(tmp_path):
    # train is split across TWO files (train-00000-of-00002, train-00001-of-00002) --
    # a glob of just the first file would silently drop half the questions.
    _write_split(tmp_path / "train-00000-of-00002.parquet",
                 [_row("a/1.", "alice", questions=("q1",), gold=("g1",), incorrect=(("bad1",),))])
    _write_split(tmp_path / "train-00001-of-00002.parquet",
                 [_row("b/1.", "bob", questions=("q2",), gold=("g2",), incorrect=(("bad2",),))])
    qa = enronqa.load_qa(root=tmp_path, split="train")
    assert {r.question for r in qa} == {"q1", "q2"}


def test_load_qa_flattens_incorrect_answers_per_question_not_globally(tmp_path):
    # Two questions in one row; each has its OWN incorrect_answers list. Flattening
    # globally would let question 1's incorrect answers leak onto question 2.
    _write_split(tmp_path / "test-00000-of-00001.parquet", [_row(
        "a/1.", "alice",
        questions=("what color", "what year"),
        gold=("blue", "1999"),
        incorrect=(("red", "green"), ("2000",)),
    )])
    qa = enronqa.load_qa(root=tmp_path, split="test")
    by_q = {r.question: r for r in qa}
    assert by_q["what color"].incorrect_answers == ("red", "green")
    assert by_q["what year"].incorrect_answers == ("2000",)
    assert by_q["what color"].gold_answers == ("blue",)
    assert by_q["what color"].tenant == "alice"
    assert by_q["what color"].email_path == "a/1."


def test_load_qa_unknown_split_raises(tmp_path):
    with pytest.raises(DataUnavailable):
        enronqa.load_qa(root=tmp_path, split="bogus")


# ------------------------------------------------------------------------ poisonedrag

def test_poisonedrag_prefix_matches_attack_py_recipe_exactly(tmp_path):
    # attack.py:92-93 (LM_targeted): adv_text_a = question + "."; adv_texts = [adv_text_a + i for i in adv_texts_b]
    # Bare concatenation -- no space between the prefix and the released text.
    fixture = {
        "test1": {
            "id": "test1",
            "question": "how many episodes are in season 4",
            "correct answer": "23",
            "incorrect answer": "24",
            "adv_texts": ["Season 4 had 24 episodes.", "There were 24 episodes total."],
        }
    }
    (tmp_path / "nq.json").write_text(json.dumps(fixture), encoding="utf-8")
    targets = poisonedrag.load_targets(root=tmp_path, corpus="nq")
    assert len(targets) == 1
    t = targets[0]
    assert t.raw_adv_texts == ("Season 4 had 24 episodes.", "There were 24 episodes total.")
    expected_prefix = "how many episodes are in season 4."
    assert t.adv_texts[0] == expected_prefix + "Season 4 had 24 episodes."
    assert t.adv_texts[1] == expected_prefix + "There were 24 episodes total."
    # No space was inserted: prefix immediately followed by the raw text's first char.
    assert not t.adv_texts[0].startswith(expected_prefix + " ")
    assert t.provenance.data_source == "real"
    assert t.provenance.dataset == "poisonedrag:nq"


def test_poisonedrag_missing_file_raises(tmp_path):
    with pytest.raises(DataUnavailable):
        poisonedrag.load_targets(root=tmp_path, corpus="nq")


def test_poisonedrag_unknown_corpus_raises(tmp_path):
    (tmp_path / "nq.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataUnavailable):
        poisonedrag.load_targets(root=tmp_path, corpus="bogus")


# ---------------------------------------------------------------------------- llmail

def _obj(retrieved=True, undetected=True, sent=True, dest=True, content=True):
    return json.dumps({
        "email.retrieved": retrieved, "defense.undetected": undetected,
        "exfil.sent": sent, "exfil.destination": dest, "exfil.content": content,
    })


def test_llmail_parses_objectives_json_string_and_filters_all_met(tmp_path):
    lines = [
        {"RowKey": "1", "body": "full success", "subject": "s1", "scenario": "level1", "objectives": _obj()},
        {"RowKey": "2", "body": "partial", "subject": "s2", "scenario": "level1",
         "objectives": _obj(sent=False)},
        {"RowKey": "3", "body": "no objectives field", "subject": "s3", "scenario": "level1"},
    ]
    path = tmp_path / "raw_submissions_phase1.jsonl"
    path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")

    all_met = llmail.load_attacks(root=tmp_path, phase=1, all_objectives_only=True)
    assert len(all_met) == 1
    assert all_met[0].body == "full success"
    assert all_met[0].all_objectives_met()

    every_row = llmail.load_attacks(root=tmp_path, phase=1, all_objectives_only=False)
    assert len(every_row) == 3


def test_llmail_dedupes_by_body(tmp_path):
    lines = [
        {"RowKey": "1", "body": "same body", "subject": "s1", "scenario": "level1", "objectives": _obj()},
        {"RowKey": "2", "body": "same body", "subject": "s2", "scenario": "level1", "objectives": _obj()},
    ]
    path = tmp_path / "raw_submissions_phase1.jsonl"
    path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")

    deduped = llmail.load_attacks(root=tmp_path, phase=1, all_objectives_only=True, dedupe=True)
    assert len(deduped) == 1
    not_deduped = llmail.load_attacks(root=tmp_path, phase=1, all_objectives_only=True, dedupe=False)
    assert len(not_deduped) == 2


def test_llmail_missing_file_raises(tmp_path):
    with pytest.raises(DataUnavailable):
        llmail.load_attacks(root=tmp_path, phase=1)


def test_llmail_load_benign(tmp_path):
    data = ["Subject of the email: Weekly update.   Body: nothing new."]
    (tmp_path / "emails_for_fp_tests.json").write_text(json.dumps(data), encoding="utf-8")
    benign = llmail.load_benign(root=tmp_path)
    assert len(benign) == 1
    assert benign[0].tenant == "public"
    assert benign[0].provenance.data_source == "real"


# ----------------------------------------------------------------------------- bipia

def test_bipia_loads_usable_task(tmp_path):
    d = tmp_path / "email"
    d.mkdir()
    row = {"context": "SUBJECT: hi|CONTENT: body text", "question": "what is this about?", "ideal": "unknown"}
    (d / "test.jsonl").write_text(json.dumps(row), encoding="utf-8")
    records = bipia.load("email", root=tmp_path)
    assert len(records) == 1
    assert records[0].task == "email"
    assert records[0].provenance.data_source == "real"


@pytest.mark.parametrize("task", ["qa", "abstract"])
def test_bipia_unavailable_tasks_report_precisely(tmp_path, task):
    with pytest.raises(DataUnavailable, match="newsqa|XSum"):
        bipia.load(task, root=tmp_path)
